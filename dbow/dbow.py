import glob
import os
import pickle
from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np

from .orb_descriptor import ORB, mean_value


class Node:
    def __init__(self, descriptors: List[ORB]):
        self.descriptors = np.array(descriptors)
        self.child_nodes: List["Node"] = []
        self.idx = None  # Initialized only for the last layer


class Word(Node):
    def __init__(self, descriptors: List[ORB]):
        self.weight: float = 0.0
        super().__init__(descriptors)

    @classmethod
    def from_node(cls, node: Node) -> "Word":
        word = cls(node.descriptors)
        word.idx = node.idx
        return word

    def update_weight(self, n_images: int) -> None:
        self.weight = np.log(n_images / len(self.descriptors))


class BoW:
    def __init__(self, data):
        self.data = np.array(data)

    def score(self, other):
        return 1 - 1 / 2 * np.linalg.norm(
            self.data / np.linalg.norm(self.data)
            - other.data / np.linalg.norm(other.data)
        )


class Vocabulary:
    def __init__(self, images, n_clusters: int, depth: int):
        orb = cv2.ORB_create()
        descriptors = []
        for image in images:
            kps, descs = orb.detectAndCompute(image, None)
            for desc in descs:
                descriptors.append(ORB.from_cv_descriptor(desc))
        descriptors = np.array(descriptors)
        self.root_node = Node(descriptors)
        words = initialize_tree(self.root_node, n_clusters, depth)
        self.words = [Word.from_node(node) for node in words]
        for word in self.words:
            word.update_weight(len(images))

    def query_descriptor(self, descriptor: ORB) -> Word:
        def _traverse_node(node, depth=0):
            min_distance = float("inf")
            if not node.child_nodes:
                return self.words[node.idx]
            for child_node in node.child_nodes:
                distance = descriptor.distance(child_node.descriptors[0])
                if distance < min_distance:
                    closest_node = child_node
                    min_distance = distance
            return _traverse_node(closest_node, depth + 1)

        return _traverse_node(self.root_node)

    def descs_to_bow(self, descriptors: List[ORB]) -> BoW:
        words = []
        for desc in descriptors:
            words.append(self.query_descriptor(desc))
        c = Counter(words)
        n_words = len(c + Counter())
        bow = []
        for word in self.words:
            tf = c[word] / n_words
            tf_idf = tf * word.weight
            bow.append(tf_idf)
        return BoW(bow)

    def image_to_bow(self, image) -> BoW:
        orb = cv2.ORB_create()
        kps, descs = orb.detectAndCompute(image, None)
        descs = [ORB.from_cv_descriptor(desc) for desc in descs]
        return self.descs_to_bow(descs)

    def save(self, path):
        with open(path, "wb") as output:
            pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as vocabulary_file:
            cls_dict = pickle.load(vocabulary_file)
        vocabulary = cls.__new__(cls)
        vocabulary.__dict__.update(cls_dict)
        return vocabulary


class Database:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary
        self.database: List[BoW] = []
        self.descriptors: List[ORB] = []

    def add(self, descriptors):
        bow = self.vocabulary.descs_to_bow(descriptors)
        self.descriptors.append(descriptors)
        self.database.append(bow)

    def query(self, descriptors):
        query_bow = self.vocabulary.descs_to_bow(descriptors)
        scores = []
        for bow in self.database:
            scores.append(query_bow.score(bow))
        return scores

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        return self.database[idx]

    def save(self, path):
        with open(path, "wb") as output:
            pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as database_file:
            cls_dict = pickle.load(database_file)
        database = cls.__new__(cls)
        database.__dict__.update(cls_dict)
        return database


def initialize_clusters(descriptors: List[ORB], n_clusters: int) -> List[List[ORB]]:
    random_idx = np.random.randint(0, len(descriptors))
    clusters = [descriptors[random_idx]]
    distances = []
    for descriptor in descriptors:
        distances.append(descriptor.distance(clusters[-1]))

    while len(clusters) < n_clusters:
        for idx, (descriptor, distance) in enumerate(zip(descriptors, distances)):
            cur_dist = descriptor.distance(clusters[-1])
            if cur_dist < distance:
                distances[idx] = cur_dist
        dist_sum = np.sum(distances)
        cut_distance = 0.0
        while not cut_distance:
            cut_distance = np.random.uniform(0, dist_sum)

        cum_distance = np.cumsum(distances)
        comparison = cum_distance >= cut_distance
        if not np.sum(comparison):
            idx = -1
        else:
            for idx, x in enumerate(comparison):
                if x:
                    break

        clusters.append(descriptors[idx])

    return clusters


def reserve_groups(n: int) -> List[List[int]]:
    alist: List[List[int]] = list()
    for i in range(n):
        alist.append(list())
    return alist


def binary_kmeans(descriptors: List[ORB], k: int) -> List[List[int]]:
    first_run = True
    while True:
        if first_run:
            clusters = initialize_clusters(descriptors, k)
        else:
            for cidx in range(len(clusters)):
                cluster_descriptors = []
                for didx in groups[cidx]:
                    cluster_descriptors.append(descriptors[didx])
                clusters[cidx] = mean_value(cluster_descriptors)
        groups: List[List[int]] = reserve_groups(len(clusters))
        current_association: List[int] = []
        for didx, descriptor in enumerate(descriptors):
            min_dist = descriptor.distance(clusters[0])
            best_cluster = 0
            for cidx in range(1, len(clusters)):
                distance = descriptor.distance(clusters[cidx])
                if distance < min_dist:
                    min_dist = distance
                    best_cluster = cidx
            groups[best_cluster].append(didx)
            current_association.append(best_cluster)
        if first_run:
            first_run = False
        else:
            if last_association == current_association:
                break
        last_association: List[int] = current_association
    return groups


def initialize_tree(root: Node, n_clusters: int, tree_depth: int) -> List[Node]:
    words = []
    idx = 0

    def _initialize_tree(node, depth=0):
        nonlocal idx
        if depth == tree_depth:
            node.idx = idx
            idx += 1
            words.append(node)
            return
        if len(node.descriptors) <= n_clusters:
            for desc in node.descriptors:
                child_node = Node([desc])
                node.child_nodes.append(child_node)
                _initialize_tree(child_node, depth + 1)
        else:
            groups = binary_kmeans(node.descriptors, n_clusters)
            for group in groups:
                child_node = Node(node.descriptors[group])
                node.child_nodes.append(child_node)
                _initialize_tree(child_node, depth + 1)

    _initialize_tree(root)
    return words
