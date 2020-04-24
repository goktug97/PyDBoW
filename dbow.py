import cv2
import numpy as np
import orb_descriptor
from typing import List
import random
import glob
import os

class Node():
    def __init__(self, descriptors):
        self.descriptors = np.array(descriptors)
        self.child_nodes = []

class Word(Node):
    def __init__(self, descriptors):
        self.weight = 0.0
        super().__init__(descriptors)

    @classmethod
    def from_node(cls, node):
        return cls(node.descriptors)

    def update_weight(self, n_images):
        self.weight = np.log(n_images/len(self.descriptors))


class Vocabulary():
    def __init__(self, images_path, n_clusters, depth):
        images_path = glob.glob(os.path.join(images_path, '*.png'))
        images = []
        for image_path in images_path:
            images.append(cv2.imread(image_path))
        orb = cv2.ORB_create()
        descriptors = []
        for image in images:
            kps, descs = orb.detectAndCompute(image, None)
            for desc in descs:
                descriptors.append(orb_descriptor.ORB.from_cv_descriptor(desc))
        descriptors = np.array(descriptors)
        self.root_node = Node(descriptors)
        words = initialize_tree(self.root_node, n_clusters, depth)
        self.words = [Word.from_node(node) for node in words]
        [word.update_weight(len(images)) for word in self.words]

def initialize_clusters(
        descriptors : List[orb_descriptor.ORB],
        n_clusters: int
        ) -> List[List[orb_descriptor.ORB]]:
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
            cut_distance = random.uniform(0, dist_sum)

        cum_distance = np.cumsum(distances)
        comparison = cum_distance >= cut_distance
        if not np.sum(comparison):
            idx = -1
        else:
            for idx, x in enumerate(comparison):
                if x: break

        clusters.append(descriptors[idx])

    return clusters

def reserve_groups(n: int) -> List[List[int]]:
    alist: List[List[int]] = list()
    for i in range(n):
        alist.append(list())
    return alist

def binary_kmeans(
        descriptors: List[orb_descriptor.ORB], k: int) -> List[List[int]]:
    first_run = True
    while True:
        if first_run:
            clusters = initialize_clusters(descriptors, k)
        else:
            for cidx in range(len(clusters)):
                cluster_descriptors = []
                for didx in groups[cidx]:
                    cluster_descriptors.append(descriptors[didx])
                clusters[cidx] = orb_descriptor.mean_value(cluster_descriptors)
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


def initialize_tree(root, n_clusters, tree_depth):
    words = []
    def _initialize_tree(node, depth=0):
        if depth == tree_depth:
            words.append(node)
            return
        if len(node.descriptors) <= n_clusters:
            for desc in node.descriptors:
                child_node = Node([desc])
                node.child_nodes.append(child_node)
                _initialize_tree(child_node, depth+1)
        else:
            groups = binary_kmeans(node.descriptors, n_clusters)
            for group in groups:
                child_node = Node(node.descriptors[group])
                node.child_nodes.append(child_node)
                _initialize_tree(child_node, depth+1)
    _initialize_tree(root)
    return words
