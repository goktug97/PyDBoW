import cv2
import numpy as np
from scipy import spatial
import struct
import orb_descriptor
from typing import List
import random

def initialize_clusters(
        descriptors : List[orb_descriptor.ORB],
        n_clusters: int
        ) -> List[orb_descriptor.ORB]:
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
        descriptors: List[orb_descriptor.ORB], k=3) -> List[List[int]]:
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

if __name__ == '__main__':
    n_clusters = 3
    cap = cv2.VideoCapture(0)

    for i in range(20):
        cap.read()

    ret, frame = cap.read()
    orb = cv2.ORB_create()
    kps, des = orb.detectAndCompute(frame, None)
    binary_descriptors : List[orb_descriptor.ORB] = []
    for idx in range(len(des)):
        binary_descriptors.append(orb_descriptor.ORB(orb_descriptor.to_binary(des[idx])))

    if len(binary_descriptors) <= n_clusters:
        groups = reserve_groups(len(binary_descriptors))
        for i in range(len(binary_descriptors)):
            groups[i].append(i)
    else:
        groups = binary_kmeans(binary_descriptors, n_clusters)

    # for group in groups:
    #     binary_kmeans(orb_descriptor.ORB(binary_descriptors)[group])
