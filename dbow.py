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

def list_of_n(n):
    alist = list()
    for i in range(n):
        alist.append(list())
    return alist

def binary_kmeans(descriptors: List[orb_descriptor.ORB], k=3):
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
        current_association = []
        groups = list_of_n(len(clusters))
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
        last_association = current_association
    return groups

cap = cv2.VideoCapture(0)

for i in range(10):
    cap.read()

# size = 128
ret, frame = cap.read()
# x, y, w, h = cv2.selectROI(frame)
# xmin, ymin, xmax, ymax = x, y, x+w, y+h
# patch = frame[ymin:ymax, xmin:xmax, :]
# patch = cv2.resize(patch, (size, size))

orb = cv2.ORB_create()

# h, w = frame.shape[0:2]
# mask = np.zeros((h, w), dtype=np.uint8)
# mask[max(0, ymin):min(ymax, h), max(0, xmin):min(xmax, w)] = 255

kps, des = orb.detectAndCompute(frame, None)
binary_descriptors = []
for idx in range(len(des)):
    binary_descriptors.append(orb_descriptor.ORB(orb_descriptor.to_binary(des[idx])))

# print(binary_descriptors)
# print(orb_descriptor.mean_value(binary_descriptors))
print(binary_kmeans(binary_descriptors, 3))
