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
print(initialize_clusters(binary_descriptors, 3))
