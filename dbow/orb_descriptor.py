import struct
from typing import List

import numpy as np


class ORB:
    def __init__(self, features):
        self.features = np.array(features).astype("int")

    def distance(self, other):
        """Calculate hamming distance between two descriptors."""
        return int(np.sum(np.logical_xor(self, other)))

    @classmethod
    def from_cv_descriptor(cls, cv_descriptor):
        features = np.array(
            list(
                format(
                    int.from_bytes(struct.pack("B" * 32, *cv_descriptor), "big"),
                    "0256b",
                )
            )
        )
        return cls(features)

    def __add__(self, other):
        return ORB(self.features + other.features)

    def __radd__(self, other):
        return self.__add__(other)

    def logical_xor(self, other):
        return np.logical_xor(self.features, other.features)

    def __eq__(self, other):
        return np.allclose(self.features, other.features)


def mean_value(descriptors):
    """Calculate mean of list of ORB Descriptors."""
    N2 = len(descriptors) / 2
    counters = np.sum(descriptors)
    mean = np.where(counters.features >= N2, 1, 0)
    return ORB(mean)
