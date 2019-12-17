import numpy as np
from typing import List
import struct

class ORB(np.ndarray):
    L = 256
    def __new__(cls, desc: np.ndarray):
        obj = np.asarray(desc, dtype=np.uint8).view(cls)
        return obj

    def distance(self, other):
        '''Calculate hamming distance between two descriptors.'''
        return int(np.sum(np.logical_xor(self, other)))

    def __array_finalize__(self, obj):
        if obj is None: return

def mean_value(descriptors: List[ORB]) -> ORB:
    '''Calculate mean of list of ORB Descriptors.'''
    if not len(descriptors): return
    N2 = len(descriptors) / 2
    counters = np.sum(descriptors, axis=0)
    mean = np.where(counters > N2, 1, 0)
    return ORB(mean)

def to_binary(cv_descriptor: np.ndarray):
    '''Convert OpenCV Unsigned char descriptor to binary descriptor.'''
    binary_desc = np.array(list(format(int.from_bytes(struct.pack('B' * 32,
        *cv_descriptor), 'big'), '0256b')), dtype=np.uint8)
    return binary_desc


if __name__ == '__main__':
    # Create random descriptors
    n_desc = 30
    descriptors = []
    for i in range(n_desc):
        descriptors.append(ORB(np.around(np.random.random((ORB.L,)))))

    print(descriptors[0].distance(descriptors[1]))

    print(mean_value(descriptors))

