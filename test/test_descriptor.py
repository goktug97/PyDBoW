#!/usr/bin/env python3

import unittest
import dbow
import numpy as np


class TestORB(unittest.TestCase):
    def test_mean(self):
        descriptors = [
            dbow.ORB([0, 1, 0, 0, 1]),
            dbow.ORB([1, 1, 1, 1, 1]),
            dbow.ORB([0, 0, 0, 1, 0]),
            dbow.ORB([0, 1, 0, 0, 1]),
        ]

        self.assertEqual(dbow.mean_value(descriptors), dbow.ORB([0, 1, 0, 1, 1]))

    def test_distance(self):
        desc1 = dbow.ORB([0, 1, 0, 0, 1])
        desc2 = dbow.ORB([1, 1, 1, 1, 1])
        self.assertEqual(desc1.distance(desc2), 3)

    def test_sum(self):
        descriptors = [
            dbow.ORB([0, 1, 0, 0, 1]),
            dbow.ORB([1, 1, 1, 1, 1]),
            dbow.ORB([0, 0, 0, 1, 0]),
            dbow.ORB([0, 1, 0, 0, 1]),
        ]
        np.allclose(np.sum(descriptors).features, np.array([1, 3, 1, 2, 3]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
