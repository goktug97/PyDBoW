#!/usr/bin/env python3

import unittest
import orb_descriptor
import numpy as np

class TestORB(unittest.TestCase):

    def test_mean(self):
        descriptors = [orb_descriptor.ORB([0, 1, 0, 0, 1]),
                       orb_descriptor.ORB([1, 1, 1, 1, 1]),
                       orb_descriptor.ORB([0, 0, 0, 1, 0]),
                       orb_descriptor.ORB([0, 1, 0, 0, 1])]

        np.testing.assert_allclose(
                orb_descriptor.mean_value(descriptors),
                orb_descriptor.ORB([0, 1, 0, 1, 1]))

    def test_distance(self):
        desc1 = orb_descriptor.ORB([0, 1, 0, 0, 1])
        desc2 = orb_descriptor.ORB([1, 1, 1, 1, 1])
        self.assertEqual(desc1.distance(desc2), 3)

if __name__ == '__main__':
    unittest.main()
