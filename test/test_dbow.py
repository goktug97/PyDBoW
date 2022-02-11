#!/usr/bin/env python3

import unittest
import dbow
import numpy as np


class TestDBOW(unittest.TestCase):
    def test_weight(self):
        descriptors = [
            dbow.ORB([0, 1, 0, 0, 1]),
            dbow.ORB([1, 1, 1, 1, 1]),
            dbow.ORB([0, 0, 0, 1, 0]),
            dbow.ORB([0, 1, 0, 0, 1]),
        ]
        word = dbow.Word(descriptors)
        word.update_weight(4)
        self.assertEqual(word.weight, 0.0)

    def test_tree(self):
        descriptors = [
            dbow.ORB([0, 1, 0, 0, 1]),
            dbow.ORB([1, 1, 1, 1, 1]),
            dbow.ORB([0, 0, 0, 1, 0]),
            dbow.ORB([0, 1, 0, 0, 1]),
        ]
        root = dbow.Node(descriptors)
        words = dbow.initialize_tree(root, 4, 4)
        for i, word in enumerate(words):
            self.assertEqual(word.idx, i)


if __name__ == "__main__":
    unittest.main(verbosity=2)
