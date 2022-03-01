import unittest

import numpy as np
from aml_storage import Indexes


class TestIndexes(unittest.TestCase):
    def test_names(self):
        indexes = Indexes(
            names=["a", "b"],
            values=np.array([[0, 0]], dtype=np.int32),
        )

        self.assertEqual(indexes.names, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
