import unittest

import numpy as np
from aml_storage import Labels


class TestLabels(unittest.TestCase):
    def test_names(self):
        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]], dtype=np.int32),
        )

        self.assertEqual(labels.names, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
