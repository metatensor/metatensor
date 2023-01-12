import doctest
import importlib
import pkgutil
import unittest

import equistore
import equistore.operations


class TestDoctests(unittest.TestCase):
    def test_doctests(self):
        result = doctest.testmod(equistore)
        self.assertEqual(result.failed, 0)

        for module in pkgutil.walk_packages(equistore.__path__):
            submodule = importlib.import_module("equistore." + module.name)
            result = doctest.testmod(submodule)
            self.assertEqual(result.failed, 0)


if __name__ == "__main__":
    unittest.main()
