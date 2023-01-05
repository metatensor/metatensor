import unittest
from os import path

import numpy as np

import equistore.operations as eop
from equistore import Labels, TensorBlock, TensorMap
from equistore.io import load


DATA_ROOT = path.join(path.dirname(__file__), "..", "data")


class TestJoinTensorMap(unittest.TestCase):
    def setUp(self):
        self.ps = load(path.join(DATA_ROOT, "qm7-power-spectrum.npz"), use_numpy=True)
        self.se = load(
            path.join(DATA_ROOT, "qm7-spherical-expansion.npz"), use_numpy=True
        )
        self.first_block = self.ps.block(0)

        # Test if Tensormaps have at least one gradient. This avoids dropping gradient
        # tests silently by removing gradients from the reference data
        self.assertIn("positions", self.ps.block(0).gradients_list())
        self.assertIn("positions", self.se.block(0).gradients_list())

        keys_first_block = Labels(
            names=self.ps.keys.names,
            values=np.array(self.ps.keys[0].tolist()).reshape(1, -1),
        )

        self.ps_first_block = TensorMap(keys_first_block, [self.first_block.copy()])

        block_extra_grad = self.first_block.copy()
        gradient = self.ps.block(0).gradient("positions")
        block_extra_grad.add_gradient(
            "foo", gradient.data, gradient.samples, gradient.components
        )
        self.ps_first_block_extra_grad = TensorMap(keys_first_block, [block_extra_grad])

    def test_single_tensormap(self):
        """Test error raise if only one tensormap is provided."""
        with self.assertRaises(ValueError) as err:
            eop.join(self.ps, axis="properties")
        self.assertIn("provide at least two", str(err.exception))

        with self.assertRaises(ValueError) as err:
            eop.join([self.ps], axis="properties")
        self.assertIn("provide at least two", str(err.exception))

    def test_join_properties(self):
        """Test public join function with three tensormaps along `properties`.

        We check for the values below."""

        ps_joined = eop.join([self.ps, self.ps, self.ps], axis="properties")

        # test property names
        names = self.ps.block(0).properties.names
        self.assertEqual(ps_joined.block(0).properties.names, ("tensor",) + names)

        # test property values
        tensor_prop = np.unique(ps_joined.block(0).properties["tensor"])
        self.assertEqual(set(tensor_prop), set((0, 1, 2)))

    def test_join_samples(self):
        """Test public join function with three tensormaps along `samples`."""
        ps_joined = eop.join([self.ps, self.ps, self.ps], axis="samples")

        # test sample values
        self.assertEqual(
            len(ps_joined.block(0).samples), 3 * len(self.ps.block(0).samples)
        )

    def test_join_error(self):
        """Test error with unknown `axis` keyword."""
        with self.assertRaises(ValueError) as err:
            eop.join([self.ps, self.ps, self.ps], axis="foo")
        self.assertIn("values for the `axis` parameter", str(err.exception))

    def test_join_properties_values(self):
        """Test values for joining along `properties`."""
        ts_1 = eop.slice(self.ps, properties=self.first_block.properties[:1])
        ts_2 = eop.slice(self.ps, properties=self.first_block.properties[1:])

        ts_joined = eop.join([ts_1, ts_2], axis="properties")
        self.assertTrue(eop.allclose(ts_joined, self.ps))

    def test_join_properties_different_samples(self):
        """Test error raise if `samples` are not the same."""
        ts = eop.slice(self.ps_first_block, samples=self.first_block.samples[:1])

        with self.assertRaises(ValueError) as err:
            eop.join([self.ps_first_block, ts], axis="properties")
        self.assertIn("samples", str(err.exception))

    def test_join_properties_different_components(self):
        """Test error raise if `components` are not the same."""
        self.se.components_to_properties(["spherical_harmonics_m"])
        se = load(path.join(DATA_ROOT, "qm7-spherical-expansion.npz"), use_numpy=True)
        with self.assertRaises(ValueError) as err:
            eop.join([self.se, se], axis="properties")
        self.assertIn("components", str(err.exception))

    def test_join_properties_different_gradients(self):
        """Test error raise if `gradients` are not the same."""
        with self.assertRaises(ValueError) as err:
            eop.join(
                [self.ps_first_block, self.ps_first_block_extra_grad], axis="properties"
            )
        self.assertIn("gradients", str(err.exception))

    def test_join_samples_values(self):
        """Test values for joining along `samples`."""
        keys = Labels(
            names=self.ps.keys.names,
            values=np.array(self.ps.keys[0].tolist()).reshape(1, -1),
        )

        ts = TensorMap(keys, [self.first_block.copy()])
        ts_1 = eop.slice(ts, samples=self.first_block.samples[:1])
        ts_2 = eop.slice(ts, samples=self.first_block.samples[1:])

        ts_joined = eop.join([ts_1, ts_2], axis="samples")
        self.assertTrue(eop.allclose(ts, ts_joined))

    def test_join_samples_different_properties(self):
        """Test error raise if `proprties` are not the same."""
        ts = eop.slice(self.ps_first_block, properties=self.first_block.properties[:1])

        with self.assertRaises(ValueError) as err:
            eop.join([self.ps_first_block, ts], axis="samples")
        self.assertIn("properties", str(err.exception))

    def test_join_samples_different_components(self):
        """Test error raise if `components` are not the same."""
        self.se.components_to_properties(["spherical_harmonics_m"])
        se = load(path.join(DATA_ROOT, "qm7-spherical-expansion.npz"), use_numpy=True)
        with self.assertRaises(ValueError) as err:
            eop.join([self.se, se], axis="samples")
        self.assertIn("components", str(err.exception))

    def test_join_samples_different_gradients(self):
        """Test error raise if `gradients` are not the same."""
        with self.assertRaises(ValueError) as err:
            eop.join(
                [self.ps_first_block, self.ps_first_block_extra_grad], axis="samples"
            )
        self.assertIn("gradients", str(err.exception))


class TestJoinLabels(unittest.TestCase):
    """Test edge cases of label joining."""

    def setUp(self):
        self.sample_labels = Labels(names=["prop"], values=np.arange(2).reshape(-1, 1))
        self.keys = Labels(names=["prop"], values=np.arange(1).reshape(-1, 1))

    def test_same_names_same_values(self):
        """Test Label joining using labels with same names but same values."""

        names = ("structure", "prop_1")
        property_labels = Labels(names, np.vstack([np.arange(5), np.arange(5)]).T)

        block = TensorBlock(
            values=np.zeros([2, 5]),
            samples=self.sample_labels,
            components=[],
            properties=property_labels,
        )

        ts = TensorMap(self.keys, [block])
        joined_ts = eop.join([ts, ts], axis="properties")

        joined_labels = joined_ts.block(0).properties

        self.assertEqual(joined_labels.names, ("tensor",) + names)

        ref = np.array(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 2, 2],
                [0, 3, 3],
                [0, 4, 4],
                [1, 0, 0],
                [1, 1, 1],
                [1, 2, 2],
                [1, 3, 3],
                [1, 4, 4],
            ]
        )

        self.assertTrue(np.equal(joined_labels.tolist(), ref).all())

    def test_same_names_unique_values(self):
        """Test Label joining using labels with same names and unique values."""
        names = ("structure", "prop_1")
        property_labels_1 = Labels(names, np.vstack([np.arange(5), np.arange(5)]).T)
        property_labels_2 = Labels(
            names, np.vstack([np.arange(5, 10), np.arange(5, 10)]).T
        )

        block_1 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=self.sample_labels,
            components=[],
            properties=property_labels_1,
        )

        block_2 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=self.sample_labels,
            components=[],
            properties=property_labels_2,
        )

        joined_ts = eop.join(
            [TensorMap(self.keys, [block_1]), TensorMap(self.keys, [block_2])],
            axis="properties",
        )

        joined_labels = joined_ts.block(0).properties

        self.assertEqual(joined_labels.names, ("structure", "prop_1"))
        self.assertTrue(
            np.equal(
                joined_labels.tolist(), np.vstack([np.arange(10), np.arange(10)]).T
            ).all()
        )

    def test_different_names(self):
        """Test Label joining using labels with different names."""
        values = np.vstack([np.arange(5), np.arange(5)]).T
        property_labels_1 = Labels(("structure", "prop_1"), -values)
        property_labels_2 = Labels(("structure", "prop_2"), values)

        block_1 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=self.sample_labels,
            components=[],
            properties=property_labels_1,
        )

        block_2 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=self.sample_labels,
            components=[],
            properties=property_labels_2,
        )

        joined_ts = eop.join(
            [TensorMap(self.keys, [block_1]), TensorMap(self.keys, [block_2])],
            axis="properties",
        )

        joined_labels = joined_ts.block(0).properties

        self.assertEqual(joined_labels.names, ("tensor", "property"))

        ref = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]
        )

        self.assertTrue(np.equal(joined_labels.tolist(), ref).all())


if __name__ == "__main__":
    unittest.main()
