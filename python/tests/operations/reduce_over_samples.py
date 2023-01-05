import os
import unittest

import numpy as np

import equistore.io
import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestSumSamples(unittest.TestCase):
    def test_sum_samples_block(self):
        tensor_se = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )
        tensor_ps = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )
        bl1 = tensor_ps[0]

        reduce_tensor_se = fn.sum_over_samples(tensor_se, sample_names=["structure"])
        reduce_tensor_ps = fn.sum_over_samples(tensor_ps, sample_names=["structure"])

        self.assertTrue(
            np.all(
                np.sum(bl1.values[:4], axis=0) == reduce_tensor_ps.block(0).values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(bl1.values[4:10], axis=0) == reduce_tensor_ps.block(0).values[1]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(bl1.values[22:26], axis=0) == reduce_tensor_ps.block(0).values[5]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(bl1.values[38:46], axis=0) == reduce_tensor_ps.block(0).values[8]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(bl1.values[46:], axis=0) == reduce_tensor_ps.block(0).values[9]
            )
        )

        # Test the gradients
        gr1 = tensor_ps[0].gradient("positions").data

        self.assertTrue(
            np.all(
                np.sum(gr1[[0, 4, 8, 12]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").data[0]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(gr1[[2, 6, 10, 14]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").data[2]
            )
        )

        self.assertTrue(
            np.all(
                np.sum(gr1[[3, 7, 11, 15]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").data[3]
            )
        )

        self.assertTrue(
            np.all(
                np.sum(gr1[[96, 99, 102]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").data[40]
            )
        )

        self.assertTrue(
            np.all(
                np.sum(gr1[-1], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").data[-1]
            )
        )

        # The TensorBlock with key=(8,8,8) has nothing to be summed over
        self.assertTrue(
            np.allclose(
                tensor_ps.block(
                    species_center=8, species_neighbor_1=8, species_neighbor_2=8
                ).values,
                reduce_tensor_ps.block(
                    species_center=8, species_neighbor_1=8, species_neighbor_2=8
                ).values,
            )
        )

        for ii, bl2 in enumerate(
            [tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]
        ):
            self.assertTrue(
                np.all(
                    np.sum(bl2.values[:4], axis=0)
                    == reduce_tensor_se.block(ii).values[0]
                )
            )
            self.assertTrue(
                np.all(
                    np.sum(bl2.values[26:32], axis=0)
                    == reduce_tensor_se.block(ii).values[6]
                )
            )
            self.assertTrue(
                np.all(
                    np.sum(bl2.values[32:38], axis=0)
                    == reduce_tensor_se.block(ii).values[7]
                )
            )
            self.assertTrue(
                np.all(
                    np.sum(bl2.values[46:], axis=0)
                    == reduce_tensor_se.block(ii).values[9]
                )
            )

    def test_reduction_block_two_samples(self):
        block_1 = TensorBlock(
            values=np.array(
                [
                    [1, 2, 4],
                    [3, 5, 6],
                    [-1.3, 26.7, 4.54],
                    [3.5, 5.3, 6.87],
                    [6.1, 35.2, 44.5],
                    [7.3, -7.65, 6.45],
                    [11, 276.0, 4.09],
                    [33, 55.5, -5.6],
                ]
            ),
            samples=Labels(
                ["samples1", "samples2", "samples3"],
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 2],
                        [0, 1, 1],
                        [0, 1, 0],
                        [2, 1, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                    ],
                    dtype=np.int32,
                ),
            ),
            components=[],
            properties=Labels(
                ["properties"], np.array([[0], [1], [5]], dtype=np.int32)
            ),
        )

        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0]], dtype=np.int32)
        )
        X = TensorMap(keys, [block_1])

        reduce_X_12 = fn.sum_over_samples(X, sample_names=["samples1", "samples2"])
        reduce_X_23 = fn.sum_over_samples(X, sample_names=["samples2", "samples3"])

        self.assertTrue(
            np.all(
                np.sum(X.block(0).values[:3], axis=0) == reduce_X_12.block(0).values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(X.block(0).values[3:5], axis=0) == reduce_X_12.block(0).values[1]
            )
        )
        self.assertTrue(np.all(X.block(0).values[5] == reduce_X_12.block(0).values[4]))
        self.assertTrue(np.all(X.block(0).values[6] == reduce_X_12.block(0).values[3]))
        self.assertTrue(np.all(X.block(0).values[7] == reduce_X_12.block(0).values[2]))

        self.assertTrue(
            np.all(
                np.sum(X.block(0).values[[0, 7]], axis=0)
                == reduce_X_23.block(0).values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(X.block(0).values[[3, 5, 6]], axis=0)
                == reduce_X_23.block(0).values[4]
            )
        )

        self.assertTrue(np.all(X.block(0).values[1] == reduce_X_23.block(0).values[1]))
        self.assertTrue(np.all(X.block(0).values[2] == reduce_X_23.block(0).values[2]))
        self.assertTrue(np.all(X.block(0).values[4] == reduce_X_23.block(0).values[3]))

        # check metadata
        self.assertTrue(
            np.all(reduce_X_12.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(
            np.all(reduce_X_23.block(0).properties == X.block(0).properties)
        )

        samples_12 = Labels(
            names=["samples1", "samples2"],
            values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]], dtype=np.int32),
        )
        samples_23 = Labels(
            names=["samples2", "samples3"],
            values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int32),
        )
        self.assertTrue(np.all(reduce_X_12.block(0).samples == samples_12))
        self.assertTrue(np.all(reduce_X_23.block(0).samples == samples_23))


class TestMeanSamples(unittest.TestCase):
    def test_mean_samples_block(self):
        tensor_se = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )
        tensor_ps = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )
        bl1 = tensor_ps[0]

        reduce_tensor_se = fn.mean_over_samples(tensor_se, sample_names=["structure"])
        reduce_tensor_ps = fn.mean_over_samples(tensor_ps, sample_names=["structure"])

        self.assertTrue(
            np.all(
                np.mean(bl1.values[:4], axis=0) == reduce_tensor_ps.block(0).values[0]
            )
        )

        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(bl1.values[4:10], axis=0),
                    reduce_tensor_ps.block(0).values[1],
                    rtol=1e-13,
                )
            )
        )
        self.assertTrue(
            np.all(
                np.mean(bl1.values[22:26], axis=0)
                == reduce_tensor_ps.block(0).values[5]
            )
        )
        self.assertTrue(
            np.all(
                np.mean(bl1.values[38:46], axis=0)
                == reduce_tensor_ps.block(0).values[8]
            )
        )
        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(bl1.values[46:], axis=0),
                    reduce_tensor_ps.block(0).values[9],
                    rtol=1e-13,
                )
            )
        )

        # Test the gradients
        gr1 = tensor_ps[0].gradient("positions").data

        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(gr1[[0, 4, 8, 12]], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").data[0],
                    rtol=1e-13,
                )
            )
        )
        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(gr1[[2, 6, 10, 14]], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").data[2],
                )
            )
        )

        self.assertTrue(
            np.all(
                np.mean(gr1[[3, 7, 11, 15]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").data[3]
            )
        )

        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(gr1[[96, 99, 102]], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").data[40],
                    rtol=1e-13,
                )
            )
        )

        self.assertTrue(
            np.all(
                np.mean(gr1[-1], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").data[-1]
            )
        )

        # The TensorBlock with key=(8,8,8) has nothing to be averaged over
        self.assertTrue(
            np.allclose(
                tensor_ps.block(
                    species_center=8, species_neighbor_1=8, species_neighbor_2=8
                ).values,
                reduce_tensor_ps.block(
                    species_center=8, species_neighbor_1=8, species_neighbor_2=8
                ).values,
            )
        )

        for ii, bl2 in enumerate(
            [tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]
        ):
            self.assertTrue(
                np.all(
                    np.mean(bl2.values[:4], axis=0)
                    == reduce_tensor_se.block(ii).values[0]
                )
            )
            self.assertTrue(
                np.all(
                    np.allclose(
                        np.mean(bl2.values[26:32], axis=0),
                        reduce_tensor_se.block(ii).values[6],
                        rtol=1e-13,
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.allclose(
                        np.mean(bl2.values[32:38], axis=0),
                        reduce_tensor_se.block(ii).values[7],
                        rtol=1e-13,
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.allclose(
                        np.mean(bl2.values[46:], axis=0),
                        reduce_tensor_se.block(ii).values[9],
                        rtol=1e-13,
                    )
                )
            )

    def test_reduction_block_two_samples(self):
        block_1 = TensorBlock(
            values=np.array(
                [
                    [1, 2, 4],
                    [3, 5, 6],
                    [-1.3, 26.7, 4.54],
                    [3.5, 5.3, 6.87],
                    [6.1, 35.2, 44.5],
                    [7.3, -7.65, 6.45],
                    [11, 276.0, 4.09],
                    [33, 55.5, -5.6],
                ]
            ),
            samples=Labels(
                ["samples1", "samples2", "samples3"],
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 2],
                        [0, 1, 1],
                        [0, 1, 0],
                        [2, 1, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                    ],
                    dtype=np.int32,
                ),
            ),
            components=[],
            properties=Labels(
                ["properties"], np.array([[0], [1], [5]], dtype=np.int32)
            ),
        )

        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0]], dtype=np.int32)
        )
        X = TensorMap(keys, [block_1])

        reduce_X_12 = fn.mean_over_samples(X, sample_names=["samples1", "samples2"])
        reduce_X_23 = fn.mean_over_samples(X, sample_names=["samples2", "samples3"])

        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(X.block(0).values[:3], axis=0),
                    reduce_X_12.block(0).values[0],
                    rtol=1e-13,
                )
            )
        )
        self.assertTrue(
            np.all(
                np.mean(X.block(0).values[3:5], axis=0)
                == reduce_X_12.block(0).values[1]
            )
        )
        self.assertTrue(np.all(X.block(0).values[5] == reduce_X_12.block(0).values[4]))
        self.assertTrue(np.all(X.block(0).values[6] == reduce_X_12.block(0).values[3]))
        self.assertTrue(np.all(X.block(0).values[7] == reduce_X_12.block(0).values[2]))

        self.assertTrue(
            np.all(
                np.mean(X.block(0).values[[0, 7]], axis=0)
                == reduce_X_23.block(0).values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(X.block(0).values[[3, 5, 6]], axis=0),
                    reduce_X_23.block(0).values[4],
                    rtol=1e-13,
                )
            )
        )

        self.assertTrue(np.all(X.block(0).values[1] == reduce_X_23.block(0).values[1]))
        self.assertTrue(np.all(X.block(0).values[2] == reduce_X_23.block(0).values[2]))
        self.assertTrue(np.all(X.block(0).values[4] == reduce_X_23.block(0).values[3]))

        # check metadata
        self.assertTrue(
            np.all(reduce_X_12.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(
            np.all(reduce_X_23.block(0).properties == X.block(0).properties)
        )

        samples_12 = Labels(
            names=["samples1", "samples2"],
            values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]], dtype=np.int32),
        )
        samples_23 = Labels(
            names=["samples2", "samples3"],
            values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int32),
        )
        self.assertTrue(np.all(reduce_X_12.block(0).samples == samples_12))
        self.assertTrue(np.all(reduce_X_23.block(0).samples == samples_23))


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
