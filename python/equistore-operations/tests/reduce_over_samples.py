import os
import unittest

import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestSumSamples(unittest.TestCase):
    def test_sum_samples_block(self):
        tensor_se = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )
        tensor_ps = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )
        bl1 = tensor_ps[0]

        # check both passing a list and a single string for sample_names
        reduce_tensor_se = equistore.sum_over_samples(
            tensor_se, sample_names=["center"]
        )
        reduce_tensor_ps = equistore.sum_over_samples(tensor_ps, sample_names="center")

        # checks that reduction over a block is the same as the tensormap operation
        reduce_block_se = equistore.sum_over_samples_block(
            tensor_se.block(0), sample_names="center"
        )
        self.assertTrue(
            np.allclose(reduce_block_se.values, reduce_tensor_se.block(0).values)
        )

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
        gr1 = tensor_ps[0].gradient("positions").values

        self.assertTrue(
            np.all(
                np.sum(gr1[[0, 4, 8, 12]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(gr1[[2, 6, 10, 14]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").values[2]
            )
        )

        self.assertTrue(
            np.all(
                np.sum(gr1[[3, 7, 11, 15]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").values[3]
            )
        )

        self.assertTrue(
            np.all(
                np.sum(gr1[[96, 99, 102]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").values[40]
            )
        )

        self.assertTrue(
            np.all(
                np.sum(gr1[-1], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").values[-1]
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
                ),
            ),
            components=[],
            properties=Labels(["p"], np.array([[0], [1], [5]])),
        )

        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
        X = TensorMap(keys, [block_1])

        reduce_X_12 = equistore.sum_over_samples(X, sample_names="samples3")
        reduce_X_23 = equistore.sum_over_samples(X, sample_names=["samples1"])
        reduce_X_2 = equistore.sum_over_samples(
            X, sample_names=["samples1", "samples3"]
        )

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

        self.assertTrue(
            np.all(
                np.sum(X.block(0).values[[0, 1, 2, 7]], axis=0)
                == reduce_X_2.block(0).values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.sum(X.block(0).values[3:7], axis=0) == reduce_X_2.block(0).values[1]
            )
        )
        # check metadata
        self.assertTrue(
            np.all(reduce_X_12.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(
            np.all(reduce_X_23.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(np.all(reduce_X_2.block(0).properties == X.block(0).properties))

        samples_12 = Labels(
            names=["samples1", "samples2"],
            values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
        )
        samples_23 = Labels(
            names=["samples2", "samples3"],
            values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
        )
        samples_2 = Labels(
            names=["samples2"],
            values=np.array([[0], [1]]),
        )
        self.assertTrue(np.all(reduce_X_12.block(0).samples == samples_12))
        self.assertTrue(np.all(reduce_X_23.block(0).samples == samples_23))
        self.assertTrue(np.all(reduce_X_2.block(0).samples == samples_2))


class TestMeanSamples(unittest.TestCase):
    def test_mean_samples_block(self):
        tensor_se = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )
        tensor_ps = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )
        bl1 = tensor_ps[0]

        # check both passing a list and a single string for sample_names
        reduce_tensor_se = equistore.mean_over_samples(tensor_se, sample_names="center")
        reduce_tensor_ps = equistore.mean_over_samples(
            tensor_ps, sample_names=["center"]
        )

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
        gr1 = tensor_ps[0].gradient("positions").values

        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(gr1[[0, 4, 8, 12]], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").values[0],
                    rtol=1e-13,
                )
            )
        )
        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(gr1[[2, 6, 10, 14]], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").values[2],
                )
            )
        )

        self.assertTrue(
            np.all(
                np.mean(gr1[[3, 7, 11, 15]], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").values[3]
            )
        )

        self.assertTrue(
            np.all(
                np.allclose(
                    np.mean(gr1[[96, 99, 102]], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").values[40],
                    rtol=1e-13,
                )
            )
        )

        self.assertTrue(
            np.all(
                np.mean(gr1[-1], axis=0)
                == reduce_tensor_ps.block(0).gradient("positions").values[-1]
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
                ),
            ),
            components=[],
            properties=Labels(["p"], np.array([[0], [1], [5]])),
        )

        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
        X = TensorMap(keys, [block_1])

        reduce_X_12 = equistore.mean_over_samples(X, sample_names=["samples3"])
        reduce_X_23 = equistore.mean_over_samples(X, sample_names="samples1")
        reduce_X_2 = equistore.mean_over_samples(
            X, sample_names=["samples1", "samples3"]
        )

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

        self.assertTrue(
            np.all(
                np.mean(X.block(0).values[[0, 1, 2, 7]], axis=0)
                == reduce_X_2.block(0).values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.mean(X.block(0).values[3:7], axis=0) == reduce_X_2.block(0).values[1]
            )
        )

        # check metadata
        self.assertTrue(
            np.all(reduce_X_12.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(
            np.all(reduce_X_23.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(np.all(reduce_X_2.block(0).properties == X.block(0).properties))

        samples_12 = Labels(
            names=["samples1", "samples2"],
            values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
        )
        samples_23 = Labels(
            names=["samples2", "samples3"],
            values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
        )
        samples_2 = Labels(
            names=["samples2"],
            values=np.array([[0], [1]]),
        )
        self.assertTrue(np.all(reduce_X_12.block(0).samples == samples_12))
        self.assertTrue(np.all(reduce_X_23.block(0).samples == samples_23))
        self.assertTrue(np.all(reduce_X_2.block(0).samples == samples_2))


class TestReductionAllSamples(unittest.TestCase):
    def test_reduction_allsamples(self):
        block_1 = TensorBlock(
            values=np.array(
                [
                    [1, 2, 4],
                    [3, 5, 6],
                    [-1.3, 26.7, 4.54],
                ]
            ),
            samples=Labels.arange("s", 3),
            components=[],
            properties=Labels(["p"], np.array([[0], [1], [5]])),
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
        X = TensorMap(keys, [block_1])

        sum_X = equistore.sum_over_samples(X, sample_names=["s"])
        mean_X = equistore.mean_over_samples(X, sample_names=["s"])
        var_X = equistore.var_over_samples(X, sample_names=["s"])
        std_X = equistore.std_over_samples(X, sample_names=["s"])

        self.assertTrue(equistore.equal_metadata(sum_X, mean_X))
        self.assertTrue(equistore.equal_metadata(sum_X, std_X))
        self.assertTrue(equistore.equal_metadata(mean_X, var_X))
        self.assertTrue(sum_X[0].samples == Labels.single())
        self.assertTrue(std_X[0].samples == Labels.single())

        self.assertTrue(np.all(sum_X[0].values == np.sum(X[0].values, axis=0)))
        self.assertTrue(np.all(mean_X[0].values == np.mean(X[0].values, axis=0)))
        self.assertTrue(np.allclose(std_X[0].values, np.std(X[0].values, axis=0)))
        self.assertTrue(np.allclose(var_X[0].values, np.var(X[0].values, axis=0)))


class TestStdSamples(unittest.TestCase):
    def test_std_samples_block(self):
        tensor_se = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )
        tensor_ps = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )
        tensor_se = equistore.remove_gradients(tensor_se)

        bl1 = tensor_ps[0]

        # check both passing a list and a single string for sample_names
        reduce_tensor_se = equistore.std_over_samples(tensor_se, sample_names="center")
        reduce_tensor_ps = equistore.std_over_samples(
            tensor_ps, sample_names=["center"]
        )

        self.assertTrue(
            np.allclose(
                np.std(bl1.values[:4], axis=0),
                reduce_tensor_ps.block(0).values[0],
                rtol=1e-13,
            )
        )

        self.assertTrue(
            np.all(
                np.allclose(
                    np.std(bl1.values[4:10], axis=0),
                    reduce_tensor_ps.block(0).values[1],
                    rtol=1e-13,
                )
            )
        )
        self.assertTrue(
            np.allclose(
                np.std(bl1.values[22:26], axis=0),
                reduce_tensor_ps.block(0).values[5],
                rtol=1e-13,
            )
        )
        self.assertTrue(
            np.allclose(
                np.std(bl1.values[38:46], axis=0),
                reduce_tensor_ps.block(0).values[8],
                rtol=1e-13,
            )
        )
        self.assertTrue(
            np.all(
                np.allclose(
                    np.std(bl1.values[46:], axis=0),
                    reduce_tensor_ps.block(0).values[9],
                    rtol=1e-13,
                )
            )
        )

        # Test the gradients
        gr1 = tensor_ps[0].gradient("positions")

        XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[0, 4, 8, 12])
        self.assertTrue(
            np.all(
                np.allclose(
                    (
                        np.mean(XdX, axis=0)
                        - np.mean(bl1.values[:4], axis=0)
                        * np.mean(gr1.values[[0, 4, 8, 12]], axis=0)
                    )
                    / np.std(bl1.values[:4], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").values[0],
                    rtol=1e-13,
                )
            )
        )
        XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[2, 6, 10, 14])
        self.assertTrue(
            np.all(
                np.allclose(
                    (
                        np.mean(XdX, axis=0)
                        - np.mean(bl1.values[:4], axis=0)
                        * np.mean(gr1.values[[2, 6, 10, 14]], axis=0)
                    )
                    / np.std(bl1.values[:4], axis=0),
                    reduce_tensor_ps.block(0).gradient("positions").values[2],
                )
            )
        )

        XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[3, 7, 11, 15])
        self.assertTrue(
            np.allclose(
                (
                    np.mean(XdX, axis=0)
                    - np.mean(bl1.values[:4], axis=0)
                    * np.mean(gr1.values[[3, 7, 11, 15]], axis=0)
                )
                / np.std(bl1.values[:4], axis=0),
                reduce_tensor_ps.block(0).gradient("positions").values[3],
            )
        )

        XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[96, 99, 102])
        idx = [
            i
            for i in range(len(bl1.samples))
            if bl1.samples[i][0] == bl1.samples[gr1.samples[96][0]][0]
        ]

        self.assertTrue(
            np.allclose(
                (
                    np.mean(XdX, axis=0)
                    - np.mean(bl1.values[idx], axis=0)
                    * np.mean(gr1.values[[96, 99, 102]], axis=0)
                )
                / np.std(bl1.values[idx], axis=0),
                reduce_tensor_ps.block(0).gradient("positions").values[40],
                rtol=1e-13,
            )
        )

        # The TensorBlock with key=(8,8,8) has nothing to be averaged over
        values = reduce_tensor_ps.block(
            species_center=8, species_neighbor_1=8, species_neighbor_2=8
        ).values
        self.assertTrue(
            np.allclose(
                np.zeros(values.shape),
                values,
            )
        )

        for ii, bl2 in enumerate(
            [tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]
        ):
            self.assertTrue(
                np.allclose(
                    np.std(bl2.values[:4], axis=0),
                    reduce_tensor_se.block(ii).values[0],
                    rtol=1e-13,
                )
            )
            self.assertTrue(
                np.all(
                    np.allclose(
                        np.std(bl2.values[26:32], axis=0),
                        reduce_tensor_se.block(ii).values[6],
                        rtol=1e-13,
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.allclose(
                        np.std(bl2.values[32:38], axis=0),
                        reduce_tensor_se.block(ii).values[7],
                        rtol=1e-13,
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.allclose(
                        np.std(bl2.values[46:], axis=0),
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
                ["s_1", "s_2", "s_3"],
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
                ),
            ),
            components=[],
            properties=Labels(["p"], np.array([[0], [1], [5]])),
        )

        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
        X = TensorMap(keys, [block_1])

        reduce_X_12 = equistore.std_over_samples(X, sample_names=["s_3"])
        reduce_X_23 = equistore.std_over_samples(X, sample_names="s_1")
        reduce_X_2 = equistore.std_over_samples(X, sample_names=["s_1", "s_3"])

        self.assertTrue(
            np.all(
                np.allclose(
                    np.std(X.block(0).values[:3], axis=0),
                    reduce_X_12.block(0).values[0],
                    rtol=1e-13,
                )
            )
        )
        self.assertTrue(
            np.allclose(
                np.std(X.block(0).values[3:5], axis=0),
                reduce_X_12.block(0).values[1],
                rtol=1e-13,
            )
        )
        self.assertTrue(np.all(np.array([0.0]) == reduce_X_12.block(0).values[4]))
        self.assertTrue(np.all(np.array([0.0]) == reduce_X_12.block(0).values[3]))
        self.assertTrue(np.all(np.array([0.0]) == reduce_X_12.block(0).values[2]))

        self.assertTrue(
            np.all(
                np.std(X.block(0).values[[0, 7]], axis=0)
                == reduce_X_23.block(0).values[0]
            )
        )
        self.assertTrue(
            np.all(
                np.allclose(
                    np.std(X.block(0).values[[3, 5, 6]], axis=0),
                    reduce_X_23.block(0).values[4],
                    rtol=1e-13,
                )
            )
        )

        self.assertTrue(np.all(np.array([0.0]) == reduce_X_23.block(0).values[1]))
        self.assertTrue(np.all(np.array([0.0]) == reduce_X_23.block(0).values[2]))
        self.assertTrue(np.all(np.array([0.0]) == reduce_X_23.block(0).values[3]))

        self.assertTrue(
            np.allclose(
                np.std(X.block(0).values[[0, 1, 2, 7]], axis=0),
                reduce_X_2.block(0).values[0],
                rtol=1e-13,
            )
        )
        self.assertTrue(
            np.all(
                np.std(X.block(0).values[3:7], axis=0) == reduce_X_2.block(0).values[1]
            )
        )

        # check metadata
        self.assertTrue(
            np.all(reduce_X_12.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(
            np.all(reduce_X_23.block(0).properties == X.block(0).properties)
        )
        self.assertTrue(np.all(reduce_X_2.block(0).properties == X.block(0).properties))

        samples_12 = Labels(
            names=["s_1", "s_2"],
            values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
        )
        samples_23 = Labels(
            names=["s_2", "s_3"],
            values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
        )
        samples_2 = Labels(
            names=["s_2"],
            values=np.array([[0], [1]]),
        )
        self.assertTrue(np.all(reduce_X_12.block(0).samples == samples_12))
        self.assertTrue(np.all(reduce_X_23.block(0).samples == samples_23))
        self.assertTrue(np.all(reduce_X_2.block(0).samples == samples_2))

    def test_reduction_of_one_element(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2, 4], [3, 5, 6], [-1.3, 26.7, 4.54]]),
            samples=Labels(["s_1", "s_2"], np.array([[0, 0], [1, 1], [2, 2]])),
            components=[],
            properties=Labels(["p"], np.array([[0], [1], [5]])),
        )

        block_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7.8]]),
                samples=Labels(["sample", "g"], np.array([[0, 0], [1, 1], [2, 2]])),
                components=[],
                properties=block_1.properties,
            ),
        )

        keys = Labels(names=["key_1"], values=np.array([[0]]))
        X = TensorMap(keys, [block_1])

        add_X = equistore.sum_over_samples(X, sample_names=["s_1"])
        mean_X = equistore.mean_over_samples(X, sample_names=["s_1"])
        var_X = equistore.var_over_samples(X, sample_names=["s_1"])
        std_X = equistore.std_over_samples(X, sample_names=["s_1"])

        # print(add_X[0])
        # print(X[0].values, add_X[0].values)
        self.assertTrue(np.all(X[0].values == add_X[0].values))
        self.assertTrue(np.all(X[0].values == mean_X[0].values))
        self.assertTrue(equistore.equal(add_X, mean_X))
        self.assertTrue(equistore.equal_metadata(add_X, var_X))
        self.assertTrue(equistore.equal_metadata(mean_X, std_X))

        self.assertTrue(np.all(np.zeros((3, 3)) == std_X[0].values))
        self.assertTrue(equistore.equal(var_X, std_X))

        # Gradients
        grad_sample_label = Labels(
            names=["sample", "g"],
            values=np.array([[0, 0], [1, 1], [2, 2]]),
        )
        self.assertTrue(std_X[0].gradient("g").samples.names == grad_sample_label.names)
        self.assertTrue(np.all(std_X[0].gradient("g").samples == grad_sample_label))
        self.assertTrue(
            np.all(X[0].gradient("g").values == add_X[0].gradient("g").values)
        )
        self.assertTrue(
            np.all(X[0].gradient("g").values == mean_X[0].gradient("g").values)
        )
        self.assertTrue(np.all(np.zeros((3, 3)) == std_X[0].gradient("g").values))
        self.assertTrue(np.all(np.zeros((3, 3)) == var_X[0].gradient("g").values))


class TestZeroSamples(unittest.TestCase):
    def test_zeros_sample_block(self):
        block = TensorBlock(
            values=np.zeros([0, 1]),
            properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
            samples=Labels(["s"], np.empty((0, 1))),
            components=[],
        )

        result_block = TensorBlock(
            values=np.zeros([0, 1]),
            properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
            samples=Labels([], np.empty((0, 0))),
            components=[],
        )

        tensor = TensorMap(Labels.single(), [block])
        result_tensor = TensorMap(Labels.single(), [result_block])

        tensor_sum = equistore.sum_over_samples(tensor, "s")
        tensor_mean = equistore.mean_over_samples(tensor, "s")
        tensor_std = equistore.std_over_samples(tensor, "s")
        tensor_var = equistore.var_over_samples(tensor, "s")

        self.assertTrue(equistore.equal(result_tensor, tensor_sum))
        self.assertTrue(equistore.equal(result_tensor, tensor_mean))
        self.assertTrue(equistore.equal(result_tensor, tensor_var))
        self.assertTrue(equistore.equal(result_tensor, tensor_std))

        block = TensorBlock(
            values=np.zeros([0, 1]),
            properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
            samples=Labels(["s_1", "s_2"], np.empty((0, 1))),
            components=[],
        )

        result_block = TensorBlock(
            values=np.zeros([0, 1]),
            properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
            samples=Labels(["s_2"], np.empty((0, 1))),
            components=[],
        )

        tensor = TensorMap(Labels.single(), [block])
        result_tensor = TensorMap(Labels.single(), [result_block])

        tensor_sum = equistore.sum_over_samples(tensor, "s_1")
        tensor_mean = equistore.mean_over_samples(tensor, "s_1")
        tensor_std = equistore.std_over_samples(tensor, "s_1")
        tensor_var = equistore.var_over_samples(tensor, "s_1")

        self.assertTrue(equistore.equal(result_tensor, tensor_sum))
        self.assertTrue(equistore.equal(result_tensor, tensor_mean))
        self.assertTrue(equistore.equal(result_tensor, tensor_var))
        self.assertTrue(equistore.equal(result_tensor, tensor_std))

    def test_zeros_sample_block_gradient(self):
        block = TensorBlock(
            values=np.array(
                [[1, 2, 4], [3, 5, 6], [-1.3, 26.7, 4.54], [3.5, 5.3, 6.87]]
            ),
            samples=Labels(
                ["s_1", "s_2"],
                np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            ),
            components=[],
            properties=Labels(["p"], np.array([[0], [1], [5]])),
        )

        block.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.zeros((0, 3)),
                samples=Labels(["sample", "g"], np.empty((0, 2))),
                components=[],
                properties=block.properties,
            ),
        )

        sum_block = TensorBlock(
            values=np.array([[-0.3, 28.7, 8.54], [6.5, 10.3, 12.87]]),
            samples=Labels.arange("s_2", 2),
            components=[],
            properties=Labels(["p"], np.array([[0], [1], [5]])),
        )

        sum_block.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.zeros((0, 3)),
                samples=Labels(["sample", "g"], np.empty((0, 2))),
                components=[],
                properties=sum_block.properties,
            ),
        )

        tensor = TensorMap(Labels.single(), [block])
        tensor_sum_result = TensorMap(Labels.single(), [sum_block])

        tensor_sum = equistore.sum_over_samples(tensor, "s_1")
        tensor_mean = equistore.mean_over_samples(tensor, "s_1")
        tensor_std = equistore.std_over_samples(tensor, "s_1")
        tensor_var = equistore.var_over_samples(tensor, "s_1")

        self.assertTrue(equistore.allclose(tensor_sum_result, tensor_sum, atol=1e-14))
        self.assertTrue(equistore.equal_metadata(tensor_sum, tensor_mean))
        self.assertTrue(equistore.equal_metadata(tensor_sum, tensor_var))
        self.assertTrue(equistore.equal_metadata(tensor_sum, tensor_std))


# TODO: add tests with torch & torch scripting/tracing
def get_XdX(block, gradient, der_index):
    XdX = []
    for ig in der_index:
        idx = gradient.samples[ig][0]
        XdX.append(block.values[idx] * gradient.values[ig])
    return np.stack(XdX)


if __name__ == "__main__":
    unittest.main()
