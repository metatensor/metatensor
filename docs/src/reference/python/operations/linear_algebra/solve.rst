solve
=====

.. autoclass:: equistore.solve
    :members:

.. code-block:: python

    """
    >>> import numpy as np
    ... import equistore
    ... from equistore import TensorBlock, TensorMap, Labels

    ... X_values = np.random.rand(100, 2)
    ... y_values = (X_values @ np.array([10.0, 42.0]) + 0.1*np.random.normal(size=(100,))).reshape((100, 1))

    ... covariance = X_values.T @ X_values 
    ... y_regression = X_values.T @ y_values

    ... X = TensorMap(
    ...     keys = Labels(
    ...         names = ["dummy"],
    ...         values = np.array([[0]])
    ...     ),
    ...     blocks = [TensorBlock(
    ...         samples = Labels(
    ...             names = ["sample"],
    ...             values = np.arange(0, 2).reshape(2, 1)
    ...         ),
    ...         components = [],
    ...         properties = Labels(
    ...             names = ["property"],
    ...             values = np.arange(0, 2).reshape(2, 1)
    ...         ),
    ...         values = covariance
    ...     )]
    ... )

    ... y = TensorMap(
    ...     keys = Labels(
    ...         names = ["dummy"],
    ...         values = np.array([[0]])
    ...     ),
    ...     blocks = [TensorBlock(
    ...         samples = Labels(
    ...             names = ["sample"],
    ...             values = np.arange(0, 2).reshape(2, 1)
    ...         ),
    ...         components = [],
    ...         properties = Labels(
    ...             names = ["property"],
    ...             values = np.arange(0, 1).reshape(1, 1)
    ...         ),
    ...         values = y_regression
    ...     )]
    ... )

    ... c = equistore.solve(X, y)

    ... print(c.block().values)
    """
