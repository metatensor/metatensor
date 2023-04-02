import numpy as np

from equistore import Labels


def one_hot(labels: Labels, dimension: Labels) -> Labels:
    """Example:

    >>> import numpy as np
    >>> import equistore
    >>> from equistore import Labels
    ...
    >>> # Let's say we have 6 atoms, whose chemical indentities
    >>> # are C, H, H, H, C, H:
    >>> original_labels = Labels(
    ...     names=["atom", "species"],
    ...     values=np.array([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]])
    ... )
    >>> # Set up a Labels object with all possible elements,
    >>> # including, for example, also O:
    >>> possible_labels = Labels(
    ...     names=["species"],
    ...     values=np.array([[1], [6], [8]])
    ... )
    >>> # Get the one-hot encoded labels:
    >>> one_hot_encoding = one_hot(original_labels, possible_labels)
    >>> print(one_hot_encoding)
    [[0. 1. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]
     [1. 0. 0.]]
    """

    if dimension.asarray().shape[1] != 1:
        raise ValueError()

    name = dimension.names[0]
    original_labels = labels[name]
    possible_labels = dimension[name]

    indices = np.where(
        original_labels.reshape(original_labels.size, 1) == possible_labels
    )[1]
    one_hot_array = np.eye(possible_labels.shape[0])[indices]

    return one_hot_array
