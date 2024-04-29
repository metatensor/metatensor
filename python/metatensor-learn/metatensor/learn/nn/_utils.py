from typing import List


def _check_module_map_parameter(
    parameter, name: str, types: type, num_keys: int, keys_param_name: str
) -> List:
    """
    A helper function that checks the type of an input parameter passed to classes that
    are children of :py:class:`ModuleMap`, and raises the appropriate errors.

    Raises a TypeError if not one of the valid `types`. Raises a ValueError if passed as
    a list and not the same length as `num_keys`.

    :param parameter: the parameter value to check
    :param name: str, the name of the parameter used in the ModuleMap child class
    :param types: the type that is valid for the `parameter`
    :param num_keys: int, the number of relevant keys in the ModuleMap child class for
        which the parameter is used. Used to check the length of `parameter` if passed
        as a list.
    :param keys_param_name: str, the name of the relevant keys parameter for the
        ModuleMap child class. Used to raise a ValueError if the length of `parameter`
        does not match the length of the keys.

    :return: `parameter` as a list, one for each key.
    """

    # Convert to list if not already
    if isinstance(parameter, types):
        parameter = [parameter] * num_keys

    # Check if it is a list
    if isinstance(parameter, list):
        if len(parameter) != num_keys:
            raise ValueError(
                f"`{name}` must have same length as `{keys_param_name}`,"
                f" but len({name}) != len({keys_param_name})"
                f" [{len(parameter)} != {num_keys}]"
            )
        # Check each element
        for p in parameter:
            if not isinstance(p, types):
                raise TypeError(
                    f"`{name}` must be of type {types} or List of {types}, but not"
                    f" {type(p)}."
                )
    # Raise if not a list
    else:
        raise TypeError(
            f"`{name}` must be type {types} or List of {types},"
            f" but not {type(parameter)}."
        )

    return parameter
