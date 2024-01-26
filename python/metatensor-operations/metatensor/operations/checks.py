"""
Metatensor performs `shape` as well as `consistency` checks for the metadata during each
operation. The check status can be verified with :py:func:`metatensor.checks_enabled()`
and controlled by :py:class:`metatensor.unsafe_enable_checks()` and
:py:class:`metatensor.unsafe_disable_checks()`.

.. warning::
    Using metatensor without checks is unsafe, can lead to unwanted results, and may
    produce cryptic error messages. Therefore, disabling checks is not recommended and
    should only be used by advanced users! If you see strange results after disabling
    checks, try running the code again with checks enabled.

Checks can either be disabled temporarily
via a compound statements or globally and permanently by calling the speicific function
directly.

The checks can also controlled wit the environment variable
``METATENSOR_UNSAFE_DISABLE_CHECKS``. To disable checks set

.. code-block:: bash

    export METATENSOR_UNSAFE_DISABLE_CHECKS=1


Note that :py:class:`metatensor.unsafe_enable_checks()` or
:py:class:`metatensor.unsafe_disable_checks()` overwrite the defintion of the enviroment
variable.
"""

import os


def _parse_bool_env_var(key):
    """Parses a boolean environment variable. Returns ``False`` if it doesn't exist."""
    value = os.environ.get(key, default="")
    return value.lower() in ["true", "1", "yes", "on"]


# global variable storing the state if checks should be performed or not
_CHECKS_ENABLED = not _parse_bool_env_var("METATENSOR_UNSAFE_DISABLE_CHECKS")


def checks_enabled() -> bool:
    """Check status if metatensor checks should be performed."""
    return _CHECKS_ENABLED


class _SetChecks:
    """Private parent class for setting metatensor check control to a certain state.

    Refer to the docstring of ``disble_checks`` for more details."""

    def __init__(self, state):
        global _CHECKS_ENABLED
        self.original_check_state = _CHECKS_ENABLED
        self.state = state
        _CHECKS_ENABLED = self.state

    def __repr__(self) -> str:
        return f"checks {'en' if self.state else 'dis'}abled"

    def __enter__(self):
        """Enter the context and set the checks to the desired state."""
        global _CHECKS_ENABLED
        _CHECKS_ENABLED = self.state
        return self

    def __exit__(self, type, value, traceback):
        """Exit the context and restore the previous checks status."""
        global _CHECKS_ENABLED
        _CHECKS_ENABLED = self.original_check_state


class unsafe_enable_checks(_SetChecks):
    """Enable metatensor checks.

    Checks are default enabled. Calling this function permanatly enables all metatensor
    operations checks.

    >>> import metatensor
    >>> metatensor.unsafe_enable_checks()
    checks enabled
    >>> print(metatensor.checks_enabled())
    True

    You can also use a compound statement to enable checks only temporarily. This can be
    useful in nested constructions.

    >>> import metatensor
    >>> print(metatensor.checks_enabled())
    True
    >>> with metatensor.unsafe_disable_checks():
    ...     # checks are disabled here
    ...     print(metatensor.checks_enabled())
    ...     with metatensor.unsafe_enable_checks():
    ...         # checks enabled here again
    ...         print(metatensor.checks_enabled())
    ...
    False
    True
    >>> print(metatensor.checks_enabled())
    True
    """

    def __init__(self):
        super().__init__(state=True)


class unsafe_disable_checks(_SetChecks):
    """Disable metatensor checks.

    Calling this function permanatly disables all metatensor operations checks.

    >>> import metatensor
    >>> metatensor.unsafe_disable_checks()
    checks disabled
    >>> print(metatensor.checks_enabled())
    False

    You can also use a compound statement to disable checks only temporarily.

    >>> metatensor.unsafe_enable_checks()
    checks enabled
    >>> with metatensor.unsafe_disable_checks():
    ...     # checks are disabled here
    ...     print(metatensor.checks_enabled())
    ...
    False
    >>> print(metatensor.checks_enabled())
    True
    """

    def __init__(self):
        super().__init__(state=False)
