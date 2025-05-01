import os
from importlib import reload

import pytest

import metatensor as mts


def set_initial_state(state):
    mts.operations._checks._CHECKS_ENABLED = state


@pytest.mark.parametrize("inital_state", [True, False])
def test_checks_enabled(inital_state):
    """Test checks_enabled() function."""
    set_initial_state(inital_state)
    assert mts.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_enable_checks(inital_state):
    """Test unsafe_enable_checks() function."""
    set_initial_state(inital_state)
    mts.unsafe_enable_checks()
    assert mts.checks_enabled() is True


def test_enable_repr():
    assert mts.unsafe_enable_checks().__repr__() == "checks enabled"


def test_disable_repr():
    assert mts.unsafe_disable_checks().__repr__() == "checks disabled"


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_disable_checks(inital_state):
    """Test unsafe_disable_checks() function"""
    set_initial_state(inital_state)
    mts.unsafe_disable_checks()
    assert mts.checks_enabled() is False


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_enable_checks_context_manager(inital_state):
    """Test unsafe_enable_checks() context manager"""
    set_initial_state(inital_state)
    with mts.unsafe_enable_checks():
        assert mts.checks_enabled() is True
    assert mts.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_disable_checks_context_manager(inital_state):
    """Test unsafe_disable_checks() context manager"""
    set_initial_state(inital_state)
    with mts.unsafe_disable_checks():
        assert mts.checks_enabled() is False
    assert mts.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_nested_context_managers_enable_disable(inital_state):
    """Test nested enable-disable context managers"""
    set_initial_state(inital_state)
    with mts.unsafe_enable_checks():
        assert mts.checks_enabled() is True
        with mts.unsafe_disable_checks():
            assert mts.checks_enabled() is False
        assert mts.checks_enabled() is True
    assert mts.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_nested_context_managers_disable_enable(inital_state):
    """Test nested enable-disable context managers"""
    set_initial_state(inital_state)
    with mts.unsafe_disable_checks():
        assert mts.checks_enabled() is False
        with mts.unsafe_enable_checks():
            assert mts.checks_enabled() is True
        assert mts.checks_enabled() is False
    assert mts.checks_enabled() is inital_state


@pytest.mark.parametrize("original_state", [True, False])
def test_environment_variable(original_state):
    """Test environment variable METATENSOR_unsafe_disable_checks"""
    os.environ["METATENSOR_UNSAFE_DISABLE_CHECKS"] = str(not original_state)

    reload(mts.operations._checks)
    assert mts.checks_enabled() == original_state
