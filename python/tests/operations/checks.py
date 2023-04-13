import os
from importlib import reload

import pytest

import equistore


def set_initial_state(state):
    equistore.operations.checks._CHECKS_ENABLED = state


@pytest.mark.parametrize("inital_state", [True, False])
def test_checks_enabled(inital_state):
    """Test checks_enabled() function."""
    set_initial_state(inital_state)
    assert equistore.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_enable_checks(inital_state):
    """Test unsafe_enable_checks() function."""
    set_initial_state(inital_state)
    equistore.unsafe_enable_checks()
    assert equistore.checks_enabled() is True


def test_enable_repr():
    assert equistore.unsafe_enable_checks().__repr__() == "checks enabled"


def test_disable_repr():
    assert equistore.unsafe_disable_checks().__repr__() == "checks disabled"


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_disable_checks(inital_state):
    """Test unsafe_disable_checks() function"""
    set_initial_state(inital_state)
    equistore.unsafe_disable_checks()
    assert equistore.checks_enabled() is False


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_enable_checks_context_manager(inital_state):
    """Test unsafe_enable_checks() context manager"""
    set_initial_state(inital_state)
    with equistore.unsafe_enable_checks():
        assert equistore.checks_enabled() is True
    assert equistore.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_disable_checks_context_manager(inital_state):
    """Test unsafe_disable_checks() context manager"""
    set_initial_state(inital_state)
    with equistore.unsafe_disable_checks():
        assert equistore.checks_enabled() is False
    assert equistore.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_nested_context_managers_enable_disable(inital_state):
    """Test nested enable-disable context managers"""
    set_initial_state(inital_state)
    with equistore.unsafe_enable_checks():
        assert equistore.checks_enabled() is True
        with equistore.unsafe_disable_checks():
            assert equistore.checks_enabled() is False
        assert equistore.checks_enabled() is True
    assert equistore.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_nested_context_managers_disable_enable(inital_state):
    """Test nested enable-disable context managers"""
    set_initial_state(inital_state)
    with equistore.unsafe_disable_checks():
        assert equistore.checks_enabled() is False
        with equistore.unsafe_enable_checks():
            assert equistore.checks_enabled() is True
        assert equistore.checks_enabled() is False
    assert equistore.checks_enabled() is inital_state


@pytest.mark.parametrize("original_state", [True, False])
def test_environment_variable(original_state):
    """Test environment variable EQUISTORE_unsafe_disable_checks"""
    os.environ["EQUISTORE_UNSAFE_DISABLE_CHECKS"] = str(not original_state)

    reload(equistore.operations.checks)
    assert equistore.checks_enabled() == original_state
