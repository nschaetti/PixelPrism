
import importlib
from typing import Optional
import numpy as np
import numpy.testing as npt
import pytest

from pixelprism.math.context import (
    Context,
    ContextAlreadyActiveError,
    ContextNotActiveError,
    ContextValueNotFoundError,
    dump_context_tree,
    restore_context_stack,
    snapshot_context_stack,
    trace_variable,
)
from pixelprism.math.dtype import DType, to_numpy
from pixelprism.math.tensor import Tensor

ctx = importlib.import_module("pixelprism.math.context")


@pytest.fixture(autouse=True)
def reset_context_stack():
    """Ensure every test begins from a clean context stack.

    Notes
    -----
    Snapshots the current stack before each test, clears the root context, and
    restores the snapshot afterward so global state never leaks across tests.
    """
    snapshot = snapshot_context_stack()
    ctx.root().clear()
    yield
    restore_context_stack(snapshot)
# end def reset_context_stack


def make_tensor(value, dtype: Optional[DType] = DType.R, mutable=True) -> Tensor:
    """Create a tensor helper for tests."""
    array = np.array(value, dtype=to_numpy(dtype))
    return Tensor(data=array, dtype=dtype)
# end def make_tensor


def test_context_manager_switches_current():
    """Validate that entering and exiting contexts swaps the active stack.

    Notes
    -----
    Ensures ``Context.current`` points to the managed context inside the
    ``with`` block and reverts to the root afterwards.
    """
    root = Context.root()
    child = Context()
    assert not child.is_active

    with child:
        assert Context.current() is child
        assert child.is_active
    # end with

    assert Context.current() is root
    assert not child.is_active
# end def test_context_manager_switches_current

def test_context_reenter_raises():
    """Check that re-entering an active context raises the custom error.

    Notes
    -----
    Calling ``__enter__`` twice should raise ``ContextAlreadyActiveError``.
    """
    c = Context()
    with pytest.raises(ContextAlreadyActiveError):
        with c:
            c.__enter__()
        # end with
    # end with
# end def test_context_reenter_raises

def test_context_exit_without_enter():
    """Check that exiting an inactive context raises ``ContextNotActiveError``."""
    c = Context()
    with pytest.raises(ContextNotActiveError):
        c.__exit__(None, None, None)
    # end with
# end def test_context_exit_without_enter

def test_set_lookup_and_get_helpers():
    """Exercise setter/getter helpers on the active context.

    Notes
    -----
    Verifies that ``set_value``, ``lookup``, ``get_value``, and ``create_variable``
    share the same underlying storage.
    """
    weights = make_tensor([1.0, 2.0])
    ctx.set_value("tensor:w", weights)
    assert ctx.lookup("tensor:w") is weights
    assert ctx.get_value("tensor:w") is weights

    child = Context()
    assert child.lookup("tensor:w") is weights

    ctx.create_variable("tensor:b")
    assert ctx.lookup("tensor:b") is None
# end def test_set_lookup_and_get_helpers

def test_get_missing_value_raises():
    """Ensure missing lookups raise ``ContextValueNotFoundError``."""
    with pytest.raises(ContextValueNotFoundError):
        ctx.get_value("missing")
    # end with
# end def test_get_missing_value_raises

def test_clear_remove_and_remove_deep():
    """Cover removal operations on contexts and their parents.

    Notes
    -----
    Ensures ``remove``, ``remove_deep``, and ``clear`` mutate the underlying
    dictionaries as expected.
    """
    root = Context.root()
    root.set("shared", make_tensor([0.0]))
    child = Context()
    child.set("local", make_tensor([1.0]))

    child.remove("local")
    assert child.lookup("local") is None

    child.remove_deep("shared")
    assert root.lookup("shared") is None

    child.set("local", make_tensor([2.0]))
    child.clear()
    assert child.lookup("local") is None
# end def test_clear_remove_and_remove_deep

def test_items_merge_parent_values():
    """Validate that ``items`` merges values from parent contexts."""
    root = Context.root()
    root.set("root-only", make_tensor([1.0]))
    child = Context()
    child.set("child-only", make_tensor([2.0]))

    merged = dict(child.items())
    assert set(merged.keys()) == {"root-only", "child-only"}
    assert merged["root-only"] is root.lookup("root-only")
    assert merged["child-only"] is child.lookup("child-only")
# end def test_items_merge_parent_values

def test_trace_reports_resolution_chain():
    """Test diagnostic tracing provides the expected metadata.

    Notes
    -----
    Ensures ``Context.trace`` identifies the first context providing a value
    and records chain entries for each ancestor.
    """
    root = Context.root()
    root.set("tensor:w", make_tensor([0.0], dtype=DType.R))
    child = Context()
    child.set("tensor:w", make_tensor([1.0], dtype=DType.R))

    info = child.trace("tensor:w")
    assert info["found"] is True
    assert info["value"] is child.lookup("tensor:w")
    assert info["chain"][0]["has_value"] is True
    assert info["chain"][1]["has_value"] is True

    missing = child.trace("absent")
    assert missing["found"] is False
    assert all(entry["has_value"] is False for entry in missing["chain"])
# end def test_trace_reports_resolution_chain

def test_trace_variable_helper_uses_current_context():
    """Ensure ``trace_variable`` defaults to the active context.

    Notes
    -----
    Enters a child context and confirms the helper inspects the same stack.
    """
    child = Context()
    child.set("tensor:w", make_tensor([5.0]))
    with child:
        info = trace_variable("tensor:w")
        assert info["found"] is True
        assert info["context_id"] == id(child)
    # end with
# end def test_trace_variable_helper_uses_current_context

def test_dump_tree_formats_output():
    """Check that ``dump_tree`` renders context metadata and values."""
    child = Context()
    child.set("tensor:z", make_tensor([[1.0, 2.0], [3.0, 4.0]]))
    dump = child.dump_tree()
    assert "- Context" in dump
    assert "tensor:z" in dump
    assert "shape" in dump

    minimal = child.dump_tree(include_values=False)
    assert "tensor:z" not in minimal
# end def test_dump_tree_formats_output

def test_dump_context_tree_helper_matches_method():
    """Verify the module-level helper mirrors ``Context.dump_tree``."""
    child = Context()
    child.set("tensor:z", make_tensor([1.0]))
    with child:
        assert dump_context_tree() == child.dump_tree()
    # end with
# end def test_dump_context_tree_helper_matches_method

def test_serialization_round_trip():
    """Round-trip a context hierarchy through ``to_dict``/``from_dict``."""
    root = Context.root()
    root.set("root", make_tensor([0.0], dtype=DType.R))
    child = Context()
    child.set("leaf", make_tensor([1.0, 2.0], mutable=False))

    as_dict = child.to_dict()
    restored = Context.from_dict(as_dict)

    npt.assert_allclose(restored.get("leaf").value, child.get("leaf").value)
    npt.assert_allclose(restored.lookup("root").value, root.lookup("root").value)
# end def test_serialization_round_trip

def test_snapshot_restore_reestablishes_stack():
    """Ensure snapshot/restore rebuilds the active stack faithfully.

    Notes
    -----
    Takes a snapshot inside a nested context, mutates the stack, and confirms
    ``restore_context_stack`` reinstates the captured structure.
    """
    root = Context.root()
    root.set("root", make_tensor([0.0]))
    with Context() as nested:
        nested.set("leaf", make_tensor([1.0]))
        snapshot = ctx.snapshot_context_stack(nested)
    # end with
    # mutate current stack after snapshot
    root.set("root", make_tensor([2.0]))

    restored_current = restore_context_stack(snapshot)
    assert Context.current() is restored_current
    npt.assert_allclose(ctx.get_value("leaf").value, [1.0])
    npt.assert_allclose(ctx.get_value("root").value, [0.0])
# end def test_snapshot_restore_reestablishes_stack

def test_push_and_pop_helpers_manage_stack():
    """Validate ``push_context`` and ``pop_context`` update the stack."""
    pushed = ctx.push_context()
    try:
        assert Context.current() is pushed
        pushed.set("x", make_tensor([1.0]))
    finally:
        ctx.pop_context(pushed)
    # end try
    assert Context.current() is Context.root()
# end def test_push_and_pop_helpers_manage_stack

def test_new_context_defaults_and_accessors():
    """Check helper functions for creating and accessing contexts."""
    current = Context.current()
    fresh = ctx.new_context()
    assert fresh.parent is current
    assert fresh.is_active is False
    assert ctx.context() is Context.current()
    assert ctx.root() is Context.root()
# end def test_new_context_defaults_and_accessors

def test_lookup_propagates_to_parent():
    """Ensure ``lookup`` walks the parent chain."""
    root = Context.root()
    root.set("shared", make_tensor([7.0]))
    child = Context()
    assert child.lookup("shared") is root.lookup("shared")
# end def test_lookup_propagates_to_parent

def test_create_variable_accepts_none():
    """Confirm ``create_variable`` can declare optional placeholders."""
    ctx.set_value("optional")
    assert ctx.lookup("optional") is None
# end def test_create_variable_accepts_none
