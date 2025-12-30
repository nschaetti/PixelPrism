"""Tests for pixelprism.math.Op."""

from typing import Mapping, Tuple

from pixelprism.math import Op, Shape, Value


class IncrementOp(Op):
    """Simple operation that increments a scalar."""

    def __init__(self, child):
        super().__init__((child,), child.shape, child.dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Add one to the child value."""
        (value,) = child_values
        data = value.get() + 1
        return Value(data, self.shape, self.dtype)
    # end def _eval_impl
# end class IncrementOp


class DummyExpr(Op):
    """Minimal expression to feed IncrementOp."""

    def __init__(self, value: Value):
        super().__init__((), value.shape, value.dtype)
        self._value = value
    # end def __init__

    def evaluate(self, env: Mapping):  # type: ignore[override]
        """Return the stored value."""
        return self._value
    # end def evaluate

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:  # type: ignore[override]
        """Direct passthrough."""
        return self._value
    # end def _eval_impl
# end class DummyExpr


def test_op_evaluation_delegation():
    """Ensure Op.evaluate delegates to _eval_impl."""
    value = Value(1, Shape(()), "int")
    source = DummyExpr(value)
    op = IncrementOp(source)
    result = op.evaluate({})
    assert result.get() == 2
# end def test_op_evaluation_delegation

