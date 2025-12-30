"""Tests for pixelprism.math.MatMul."""

from pixelprism.math import MatMul
from tests.math.utils import FakeTensor, ValueExpr, make_value


def test_matmul_shape_and_data():
    """Ensure MatMul validates shapes and performs multiplication."""
    left = make_value(FakeTensor(2), (1, 2, 3))
    right = make_value(FakeTensor(4), (1, 3, 5))
    expr = MatMul(ValueExpr(left), ValueExpr(right))
    assert expr.shape.as_tuple() == (1, 2, 5)
    result = expr.evaluate({})
    assert result.get().value == 8
# end def test_matmul_shape_and_data
