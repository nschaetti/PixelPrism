"""Tests for pixelprism.math.Shape."""

from pixelprism.math import Shape


def test_shape_properties():
    """Validate basic shape properties."""
    shape = Shape((2, None, 4))
    assert shape.rank == 3
    assert shape.size is None
    assert shape[0] == 2
    assert shape.as_tuple() == (2, None, 4)
    assert str(shape) == "Shape(2x?x4)"
    assert repr(shape) == "Shape((2, None, 4))"
    assert shape == Shape((2, None, 4))
    assert shape != Shape((2, 1, 4))
# end def test_shape_properties


def test_shape_elementwise_and_concat():
    """Test elementwise merge and concatenation."""
    left = Shape((2, 3))
    right = Shape((2, 3))
    assert left.is_elementwise_compatible(right)
    merged = left.merge_elementwise(right)
    assert merged.as_tuple() == (2, 3)
    concat = left.concat_result(right, axis=1)
    assert concat.as_tuple() == (2, 6)
# end def test_shape_elementwise_and_concat


def test_shape_stack_matmul_transpose():
    """Validate stack, matmul, and transpose helpers."""
    shapes = [Shape((2, 4)), Shape((2, 4))]
    stacked = Shape.stack_result(shapes, axis=0)
    assert stacked.as_tuple() == (2, 2, 4)

    left = Shape((2, 3, 4))
    right = Shape((2, 4, 5))
    matmul = left.matmul_result(right)
    assert matmul.as_tuple() == (2, 3, 5)

    transposed = left.transpose((1, 0, 2))
    assert transposed.as_tuple() == (3, 2, 4)
# end def test_shape_stack_matmul_transpose


def test_shape_reshape_validation():
    """Ensure reshape validation checks element counts."""
    shape = Shape((2, 3))
    target = Shape((3, 2))
    assert shape.can_reshape(target)
    assert shape.reshape(target) is target
# end def test_shape_reshape_validation

