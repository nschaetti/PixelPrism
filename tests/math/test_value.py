"""Tests for pixelprism.math.Value."""

from pixelprism.math import Shape, Value


def test_value_access_and_copy():
    """Ensure Value exposes get/set/copy semantics."""
    shape = Shape((2, 2))
    value = Value([[1, 2], [3, 4]], shape, dtype="int")
    assert value.shape is shape
    assert value.dtype == "int"
    assert value.get()[0][0] == 1

    value.set([[5, 6], [7, 8]])
    assert value.get()[0][0] == 5

    clone = value.copy()
    assert clone.get() == value.get()
    assert clone is not value
# end def test_value_access_and_copy

