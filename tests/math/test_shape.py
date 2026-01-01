# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2025 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Tests for pixelprism.math.Shape."""

# Imports
import pytest
from pixelprism.math import Shape


def test_shape_init():
    shape = Shape((2, 3))
    assert shape.dims == (2, 3)
# end def test_shape_init

def test_shape_init_neg_dim():
    with pytest.raises(ValueError):
        Shape((2, -3, 4))
    # end with
# end def test_shape_init_neg_dim

def test_shape_init_none_dims():
    with pytest.raises(ValueError):
        Shape((1, None))
    # end with
# end def test_shape_init_empty_dims

def test_shape_properties_numeric_dimensions():
    shape = Shape((2, 3, 4))
    assert shape.rank == 3
    assert shape.n_dims == 3
    assert shape.size == 24
    assert shape.dims == (2, 3, 4)
    assert shape.as_tuple() == (2, 3, 4)
    assert shape[0] == 2
    assert len(shape) == 3
    assert shape == Shape((2, 3, 4))
# end def test_shape_properties_numeric_dimensions

def test_shape_properties_symbolic_dimensions():
    n = 2
    m = 5
    shape = Shape((n, m))
    assert shape.rank == 2
    assert shape.n_dims == 2
    assert shape.size == n * m
    assert shape.dims == (n, m)
    assert shape.as_tuple() == (n, m)
    assert shape[1] == m
    assert shape == Shape((2, 5))
# end def test_shape_properties_symbolic_dimensions

def test_shape_static_creators():
    scalar = Shape.scalar()
    vector = Shape.vector(3)
    matrix = Shape.matrix(2, 4)
    assert scalar.dims == ()
    assert scalar.size == 1
    assert vector == Shape((3,))
    assert matrix == Shape((2, 4))
# end def test_shape_static_creators

def test_shape_operations_numeric_dimensions():
    shape = Shape((2, 3))
    same = Shape((2, 3))
    merged = shape.merge_elementwise(same)
    assert merged == shape

    left = Shape((2, 3))
    right = Shape((3, 4))
    assert left.matmul_result(right) == Shape((2, 4))

    concat = Shape((2, 3)).concat_result(Shape((2, 5)), axis=1)
    assert concat == Shape((2, 8))

    transposed = Shape((2, 3, 4)).transpose([2, 0, 1])
    assert transposed == Shape((4, 2, 3))

    stacked = Shape.stack_shape([Shape((2, 3)), Shape((2, 3)), Shape((2, 3))], axis=-1)
    assert stacked == Shape((2, 3, 3))

    assert Shape((2, 3)).can_reshape(Shape((6,)))
    assert not Shape((2, 3)).can_reshape(Shape((3, 3)))
    assert Shape((2, 3)).reshape(Shape((6,))) == Shape((6,))

    with pytest.raises(ValueError):
        Shape((2, 3)).reshape(Shape((3, 3)))
    # end with
# end def test_shape_operations_numeric_dimensions

def test_shape_operations_symbolic_dimensions():
    batch = 4
    inner = 3
    outer = 5

    a = Shape((batch, inner))
    b = Shape((batch, inner))
    assert a.merge_elementwise(b) == a

    matmul_result = Shape((batch, inner)).matmul_result(Shape((inner, outer)))
    assert matmul_result == Shape((batch, outer))

    concat_result = Shape((batch, 2)).concat_result(Shape((batch, 4)), axis=1)
    assert concat_result == Shape((batch, 6))

    stacked = Shape.stack_shape([Shape((batch, 3)), Shape((batch, 3))], axis=0)
    assert stacked == Shape((2, batch, 3))

    assert Shape((batch, inner)).can_reshape(Shape((batch, inner)))
    assert Shape((batch, inner)).reshape(Shape((batch, inner))) == Shape((batch, inner))
# end def test_shape_operations_symbolic_dimensions

def test_is_elementwise_compatible():
    assert Shape((2, 3)).is_elementwise_compatible(Shape((2, 3)))
    assert not Shape((2, 3)).is_elementwise_compatible(Shape((2, 4)))
# end def test_is_elementwise_compatible


