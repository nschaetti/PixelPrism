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

# Imports
import pytest
from pixelprism.math import helpers


class _SequenceLike:
    """Simple iterable with __len__ used to test helper behavior."""

    def __init__(self, data):
        self._data = list(data)
    # end def __init__

    def __iter__(self):
        return iter(self._data)
    # end def __iter_-

    def __len__(self):
        return len(self._data)
    # end def __len__

# end class _SequenceLike


class _DummyValue:
    """Stub value that exposes an _ops attribute."""

    def __init__(self, ops=None):
        self._ops = ops
    # end def __init__

# end class _DummyValue


def test_is_sequence_like_detects_iterables():
    assert helpers.is_sequence_like([1, 2])
    assert helpers.is_sequence_like((1, 2))
    assert not helpers.is_sequence_like(10)
# end def test_is_sequence_like_detects_iterables


def test_as_sequence_converts_iterables_and_raises_on_scalars():
    seq = [1, 2]
    assert helpers.as_sequence(seq) is seq
    assert helpers.as_sequence((1, 2)) == [1, 2]
    assert helpers.as_sequence(_SequenceLike((3, 4, 5))) == [3, 4, 5]
    with pytest.raises(TypeError):
        helpers.as_sequence(42)
# end def test_as_sequence_converts_iterables_and_raises_on_scalars


def test_infer_dims_from_data_nested_and_empty():
    assert helpers.infer_dims_from_data([[1, 2, 3], [4, 5, 6]]) == (2, 3)
    assert helpers.infer_dims_from_data(5) == ()
    assert helpers.infer_dims_from_data([]) == (0,)
# end def test_infer_dims_from_data_nested_and_empty


def test_num_elements_handles_unknown_dimensions():
    assert helpers.num_elements((2, 3, 4)) == 24
# end def test_num_elements_handles_unknown_dimensions


def test_flatten_simple_and_build_from_flat_are_inverse_operations():
    nested = [[1, 2], [3, [4]]]
    flat = helpers.flatten_simple(nested)
    rebuilt = helpers.build_from_flat(iter(flat), (2, 2))
    assert flat == [1, 2, 3, 4]
    assert rebuilt == [[1, 2], [3, 4]]
# end def test_flatten_simple_and_build_from_flat_are_inverse_operations


def test_reshape_python_success_and_errors():
    data = [1, 2, 3, 4]
    reshaped = helpers.reshape_python(data, (2, 2))
    assert reshaped == [[1, 2], [3, 4]]
    with pytest.raises(RuntimeError):
        helpers.reshape_python(data, (2, None))
    with pytest.raises(ValueError):
        helpers.reshape_python([1, 2, 3], (2, 2))
# end def test_reshape_python_success_and_errors


def test_concat_python_handles_axis_zero_and_nested_axes():
    assert helpers.concat_python(([1, 2], [3, 4]), axis=0) == [1, 2, 3, 4]
    values = (
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    )
    assert helpers.concat_python(values, axis=1) == [[1, 2, 5, 6], [3, 4, 7, 8]]
    with pytest.raises(ValueError):
        helpers.concat_python(([[1], [2]], [[3]]), axis=1)
# end def test_concat_python_handles_axis_zero_and_nested_axes


def test_stack_python_inserts_axis_and_validates_lengths():
    assert helpers.stack_python(([1, 2], [3, 4]), axis=0) == [[1, 2], [3, 4]]
    assert helpers.stack_python(([1, 2], [3, 4]), axis=1) == [[1, 3], [2, 4]]
    with pytest.raises(ValueError):
        helpers.stack_python(([[1], [2]], [[3]]), axis=1)
# end def test_stack_python_inserts_axis_and_validates_lengths


def test_ravel_and_unravel_index_are_consistent():
    dims = (3, 4)
    index = 5
    coords = helpers.unravel_index(index, dims)
    assert coords == (1, 1)
    assert helpers.ravel_index(coords, dims) == index
    assert helpers.unravel_index(0, ()) == ()
# end def test_ravel_and_unravel_index_are_consistent


def test_transpose_python_basic_and_zero_dimension_cases():
    data = [[1, 2, 3], [4, 5, 6]]
    assert helpers.transpose_python(data, (1, 0)) == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]
    zero_dim = [[], []]
    assert helpers.transpose_python(zero_dim, (1, 0)) == []
# end def test_transpose_python_basic_and_zero_dimension_cases


def test_select_ops_returns_first_available_helper():
    first = _DummyValue(ops="backend")
    second = _DummyValue()
    assert helpers.select_ops([first, second]) == "backend"
    assert helpers.select_ops([_DummyValue(), _DummyValue()]) is None
# end def test_select_ops_returns_first_available_helper
