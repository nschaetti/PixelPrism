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


from __future__ import annotations

from types import EllipsisType
from typing import Sequence, Tuple, Union, TypeAlias
import numpy as np


__all__ = [
    "ScalarLike",
    "ScalarListLike",
    "IndexAtom",
    "Index",
    "TensorLike",
]


# Type numeric
# Scalar Python/NumPy accepted as atomic numeric values in symbolic expressions.
# Includes bool and complex to match Tensor/domain operator coverage.
ScalarLike = float | int | np.number | bool | complex

# Recursive Python list representation of scalar tensor-like data.
# Example: [1, 2, [3, 4.5]]
ScalarListLike: TypeAlias = list[Union[ScalarLike, "ScalarListLike"]]

# Tensor data
# are Any accepted tensor payload: scalar, nested Python lists, or NumPy array.
TensorLike: TypeAlias = Union[ScalarLike, ScalarListLike, np.ndarray]


# Single indexing atom accepted in one axis position.
# - int: pick one element on an axis
# - slice: ranged selection
# - None: insert a new axis (numpy.newaxis)
# - Ellipsis: expand to remaining axes
# - Sequence[int]: integer fancy indexing on one axis
# - Sequence[bool]: boolean mask indexing on one axis
IndexAtom: TypeAlias = Union[
    int,
    slice,
    None,
    EllipsisType,
    Sequence[int],
    Sequence[bool],
]

# Full tensor index:
# - a single atom (e.g. x[3], x[1:5], x[..., 0])
# - or a tuple of atoms for multi-axis indexing (e.g. x[:, 1, ..., None])
Index: TypeAlias = Union[
    IndexAtom,
    Tuple[IndexAtom, ...],
]
