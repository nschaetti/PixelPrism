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
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Runtime tensor value container."""

# Imports
from __future__ import annotations
from copy import deepcopy
from typing import Any
from .shape import Shape


__all__ = ["Value"]


class Value:
    """Mutable runtime tensor value."""

    def __init__(
            self,
            data: Any,
            shape: Shape,
            dtype: Any | None = None,
            ops: Any | None = None
    ):
        """Initialize a Value.

        Args:
            data: Backend tensor data.
            shape: Symbolic tensor shape.
            dtype: Backend dtype metadata.
            ops: Optional backend operations helper.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be an instance of Shape.")
        # end if
        self._data = data
        self._shape = shape
        self._dtype = dtype
        self._ops = ops
    # end def __init__

    @property
    def shape(self) -> Shape:
        """Return the associated Shape.

        Returns:
            Shape: Value shape.
        """
        return self._shape
    # end def shape

    @property
    def dtype(self) -> Any:
        """Return the backend dtype metadata.

        Returns:
            Any: Backend dtype descriptor.
        """
        return self._dtype
    # end def dtype

    def get(self) -> Any:
        """Return the backend data.

        Returns:
            Any: Underlying backend tensor data.
        """
        return self._data
    # end def get

    def set(self, data: Any) -> None:
        """Update the backend data.

        Args:
            data: New backend data sharing the same shape.
        """
        self._data = data
    # end def set

    def copy(self) -> "Value":
        """Return a deep copy of the Value.

        Returns:
            Value: Independent Value containing copied data.
        """
        data_copy = deepcopy(self._data)
        return Value(data_copy, self._shape, self._dtype, self._ops)
    # end def copy

# end class Value

