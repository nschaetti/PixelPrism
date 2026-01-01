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
# This program is free software : you can redistribute it and/or modify
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
"""Runtime tensor value container."""


# Imports
from __future__ import annotations
from copy import deepcopy
from typing import Any, Tuple

from .shape import Shape
from .dtype import DType


__all__ = ["Value"]


class Value:
    """Mutable runtime tensor value."""

    def __init__(
            self,
            data: Any,
            shape: Shape,
            dtype: DType,
            mutable: bool = True,
            ops: Any | None = None,
    ):
        """Initialize a Value.

        Parameters
        ----------
        data : Any
            Backend tensor data or array-like object.
        shape : Shape
            Symbolic tensor shape describing ``data``.
        dtype : DType
            Backend dtype metadata associated with ``data``.
        mutable : bool, optional
            Define whether the value can be modified.
        ops : Any | None, optional
            Optional backend operations helper carrying tensor operations used
            by higher-level functions. May be ``None`` when no backend is
            registered.

        Raises
        ------
        TypeError
            If ``shape`` is not an instance of :class:`Shape`.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be an instance of Shape.")
        # end if
        self._data = data
        self._shape = shape
        self._dtype = dtype
        self._mutable = mutable
        self._ops = ops
        self._check_data_shape(data)
    # end def __init__

    # region PROPERTIES

    @property
    def shape(self) -> Shape:
        """Return the associated Shape.

        Returns
        -------
        Shape
            Symbolic shape describing the value.
        """
        return self._shape
    # end def shape

    @property
    def dtype(self) -> DType:
        """Return the backend dtype metadata.

        Returns
        -------
        Any
            Backend dtype descriptor stored for this value.
        """
        return self._dtype
    # end def dtype

    @property
    def mutable(self) -> bool:
        """Return whether the value can be modified.

        Returns
        -------
        bool
            Whether the value can be modified.
        """
        return self._mutable
    # end def mutable

    # endregion PROPERTIES

    # region PUBLIC

    def get(self) -> Any:
        """Return the backend data.

        Returns
        -------
        Any
            Underlying backend tensor data.
        """
        return self._data
    # end def get

    def set(self, data: Any) -> None:
        """Update the backend data.

        Parameters
        ----------
        data : Any
            New backend data sharing the same shape contract.
        """
        if not self._mutable:
            raise RuntimeError("Trying to modify an immutable Value.")
        # end if
        self._data = data
    # end def set

    def copy(self) -> "Value":
        """Return a deep copy of the Value.

        Returns
        -------
        Value
            Independent Value containing deep-copied data and the same metadata.
        """
        data_copy = deepcopy(self._data)
        return Value(
            data=data_copy,
            shape=self._shape,
            dtype=self._dtype,
            mutable=self._mutable,
            ops=self._ops
        )
    # end def copy

    # endregion PUBLIC

    # region PRIVATE

    def _check_data_shape(
            self,
            data: Any
    ) -> None:
        """Best-effort check that data matches the declared shape.

        Parameters
        ----------
        data : Any
            Backend tensor data or array-like object.
        """
        if not isinstance(data, (list, tuple)):
            return
        # end if

        # Infer data dimension
        def infer_dim(d) -> Tuple:
            if not isinstance(d, (list, tuple)):
                return ()
            # end if
            if len(d) == 0:
                return (0,)
            # end if
            entry_dims = list(map(infer_dim, d))
            if not all(v == entry_dims[0] for v in entry_dims):
                raise ValueError("Cannot infer shape from nested sequences.")
            # end if
            return (len(d),) + infer_dim(d[0])
        # end def infer_dim

        # Infer data shape
        data_shape = infer_dim(data)
        given_shape = self._shape.as_tuple()

        # Check data dimensions
        if len(data_shape) != self._shape.rank:
            raise ValueError(f"Expected shape {given_shape}, got {data_shape}.")
        # end if

        for dim_i, (dim_a, dim_b) in enumerate(zip(data_shape, given_shape)):
            if dim_b is not None and dim_a != dim_b:
                raise ValueError(f"Dimension mismatch at index {dim_i}: {dim_a} vs {dim_b}.")
            # end if
        # end for
    # end def _check_data_shape

    # endregion PRIVATE

    # region OVERRIDE

    def __repr__(self) -> str:
        """Return a string representation of the Value.

        Returns
        -------
        str
            A string representation of the Value.
        """
        if self._mutable:
            return f"value({self._data})"
        else:
            return f"const({self._data}, immutable)"
        # end if
    # end def __repr__

    def __str__(self) -> str:
        """Return a string representation of the Value.

        Returns
        -------
        str
            A string representation of the Value.
        """
        return f"value({self._data})"
    # end def __str__

    # endregion OVERRIDE

# end class Value

