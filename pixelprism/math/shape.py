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
"""Tensor shape representation."""

# Imports
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass


__all__ = ["Shape", "Dim", "Dims", "AnyShape"]



Dim = int
Dims = Tuple[Dim, ...]
AnyShape = 'Shape' | Dims


def _num_elements(dims: Sequence[int | None]) -> int | None:
    """Compute the product of symbolic dimensions when possible.

    Parameters
    ----------
    dims : Sequence[int | None]
        Sequence of tensor dimensions, which may include ``None`` for unknown
        values.

    Returns
    -------
    int | None
        Number of elements or ``None`` when any dimension is unknown.
    """
    total = 1
    for dim in dims:
        total *= dim
    # end for
    return total
# end def _num_elements


class Shape:
    """
    Immutable descriptor for symbolic tensor shapes.

    A :class:`Shape` records the axis lengths associated with a math expression.
    Each axis can be a concrete ``int``.  Shapes are lightweight, hashable, and safe to
    share across nodes, ensuring downstream passes can reason about tensor
    metadata without mutating the original objects.

    Validation & normalization
    --------------------------
    The constructor eagerly validates every dimension through ``_check_dim`` to
    guarantee negative or otherwise malformed values never propagate past the
    creation site.  Internally, axes are stored as an immutable tuple, giving
    deterministic hashing/printing behaviour and keeping equality semantics
    simple.

    Convenience properties
    ----------------------
    ``dims`` exposes the canonical tuple representation, ``rank``/``n_dims``
    provide fast access to the tensor arity, and ``size`` returns the product
    of all known axes (``None`` when at least one axis is symbolic).  These
    helpers prevent ad‑hoc recomputation scattered around the code base.

    Compatibility helpers
    ---------------------
    Many operators need to verify that their operands can participate in
    elementwise arithmetic.  ``is_elementwise_compatible`` performs the check by
    comparing ranks and allowing either matching integers or symbolic ``None``
    values.  ``merge_elementwise`` builds on top of that by returning a new
    :class:`Shape` where each axis is the tightened version of both operands,
    raising :class:`ValueError`` when a conflict is detected.  Having these
    utilities on the shape class keeps validation logic centralized and
    consistent across operators.

    Subclassing
    -----------
    ``Shape`` is intentionally concrete.  Higher-level abstractions should wrap
    it rather than subclassing to avoid diverging validation paths.  Should
    additional metadata (like layout or batching semantics) be required, they
    can be attached via separate objects keyed by ``Shape`` instances.
    """

    def __init__(self, dims: Iterable[Dim]):
        """Initialize a Shape.

        Parameters
        ----------
        dims : Iterable[Dim]
            Iterable of dimension sizes to store in the shape.
        """
        dims_tuple = tuple(dims)
        for dim in dims_tuple:
            self._check_dim(dim)
        # end for
        self._dims: Dims = dims_tuple
        self._n_dims: int = len(dims_tuple)
    # end def __init__

    # region PROPERTIES

    @property
    def dims(self) -> Dims:
        """Return the dimensions tuple.

        Returns
        -------
        Dims
            Tuple describing tensor dimensions (each value is an ``int`` or
            ``None``).
        """
        return self._dims
    # end def dims

    @property
    def rank(self) -> int:
        """Return the tensor rank.

        Returns
        -------
        int
            Number of dimensions in the shape.
        """
        return len(self._dims)
    # end def rank

    @property
    def size(self) -> Optional[int]:
        """Return the total number of elements when known.

        Returns
        -------
        Optional[int]
            Number of elements represented by the shape.
        """
        return _num_elements(self._dims)
    # end def size

    @property
    def n_dims(self) -> int:
        """Return the number of dimensions.

        Returns
        -------
        int
            Number of dimensions in the shape.
        """
        return self.rank
    # end def n_dims

    # endregion PROPERTIES

    # region PUBLIC

    def as_tuple(self) -> tuple[Dim, ...]:
        """Return the shape as a tuple.

        Returns
        -------
        tuple[Dim, ...]
            The shape as a tuple of dimensions.
        """
        return tuple([d for d in self._dims])
    # end def as_tuple

    def is_elementwise_compatible(self, other: "Shape") -> bool:
        """Check whether elementwise operations are allowed.

        Parameters
        ----------
        other : Shape
            Shape to compare.

        Returns
        -------
        bool
            ``True`` when ranks are identical and dimensions are compatible for
            elementwise operations.
        """
        if self.rank != other.rank:
            return False
        # end if
        for dim_a, dim_b in zip(self._dims, other._dims):
            if not self._dims_equal(dim_a, dim_b):
                return False
            # end if
        # end for
        return True
    # end def is_elementwise_compatible

    def merge_elementwise(self, other: "Shape") -> "Shape":
        """Return the merged shape for elementwise operations.

        Parameters
        ----------
        other : Shape
            Shape to merge.

        Returns
        -------
        Shape
            Resulting shape compatible with both inputs.

        Raises
        ------
        ValueError
            If shapes are incompatible for elementwise operations.
        """
        if self.rank != other.rank:
            raise ValueError("Elementwise operations require equal ranks.")
        # end if
        merged = tuple(self._merge_dims(dim_a, dim_b) for dim_a, dim_b in zip(self._dims, other._dims))
        return Shape(merged)
    # end def merge_elementwise

    def matmul_result(self, other: "Shape") -> "Shape":
        """Return the result shape of a matrix multiplication.

        Parameters
        ----------
        other : Shape
            Right-hand operand shape.

        Returns
        -------
        Shape
            Resulting shape of the matrix multiplication.

        Raises
        ------
        ValueError
            If ranks are below 2, ranks differ, or inner dimensions are
            incompatible.
        """
        if self.rank < 2 or other.rank < 2:
            raise ValueError("MatMul requires rank >= 2 for both operands.")
        # end if
        if self.rank != other.rank:
            raise ValueError("MatMul requires operands with the same rank.")
        # end if
        batch_rank = self.rank - 2
        batch_dims: List[Dim] = []
        for idx in range(batch_rank):
            batch_dims.append(self._merge_dims(self._dims[idx], other._dims[idx]))
        # end for
        left_inner = self._dims[-1]
        right_inner = other._dims[-2]
        if not self._dims_equal(left_inner, right_inner):
            raise ValueError("Inner dimensions do not match for MatMul.")
        # end if
        result = tuple(batch_dims) + (self._dims[-2], other._dims[-1])
        return Shape(result)
    # end def matmul_result

    def concat_result(self, other: "Shape", axis: int) -> "Shape":
        """Return the result shape of concatenation along an axis.

        Parameters
        ----------
        other : Shape
            Shape to concatenate with.
        axis : int
            Concatenation axis.

        Returns
        -------
        Shape
            Concatenated shape along the specified axis.

        Raises
        ------
        ValueError
            If ranks differ or dimensions other than the concatenation axis are
            incompatible.
        """
        if self.rank != other.rank:
            raise ValueError("Concat requires operands with equal rank.")
        # end if
        axis_norm = self._normalize_axis(axis, self.rank)
        dims: List[Dim] = []
        for idx, (dim_a, dim_b) in enumerate(zip(self._dims, other._dims)):
            if idx == axis_norm:
                dims.append(self._sum_dims(dim_a, dim_b))
            else:
                dims.append(self._merge_dims(dim_a, dim_b))
            # end if
        # end for
        return Shape(tuple(dims))
    # end def concat_result

    def transpose(self, permutation: Sequence[int]) -> "Shape":
        """Return the shape after applying an axis permutation.

        Parameters
        ----------
        permutation : Sequence[int]
            Axis permutation describing the new order.

        Returns
        -------
        Shape
            Permuted shape following the given axis order.

        Raises
        ------
        ValueError
            If the permutation length is incorrect or contains invalid axis
            indices.
        """
        if len(permutation) != self.rank:
            raise ValueError("Permutation must include every axis exactly once.")
        # end if
        if sorted(permutation) != list(range(self.rank)):
            raise ValueError("Permutation contains invalid axis indices.")
        # end if
        dims = tuple(self._dims[idx] for idx in permutation)
        return Shape(dims)
    # end def transpose

    def can_reshape(self, new_shape: "Shape") -> bool:
        """Check whether reshape is symbolically valid.

        Parameters
        ----------
        new_shape : Shape
            Target shape to test against.

        Returns
        -------
        bool
            ``True`` if both shapes represent the same number of elements.
        """
        own_size = self.size
        target_size = new_shape.size
        return own_size == target_size
    # end def can_reshape

    def reshape(self, new_shape: "Shape") -> "Shape":
        """Return the symbolic shape after reshape.

        Parameters
        ----------
        new_shape : Shape
            Target shape.

        Returns
        -------
        Shape
            Target shape when the reshape is valid.

        Raises
        ------
        ValueError
            If the reshape would change the number of elements.
        """
        if not self.can_reshape(new_shape):
            raise ValueError("Reshape requires matching number of elements.")
        # end if
        return new_shape
    # end def reshape

    # endregion PUBLIC

    # region STATIC

    @staticmethod
    def scalar() -> "Shape":
        return Shape(())
    # end def scalar

    @staticmethod
    def vector(n: Dim) -> "Shape":
        return Shape((n,))
    # end def vector

    @staticmethod
    def matrix(n: Dim, m: Dim) -> "Shape":
        return Shape((n, m))
    # end def matrix

    @staticmethod
    def _check_dim(dim: Dim) -> None:
        """Validate a single dimension value.

        Parameters
        ----------
        dim : Dim
            Dimension to validate. Allowed values are non-negative integers.

        Raises
        ------
        ValueError
            If the dimension is negative or not an integer/``None``.
        """
        if dim is None:
            raise ValueError("Shape dimensions cannot be None.")
        # end if
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("Shape dimensions must be non-negative integers or None.")
        # end if
    # end def _check_dim

    @staticmethod
    def _dims_equal(dim_a: Dim, dim_b: Dim) -> bool:
        """Check whether two dimensions are symbolically compatible.

        Parameters
        ----------
        dim_a : Dim
            First dimension.
        dim_b : Dim
            Second dimension.

        Returns
        -------
        bool
            ``True`` if both dimensions can represent the same size.
        """
        return dim_a == dim_b
    # end def _dims_equal

    @staticmethod
    def _merge_dims(dim_a: Dim, dim_b: Dim) -> Dim:
        """Merge two dimensions into the most specific shared size.

        Parameters
        ----------
        dim_a : Dim
            First dimension.
        dim_b : Dim
            Second dimension.

        Returns
        -------
        Dim
            Dimension compatible with both inputs.

        Raises
        ------
        ValueError
            If the dimensions are incompatible.
        """
        if dim_a != dim_b:
            raise ValueError(f"Incompatible dimensions: {dim_a} vs {dim_b}.")
        # end if
        return dim_a
    # end def _merge_dims

    @staticmethod
    def _sum_dims(dim_a: Dim, dim_b: Dim) -> Dim:
        """Sum two dimensions symbolically.

        Parameters
        ----------
        dim_a : Dim
            First dimension.
        dim_b : Dim
            Second dimension.

        Returns
        -------
        Dim
            Sum of both dimensions when known, otherwise ``None``.
        """
        return dim_a + dim_b
    # end def _sum_dims

    @staticmethod
    def _normalize_axis(axis: int, rank: int, allow_new_axis: bool = False) -> int:
        """Normalize axis indices, supporting negatives.

        Parameters
        ----------
        axis : int
            Requested axis index, possibly negative.
        rank : int
            Tensor rank.
        allow_new_axis : bool, optional
            Whether an axis equal to the current rank is acceptable (used for
            operations that add a dimension). Defaults to ``False``.

        Returns
        -------
        int
            Normalized non-negative axis index within the allowed bounds.

        Raises
        ------
        ValueError
            If the axis is outside the valid range.
        """
        upper = rank + (1 if allow_new_axis else 0)
        if not -upper <= axis < upper:
            raise ValueError(f"Axis {axis} out of bounds for rank {rank}.")
        # end if
        if axis < 0:
            axis += upper
        # end if
        return axis
    # end def _normalize_axis

    @staticmethod
    def stack_shape(shapes: Sequence["Shape"], axis: int) -> "Shape":
        """Return the result shape of stacking tensors.

        Parameters
        ----------
        shapes : Sequence[Shape]
            Shapes of tensors to stack. All shapes must be elementwise
            compatible.
        axis : int
            Axis index for the new dimension.

        Returns
        -------
        Shape
            Resulting stacked shape including the new dimension size.

        Raises
        ------
        ValueError
            If no shapes are provided or shapes are incompatible.
        """
        if not shapes:
            raise ValueError("Stack requires at least one shape.")
        # end if
        base = shapes[0]
        axis_norm = Shape._normalize_axis(axis, base.rank, allow_new_axis=True)
        for shape in shapes[1:]:
            base = base.merge_elementwise(shape)
        # end for
        dims = list(base.dims)
        dims.insert(axis_norm, len(shapes))
        return Shape(tuple(dims))
    # end def stack_shape

    def copy(self):
        """Return a copy of the shape."""
        pass
    # end def copy

    # endregion STATIC

    # region OVERRIDE

    def __len__(self) -> int:
        """Return the rank for len().

        Returns
        -------
        int
            Tensor rank.
        """
        return self.rank
    # end def __len__

    def __getitem__(self, index: int) -> Dim:
        """Return the dimension at the provided index.

        Parameters
        ----------
        index : int
            Axis index to access.

        Returns
        -------
        Dim
            Dimension size at the given axis.
        """
        return self._dims[index]
    # end def __getitem__

    def __eq__(self, other: object) -> bool:
        """Compare shapes for equality.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when both shapes share identical dimensions.
        """
        if not isinstance(other, Shape):
            return False
        # end if
        return self._dims == other._dims
    # end def __eq__

    def __hash__(self) -> int:
        """Return a hash for the shape.

        Returns
        -------
        int
            Hash value derived from the dimensions.
        """
        return hash(self._dims)
    # end def __hash__

    def __repr__(self) -> str:
        """Return the repr() form of the shape.

        Returns
        -------
        str
            Developer-friendly representation including raw dimensions.
        """
        return f"{self._dims}"
    # end def __repr__

    def __str__(self) -> str:
        """Return the str() form of the shape.

        Returns
        -------
        str
            Readable representation.
        """
        return self.__repr__()
    # end def __str__

    # endregion OVERRIDE

# end class Shape

