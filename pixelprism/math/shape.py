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
"""Symbolic tensor shape representation."""

# Imports
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
from ._helpers import num_elements


Dim = Optional[int]
Dims = Tuple[Dim, ...]


__all__ = ["Shape", "Dim", "Dims"]


class Shape:
    """Represents a symbolic tensor shape."""

    def __init__(self, dims: Iterable[Dim]):
        """Initialize a Shape.

        Args:
            dims: Iterable of dimension sizes.
        """
        dims_tuple = tuple(dims)
        for dim in dims_tuple:
            self._check_dim(dim)
        # end for
        self._dims: Dims = dims_tuple
    # end def __init__

    @staticmethod
    def _check_dim(dim: Dim) -> None:
        """Validate a single dimension value.

        Args:
            dim: Dimension to validate.

        Raises:
            ValueError: If the dimension is invalid.
        """
        if dim is None:
            return
        # end if
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("Shape dimensions must be non-negative integers or None.")
        # end if
    # end def _check_dim

    @staticmethod
    def _dims_equal(dim_a: Dim, dim_b: Dim) -> bool:
        """Check whether two dimensions are symbolically compatible.

        Args:
            dim_a: First dimension.
            dim_b: Second dimension.

        Returns:
            bool: True if both dimensions can represent the same size.
        """
        if dim_a is None or dim_b is None:
            return True
        # end if
        return dim_a == dim_b
    # end def _dims_equal

    @staticmethod
    def _merge_dims(dim_a: Dim, dim_b: Dim) -> Dim:
        """Merge two dimensions into the most specific shared size.

        Args:
            dim_a: First dimension.
            dim_b: Second dimension.

        Returns:
            Dim: Dimension compatible with both inputs.

        Raises:
            ValueError: If the dimensions are incompatible.
        """
        if dim_a is None:
            return dim_b
        # end if
        if dim_b is None:
            return dim_a
        # end if
        if dim_a != dim_b:
            raise ValueError(f"Incompatible dimensions: {dim_a} vs {dim_b}.")
        # end if
        return dim_a
    # end def _merge_dims

    @staticmethod
    def _sum_dims(dim_a: Dim, dim_b: Dim) -> Dim:
        """Sum two dimensions symbolically.

        Args:
            dim_a: First dimension.
            dim_b: Second dimension.

        Returns:
            Dim: Sum of known dimensions or None when unknown.
        """
        if dim_a is None or dim_b is None:
            return None
        # end if
        return dim_a + dim_b
    # end def _sum_dims

    @staticmethod
    def _normalize_axis(axis: int, rank: int, allow_new_axis: bool = False) -> int:
        """Normalize axis indices, supporting negatives.

        Args:
            axis: Requested axis index.
            rank: Tensor rank.
            allow_new_axis: Whether axis == rank is acceptable.

        Returns:
            int: Normalized non-negative axis index.

        Raises:
            ValueError: If the axis exceeds allowed bounds.
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

    @property
    def dims(self) -> Dims:
        """Return the dimensions tuple.

        Returns:
            Dims: Tuple describing tensor dimensions.
        """
        return self._dims
    # end def dims

    @property
    def rank(self) -> int:
        """Return the tensor rank.

        Returns:
            int: Number of dimensions.
        """
        return len(self._dims)
    # end def rank

    def __len__(self) -> int:
        """Return the rank for len().

        Returns:
            int: Tensor rank.
        """
        return self.rank
    # end def __len__

    def __getitem__(self, index: int) -> Dim:
        """Return the dimension at the provided index.

        Args:
            index: Axis index.

        Returns:
            Dim: Dimension size or None.
        """
        return self._dims[index]
    # end def __getitem__

    @property
    def size(self) -> Optional[int]:
        """Return the total number of elements when known.

        Returns:
            Optional[int]: Number of elements or None if unknown.
        """
        return num_elements(self._dims)
    # end def size

    def as_tuple(self) -> Dims:
        """Expose the raw dimensions tuple.

        Returns:
            Dims: Underlying tuple of dimensions.
        """
        return self._dims
    # end def as_tuple

    def __eq__(self, other: object) -> bool:
        """Compare shapes for equality.

        Args:
            other: Object to compare against.

        Returns:
            bool: True when both shapes share identical dimensions.
        """
        if not isinstance(other, Shape):
            return False
        # end if
        return self._dims == other._dims
    # end def __eq__

    def __hash__(self) -> int:
        """Return a hash for the shape.

        Returns:
            int: Hash value.
        """
        return hash(self._dims)
    # end def __hash__

    def __repr__(self) -> str:
        """Return the repr() form of the shape.

        Returns:
            str: Developer-friendly representation.
        """
        return f"Shape({self._dims})"
    # end def __repr__

    def __str__(self) -> str:
        """Return the str() form of the shape.

        Returns:
            str: Readable representation.
        """
        dims_str = "x".join("?" if dim is None else str(dim) for dim in self._dims)
        return f"Shape({dims_str})"
    # end def __str__

    def is_elementwise_compatible(self, other: "Shape") -> bool:
        """Check whether elementwise operations are allowed.

        Args:
            other: Shape to compare.

        Returns:
            bool: True when ranks and dimensions are compatible.
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

        Args:
            other: Shape to merge.

        Returns:
            Shape: Resulting shape.

        Raises:
            ValueError: If shapes are incompatible.
        """
        if self.rank != other.rank:
            raise ValueError("Elementwise operations require equal ranks.")
        # end if
        merged = tuple(self._merge_dims(dim_a, dim_b) for dim_a, dim_b in zip(self._dims, other._dims))
        return Shape(merged)
    # end def merge_elementwise

    def matmul_result(self, other: "Shape") -> "Shape":
        """Return the result shape of a matrix multiplication.

        Args:
            other: Right-hand operand shape.

        Returns:
            Shape: Resulting matmul shape.

        Raises:
            ValueError: If shapes are incompatible.
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

        Args:
            other: Shape to concatenate with.
            axis: Concatenation axis.

        Returns:
            Shape: Concatenated shape.

        Raises:
            ValueError: If ranks differ.
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

    @staticmethod
    def stack_result(shapes: Sequence["Shape"], axis: int) -> "Shape":
        """Return the result shape of stacking tensors.

        Args:
            shapes: Shapes of tensors to stack.
            axis: Axis index for the new dimension.

        Returns:
            Shape: Resulting stacked shape.
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
    # end def stack_result

    def transpose(self, permutation: Sequence[int]) -> "Shape":
        """Return the shape after applying an axis permutation.

        Args:
            permutation: Axis permutation.

        Returns:
            Shape: Permuted shape.

        Raises:
            ValueError: If permutation is invalid.
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

        Args:
            new_shape: Target shape.

        Returns:
            bool: True if reshape preserves element count.
        """
        own_size = self.size
        target_size = new_shape.size
        if own_size is None or target_size is None:
            return True
        # end if
        return own_size == target_size
    # end def can_reshape

    def reshape(self, new_shape: "Shape") -> "Shape":
        """Return the symbolic shape after reshape.

        Args:
            new_shape: Target shape.

        Returns:
            Shape: Target shape if compatible.

        Raises:
            ValueError: If reshape is invalid.
        """
        if not self.can_reshape(new_shape):
            raise ValueError("Reshape requires matching number of elements.")
        # end if
        return new_shape
    # end def reshape

# end class Shape

