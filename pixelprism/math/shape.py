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
"""Tensor shape representation."""

# Imports
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Union, Dict, TypeAlias, Tuple, TYPE_CHECKING, Mapping
import numpy as np

from .dtype import TypeLike, to_numpy, DType
from .typing import MathExpr, LeafKind, SimplifyOptions
from .math_exceptions import SymbolicMathInvalidDimensionError, SymbolicMathNotImplementedError
from .random import rand_name


if TYPE_CHECKING:
    from .math_leaves import Variable, Constant
    from .tensor import Tensor
# end if


__all__ = [
    "DimExpr",
    "DimInt",
    "DimLike",
    "Shape",
    "ShapeLike"
]


# Dimensions
DimExpr: TypeAlias = "MathExpr"
DimInt = int
DimLike: TypeAlias = Union[int, "MathExpr"]
ShapeLike = Union['Shape', Sequence[DimLike]]


class Shape(MathExpr):
    """
    TODO: documentation
    """

    def __init__(
            self,
            name: Optional[str] = None,
            *,
            dims: Sequence[DimLike]
    ):
        """Initialize a TensorShape.

        Parameters
        ----------
        name: Optional[str]
            Name of the shape.
        dims : Iterable[TensorDim]
            Iterable of dimension sizes to store in the shape.
        """
        self._name = name if name is not None else rand_name("shape")
        dims_tuple = tuple(dims)
        for dim in dims_tuple:
            self._check_dim(dim)
        # end for
        dims_tuple = [dim for dim in dims_tuple]
        self._dims: Sequence[DimLike] = dims_tuple
    # end def __init__

    # region PROPERTIES

    @property
    def dims(self) -> Sequence[DimLike]:
        """Return the dimensions' tuple.

        Returns
        -------
        TensorDims
            Tuple describing tensor dimensions.
        """
        return self._dims
    # end def dims

    @property
    def size(self) -> Optional[int]:
        """Return the total number of elements when known.

        Returns
        -------
        Optional[int]
            Number of elements represented by the shape.
        """
        return self._num_elements()
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

    @property
    def is_scalar(self) -> bool:
        """Return whether the shape is scalar (rank-0)."""
        return self.rank == 0
    # end def is_scalar

    @property
    def is_vector(self) -> bool:
        """Return whether the shape is a vector (rank-1)."""
        return self.rank == 1
    # end def is_vector

    @property
    def is_matrix(self) -> bool:
        """Return whether the shape is a matrix (rank-2)."""
        return self.rank == 2
    # end def is_matrix

    @property
    def is_higher_order(self) -> bool:
        """Return whether the shape is higher-order (rank > 2)."""
        return self.rank > 2
    # end def is_higher_order

    @property
    def is_literal(self) -> bool:
        """Return whether dimensions are known at compile time."""
        truth_value = [isinstance(d, int) for d in self._dims]
        return all(truth_value)
    # end def is_literal

    # endregion PROPERTIES

    # region MATH_EXPR

    @property
    def shape(self) -> 'Shape':
        """Return the dimensions' list."""
        return Shape(dims=[dim.eval().item() if isinstance(dim, MathExpr) else dim for dim in self._dims])
    # end def shape

    @property
    def dtype(self) -> "DType":
        """Return the data type of the shape."""
        return DType.Z
    # end def dtype

    @property
    def name(self) -> str:
        """Return the name root of the shape."""
        return "shape"
    # end def name

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

    def eval(self) -> 'Tensor':
        """Evaluate the symbolic shape as a tensor."""
        from .tensor import Tensor
        dim_values = [d.eval().item() if isinstance(d, MathExpr) else d for d in self._dims]
        return Tensor(dim_values)
    # end def eval

    def diff(self, wrt: 'MathExpr') -> 'MathExpr':
        """Differentiate the symbolic shape with respect to a variable."""
        raise SymbolicMathNotImplementedError("Shape does not support differentiation.")
    # end def diff

    def variables(self) -> Sequence["Variable"]:
        """
        Return all variable leaves reachable from this expression.

        Returns
        -------
        Sequence["Variable"]
            List of variable leaves reachable from this expression.
        """
        return []
    # end def variables

    def constants(self) -> Sequence["Constant"]:
        """Return a list of constants used in the expression."""
        consts = [
            d.constants() for d in self._dims if isinstance(d, MathExpr)
        ]
        return list(set([c for d in consts for c in d]))
    # end def constants

    def contains(
            self,
            leaf: Union[str, "MathExpr"],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Check if the expression contains a given leaf node or variable."""
        truth_values = [
            d.contains(leaf, by_ref, check_operator, look_for) for d in self._dims if isinstance(d, MathExpr)
        ]
        return any(truth_values)
    # end def contains

    def contains_variable(
            self,
            variable: Union[str, 'MathExpr'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return whether the expression contains a given variable."""
        return self.contains(variable, by_ref, check_operator, look_for=LeafKind.VARIABLE)
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, 'MathExpr'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return whether the expression contains a given constant."""
        return self.contains(constant, by_ref, check_operator, look_for=LeafKind.CONSTANT)
    # end def contains_constant

    #
    # Immutable transforms
    #

    # Apply symbolic rewrite rules and return a simplified expression.
    # This operation is pure: it never mutates the current tree.
    # `options` controls which rules are enabled/disabled for this pass.
    def simplify(self, options: SimplifyOptions | None = None) -> MathExpr:
        return self
    # end def simplify

    # Normalize an expression form without changing semantics.
    # Typical effects: associative flattening, deterministic operand ordering, etc.
    def canonicalize(self) -> MathExpr:
        return self
    # end def canonicalize

    # Fold constant-only subexpressions into constant leaves.
    # This is a focused transform and may be used independently of full simplifying.
    def fold_constants(self) -> MathExpr:
        return self
    # end def fold_constants

    # Replace matching subexpressions using `mapping` and return a new tree.
    # - `by_ref=True`: match by object identity.
    # - `by_ref=False`: match by symbolic/tree equality policy.
    def substitute(
            self,
            mapping: Mapping[MathExpr, MathExpr],
            *,
            by_ref: bool = True
    ) -> MathExpr:
        """Apply a substitution mapping to the expression."""
        for old_expr, new_expr in mapping.items():
            if by_ref and old_expr is self:
                return new_expr
            # end if
            if (not by_ref) and old_expr == self:
                return new_expr
            # end if
        # end for
        return self
    # end def substitute

    # Return a new expression where occurrences of `old_name` are renamed to `new_name`.
    # This transform is immutable and does not alter the current instance.
    def renamed(self, old_name: str, new_name: str) -> MathExpr:
        """Rename a variable in the expression."""
        return self
    # end def rename

    def eq_tree(self, other: MathExpr) -> bool:
        """Return strict symbolic tree equality."""
        return isinstance(other, Shape) and tuple(self._dims) == tuple(other._dims)
    # end def eq_tree

    def equivalent(self, other: MathExpr) -> bool:
        """Return symbolic equivalence for shape expressions."""
        return self.eq_tree(other)
    # end def equivalent

    def is_constant(self) -> bool:
        """Returns ``True`` if the expression is a constant."""
        return True
    # end def is_constant

    def is_variable(self) -> bool:
        """Returns ``True`` if the expression is a variable."""
        return False
    # end def is_variable

    def is_node(self) -> bool:
        """Returns ``True`` if the expression is a node."""
        return True
    # end def is_node

    def is_leaf(self) -> bool:
        """Returns ``True`` if the expression is a leaf."""
        return False
    # end def is_leaf

    def depth(self) -> int:
        """Returns maximum depth of the expression tree."""
        dims_depth = [d.depth() if isinstance(d, MathExpr) else 1 for d in self._dims]
        return max(dims_depth) + 1
     # end def depth

    def copy(self, deep: bool = False):
        """Return a copy of the shape.

        Returns
        -------
        Shape
            Copy of the current shape.
        """
        if not deep:
            return Shape(dims=self._dims)
        else:
            return Shape(dims=[d.copy(deep=deep) if isinstance(d, MathExpr) else d for d in self._dims])
        # end if
    # end def copy

    def __str__(self) -> str:
        """Return the str() form of the shape.

        Returns
        -------
        str
            Readable representation.
        """
        return f"{self._dims}"
    # end def __str__

    def __repr__(self) -> str:
        """Return the repr() form of the shape.

        Returns
        -------
        str
            Developer-friendly representation including raw dimensions.
        """
        if self.rank == 0:
            return "scalar_shape()"
        elif self.rank == 1:
            return f"vector_shape({self._dims[0]})"
        elif self.rank == 2:
            return f"matrix_shape({self._dims[0]}, {self._dims[1]})"
        else:
            return f"tensor_shape({self._dims})"
        # end if
    # end def __repr__

    # endregion MATH_EXPR

    # region PUBLIC

    def transpose(self, axes: Optional[List[int]] = None) -> "Shape":
        """Return the shape with axes permuted.

        Parameters
        ----------
        axes : list[int], optional
            Axis permutation. When ``None``, the axis order is reversed.

        Returns
        -------
        Shape
            New shape with permuted axes.

        Raises
        ------
        ValueError
            If ``axes`` does not represent a valid permutation.
        """
        if axes is not None:
            self._check_transpose(axes)
            new_shape = [self.dims[i] for i in axes]
        else:
            new_shape = list(self.dims)
            new_shape.reverse()
        # end if
        return Shape(dims=tuple(new_shape))
    # end def transpose

    def transpose_(self):
        """
        Transpose the shape in-place.

        Returns
        -------
        None
            This operation updates the instance in-place.
        """
        self._dims = self.transpose().dims
    # end def transpose_

    def drop_axis(self, axis: int) -> "Shape":
        """Return a new shape with the specified axis removed.

        Parameters
        ----------
        axis : int
            Axis index to drop.

        Returns
        -------
        Shape
            New shape with the axis removed.

        Raises
        ------
        ValueError
            If the axis is out of bounds.
        """
        if axis < 0 or axis >= self.rank:
            raise ValueError(f"Axis {axis} out of bounds for rank {self.rank}.")
        # end if
        if axis == self.rank - 1:
            return Shape(dims=self._dims[:axis])
        elif axis == 0:
            return Shape(dims=self._dims[1:])
        else:
            return Shape(dims=list(self._dims[:axis]) + list(self._dims[axis + 1 :]))
        # end if
    # end def drop_axis

    def drop_axis_(self, axis: int) -> None:
        """Remove the specified axis from the shape in-place.

        Parameters
        ----------
        axis : int
            Axis index to drop.

        Returns
        -------
        None
            This operation updates the instance in-place.
        """
        self._dims = self.drop_axis(axis)._dims
    # end def drop_axis

    def insert_axis(self, axis: int, size: DimLike) -> "Shape":
        """Return a new shape with the specified axis inserted.

        Parameters
        ----------
        axis : int
            Axis index to insert.
        size : Dim
            Size of the inserted axis.

        Returns
        -------
        Shape
            New shape with the axis inserted.

        Raises
        ------
        ValueError
            If the axis is out of bounds.
        """
        if axis < 0 or axis > self.rank:
            raise ValueError(f"Axis {axis} out of bounds for rank {self.rank}.")
        # end if
        from .build import as_expr
        size = as_expr(size)
        return Shape(dims=list(self._dims[:axis]) + [size] + list(self._dims[axis:]))
    # end def insert_axis

    def insert_axis_(self, axis: int, size: DimLike) -> None:
        """Insert the specified axis into the shape in-place.

        Parameters
        ----------
        axis : int
            Axis index to insert.
        size : Dim
            Size of the inserted axis.

        Returns
        -------
        None
            This operation updates the instance in-place.
        """
        from .build import as_expr
        size = as_expr(size)
        self._dims = self.insert_axis(axis, size)._dims
    # end def insert_axis_

    def as_tuple(self) -> tuple[DimExpr, ...]:
        """Return the shape as a tuple.

        Returns
        -------
        tuple[DimExpr, ...]
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
        return Shape(dims=merged)
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
        batch_dims: List[DimLike] = []
        for idx in range(batch_rank):
            batch_dims.append(self._merge_dims(self._dims[idx], other._dims[idx]))
        # end for
        left_inner = self._dims[-1]
        right_inner = other._dims[-2]
        if not self._dims_equal(left_inner, right_inner):
            raise ValueError("Inner dimensions do not match for MatMul.")
        # end if
        result = tuple(batch_dims) + (self._dims[-2], other._dims[-1])
        return Shape(dims=result)
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
        dims: List[DimLike] = []
        for idx, (dim_a, dim_b) in enumerate(zip(self._dims, other._dims)):
            if idx == axis_norm:
                dims.append(self._sum_dims(dim_a, dim_b))
            else:
                dims.append(self._merge_dims(dim_a, dim_b))
            # end if
        # end for
        return Shape(dims=tuple(dims))
    # end def concat_result

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

    def reshape_(self, new_shape: "Shape") -> None:
        """
        Reshape the shape in-place.

        Parameters
        ----------
        new_shape : Shape
            Target shape.

        Returns
        -------
        None
            This operation updates the instance in-place.
        """
        self._dims = self.reshape(new_shape)._dims
    # end def reshape_

    def equal_or_broadcastable(self, other: "Shape") -> bool:
        """Check whether the shape is equal or broadcastable to another shape."""
        # TODO: implement this properly
        pass
    # end def equal_or_broadcastable

    # endregion PUBLIC

    # region PRIVATE

    def _num_elements(self) -> int | None:
        """Compute the product of symbolic dimensions when possible.

        Returns
        -------
        int | None
            Number of elements or ``None`` when any dimension is unknown.
        """
        total = 1
        dims = self.eval()
        for dim in dims.tolist():
            total *= dim
        # end for
        return total
    # end def _num_elements

    def _check_transpose(self, axes: Sequence[int]) -> None:
        """Validate a permutation of axes.

        Parameters
        ----------
        axes : Sequence[int]
            Proposed axis permutation.

        Raises
        ------
        ValueError
            If the permutation is invalid for this shape.
        """
        if len(axes) != self.n_dims:
            raise ValueError(
                f"Permutation must include every axis exactly once (got {len(axes)} axes, expected {self.dims})."
            )
        # end if
        if sorted(axes) != list(range(self.n_dims)):
            raise ValueError(f"Permutation contains invalid axis indices: {axes}")
        # end if
    # end def _check_transpose

    # endregion PRIVATE

    # region STATIC

    @staticmethod
    def create(shape: ShapeLike) -> "Shape":
        """Create a shape from a tuple, sequence, or compatible object.

        Parameters
        ----------
        shape : ShapeLike
            Shape input (tuple, sequence, scalar dimension, or Shape).

        Returns
        -------
        Shape
            Normalized Shape instance.

        Raises
        ------
        TypeError
            If the input type is unsupported.
        """
        if isinstance(shape, tuple):
            return Shape(dims=shape)
        elif isinstance(shape, Sequence):
            return Shape(dims=shape)
        elif isinstance(shape, int):
            return Shape(dims=(shape,))
        elif isinstance(shape, Shape):
            return shape.copy()
        elif hasattr(shape, "dims"):
            return Shape(dims=getattr(shape, "dims"))
        else:
            raise TypeError(f"Unsupported shape type: {type(shape)}")
        # end if
    # end def create

    @staticmethod
    def scalar() -> "Shape":
        """Return a scalar (rank-0) shape.

        Returns
        -------
        'Shape'
            Shape with no dimensions.
        """
        return Shape(dims=())
    # end def scalar

    @staticmethod
    def vector(n: DimLike) -> "Shape":
        """Return a vector shape.

        Parameters
        ----------
        n : Dim
            Length of the vector.

        Returns
        -------
        Shape
            Shape with a single dimension of size ``n``.
        """
        return Shape(dims=(n,))
    # end def vector

    @staticmethod
    def matrix(n: DimLike, m: DimLike) -> "Shape":
        """Return a matrix shape.

        Parameters
        ----------
        n : Dim
            Row count.
        m : Dim
            Column count.

        Returns
        -------
        'Shape'
            Shape with two dimensions ``(n, m)``.
        """
        return Shape(dims=(n, m))
    # end def matrix

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
        return Shape(dims=tuple(dims))
    # end def stack_shape

    @staticmethod
    def _check_dim(dim: DimLike) -> None:
        """Validate a single-dimension value.

        Parameters
        ----------
        dim : Dim
            Dimension to validate. Allowed values are non-negative integers.

        Raises
        ------
        ValueError
            If the dimension is negative or not an integer.
        """
        if dim is None:
            raise SymbolicMathInvalidDimensionError("Shape dimensions cannot be None.")
        # end if
        if isinstance(dim, int) and dim < 0:
            raise SymbolicMathInvalidDimensionError(
                f"Shape dimensions must be non-negative if integers ({dim} given)."
            )
        # end if
        if isinstance(dim, MathExpr):
            # Only expr with constants
            if not dim.is_constant:
                raise SymbolicMathInvalidDimensionError(f"Shape dimensions must be constant: {dim}")
            # end if
            if dim.eval().item() <= 0:
                raise SymbolicMathInvalidDimensionError(f"Shape dimensions must be positive: {dim}")
            # end if
        # end if
    # end def _check_dim

    @staticmethod
    def _dims_equal(dim_a: DimLike, dim_b: DimLike) -> bool:
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
        if isinstance(dim_a, MathExpr) and isinstance(dim_b, MathExpr):
            return dim_a.eval().item() == dim_b.eval().item()
        elif isinstance(dim_a, int) and isinstance(dim_b, int):
            return dim_a == dim_b
        elif isinstance(dim_a, MathExpr) and isinstance(dim_b, int):
            return dim_a.eval().item() == dim_b
        elif isinstance(dim_a, int) and isinstance(dim_b, MathExpr):
            return dim_b.eval().item() == dim_a
        else:
            return False
        # end if
    # end def _dims_equal

    @staticmethod
    def _merge_dims(dim_a: DimLike, dim_b: DimLike) -> DimLike:
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
        if not Shape._dims_equal(dim_a, dim_b):
            raise ValueError(f"Incompatible dimensions: {dim_a} vs {dim_b}.")
        # end if
        return dim_a
    # end def _merge_dims

    @staticmethod
    def _sum_dims(dim_a: DimLike, dim_b: DimLike) -> DimLike:
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
            Sum of both dimensions.
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

    def __iter__(self):
        """Return an iterator over the dimensions.

        Returns
        -------
        Iterator[Dim]
            Iterator over the dimensions.
        """
        return iter(self._dims)
    # end def __iter__

    def __contains__(self, dim: DimLike) -> bool:
        """Check whether a dimension is present in the shape.
        """
        return dim in self._dims
    # end def __contains__

    def __bool__(self) -> bool:
        """Return whether the shape is non-empty."""
        return bool(self._dims)
    # end def __bool_

    def __getitem__(self, index: int) -> DimLike:
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
        lhs_dims = tuple(self._dims)
        if isinstance(other, tuple):
            return lhs_dims == tuple(other)
        elif isinstance(other, list):
            return lhs_dims == tuple(other)
        # end if
        if isinstance(other, Shape):
            return lhs_dims == tuple(other._dims)
        # end if
        if hasattr(other, "dims"):
            try:
                return lhs_dims == tuple(getattr(other, "dims"))
            except TypeError:
                return False
            # end try
        # end if
        return False
    # end def __eq__

    def __ne__(self, other: object) -> bool:
        """Compare shapes for inequality.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when both shapes have different dimensions.
        """
        return not self.__eq__(other)
    # end def __ne__

    def __hash__(self) -> int:
        """Return a hash for the shape.

        Returns
        -------
        int
            Hash value derived from the dimensions.
        """
        return hash(self._dims)
    # end def __hash__

    # endregion OVERRIDE

    # region NUMPY

    def __array__(self, dtype: Optional[TypeLike] = None) -> np.ndarray:
        """Convert the shape to a NumPy array.
        """
        return np.array(self._dims, dtype=to_numpy(dtype if dtype is not None else np.int32))
    # end def __array__

    # endregion NUMPY

# end class Shape
