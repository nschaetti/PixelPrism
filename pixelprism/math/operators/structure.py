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
"""
Structure operator implementations.
"""

# Imports
from abc import ABC
from typing import Optional, Sequence, Union, Any, List, Tuple

import numpy as np

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor
from ..math_slice import SliceExpr
from ..math_node import MathNode
from ..typing import MathExpr, LeafKind, OperatorSpec, AritySpec, OpAssociativity
from .base import Operands, operator_registry, OperatorBase


__all__ = [
    "Getitem",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
]


def _structure_spec(name: str, *, exact: int | None, min_operands: int, variadic: bool) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        symbol=name,
        arity=AritySpec(exact=exact, min_operands=min_operands, variadic=variadic),
        precedence=40,
        associativity=OpAssociativity.NONE,
        commutative=False,
        associative=False,
        is_diff=False,
    )
# end def _structure_spec


class StructureOperator(OperatorBase, ABC):
    """
    Linear algebra operator.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parameters: dict[str, Any] = kwargs
        if not self.check_parameters(**kwargs):
            raise ValueError(f"Invalid parameters for operator {self.NAME}: {kwargs}")
        # end if
    # end def __init__

    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric operators must implement contains(..).")
    # end def contains

    def check_parameters(self, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        raise NotImplementedError("Parametric operators must implement check_parameters(..).")
    # end def check_shapes

    def __str__(self) -> str:
        formatted = self._format_parameters()
        if formatted:
            return f"{self.NAME}({formatted})"
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        formatted = self._format_parameters()
        cls_name = self.__class__.__name__
        if formatted:
            return f"{cls_name}({formatted})"
        return f"{cls_name}()"
    # end def __repr__

    def _format_parameters(self) -> str:
        if not self._parameters:
            return ""
        return ", ".join(f"{key}={value!r}" for key, value in self._parameters.items())
    # end def _format_parameters

    def _needs_parentheses(self, *args, **kwargs):
        return None
    # end def _needs_parentheses

    def print(self, operands: Operands, **kwargs) -> str:
        return str(self)
    # end def print

# end class StructureOperator


class Reshape(StructureOperator):

    SPEC = _structure_spec("reshape", exact=1, min_operands=1, variadic=False)

    NAME = "reshape"
    ARITY = 1

    def __init__(self, shape: Sequence[int]):
        super().__init__(shape=shape)
    # end def __init__

    def check_parameters(self, **kwargs) -> bool:
        # Only one -1
        count_minus = sum([a == -1 for a in self._parameters["shape"]])
        # No -2, -3, etc
        if any([a < -1 for a in self._parameters["shape"]]):
            return False
        # end if
        return count_minus <= 1
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != 1:
            return False
        # end if
        return True
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor = operands[0].eval()
        target_shape = self._compute_shape(self._parameters["shape"], tensor.shape)
        if target_shape == tensor.shape:
            return tensor
        # end
        return tensor.reshape(target_shape)
    # end def _eval

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return self._compute_shape(self._parameters["shape"], operands[0].shape)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        new_shape = self._compute_shape(self._parameters["shape"], operands[0].shape)
        return new_shape.size == operands[0].size
    # end def check_shapes

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Flatten does not support backward.")
    # end def _backward

    def _compute_shape(self, new_shape: Sequence[int], shape: Shape) -> Shape:
        """Compute the new shape."""
        n_elements = shape.size
        if any([a == -1 for a in new_shape]):
            known_elements = [a for a in new_shape if a != -1]
            size_known_elements = int(np.prod(known_elements))
            new_dim_size = n_elements // size_known_elements
            final_size = list()
            for i, known in enumerate(new_shape):
                if known == -1:
                    final_size.append(new_dim_size)
                else:
                    final_size.append(known)
                # end if
            # end for
            return Shape(final_size)
        else:
            return Shape(new_shape)
        # end if
    # end def _compute_shape

# end class Flatten


class Getitem(StructureOperator):
    """Getitem operator."""

    SPEC = _structure_spec("getitem", exact=1, min_operands=1, variadic=False)

    NAME = "getitem"
    ARITY = 1

    def __init__(self, indices: List[Union[SliceExpr, int]]):
        super().__init__(indices=indices)
        self._indices = indices
    # end def __init__

    @property
    def indices(self):
        return self._indices
    # end def indices

    # region PUBLIC

    def check_parameters(self, indices: List[Union[SliceExpr, int]]) -> bool:
        def _check_slice(s: Union[SliceExpr, int]) -> bool:
            if isinstance(s, int):
                return True
            # end if
            return s.step is None or self._get_scalar(s.step) != 0
        # end def _check_slice
        return all([
            _check_slice(o)
            for o in indices
        ])
    # end for

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        new_shape = list(operands[0].shape.dims)
        for n_i, (i, n) in enumerate(zip(self._indices, operands[0].shape.dims)):
            if isinstance(i, int):
                new_shape[n_i] = 0
            else:
                new_shape[n_i] = self._compute_new_dim(
                    start=self._get_scalar(i.start) if i.start is not None else 0,
                    stop=self._get_scalar(i.stop) if i.stop is not None else None,
                    step=self._get_scalar(i.step) if i.step is not None else 1,
                    n=n
                )
            # end if
        # end for
        if len(new_shape) > 1 and 0 in new_shape:
            new_shape.remove(0)
        # end if
        return Shape(new_shape)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        for n_i, i in enumerate(self._indices):
            start = self._get_scalar(i.start) if isinstance(i, SliceExpr) else i
            if start < -operands[0].shape[n_i] or start >= operands[0].shape[n_i]:
                return False
            # end if
        # end for
        return True
    # end def check_shapes

    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        return any([s.contains(expr, by_ref=by_ref, look_for=look_for) for s in self._indices])
    # end def contains

    # endregion PUBLIC

    # region PRIVATE

    def _get_scalar(self, e: Union[MathNode, int]) -> Union[int, float]:
        if isinstance(e, int):
            return e
        else:
            return e.eval().item()
        # end if
    # end def _get_scalar

    def _compute_new_dim(self, start: int, stop: Optional[int], step: int, n: int):
        """
        Compute the new dimension.
        """
        start = self._compute_start(start=start, n=n)
        stop = self._compute_stop(start=start, stop=stop, step=step, n=n)
        range_is = np.arange(start, stop, step)
        range_is = range_is[range_is >= 0]
        range_is = range_is[range_is < n]
        return range_is.size
    # end if

    def _compute_start(self, start: int, n: int) -> int:
        return start + n if start < 0 else start
    # end def _compute_start

    def _compute_stop(self, start: int, stop: Optional[int], step: int, n: int) -> int:
        """Compute the stop value for a slice."""
        if stop is None:
            return -1 if step < 0 else n
        else:
            return stop + n if stop < 0 else stop
        # end if
    # end def _compute_stop

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        indices: List[Union[SliceExpr, int]] = list()
        for i in self._indices:
            indices.append(i.to_slice() if isinstance(i, SliceExpr) else i)
        # end for
        return operands[0].eval()[*indices]
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("GetItem does not support backward.")
    # end def _backward

    # endregion PRIVATE

    # region OVERRIDE

    def __str__(self) -> str:
        return f"{self.NAME}(indices={self._format_indices()})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(indices={self._format_indices()})"
    # end def __repr__

    def _format_indices(self) -> list[str]:
        formatted: list[str] = []
        for index in self._indices:
            if isinstance(index, SliceExpr):
                py_slice = index.to_slice()
                formatted.append(f"slice({py_slice.start}, {py_slice.stop}, {py_slice.step})")
            else:
                formatted.append(str(index))
        return formatted
    # end def _format_indices

    # endregion OVERRIDE

# end class GetItem


class Flatten(StructureOperator):

    SPEC = _structure_spec("flatten", exact=1, min_operands=1, variadic=False)

    NAME = "flatten"
    ARITY = 1

    def __init__(self):
        super().__init__()
    # end def __init__

    def check_parameters(self, **kwargs) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor = operands[0].eval()
        target_dims = self._target_dims(tensor.dims)
        if target_dims == tensor.dims:
            return tensor
        return tensor.reshape(Shape(target_dims))
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Flatten does not support backward.")
    # end def _backward

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape(self._target_dims(operands[0].shape.dims))
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        return True
    # end def check_shapes

    def _target_dims(self, dims: Sequence[int]) -> Tuple[int, ...]:
        if not dims:
            return (1,)
        flat_size = int(np.prod(dims))
        return (flat_size,)
    # end def _target_dims

# end class Flatten


class Squeeze(StructureOperator):
    """Remove axes of length one optionally restricted by ``axes``."""

    SPEC = _structure_spec("squeeze", exact=1, min_operands=1, variadic=False)

    NAME = "squeeze"
    ARITY = 1

    def __init__(self, axes: Optional[Sequence[int]] = None):
        axes_tuple = tuple(int(axis) for axis in axes) if axes is not None else None
        super().__init__(axes=axes_tuple)
        self._axes = axes_tuple
    # end def __init__

    def check_parameters(self, axes: Optional[Sequence[int]] = None) -> bool:
        if axes is None:
            return True
        return len(set(axes)) == len(tuple(axes))
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor = operands[0].eval()
        target_dims = self._target_dims(tensor.dims)
        if target_dims == tensor.dims:
            return tensor
        return tensor.reshape(Shape(target_dims))
    # end def _eval

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape(self._target_dims(operands[0].shape.dims))
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        dims = operands[0].shape.dims
        axes = self._normalized_axes(dims)
        if self._axes is None:
            return True
        for axis in axes:
            if dims[axis] != 1:
                raise ValueError(f"Axis {axis} has dimension {dims[axis]}, cannot squeeze.")
        # end for
        return True
    # end def check_shapes

    def _normalized_axes(self, dims: Sequence[int]) -> List[int]:
        rank = len(dims)
        if self._axes is None:
            return [idx for idx, dim in enumerate(dims) if dim == 1]
        normalized: List[int] = []
        for axis in self._axes:
            ax = axis if axis >= 0 else axis + rank
            if ax < 0 or ax >= rank:
                raise ValueError(f"Axis {axis} is out of bounds for rank {rank}.")
            normalized.append(ax)
        # end for
        seen = set()
        for ax in normalized:
            if ax in seen:
                raise ValueError("Duplicate axes after normalization in squeeze.")
            seen.add(ax)
        # end for
        return normalized
    # end def _normalized_axes

    def _target_dims(self, dims: Sequence[int]) -> Tuple[int, ...]:
        axes = sorted(set(self._normalized_axes(dims)), reverse=True)
        if not axes:
            return tuple(dims)
        dims_list = list(dims)
        for axis in axes:
            if axis < len(dims_list):
                dims_list.pop(axis)
        # end for
        return tuple(dims_list)
    # end def _target_dims

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Squeeze does not support backward.")
    # end def _backward

# end class Squeeze


class Unsqueeze(StructureOperator):
    """Insert size-one axes at the requested positions."""

    SPEC = _structure_spec("unsqueeze", exact=1, min_operands=1, variadic=False)

    NAME = "unsqueeze"
    ARITY = 1

    def __init__(self, axes: Sequence[int]):
        if not axes:
            raise ValueError("Unsqueeze requires at least one axis.")
        axes_tuple = tuple(int(axis) for axis in axes)
        super().__init__(axes=axes_tuple)
        self._axes = axes_tuple
    # end def __init__

    def check_parameters(self, axes: Sequence[int]) -> bool:
        return len(set(axes)) == len(tuple(axes))
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor = operands[0].eval()
        target_dims = self._target_dims(tensor.dims)
        return tensor.reshape(Shape(target_dims))
    # end def _eval

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape(self._target_dims(operands[0].shape.dims))
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        self._normalized_axes(len(operands[0].shape.dims))  # validation
        return True
    # end def check_shapes

    def _normalized_axes(self, base_rank: int) -> List[int]:
        total_rank = base_rank + len(self._axes)
        normalized: List[int] = []
        for axis in self._axes:
            ax = axis if axis >= 0 else axis + total_rank
            if ax < 0 or ax > total_rank:
                raise ValueError(f"Axis {axis} is out of bounds for resulting rank {total_rank}.")
            normalized.append(ax)
        # end for
        normalized.sort()
        for idx in range(1, len(normalized)):
            if normalized[idx] == normalized[idx - 1]:
                raise ValueError("Duplicate axes after normalization in unsqueeze.")
        # end for
        return normalized
    # end def _normalized_axes

    def _target_dims(self, dims: Sequence[int]) -> Tuple[int, ...]:
        normalized = self._normalized_axes(len(dims))
        dims_list = list(dims)
        for axis in normalized:
            dims_list.insert(axis, 1)
        # end for
        return tuple(dims_list)
    # end def _target_dims

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Unsqueeze does not support backward.")
    # end def _backward

# end class Unsqueeze


operator_registry.register(Reshape)
operator_registry.register(Getitem)
operator_registry.register(Flatten)
operator_registry.register(Squeeze)
operator_registry.register(Unsqueeze)
