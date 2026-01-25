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
from ..tensor import (
    Tensor,
    concatenate as tensor_concatenate,
    hstack as tensor_hstack,
    vstack as tensor_vstack,
)
from ..math_expr import SliceExpr, MathNode, MathExpr
from .base import Operands, Operand, operator_registry, Operator, ParametricOperator

__all__ = [
    "Getitem",
    "Flatten",
    "Concatenate",
    "HStack",
    "VStack",
    "Squeeze",
    "Unsqueeze",
]


class StructureOperator(Operator, ABC):
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
            expr: "MathNode",
            by_ref: bool = False,
            look_for: Optional[str] = None
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

# end class StructureOperator


class Getitem(StructureOperator):
    """Getitem operator."""

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
        new_shape = list(operands[0].input_shape.dims)
        for n_i, (i, n) in enumerate(zip(self._indices, operands[0].input_shape.dims)):
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
            if start < -operands[0].input_shape[n_i] or start >= operands[0].input_shape[n_i]:
                return False
            # end if
        # end for
        return True
    # end def check_shapes

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
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

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
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


class SetItem(StructureOperator):
    """Setitem operator."""

    NAME = "setitem"
    ARITY = 2

    def __init__(self, tensor: MathExpr, value: MathExpr, indices: Union[SliceExpr, int]) -> None:
        """Construct a setitem operator."""
        super().__init__(
            indices=indices
        )
        self._indices = indices
        self._tensor = tensor
        self._value = value
    # end def __init__

    # region PROPERTIES

    @property
    def tensor(self) -> MathExpr:
        return self._tensor
    # end def

    @property
    def value(self) -> MathExpr:
        return self._value
    # end def value

    @property
    def indices(self):
        return self._indices
    # end def indices

    # endregion PROPERTIES

    # region PUBLIC

    def check_operands(self, operands: Operands) -> bool:
        pass

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        pass

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        pass

    def infer_dtype(self, operands: Operands) -> DType:
        pass

    def infer_shape(self, operands: Operands) -> Shape:
        pass

    def check_shapes(self, operands: Operands) -> bool:
        pass

    # endregion PUBLIC

# end class SetItem


class Flatten(StructureOperator):

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
        target_dims = self._target_dims(tensor.input_shape.dims)
        if target_dims == tensor.input_shape.dims:
            return tensor
        return tensor.reshape(Shape(target_dims))
    # end def _eval

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError("Flatten does not support backward.")
    # end def _backward

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape(self._target_dims(operands[0].input_shape.dims))
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


class Concatenate(StructureOperator):
    """Concatenate tensors along an axis, mirroring ``np.concatenate``."""

    NAME = "concatenate"
    ARITY = -1  # variable number of operands
    IS_VARIADIC = True

    def __init__(self, axis: Optional[int] = 0):
        super().__init__(axis=axis)
        self._axis = axis
    # end def __init__

    @classmethod
    def check_arity(cls, operands: Operands):
        """Allow a variable number of operands with at least one entry."""
        return len(operands) >= 1
    # end def check_arity

    def eval(self, operands: Operands, **kwargs) -> Tensor:
        """Override default arity enforcement for variadic operators."""
        if not operands:
            raise ValueError("Concatenate expects at least one operand.")
        # end if
        return self._eval(operands=operands, **kwargs)
    # end def eval

    def check_parameters(self, axis: Optional[int] = 0) -> bool:
        return axis is None or isinstance(axis, int)
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) >= 1
    # end def check_operands

    def contains(
            self,
            expr: "MathNode",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return False
    # end def contains

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensors = [operand.eval() for operand in operands]
        return tensor_concatenate(tensors, axis=self._axis)
    # end def _eval

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError("Concatenate does not support backward.")
    # end def _backward

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        for operand in operands[1:]:
            dtype = DType.promote(dtype, operand.dtype)
        # end for
        return dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        if not operands:
            raise ValueError("Cannot infer shape without operands.")
        # end if
        if self._axis is None:
            return Shape((self._flattened_size(operands),))
        # end if
        base_shape = operands[0].input_shape
        axis = self._normalized_axis(base_shape.rank)
        dims = list(base_shape.dims)
        dims[axis] = sum(operand.input_shape[axis] for operand in operands)
        return Shape(tuple(dims))
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        if not operands:
            raise ValueError("Concatenate requires at least one operand.")
        # end if
        if self._axis is None:
            return True
        reference = operands[0].input_shape
        axis = self._normalized_axis(reference.rank)
        for operand in operands[1:]:
            if operand.input_shape.rank != reference.rank:
                raise ValueError("All operands must share the same rank for concatenation.")
            # end if
            for idx, (dim_ref, dim_other) in enumerate(zip(reference.dims, operand.input_shape.dims)):
                if idx == axis:
                    continue
                if dim_ref != dim_other:
                    raise ValueError("Operand dimensions must match on non-concatenated axes.")
                # end if
            # end for
        # end for
        return True
    # end def check_shapes

    def __str__(self) -> str:
        return f"{self.NAME}(axis={self._axis})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self._axis})"
    # end def __repr__

    def _normalized_axis(self, rank: int) -> int:
        if self._axis is None:
            raise ValueError("Axis is None when normalization was requested.")
        return Shape._normalize_axis(self._axis, rank)
    # end def _normalized_axis

    def _flattened_size(self, operands: Operands) -> int:
        total = 0
        for operand in operands:
            size = operand.input_shape.size
            if size is None:
                raise ValueError("Cannot concatenate unknown-size tensors along axis=None.")
            # end if
            total += size
        # end for
        return total
    # end def _flattened_size

# end class Concatenate


class HStack(Concatenate):
    """Concatenate tensors along axis 1, matching :func:`numpy.hstack` semantics."""

    NAME = "hstack"

    def __init__(self):
        super().__init__(axis=1)
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensors = [operand.eval() for operand in operands]
        return tensor_hstack(tensors)
    # end def _eval

# end class HStack


class VStack(Concatenate):
    """Concatenate tensors along axis 0, matching :func:`numpy.vstack` semantics."""

    NAME = "vstack"

    def __init__(self):
        super().__init__(axis=0)
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensors = [operand.eval() for operand in operands]
        return tensor_vstack(tensors)
    # end def _eval

# end class VStack


class Squeeze(StructureOperator):
    """Remove axes of length one optionally restricted by ``axes``."""

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
        target_dims = self._target_dims(tensor.input_shape.dims)
        if target_dims == tensor.input_shape.dims:
            return tensor
        return tensor.reshape(Shape(target_dims))
    # end def _eval

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape(self._target_dims(operands[0].input_shape.dims))
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        dims = operands[0].input_shape.dims
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

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError("Squeeze does not support backward.")
    # end def _backward

# end class Squeeze


class Unsqueeze(StructureOperator):
    """Insert size-one axes at the requested positions."""

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
        target_dims = self._target_dims(tensor.input_shape.dims)
        return tensor.reshape(Shape(target_dims))
    # end def _eval

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape(self._target_dims(operands[0].input_shape.dims))
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        self._normalized_axes(len(operands[0].input_shape.dims))  # validation
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

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError("Unsqueeze does not support backward.")
    # end def _backward

# end class Unsqueeze


operator_registry.register(Getitem)
operator_registry.register(Flatten)
operator_registry.register(Concatenate)
operator_registry.register(HStack)
operator_registry.register(VStack)
operator_registry.register(Squeeze)
operator_registry.register(Unsqueeze)
