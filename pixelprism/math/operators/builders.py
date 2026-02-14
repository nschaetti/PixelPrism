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
Tensor building operators.
"""

# Imports
from abc import ABC
from typing import Optional, Sequence, Union
import numpy as np

from ..dtype import DType, to_numpy, promote
from ..math_node import MathNode
from ..shape import Shape
from ..tensor import (
    Tensor,
    concatenate as tensor_concatenate,
    hstack as tensor_hstack,
    vstack as tensor_vstack,
)
from ..context import new_context, set_value
from ..math_leaves import const
from ..random import random_const_name
from .base import Operands, OperatorBase, ParametricOperator, operator_registry


Element = Union[MathNode, float, int]
Elements = Sequence[Element]


__all__ = [
    "Builder",
    "ParametricBuilder",
    "BuildTensor",
    "Vector",
    "Matrix",
    "Full",
    "Concatenate",
    "HStack",
    "VStack",
    "Zeros",
    "Ones",
    "Eye",
    "Identity",
    "Map",
    "Linspace",
    "Logspace",
    "SparseCOO",
    "FromFunction",
    "Diag",
]


class Builder(OperatorBase, ABC):
    """Tensor builder operator."""
    pass
# end class TensorBuilder


class ParametricBuilder(Builder, ParametricOperator, ABC):
    """Parametric tensor builder operator."""
    pass
# end def ParametricTensorBuilder


class BuildTensor(ParametricBuilder):
    """Tensor builder from list."""

    NAME = "build_tensor"
    ARITY = -1
    IS_VARIADIC = True

    @classmethod
    def check_arity(cls, operands: Operands) -> bool:
        """Allow a variable number of operands, requiring at least one."""
        return len(operands) >= 1
    # end def check_arity

    def __init__(
            self,
            input_shape: Optional[Shape] = None
    ):
        super(BuildTensor, self).__init__(
            shape=input_shape
        )
        self._input_shape = input_shape
    # end def __init__

    # region PROPERTY

    @property
    def input_shape(self) -> Optional[Shape]:
        return self._input_shape
    # end def shape

    @property
    def n_elements(self) -> Optional[int]:
        """Get the number of elements in the tensor."""
        if self._input_shape is None:
            return None
        # end if
        return self._input_shape.size
    # end def n_elements

    # endregion PROPERTY

    # region PUBLIC

    @classmethod
    def check_parameters(cls, input_shape: Shape) -> bool:
        return input_shape.n_dims >= 1
    # end def check_parameters

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) < 1:
            return False
        # end if
        if self.n_elements is not None and len(operands) != self.n_elements:
            return False
        # end if
        return True
    # end def check_operands

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        for operand in operands[1:]:
            dtype = promote(dtype, operand.dtype)
        # end for
        return dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        if self._input_shape is None:
            return Shape(dims=(len(operands),))
        # end if
        return self._input_shape.copy()
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        for operand in operands:
            if operand.rank != 0:
                return False
            # end if
        # end for
        if self.n_elements is not None and len(operands) != self.n_elements:
            return False
        # end if
        return True
    # end def check_shapes

    # endregion PUBLIC

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        dtype = self.infer_dtype(operands)
        numpy_dtype = to_numpy(dtype)
        values = [np.asarray(operand.eval().value, dtype=numpy_dtype) for operand in operands]
        data = np.stack(values, axis=0)
        if self._input_shape is not None:
            data = data.reshape(self._input_shape.dims)
        # end if
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("BuildTensor does not support backward propagation.")
    # end def _backward

    # endregion PRIVATE

    # region OVERRIDE

    def __str__(self) -> str:
        return f"{self.NAME}(input_shape={self._input_shape})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_shape={self._input_shape})"
    # end def __repr__

    # endregion OVERRIDE

# end class BuildTensor


operator_registry.register(BuildTensor)


class Vector(Builder):
    """
    Construct a 1-D tensor from scalar expressions.
    """

    NAME = "vector"
    ARITY = -1
    IS_VARIADIC = True

    @classmethod
    def check_arity(cls, operands: Operands) -> bool:
        return len(operands) > 0
    # end def check_arity

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        return all(op.rank == 0 for op in operands)
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(len(operands))
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        for operand in operands[1:]:
            dtype = promote(dtype, operand.dtype)
        # end for
        return dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        dtype = self.infer_dtype(operands)
        numpy_dtype = to_numpy(dtype)
        values = [np.asarray(op.eval().value, dtype=numpy_dtype) for op in operands]
        data = np.stack(values, axis=0)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Vector does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(n={self.arity if self.arity >= 0 else 'variadic'})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.arity if self.arity >= 0 else 'variadic'})"
    # end def __repr__

# end class Vector


class Matrix(ParametricBuilder):
    """
    Construct a 2-D tensor from scalar expressions.
    """

    NAME = "matrix"
    ARITY = -1
    IS_VARIADIC = True

    def __init__(self, rows: int, cols: int):
        super(Matrix, self).__init__(rows=rows, cols=cols)
        self._rows = rows
        self._cols = cols
        self._shape = Shape.matrix(rows, cols)
    # end def __init__

    @classmethod
    def check_parameters(cls, rows: int, cols: int) -> bool:
        return rows > 0 and cols > 0
    # end def check_parameters

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != self._rows * self._cols:
            return False
        return all(op.rank == 0 for op in operands)
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        for operand in operands[1:]:
            dtype = promote(dtype, operand.dtype)
        # end for
        return dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        dtype = self.infer_dtype(operands)
        numpy_dtype = to_numpy(dtype)
        values = [np.asarray(op.eval().value, dtype=numpy_dtype) for op in operands]
        data = np.stack(values, axis=0).reshape(self._rows, self._cols)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Matrix does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(rows={self._rows}, cols={self._cols})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rows={self._rows}, cols={self._cols})"
    # end def __repr__

# end class Matrix


class Full(ParametricBuilder):
    """
    Fill a tensor with a single scalar expression.
    """

    NAME = "full"
    ARITY = 1

    def __init__(self, shape: Shape):
        super(Full, self).__init__(shape=shape)
        self._shape = shape
    # end def __init__

    @classmethod
    def check_parameters(cls, shape: Shape) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1 and operands[0].rank == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        operand, = operands
        dtype = operand.dtype
        numpy_dtype = to_numpy(dtype)
        scalar_value = np.asarray(operand.eval().value, dtype=numpy_dtype).item()
        data = np.full(self._shape.dims, scalar_value, dtype=numpy_dtype)
        return Tensor(data=data, dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Full does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(shape={self._shape.dims})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self._shape.dims})"
    # end def __repr__

# end class Full


class Diag(Builder):
    """
    Build a diagonal matrix from a vector.
    """

    NAME = "diag"
    ARITY = 1

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        vector = operands[0]
        return vector.rank == 1
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        dim = operands[0].shape.dims[-1]
        return Shape.matrix(dim, dim)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        vector, = operands
        dtype = vector.dtype
        numpy_dtype = to_numpy(dtype)
        values = np.asarray(vector.eval().value, dtype=numpy_dtype)
        result = np.diag(values)
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Diag does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        return self.NAME
    # end def __repr__

# end class Diag


class Concatenate(Builder):
    """Concatenate tensors along an axis, mirroring ``np.concatenate``."""

    NAME = "concatenate"
    ARITY = -1
    IS_VARIADIC = True

    def __init__(self, axis: Optional[int] = 0):
        super().__init__(axis=axis)
        self._axis = axis
    # end def __init__

    @classmethod
    def check_arity(cls, operands: Operands):
        return len(operands) >= 1
    # end def check_arity

    def eval(self, operands: Operands, **kwargs) -> Tensor:
        if not operands:
            raise ValueError("Concatenate expects at least one operand.")
        return self._eval(operands=operands, **kwargs)
    # end def eval

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return False
    # end def contains

    def check_parameters(self, axis: Optional[int] = 0) -> bool:
        return axis is None or isinstance(axis, int)
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) >= 1
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensors = [operand.eval() for operand in operands]
        return tensor_concatenate(tensors, axis=self._axis)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Concatenate does not support backward.")
    # end def _backward

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        for operand in operands[1:]:
            dtype = promote(dtype, operand.dtype)
        return dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        if not operands:
            raise ValueError("Cannot infer shape without operands.")
        if self._axis is None:
            return Shape((self._flattened_size(operands),))
        base_shape = operands[0].input_shape
        axis = self._normalized_axis(base_shape.rank)
        dims = list(base_shape.dims)
        dims[axis] = sum(operand.input_shape[axis] for operand in operands)
        return Shape(tuple(dims))
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        if not operands:
            raise ValueError("Concatenate requires at least one operand.")
        if self._axis is None:
            return True
        reference = operands[0].input_shape
        axis = self._normalized_axis(reference.rank)
        for operand in operands[1:]:
            if operand.input_shape.rank != reference.rank:
                raise ValueError("All operands must share the same rank for concatenation.")
            for idx, (dim_ref, dim_other) in enumerate(zip(reference.dims, operand.input_shape.dims)):
                if idx == axis:
                    continue
                if dim_ref != dim_other:
                    raise ValueError("Operand dimensions must match on non-concatenated axes.")
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
            total += size
        return total
    # end def _flattened_size

# end class Concatenate


class HStack(Concatenate):
    """Concatenate tensors along axis 1."""

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
    """Concatenate tensors along axis 0."""

    NAME = "vstack"

    def __init__(self):
        super().__init__(axis=0)
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensors = [operand.eval() for operand in operands]
        return tensor_vstack(tensors)
    # end def _eval

# end class VStack


class Zeros(ParametricBuilder):
    """
    Tensor of zeros with a prescribed shape and dtype.
    """

    NAME = "zeros"
    ARITY = 0

    def __init__(self, shape: Shape, dtype: DType = DType.R):
        super().__init__(shape=shape, dtype=dtype)
        self._shape = shape
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    @classmethod
    def check_parameters(cls, shape: Shape, dtype: DType = DType.R) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        numpy_dtype = to_numpy(self._dtype)
        data = np.zeros(self._shape.dims, dtype=numpy_dtype)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Zeros does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(shape={self._shape.dims}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Zeros


class Ones(ParametricBuilder):
    """
    Tensor of ones with a prescribed shape and dtype.
    """

    NAME = "ones"
    ARITY = 0

    def __init__(self, shape: Shape, dtype: DType = DType.R):
        super().__init__(shape=shape, dtype=dtype)
        self._shape = shape
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    @classmethod
    def check_parameters(cls, shape: Shape, dtype: DType = DType.R) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        numpy_dtype = to_numpy(self._dtype)
        data = np.ones(self._shape.dims, dtype=numpy_dtype)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Ones does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(shape={self._shape.dims}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Ones


class Eye(ParametricBuilder):
    """
    Identity matrix builder.
    """

    NAME = "eye"
    ARITY = 0

    def __init__(self, rows: int, cols: Optional[int] = None, dtype: DType = DType.R):
        if rows <= 0:
            raise ValueError("Eye requires rows > 0.")
        cols = cols if cols is not None else rows
        if cols <= 0:
            raise ValueError("Eye requires cols > 0.")
        super().__init__(rows=rows, cols=cols, dtype=dtype)
        self._rows = rows
        self._cols = cols
        self._dtype = dtype
        self._shape = Shape.matrix(rows, cols)
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    @classmethod
    def check_parameters(cls, rows: int, cols: Optional[int] = None, dtype: DType = DType.R) -> bool:
        cols = cols if cols is not None else rows
        return rows > 0 and cols > 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        numpy_dtype = to_numpy(self._dtype)
        data = np.eye(self._rows, self._cols, dtype=numpy_dtype)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Eye does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(rows={self._rows}, cols={self._cols}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Eye


class Identity(Eye):
    """
    Square identity builder shortcut.
    """

    NAME = "identity"

    def __init__(self, size: int, dtype: DType = DType.R):
        super().__init__(rows=size, cols=size, dtype=dtype)
        self._size = size
    # end def __init__

    def __str__(self) -> str:
        return f"{self.NAME}(size={self._size}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Identity


class Linspace(Builder):
    """
    Generate linearly spaced values between two scalars.
    """

    NAME = "linspace"
    ARITY = 0

    def __init__(self, start: MathNode, stop: MathNode, num: MathNode):
        super().__init__(start=start, stop=stop, num=num)
        self._start = start
        self._stop = stop
        self._num = num
        self._shape = self._resolve_shape()
    # end def __init__

    def _validate_scalar(self, expr: MathNode, name: str) -> None:
        if expr.rank != 0:
            raise ValueError(f"{self.NAME} {name} must be scalar.")
    # end def _validate_scalar

    def _resolve_shape(self) -> Shape:
        self._validate_scalar(self._start, "start")
        self._validate_scalar(self._stop, "stop")
        self._validate_scalar(self._num, "num")
        if not self._num.is_constant():
            raise ValueError(f"{self.NAME} num must be a constant integer.")
        num_value = int(np.asarray(self._num.eval().value).item())
        if num_value <= 0:
            raise ValueError(f"{self.NAME} num must be positive.")
        self._length = num_value
        return Shape.vector(self._length)
    # end def _resolve_shape

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_parameters(self, **kwargs) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return promote(self._start.dtype, self._stop.dtype)
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        dtype = self.infer_dtype(())
        numpy_dtype = to_numpy(dtype)
        start_val = np.asarray(self._start.eval().value, dtype=numpy_dtype).item()
        stop_val = np.asarray(self._stop.eval().value, dtype=numpy_dtype).item()
        data = np.linspace(start_val, stop_val, self._length, dtype=numpy_dtype)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Linspace does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(length={self._length})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Linspace


class Logspace(Builder):
    """
    Generate logarithmically spaced values.
    """

    NAME = "logspace"
    ARITY = 0

    def __init__(self, start: MathNode, stop: MathNode, num: MathNode, base: MathNode):
        super().__init__(start=start, stop=stop, num=num, base=base)
        self._start = start
        self._stop = stop
        self._num = num
        self._base = base
        self._shape = self._resolve_shape()
        self._base_value = self._resolve_base()
    # end def __init__

    def _validate_scalar(self, expr: MathNode, name: str) -> None:
        if expr.rank != 0:
            raise ValueError(f"{self.NAME} {name} must be scalar.")
        # end if
    # end def _validate_scalar

    def _resolve_shape(self) -> Shape:
        for expr, name in (
                (self._start, "start"),
                (self._stop, "stop"),
                (self._num, "num"),
                (self._base, "base"),
        ):
            self._validate_scalar(expr, name)
        if not self._num.is_constant():
            raise ValueError(f"{self.NAME} num must be a constant integer.")
        # end for
        num_value = int(np.asarray(self._num.eval().value).item())
        if num_value <= 0:
            raise ValueError(f"{self.NAME} num must be positive.")
        # end if
        self._length = num_value
        return Shape.vector(self._length)
    # end def _resolve_shape

    def _resolve_base(self) -> float:
        if not self._base.is_constant():
            raise ValueError(f"{self.NAME} base must be constant.")
        base_value = float(np.asarray(self._base.eval().value).item())
        if base_value <= 0:
            raise ValueError(f"{self.NAME} base must be positive.")
        return base_value
    # end def _resolve_base

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_parameters(self, **kwargs) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = promote(self._start.dtype, self._stop.dtype)
        dtype = promote(dtype, self._base.dtype)
        return dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        dtype = self.infer_dtype(())
        numpy_dtype = to_numpy(dtype)
        start_val = np.asarray(self._start.eval().value, dtype=numpy_dtype).item()
        stop_val = np.asarray(self._stop.eval().value, dtype=numpy_dtype).item()
        data = np.logspace(
            start_val,
            stop_val,
            self._length,
            base=self._base_value,
            dtype=numpy_dtype
        )
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Logspace does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(length={self._length}, base={self._base_value})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Logspace


class Map(Builder):
    """
    Apply a symbolic body expression over each element of a tensor.
    """

    NAME = "map"
    ARITY = 1

    def __init__(self, var_name: str, body: MathNode):
        if not var_name:
            raise ValueError("Map requires a non-empty variable name.")
        super().__init__(var_name=var_name, body=body)
        self._var_name = var_name
        self._body = body
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return self._body.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != 1:
            raise ValueError("Map expects exactly one tensor operand.")
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        tensor, = operands
        if tensor.rank < 0:
            raise ValueError("Map operand must have a valid shape.")
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        tensor, = operands
        return tensor.shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._body.dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor, = operands
        tensor_value = tensor.eval().value
        numpy_dtype = to_numpy(self._body.dtype)
        result = np.empty(tensor_value.shape, dtype=numpy_dtype)
        with new_context():
            it = np.nditer(tensor_value, flags=["multi_index"])
            for value in it:
                set_value(self._var_name, np.asarray(value).item())
                evaluated = self._body.eval().value
                result[it.multi_index] = np.asarray(evaluated, dtype=numpy_dtype)
            # end for
        # end with
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=self._body.dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Map does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(var='{self._var_name}')"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Map


class SparseCOO(Builder):
    """
    Build a dense tensor from sparse COO data.
    """

    NAME = "sparse_coo"
    ARITY = -1
    IS_VARIADIC = True

    def __init__(self, shape: Shape, indices: Sequence[tuple[int, ...]]):
        super().__init__(shape=shape, indices=indices)
        self._shape = shape
        self._indices = self._validate_indices(indices)
    # end def __init__

    def _validate_indices(
            self,
            indices: Sequence[tuple[int, ...]]
    ) -> tuple[tuple[int, ...], ...]:
        rank = self._shape.rank
        validated: list[tuple[int, ...]] = []
        for idx in indices:
            if len(idx) != rank:
                raise ValueError(f"Index {idx} does not match rank {rank}.")
            # end if
            for dim, size in zip(idx, self._shape.dims):
                if dim < 0 or dim >= size:
                    raise ValueError(f"Index {idx} is out of bounds for shape {self._shape.dims}.")
                # end if
            # end for
            validated.append(tuple(int(dim) for dim in idx))
        return tuple(validated)
    # end def _validate_indices

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return False
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            shape: Shape,
            indices: Sequence[tuple[int, ...]]
    ) -> bool:
        return len(indices) >= 0
    # end def check_parameters

    @classmethod
    def check_arity(cls, operands: Operands) -> bool:
        return True
    # end def check_arity

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != len(self._indices):
            raise ValueError(
                f"{self.NAME} expects {len(self._indices)} values, got {len(operands)}"
            )
        # end if
        if not operands:
            raise ValueError("SparseCOO requires at least one value.")
        # end if
        for operand in operands:
            if operand.rank != 0:
                raise ValueError("SparseCOO values must be scalar expressions.")
            # end if
        # end for
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        for operand in operands[1:]:
            dtype = promote(dtype, operand.dtype)
        # end for
        return dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        dtype = self.infer_dtype(operands)
        numpy_dtype = to_numpy(dtype)
        zero_const = const(
            name=random_const_name(f"{self.NAME}-zero-"),
            data=0,
            dtype=dtype
        )
        data = np.full(self._shape.dims, zero_const.eval().value, dtype=numpy_dtype)
        for index, value_expr in zip(self._indices, operands):
            value = value_expr.eval().value
            data[index] = np.asarray(value, dtype=numpy_dtype)
        # end for
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("SparseCOO does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(shape={self._shape.dims}, nnz={len(self._indices)})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class SparseCOO


class FromFunction(Builder):
    """
    Build a tensor by evaluating a body expression over index variables.
    """

    NAME = "from_function"
    ARITY = 1
    DEFAULT_INDEX_NAMES = ("i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t")
    __slots__ = ("_shape", "_index_vars")

    def __init__(self, shape: Shape, index_vars: Optional[Sequence[str]] = None):
        super().__init__(shape=shape, index_vars=index_vars)
        self._shape = shape
        self._index_vars = self._resolve_index_vars(index_vars)
    # end def __init__

    def _resolve_index_vars(
            self,
            index_vars: Optional[Sequence[str]]
    ) -> tuple[str, ...]:
        # 1. If None -> create variables from default index names
        rank = self._shape.rank
        if index_vars is None:
            generated: list[str] = []
            for idx in range(rank):
                if idx < len(self.DEFAULT_INDEX_NAMES):
                    name = self.DEFAULT_INDEX_NAMES[idx]
                else:
                    name = f"idx_{idx}"
                # end if
                generated.append(name)
            return tuple(generated)
        # end if
        if len(index_vars) != rank:
            raise ValueError(f"Expected {rank} index variables, got {len(index_vars)}")
        # end if
        for var in index_vars:
            if not isinstance(var, str):
                raise TypeError("Index variables must be variable names.")
            # end if
        # end for
        return tuple(index_vars)
    # end def _resolve_index_vars

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return False
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            shape: Shape,
            index_vars: Optional[Sequence[str]] = None
    ) -> bool:
        if index_vars is None:
            return True
        # end if
        return len(index_vars) == shape.rank
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != 1:
            raise ValueError(f"{self.NAME} expects exactly one operand.")
        # end if
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        body, = operands
        if body.rank != 0:
            raise ValueError("FromFunction body must be scalar.")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        body, = operands
        return body.dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        body, = operands
        dtype = body.dtype
        numpy_dtype = to_numpy(dtype)
        result = np.empty(self._shape.dims, dtype=numpy_dtype)

        with new_context():
            for idx in np.ndindex(self._shape.dims):
                for var_name, value in zip(self._index_vars, idx):
                    set_value(var_name, value)
                # end for
                evaluated = body.eval()
                result[idx] = np.asarray(evaluated.value, dtype=numpy_dtype)
            # end for
        # end with

        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("FromFunction does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        names = ", ".join(var_name for var_name in self._index_vars)
        return f"{self.NAME}(shape={self._shape.dims}, index_vars=[{names}])"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class FromFunction


operator_registry.register(Vector)
operator_registry.register(Matrix)
operator_registry.register(Full)
operator_registry.register(Diag)
operator_registry.register(Concatenate)
operator_registry.register(HStack)
operator_registry.register(VStack)
operator_registry.register(Zeros)
operator_registry.register(Ones)
operator_registry.register(Eye)
operator_registry.register(Identity)
operator_registry.register(Map)
operator_registry.register(Linspace)
operator_registry.register(Logspace)
operator_registry.register(SparseCOO)
operator_registry.register(FromFunction)
