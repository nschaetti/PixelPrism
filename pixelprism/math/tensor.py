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
from typing import Iterable, List, Union, Any, Optional, Tuple, Callable, Sequence, Literal
import numpy as np

from .shape import Shape, ShapeLike
from .dtype import DType, TypeLike, to_numpy, convert_numpy, from_numpy
from .typing import TensorLike, NumberLike, NumberListLike


__all__ = [
    "t_tensor",
    "t_scalar",
    "Tensor",
    "TensorLike",
    "ts_scalar",
    "ts_vector",
    "ts_matrix",
    "t_full",
    "t_zeros",
    "t_ones",
    "t_concatenate",
    "t_hstack",
    "t_vstack",
    "t_pow",
    "t_square",
    "t_sqrt",
    "t_cbrt",
    "t_reciprocal",
    "t_exp",
    "t_exp2",
    "t_expm1",
    "t_log",
    "t_log2",
    "t_log10",
    "t_log1p",
    "t_sin",
    "t_cos",
    "t_tan",
    "t_arcsin",
    "t_arccos",
    "t_arctan",
    "t_atan2",
    "t_sinh",
    "t_cosh",
    "t_tanh",
    "t_arcsinh",
    "t_arccosh",
    "t_arctanh",
    "t_deg2rad",
    "t_rad2deg",
    "t_absolute",
    "t_abs",
    "t_sign",
    "t_floor",
    "t_ceil",
    "t_trunc",
    "t_rint",
    "t_round",
    "t_clip",
    "t_einsum",
    "t_equal",
    "t_not_equal",
    "t_less",
    "t_less_equal",
    "t_greater",
    "t_greater_equal",
    "t_logical_not",
    "t_logical_and",
    "t_logical_or",
    "t_logical_xor",
    "t_any",
    "t_all",
]


# region PRIVATE METHODS


def _resolve_dtype(dtype):
    """Return a numpy dtype for helper constructors."""
    return to_numpy(dtype)
# end def _resolve_dtype


def _normalize_shape(shape: ShapeLike) -> Tuple[int, ...]:
    """Normalize user-provided shape inputs into a tuple of ints."""
    dims: Tuple[int, ...] = ()
    if isinstance(shape, int):
        dims = (shape,)
    # end if
    if isinstance(shape, tuple):
        dims = tuple(shape)
    elif isinstance(shape, list):
        dims = tuple(shape)
    elif isinstance(shape, Shape):
        return shape.dims
    # end if
    assert all(isinstance(dim, int) and dim >= 0 for dim in dims), "shape must be non-negative integers"
    return dims
# end def _normalize_shape


# Tensor class
def _numpy_dtype_to_dtype(dtype: np.dtype) -> DType:
    """Convert numpy dtype to pixelprism dtype.

    Parameters
    ----------
    dtype: np.dtype
        Numpy dtype.

    Returns
    -------
    DType
        Converted dtype.
    """
    return from_numpy(dtype)
# end def _numpy_dtype_to_dtype


def _convert_data_to_numpy_array(
        data: Union[List[float], np.ndarray],
        dtype: Optional[TypeLike] = None
) -> np.ndarray:
    """Convert data to numpy array."""
    if dtype:
        dtype = _convert_dtype_to_numpy(dtype)
    # end if
    if isinstance(data, np.ndarray):
        if dtype is not None:
            if data.dtype != dtype:
                data = data.astype(dtype)
            # end if
        return data
    else:
        return np.array(data, dtype=dtype)
    # end if
# end def _convert_data


def _convert_dtype_to_numpy(dtype: TypeLike) -> np.dtype:
    """Convert dtype to numpy dtype."""
    return to_numpy(dtype)
# end def _convert_dtype_to_numpy


def _get_dtype(
        data: TensorLike,
        dtype: Optional[TypeLike] = None
) -> np.dtype:
    """Get dtype from data."""
    if dtype is not None:
        dtype = to_numpy(dtype)
    elif isinstance(data, np.ndarray):
        dtype = data.dtype
    elif isinstance(data, np.generic):
        dtype = data.dtype
    elif isinstance(data, float):
        dtype = to_numpy(DType.R)
    elif isinstance(data, int):
        dtype = to_numpy(DType.Z)
    elif isinstance(data, bool):
        dtype = np.bool_
    elif isinstance(data, complex):
        dtype = to_numpy(DType.C)
    elif isinstance(data, list):
        # TODO: check numpy convention
        dtype = to_numpy(DType.R)
    else:
        raise TypeError(f"Unsupported or unknown data type when creating tensor: {type(data)}")
    # end if
    return dtype
# end if


def _check_shape(
        data: np.ndarray,
        n_dims: int
):
    """Check that the shape of `data` matches the expected shape."""
    assert data.ndim == n_dims, f"shape mismatch: expected {n_dims}, got {data.shape}"
# end _check_shape


def _data_as_nparray(
        data: TensorLike,
        dtype: TypeLike
):
    try:
        return np.asarray(data, dtype=_resolve_dtype(dtype))
    except ValueError as e:
        raise ValueError(f"cannot convert input to NumPy array, invalid data: {e}") from e
    # end try
# end def _data_as_nparray


def _dim_tensor(
        data: TensorLike,
        ndim: int,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """Allocate a tensor with a single dimension."""
    data = _data_as_nparray(data, dtype=_resolve_dtype(dtype))
    _check_shape(data, n_dims=ndim)
    return Tensor(data=data, mutable=mutable)
# end def _dim_tensor


# endregion PRIVATE METHODS


# region PUBLIC METHODS


def t_tensor(
        data: TensorLike,
        dtype: Optional[TypeLike] = None,
        mutable: bool = True
) -> 'Tensor':
    """
    Construct a tensor wrapper around an existing NumPy array.

    Parameters
    ----------
    name : str
        Human-readable identifier assigned to the tensor.
    data : DataType
        Array buffer that will be wrapped without copying.
    dtype : AnyDType, default float
        Data type of the underlying array.
    mutable : bool, default True
        Whether subsequent operations may mutate the tensor in-place.

    Returns
    -------
    'Tensor'
        Tensor instance referencing ``data``.

    Examples
    --------
    >>> import numpy as np
    >>> from pixelprism.math import t_tensor
    >>> logits = t_tensor([[1, 2], [3, 4]])
    >>> logits.shape
    (2, 2)
    """
    return Tensor(data=data, dtype=dtype, mutable=mutable)
# end def t_tensor


def t_scalar(
        value: int | float | np.number | bool | complex,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Create a scalar tensor (zero-dimensional array) from a numeric value.

    Parameters
    ----------
    value : NumericType
        Scalar value that can be converted to a NumPy 0-D array.
    dtype : AnyDType, default float
        Target numerical dtype for the stored value.
    mutable : bool, default True
        Whether the tensor can be mutated later on.

    Returns
    -------
    Tensor
        Scalar tensor storing ``value``.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> bias = pm.t_scalar(3.5)
    >>> bias
    array(3.5)
    """
    return _dim_tensor(data=value, ndim=0, dtype=dtype, mutable=mutable)
# end def t_scalar


def t_vector(
        value: TensorLike,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Create a 1-D tensor from a vector-like input.

    Parameters
    ----------
    name : str
        Identifier assigned to the resulting tensor.
    value : NumericType
        Sequence or array that can be coerced into a one-dimensional array.
    dtype : AnyDType, default float
        Desired dtype of the resulting vector.
    mutable : bool, default True
        Whether the tensor may be mutated.

    Returns
    -------
    Tensor
        Vector tensor containing ``value``.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> weights = pm.t_vector([0.2, 0.3, 0.5])
    >>> weights.input_shape
    (3,)
    """
    return _dim_tensor(data=value, ndim=1, dtype=dtype, mutable=mutable)
# end def t_vector


def t_matrix(
        value: TensorLike,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Create a 2-D tensor from matrix-like input.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
    value : NumericType
        Array-like object that can be reshaped into two dimensions.
    dtype : AnyDType, default float
        Desired dtype for the matrix entries.
    mutable : bool, default True
        Whether the matrix can be mutated later on.

    Returns
    -------
    Tensor
        Matrix tensor populated with ``value``.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> mat = pm.matrix([[1, 0], [0, 1]])
    >>> mat
    array([[1., 0.],
           [0., 1.]])
    """
    return _dim_tensor(data=value, ndim=2, dtype=dtype, mutable=mutable)
# end def t_matrix


def t_empty(
        shape: ShapeLike,
        dtype: TypeLike = float
) -> 'Tensor':
    """
    Allocate an uninitialized tensor of the requested shape.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
    shape : AnyShape
        Dimensions of the tensor; accepts ints, tuples, lists or ``Shape``.
    dtype : AnyDType, default float
        Data type of the uninitialized buffer.

    Returns
    -------
    Tensor
        Tensor backed by ``np.empty(shape, dtype)``.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> scratch = pm.t_empty((2, 3))
    >>> scratch.input_shape
    (2, 3)
    """
    dims = _normalize_shape(shape)
    data = np.empty(dims, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=True)
# end def t_empty


def zeros(
        shape: ShapeLike,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Allocate a tensor initialized with zeros.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
    shape : AnyShape
        Desired tensor dimensions.
    dtype : AnyDType, default float
        Numerical dtype of the zeros.
    mutable : bool, default True
        Whether the tensor is mutable.

    Returns
    -------
    Tensor
        Tensor filled with zeros of the given ``shape``.

    Examples
    --------
    >>> import pixelprism.math as ppmath
    >>> zeros_tensor = ppmath.zeros((2, 2))
    >>> zeros_tensor
    array([[0., 0.],
           [0., 0.]])
    """
    dims = _normalize_shape(shape)
    data = np.zeros(dims, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=mutable)
# end def zeros


def ones(
        shape: ShapeLike,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Allocate a tensor initialized with ones.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
    shape : AnyShape
        Desired tensor dimensions.
    dtype : AnyDType, default float
        Numerical dtype of the ones.
    mutable : bool, default True
        Whether the tensor may later be mutated.

    Returns
    -------
    Tensor
        Tensor filled with ones of the requested ``shape``.

    Examples
    --------
    >>> import pixelprism.math as ppmath
    >>> ones_tensor = ppmath.ones(4)
    >>> ones_tensor
    array([1., 1., 1., 1.])
    """
    dims = _normalize_shape(shape)
    data = np.ones(dims, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=mutable)
# end def ones


def full(
        shape: ShapeLike,
        value,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Allocate a tensor whose entries are filled with a constant value.

    Parameters
    ----------
    shape : AnyShape
        Desired tensor dimensions.
    value : Any
        Value used to populate the tensor.
    dtype : AnyDType, default float
        Data type used to store ``value``.
    mutable : bool, default True
        Whether the tensor can be mutated.

    Returns
    -------
    Tensor
        Tensor filled entirely with ``value``.

    Examples
    --------
    >>> import pixelprism.math as ppmath
    >>> mask = ppmath.full((2, 3), 7)
    >>> mask
    array([[7., 7., 7.],
           [7., 7., 7.]])
    """
    dims = _normalize_shape(shape)
    data = np.full(dims, value, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=mutable)
# end def full


def nan(
        shape: ShapeLike,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Allocate a tensor filled with ``NaN`` sentinels.

    Parameters
    ----------
    shape : AnyShape
        Desired tensor dimensions.
    dtype : AnyDType, default float
        Floating-point dtype used for the ``NaN`` values.
    mutable : bool, default True
        Whether the tensor can be mutated.

    Returns
    -------
    Tensor
        Tensor filled with ``np.nan`` values.

    Examples
    --------
    >>> import pixelprism.math as ppmath
    >>> missing = ppmath.nan((2, 2))
    >>> np.isnan(missing).t_all()
    True
    """
    dims = _normalize_shape(shape)
    data = np.full(dims, np.nan, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=mutable)
# end def nan


def I(
        n: int,
        dtype: TypeLike = float,
        mutable: bool = False
) -> 'Tensor':
    """
    Construct an ``n`` by ``n`` identity matrix tensor.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
    n : int
        Dimension of the square identity matrix.
    dtype : AnyDType, default float
        Numerical dtype of the diagonal ones.
    mutable : bool, default False
        Whether the tensor can be mutated.

    Returns
    -------
    Tensor
        Identity matrix tensor of shape ``(n, n)``.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> eye = pm.I(3)
    >>> eye.data
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    assert isinstance(n, int) and n >= 0, "n must be a non-negative integer"
    data = np.eye(n, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=mutable)
# end def I


def diag(
        v,
        dtype: TypeLike = float,
        mutable: bool = True
) -> 'Tensor':
    """
    Construct a square matrix using the provided diagonal values.

    Parameters
    ----------
    v : array-like
        One-dimensional values placed along the diagonal.
    dtype : AnyDType, default float
        Dtype used to represent the diagonal values.
    mutable : bool, default True
        Whether the tensor may be mutated.

    Returns
    -------
    Tensor
        Square matrix tensor with ``v`` on the diagonal.

    Examples
    --------
    >>> import pixelprism.math as ppmath
    >>> diag_tensor = ppmath.diag([1, 2, 3])
    >>> diag_tensor
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    """
    diag_values = np.asarray(v, dtype=_resolve_dtype(dtype))
    assert diag_values.ndim == 1, "diag input must be 1-D"
    data = np.diag(diag_values)
    return Tensor(data=data, mutable=mutable)
# end def diag


def eye_like(
        x: Union['Tensor', np.ndarray],
        dtype: TypeLike = None,
        mutable: bool = True
) -> 'Tensor':
    """
    Build an identity matrix tensor with the same shape (and optional dtype) as ``x``.

    Parameters
    ----------
    x : Tensor or numpy.ndarray
        Reference tensor/array that provides the square shape.
    dtype : AnyDType, optional
        Overrides the dtype of the created identity matrix. Defaults to ``x``'s dtype.
    mutable : bool, default True
        Whether the tensor is mutable.

    Returns
    -------
    Tensor
        Identity matrix tensor shaped like ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> import pixelprism.math as pm
    >>> base = np.zeros((4, 4))
    >>> eye = pm.eye_like(base)
    >>> np.allclose(eye.value, np.eye(4))
    True
    """
    assert x.rank == 2, "eye_like expects a 2-D input"
    rows, cols = x.shape
    assert rows == cols, "eye_like requires a square matrix input"
    dtype = _resolve_dtype(dtype) if dtype else _resolve_dtype(x.dtype)
    data = np.eye(rows, dtype=dtype)
    return Tensor(data=data, mutable=mutable)
# end def eye_like


def zeros_like(
        x: Union['Tensor', np.ndarray],
        dtype: TypeLike = None,
        mutable: bool = True
) -> 'Tensor':
    """
    Allocate a zero tensor matching the shape of ``x``.

    Parameters
    ----------
    x : Tensor or numpy.ndarray
        Reference tensor/array that provides the target shape.
    dtype : AnyDType, optional
        Overrides the dtype of the resulting tensor. Defaults to ``x``'s dtype.
    mutable : bool, default True
        Whether the tensor is mutable.

    Returns
    -------
    Tensor
        Tensor filled with zeros and shaped like ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> import pixelprism.math as ppmath
    >>> template = np.arange(6).reshape(2, 3)
    >>> zeros_clone = ppmath.zeros_like(template)
    >>> zeros_clone.input_shape
    (2, 3)
    """
    base = np.asarray(x)
    dtype = _resolve_dtype(dtype) if dtype else base.dtype
    data = np.zeros(base.shape, dtype=dtype)
    return Tensor(data=data, mutable=mutable)
# end def zeros_like


def ones_like(
        x: Union['Tensor', np.ndarray],
        dtype: TypeLike = None,
        mutable: bool = True
) -> 'Tensor':
    """
    Allocate a tensor of ones sharing the shape of ``x``.

    Parameters
    ----------
    x : Tensor or numpy.ndarray
        Reference tensor/array that provides the target shape.
    dtype : AnyDType, optional
        Overrides the dtype of the resulting tensor. Defaults to ``x``'s dtype.
    mutable : bool, default True
        Whether the tensor is mutable.

    Returns
    -------
    Tensor
        Tensor filled with ones and shaped like ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> import pixelprism.math as ppmath
    >>> template = np.zeros((2, 2), dtype=np.float32)
    >>> ones_clone = ppmath.ones_like(template)
    >>> ones_clone
    array([[1., 1.],
           [1., 1.]], dtype=float32)
    """
    base = np.asarray(x)
    dtype = _resolve_dtype(dtype) if dtype else base.dtype
    data = np.ones(base.shape, dtype=dtype)
    return Tensor(data=data, mutable=mutable)
# end def ones_like


# endregion PUBLIC METHODS


class Tensor:
    """
    Declare a tensor as a class.
    """

    __array_priority__ = 1000  # ensure numpy prefers Tensor overrides

    def __init__(
            self,
            data: TensorLike, # TensorLike can be int/float, [int/float, [...]] or np.array
            *,
            dtype: Optional[TypeLike] = None,
            mutable: bool = True
    ):
        """Initialize a tensor from array-like data.

        Parameters
        ----------
        data : TensorLike
            Input parameter.
        dtype : Optional[TypeLike]
            Input parameter.
        mutable : bool
            Input parameter.
        """
        # Super
        self._dtype = from_numpy(_get_dtype(data, dtype))
        self._data = _convert_data_to_numpy_array(data=data, dtype=dtype)
        self._shape = Shape(dims=self._data.shape)
        self._mutable = mutable
    # end __init__

    # region PROPERTIES

    @property
    def value(self) -> np.ndarray:
        """
        Get the value of the tensor.
        
        Returns
        -------
        np.ndarray
            Result of the operation.
        """
        return self._data
    # end def value

    @property
    def dtype(self) -> DType:
        """Get the dtype of the tensor.
        
        Returns
        -------
        DType
            Result of the operation.
        """
        return self._dtype
    # end def dtype

    @property
    def shape(self) -> Shape:
        """Get the shape of the tensor.
        
        Returns
        -------
        Shape
            Result of the operation.
        """
        return self._shape
    # end def shape

    @property
    def dims(self) -> Tuple[int, ...]:
        """Get the dimensions of the tensor.
        
        Returns
        -------
        Tuple[int, ...]
            Result of the operation.
        """
        return self._shape.dims
    # end def dims

    @property
    def mutable(self) -> bool:
        """Get the mutable status of the tensor.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        return self._mutable
    # end def mutable

    @property
    def is_mutable(self) -> bool:
        """Get the mutable status of the tensor.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        return self._mutable
    # end def is_mutable

    @property
    def ndim(self) -> int:
        """Get the dimension of the tensor.
        
        Returns
        -------
        int
            Result of the operation.
        """
        return self._shape.rank
    # end def dim

    @property
    def rank(self) -> int:
        """Get the dimension of the tensor.
        
        Returns
        -------
        int
            Result of the operation.
        """
        return self._shape.rank
    # end def rank

    @property
    def size(self) -> int:
        """Get the size of the tensor.
        
        Returns
        -------
        int
            Result of the operation.
        """
        return self._shape.size
    # end def size

    @property
    def n_elements(self) -> int:
        """Get the number of elements in the tensor.
        
        Returns
        -------
        int
            Result of the operation.
        """
        return self._shape.size
    # end def n_elements

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, data: TensorLike) -> None:
        """Replace the tensor data in-place.
        
        Parameters
        ----------
        data : TensorLike
            Input parameter.
        
        Returns
        -------
        None
            Result of the operation.
        """

        if isinstance(data, list):
            data = _convert_data_to_numpy_array(data, dtype=to_numpy(self.dtype))
        # end if
        if data.shape != self.shape:
            raise ValueError(f"Cannot assign data of shape {data.shape} to tensor of shape {self.shape}.")
        # end if
        data = convert_numpy(self.dtype, data)
        self._data = data
    # end set

    def copy(
            self,
            order: Literal["C", "F", "A"] = "C",
            mutable: Optional[bool] = None,
            copy_data: bool = True
    ) -> 'Tensor':
        """Return a copy of this tensor.
        
        Parameters
        ----------
        order: Literal["C", "F", "A"]
            Order of the copy.
        mutable : Optional[bool]
            Input parameter.
        copy_data : bool
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(
            data=self._data.copy(order=order) if copy_data else self._data,
            mutable=mutable if mutable is not None else self._mutable
        )
    # end copy

    def item(self) -> Union[int, float]:
        """Return the first element of the tensor as a scalar.
        
        Returns
        -------
        Union[int, float]
            Result of the operation.
        """
        if self.rank == 0:
            return self._data.item()
        else:
            raise ValueError("Cannot convert a tensor to a single integer value.")
        # end if
    # end def item

    def astype(self, dtype: DType) -> 'Tensor':
        """Convert the tensor to a different dtype.
        
        Parameters
        ----------
        dtype : DType
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=self._data.astype(to_numpy(dtype)), dtype=dtype)
    # end def as_type

    def astype_(self, new_type: DType) -> 'Tensor':
        """Convert the tensor to a different dtype.
        
        Parameters
        ----------
        new_type : DType
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        self._dtype = new_type
        self._data = convert_numpy(self._dtype, self._data)
        return self
    # end def as_type_

    def as_bool(self) -> 'Tensor':
        """Convert the tensor to a boolean tensor.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=self._data.astype(np.bool_))
    # end def as_bool

    def as_float(self) -> 'Tensor':
        """Convert the tensor to a float tensor.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=self._data.astype(to_numpy(DType.R)))
    # end def as_float

    def as_int(self) -> 'Tensor':
        """Convert the tensor to an integer tensor.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=self._data.astype(to_numpy(DType.Z)))
    # end def as_int

    def as_immutable(self) -> 'Tensor':
        """Convert the tensor to an immutable tensor.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=self._data.copy(), mutable=False)
    # end def as_immutable

    def as_mutable(self) -> 'Tensor':
        """Convert the tensor to a mutable tensor.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=self._data.copy(), mutable=True)
    # end def as_mutable

    def reshape(self, shape: ShapeLike) -> 'Tensor':
        """Reshape the tensor.
        
        Parameters
        ----------
        shape : ShapeLike
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        shape = Shape.create(shape)
        if not self.shape.can_reshape(shape):
            raise ValueError(f"Cannot reshape tensor of shape {self.shape} to {shape}.")
        # end if
        return Tensor(data=self._data.reshape(shape.dims))
    # end def reshape

    def ravel(self, order: Literal['C', 'F', 'A', 'K'] = 'C') -> 'Tensor':
        """
        Returns a contiguous flattened tensor.

        Returns:
            y is a contiguous 1-D array of the same subtype as a, with shape `(a.size,)`.
            Note that matrices are special cased for backward compatibility, if a is a matrix,
            then y is a 1-D ndarray.
        """
        return Tensor(data=self._data.ravel(order=order))
    # end def ravel

    def tolist(self) -> NumberListLike:
        """Return the tensor data as nested Python lists.
        
        Returns
        -------
        NumberListLike
            Nested list of numbers.
        """
        return self._data.tolist()
    # end def tolist

    def numpy(self) -> np.ndarray:
        """Returns a numpy array representation of the tensor.

        Returns
        -------
        np.ndarray
            Numpy array representation of the tensor.
        """
        return self._data
    # end def numpy

    # endregion PUBLIC

    # region PRIVATE

    def _coerce_operand(self, other: Union["Tensor", TensorLike]) -> np.ndarray:
        """Convert operands to numpy arrays for arithmetic.
        
        Parameters
        ----------
        other : Union['Tensor', TensorLike, np.ndarray]
            Input parameter.
        
        Returns
        -------
        np.ndarray
            Result of the operation.
        """
        if isinstance(other, Tensor):
            return other.value
        # end if
        return np.asarray(other)
    # end def _coerce_operand

    def _binary_op(self, other: Union["Tensor", TensorLike], op: Callable) -> 'Tensor':
        """Apply a numpy binary operator and wrap in a Tensor.
        
        Parameters
        ----------
        other : object
            Input parameter.
        op : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        other_arr = self._coerce_operand(other)
        result = op(self._data, other_arr)
        return Tensor(data=np.asarray(result))
    # end def _binary_op

    def _binary_op_reverse(self, other: Union["Tensor", TensorLike], op: Callable) -> 'Tensor':
        """Apply a numpy binary operator with operands reversed.
        
        Parameters
        ----------
        other : object
            Input parameter.
        op : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        other_arr = self._coerce_operand(other)
        result = op(other_arr, self._data)
        return Tensor(data=np.asarray(result))
    # end def _binary_op_reverse

    def _unary_op(self, op: Callable) -> 'Tensor':
        """Apply a numpy unary operator and wrap in a Tensor.
        
        Parameters
        ----------
        op : Callable[[np.ndarray], np.ndarray]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = op(self._data)
        return Tensor(data=np.asarray(result))
    # end def _unary_op

    # endregion PRIVATE

    # region OVERRIDE

    def __add__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise addition.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.add)
    # end __add__

    def __radd__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise reverse addition.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op_reverse(other, np.add)
    # end __radd__

    def __sub__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise subtraction.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.subtract)
    # end __sub__

    def __rsub__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise reverse subtraction.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op_reverse(other, np.subtract)
    # end __rsub__

    def __mul__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise multiplication.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.multiply)
    # end __mul__

    def __rmul__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise reverse multiplication.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op_reverse(other, np.multiply)
    # end __rmul__

    def __truediv__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise division.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.divide)
    # end __truediv__

    def __rtruediv__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise reverse division.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op_reverse(other, np.divide)
    # end __rtruediv__

    def __pow__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise power.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.power)
    # end __pow__

    def __rpow__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise reverse power.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op_reverse(other, np.power)
    # end __rpow__

    def __matmul__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Matrix multiplication.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        other_arr = self._coerce_operand(other)
        result = np.matmul(self._data, other_arr)
        return Tensor(data=np.asarray(result))
    # end __matmul__

    def __rmatmul__(self, other: Union["Tensor", TensorLike]) -> 'Tensor':
        """Reverse matrix multiplication.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        other_arr = self._coerce_operand(other)
        result = np.matmul(other_arr, self._data)
        return Tensor(data=np.asarray(result))
    # end __rmatmul__

    def __neg__(self) -> 'Tensor':
        """Elementwise negation.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=np.negative(self._data))
    # end __neg__

    def __eq__(self, other: Union["Tensor", TensorLike]):
        """Compare equality against another object.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        if isinstance(other, Tensor):
            return bool(np.all(self._data == other._data))
        elif isinstance(other, np.ndarray):
            return bool(np.array_equal(self._data, other))
        elif isinstance(other, (int, float)):
            return bool(np.array_equal(self._data, np.asarray(other)))
        elif isinstance(other, list):
            return bool(np.array_equal(self._data, np.asarray(other)))
        else:
            return False
        # end if
    # end __eq__

    def __ne__(self, other: Union["Tensor", TensorLike]):
        """Compare inequality against another object.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        return not self.__eq__(other)
    # end __ne__

    # TODO: check that np.any
    def __gt__(self, other):
        """Compare greater-than against another object.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        if isinstance(other, Tensor):
            return bool(np.any(self._data > other._data))
        elif isinstance(other, np.ndarray):
            return bool(np.any(np.greater(self._data, other)))
        elif isinstance(other, (int, float)):
            return bool(np.any(np.greater(self._data, np.asarray(other))))
        elif isinstance(other, list):
            return bool(np.any(np.greater(self._data, np.asarray(other))))
        else:
            return False
        # end if
    # end def __gt__

    # TODO: check that np.all
    def __ge__(self, other):
        """Compare greater-or-equal against another object.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        if isinstance(other, Tensor):
            return bool(np.all(self._data >= other._data))
        elif isinstance(other, np.ndarray):
            return bool(np.all(np.greater_equal(self._data, other)))
        elif isinstance(other, (int, float)):
            return bool(np.all(np.greater_equal(self._data, np.asarray(other))))
        elif isinstance(other, list):
            return bool(np.all(np.greater_equal(self._data, np.asarray(other))))
        else:
            return False
        # end if
    # end def __ge__

    # TODO: check that np.any
    def __lt__(self, other):
        """Compare less-than against another object.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        if isinstance(other, Tensor):
            return bool(np.any(self._data < other._data))
        elif isinstance(other, np.ndarray):
            return bool(np.any(np.less(self._data, other)))
        elif isinstance(other, (int, float)):
            return bool(np.any(np.less(self._data, np.asarray(other))))
        elif isinstance(other, list):
            return bool(np.any(np.less(self._data, np.asarray(other))))
        else:
            return False
        # end if
    # end def __lt__

    # TODO: check that np.all
    def __le__(self, other):
        """Compare less-or-equal against another object.
        
        Parameters
        ----------
        other : object
            Input parameter.
        
        Returns
        -------
        bool
            Result of the operation.
        """
        if isinstance(other, Tensor):
            return bool(np.all(self._data <= other._data))
        elif isinstance(other, np.ndarray):
            return bool(np.all(np.less_equal(self._data, other)))
        elif isinstance(other, (int, float)):
            return bool(np.all(np.less_equal(self._data, np.asarray(other))))
        elif isinstance(other, list):
            return bool(np.all(np.less_equal(self._data, np.asarray(other))))
        else:
            return False
        # end if
    # end def __le__

    # Override the integer conversion
    def __int__(self):
        """Convert a scalar tensor to an int.
        
        Returns
        -------
        int
            Result of the operation.
        """
        if self.rank == 0:
            return int(self._data.item())
        else:
            raise TypeError("Cannot convert a tensor to a single integer value.")
        # end if
    # end __int__

    # Override the float conversion
    def __float__(self):
        """Convert a scalar tensor to a float.
        
        Returns
        -------
        float
            Result of the operation.
        """
        if self.rank == 0:
            return float(self._data.item())
        else:
            raise TypeError("Cannot convert a tensor to a single integer value.")
        # end if
    # end __float__

    def __str__(self):
        """Return the string representation.
        
        Returns
        -------
        str
            Result of the operation.
        """
        return f"tensor({str(self._data.tolist())}, dtype={self._dtype}, shape={self._shape}, mutable={self._mutable})"
    # end __str__

    def __repr__(self):
        """Return the repr() representation.
        
        Returns
        -------
        str
            Result of the operation.
        """
        return self.__str__()
    # end __repr__

    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice], ...]]):
        """Return a sliced tensor view.
        
        Parameters
        ----------
        key : object
            Input parameter.
        
        Returns
        -------
        Tensor
            Result of the operation.
        """
        return Tensor(data=self._data[key])
    # end def __getitem__

    def __setitem__(self, key, value):
        """Assign values using NumPy-style indexing.
        
        Parameters
        ----------
        key : object
            Input parameter.
        value : object
            Input parameter.
        
        Returns
        -------
        None
            Result of the operation.
        """
        self._data[key] = value
    # end def __setitem__


    # endregion OVERRIDE

    # region MATH BASE

    def pow(self, exponent: Union["Tensor", TensorLike]) -> 'Tensor':
        """Elementwise power equivalent to numpy.power.
        
        Parameters
        ----------
        exponent : Union['Tensor', TensorLike, np.ndarray]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(exponent, np.power)
    # end def pow

    def square(self) -> 'Tensor':
        """Elementwise square.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.square)
    # end def square

    def sqrt(self) -> 'Tensor':
        """Elementwise square root.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.sqrt)
    # end def sqrt

    def cbrt(self) -> 'Tensor':
        """Elementwise cubic root.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.cbrt)
    # end def cbrt

    def reciprocal(self) -> 'Tensor':
        """Elementwise reciprocal.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.reciprocal)
    # end def reciprocal

    def exp(self) -> 'Tensor':
        """Elementwise natural exponential.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.exp)
    # end def exp

    def exp2(self) -> 'Tensor':
        """Elementwise base-2 exponential.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.exp2)
    # end def exp2

    def expm1(self) -> 'Tensor':
        """Elementwise exp(x) - 1 with better precision for small x.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.expm1)
    # end def expm1

    def log(self) -> 'Tensor':
        """Elementwise natural logarithm.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.log)
    # end def log

    def log2(self) -> 'Tensor':
        """Elementwise base-2 logarithm.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.log2)
    # end def log2

    def log10(self) -> 'Tensor':
        """Elementwise base-10 logarithm.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.log10)
    # end def log10

    def log1p(self) -> 'Tensor':
        """Elementwise log(1 + x) with higher accuracy for small x.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.log1p)
    # end def log1p

    def absolute(self) -> 'Tensor':
        """Elementwise absolute value.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.absolute)
    # end def absolute

    def abs(self) -> 'Tensor':
        """Alias for absolute to mirror numpy.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self.absolute()
    # end def abs

    def __abs__(self) -> 'Tensor':
        """Support Python's built-in abs().
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self.absolute()
    # end def __abs__

    # endregion MATH BASE

    # region MATH COMPARISON

    def equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise equality.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.equal)
    # end def equal

    def not_equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise inequality.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.not_equal)
    # end def not_equal

    def greater_equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise greater-than-or-equal-to.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.greater_equal)
    # end def greater_equal

    def greater(self, other: 'Tensor') -> 'Tensor':
        """Elementwise greater-than.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.greater)
    # end def greater

    def less_equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise less-than-or-equal-to.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.less_equal)
    # end def less_equal

    def less(self, other: 'Tensor') -> 'Tensor':
        """Elementwise less-than.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.less)
    # end def less

    def logical_not(self) -> 'Tensor':
        """Elementwise logical inversion.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.logical_not)
    # end def logical_not

    def logical_and(self, other: 'Tensor') -> 'Tensor':
        """Elementwise logical conjunction.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.logical_and)
    # end def logical_and

    def logical_or(self, other: 'Tensor') -> 'Tensor':
        """Elementwise logical disjunction.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.logical_or)
    # end def logical_or

    def logical_xor(self, other: 'Tensor') -> 'Tensor':
        """Elementwise logical exclusive-or.
        
        Parameters
        ----------
        other : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.logical_xor)
    # end def logical_xor

    # endregion MATH COMPARISON

    # region MATH TRIGO

    def sin(self) -> 'Tensor':
        """Elementwise sine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.sin)
    # end def sin

    def cos(self) -> 'Tensor':
        """Elementwise cosine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.cos)
    # end def cos

    def tan(self) -> 'Tensor':
        """Elementwise tangent.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.tan)
    # end def tan

    def arcsin(self) -> 'Tensor':
        """Elementwise inverse sine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.arcsin)
    # end def arcsin

    def arccos(self) -> 'Tensor':
        """Elementwise inverse cosine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.arccos)
    # end def arccos

    def arctan(self) -> 'Tensor':
        """Elementwise inverse tangent.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.arctan)
    # end def arctan

    def arctan2(
            self,
            other: Union["Tensor", TensorLike, np.ndarray]
    ) -> 'Tensor':
        """Elementwise two-argument arctangent.
        
        Parameters
        ----------
        other : Union['Tensor', TensorLike, np.ndarray]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.arctan2)
    # end def arctan2

    def sinh(self) -> 'Tensor':
        """Elementwise hyperbolic sine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.sinh)
    # end def sinh

    def cosh(self) -> 'Tensor':
        """Elementwise hyperbolic cosine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.cosh)
    # end def cosh

    def tanh(self) -> 'Tensor':
        """Elementwise hyperbolic tangent.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.tanh)
    # end def tanh

    def arcsinh(self) -> 'Tensor':
        """Elementwise inverse hyperbolic sine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.arcsinh)
    # end def arcsinh

    def arccosh(self) -> 'Tensor':
        """Elementwise inverse hyperbolic cosine.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.arccosh)
    # end def arccosh

    def arctanh(self) -> 'Tensor':
        """Elementwise inverse hyperbolic tangent.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.arctanh)
    # end def arctanh

    def deg2rad(self) -> 'Tensor':
        """Convert angles from degrees to radians.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.deg2rad)
    # end def deg2rad

    def rad2deg(self) -> 'Tensor':
        """Convert angles from radians to degrees.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.rad2deg)
    # end def rad2deg

    # endregion MATH TRIGO

    # region MATH DISCRETE

    def sign(self) -> 'Tensor':
        """Elementwise sign indicator.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.sign)
    # end def sign

    def floor(self) -> 'Tensor':
        """Elementwise floor.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.floor)
    # end def floor

    def ceil(self) -> 'Tensor':
        """Elementwise ceiling.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.ceil)
    # end def ceil

    def trunc(self) -> 'Tensor':
        """Elementwise truncate towards zero.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.trunc)
    # end def trunc

    def rint(self) -> 'Tensor':
        """Elementwise rounding to nearest integer.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._unary_op(np.rint)
    # end def rint

    def round(self, decimals: int = 0) -> 'Tensor':
        """Elementwise rounding with configurable precision.
        
        Parameters
        ----------
        decimals : int
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=np.array(np.round(self._data, decimals=decimals)))
    # end def round

    def clip(
            self,
            min_value: Optional[Union['Tensor', TensorLike, np.ndarray]] = None,
            max_value: Optional[Union['Tensor', TensorLike, np.ndarray]] = None
    ) -> 'Tensor':
        """Clip tensor values between bounds.
        
        Parameters
        ----------
        min_value : Optional[Union['Tensor', TensorLike, np.ndarray]]
            Input parameter.
        max_value : Optional[Union['Tensor', TensorLike, np.ndarray]]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be provided.")
        min_arr = self._coerce_operand(min_value) if min_value is not None else None
        max_arr = self._coerce_operand(max_value) if max_value is not None else None
        result = np.clip(self._data, min_arr, max_arr)
        return Tensor(data=np.asarray(result))
    # end def clip

    # endregion MATH DISCRETE

    # region MATH LINEAR

    def matmul(self, other: Union["Tensor", TensorLike, np.ndarray]) -> 'Tensor':
        """Matrix multiplication.
        
        Parameters
        ----------
        other : Union['Tensor', TensorLike, np.ndarray]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return self._binary_op(other, np.matmul)
    # end def matmul

    def einsum(self, equation: str, *operands: Union['Tensor', TensorLike, np.ndarray]) -> 'Tensor':
        """Compute the tensor product of the given operands using the given equation.
        
        Parameters
        ----------
        equation : str
            Input parameter.
        *operands : Union['Tensor', TensorLike, np.ndarray]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        operands = [_as_numpy_operand(self)] + [_as_numpy_operand(o) for o in operands]
        result = np.einsum(equation, *operands)
        return Tensor(data=np.asarray(result))
    # end def einsum

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> 'Tensor':
        """Compute the trace of a matrix.
        
        Parameters
        ----------
        offset : int
            Input parameter.
        axis1 : int
            Input parameter.
        axis2 : int
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.trace(self._data, offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def trace

    def transpose(self, axes: Optional[List[int]] = None) -> 'Tensor':
        """Transpose the tensor.
        
        Parameters
        ----------
        axes : Optional[List[int]]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.transpose(self._data, axes=axes)
        return Tensor(data=np.asarray(result))
    # end def transpose

    def inverse(self) -> 'Tensor':
        """Compute the inverse of a square matrix.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.linalg.inv(self._data)
        return Tensor(data=np.asarray(result))
    # end def inverse

    def det(self) -> 'Tensor':
        """Compute the determinant of a square matrix.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.linalg.det(self._data)
        return Tensor(data=np.asarray(result))
    # end def det

    def norm(self, order: Union[int, float] = 2) -> 'Tensor':
        """Compute the norm of the tensor.
        
        Parameters
        ----------
        order : Union[int, float]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.linalg.norm(self._data, ord=order)
        dtype = self._dtype if self._dtype in {DType.R, DType.C} else DType.R
        numpy_dtype = to_numpy(dtype)
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=dtype)
    # end def norm

    # endregion MATH LINEAR

    # region MATH REDUCTION

    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the sum of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.sum(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def sum

    def mean(self, axis:  Optional[int] = None) -> 'Tensor':
        """Compute the mean of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.mean(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def mean

    def std(self, axis: Optional[int] = None, ddof: int = 0) -> 'Tensor':
        """Compute the standard deviation of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        ddof : int
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.std(self._data, axis=axis, ddof=ddof)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def std

    def median(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the median of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.median(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def median

    def q1(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the first quartile of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.percentile(self._data, 25, axis=axis)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def q1

    def q3(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the third quartile of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.percentile(self._data, 75, axis=axis)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def q3

    def max(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the maximum of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.max(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def max

    def min(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the minimum of the tensor along the given axis.
        
        Parameters
        ----------
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.min(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._dtype)))
    # end def min

    def any(self) -> 'Tensor':
        """Return True if any element evaluates to True.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.any(self._data)
        return Tensor(data=np.asarray(result, dtype=np.bool_))
    # end def any

    def all(self) -> 'Tensor':
        """Return True if all elements evaluate to True.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        result = np.all(self._data)
        return Tensor(data=np.asarray(result, dtype=np.bool_))
    # end def all

    # endregion MATH REDUCTION

    # region RESHAPE

    def flatten(self) -> 'Tensor':
        """Flatten the tensor into a 1D array.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=np.reshape(self._data, (-1,)))
    # end def flatten

    def concatenate(self, *others: "Tensor", axis: Optional[int] = 0) -> 'Tensor':
        """Concatenate the current tensor with additional tensors.
        
        Parameters
        ----------
        *others : 'Tensor'
            Input parameter.
        axis : Optional[int]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        tensors: Tuple["Tensor", ...] = (self,) + tuple(others)
        return t_concatenate(tensors, axis=axis)
    # end def concatenate

    def hstack(self, *others: "Tensor") -> 'Tensor':
        """Concatenate tensors along axis 1.
        
        Parameters
        ----------
        *others : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        tensors: Tuple["Tensor", ...] = (self,) + tuple(others)
        return t_hstack(tensors)
    # end def hstack

    def vstack(self, *others: "Tensor") -> 'Tensor':
        """Concatenate tensors along axis 0.
        
        Parameters
        ----------
        *others : 'Tensor'
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        tensors: Tuple["Tensor", ...] = (self,) + tuple(others)
        return t_vstack(tensors)
    # end def vstack

    # endregion RESHAPE

    # region STATIC

    @staticmethod
    def from_numpy(data: np.ndarray, dtype: Optional[TypeLike] = None) -> 'Tensor':
        """Create a tensor from numpy array.
        
        Parameters
        ----------
        data : np.ndarray
            Input parameter.
        dtype : Optional[TypeLike]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=data, dtype=dtype)
    # end def from_numpy

    @staticmethod
    def full(fill_value, shape: ShapeLike, dtype: Optional[TypeLike] = None) -> 'Tensor':
        """Create a tensor of full.

        Parameters
        ----------
        fill_value: Any
            Value to fill the tensor with.
        shape : ShapeLike
            Input parameter.
        dtype : Optional[TypeLike]
            Input parameter.

        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        shape = Shape.create(shape)
        np_dtype = _convert_dtype_to_numpy(dtype) if dtype else None
        return Tensor(data=np.full(shape, fill_value, dtype=np_dtype))
    # end def full

    @staticmethod
    def zeros(shape: ShapeLike, dtype: Optional[TypeLike] = None) -> 'Tensor':
        """Create a tensor of zeros.
        
        Parameters
        ----------
        shape : ShapeLike
            Input parameter.
        dtype : Optional[TypeLike]
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor.full(0, shape, dtype=dtype)
    # end def zeros

    @staticmethod
    def ones(shape: ShapeLike, dtype: Optional[TypeLike] = None) -> 'Tensor':
        """Create a tensor of ones.

        Parameters
        ----------
        shape : ShapeLike
            Input parameter.
        dtype : Optional[TypeLike]
            Input parameter.

        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor.full(1, shape, dtype=dtype)
    # end def ones

    @staticmethod
    def from_list(data: List, dtype: DType) -> 'Tensor':
        """Create a tensor from list.
        
        Parameters
        ----------
        data : List
            Input parameter.
        dtype : DType
            Input parameter.
        
        Returns
        -------
        'Tensor'
            Result of the operation.
        """
        return Tensor(data=data, dtype=dtype)
    # end def from_list

    # endregion STATIC

    # region NUMPY

    def __array__(self, dtype: Optional[TypeLike] = None) -> np.ndarray:
        """Convert the shape to a NumPy array.
        """
        return np.array(self._data, dtype=to_numpy(dtype if dtype is not None else self._dtype))
    # end def __array__

    def __array_function__(self, func, types, args, kwargs):
        """Return the result of applying the function to the inputs."""
        # As a list of array
        array = [
            x._data if isinstance(x, Tensor) else x
            for x in args
        ]

        return Tensor(data=func(*array, **kwargs))
    # end def __array_function__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Return the result of applying the ufunc to the inputs."""
        if method != "__call__":
            return NotImplemented
        # end if

        # As a list of array
        arrays = [
            x._data if isinstance(x, Tensor) else x
            for x in inputs
        ]

        return Tensor(data=ufunc(*arrays, **kwargs))
    # end def __array_ufunc__

    # endregion NUMPY

# end Tensor


#
# Helpers
#

# region _HELPERS


def _require_tensor(value: Any) -> Tensor:
    """Ensure the provided value is a Tensor instance."""
    if not isinstance(value, Tensor):
        raise TypeError("Tensor math functions expect Tensor inputs.")
    return value
# end def _require_tensor


def _call_tensor_method(name: str, tensor: Tensor, *args, **kwargs) -> Tensor:
    """Helper to invoke a Tensor math method by name."""
    method = getattr(_require_tensor(tensor), name)
    return method(*args, **kwargs)
# end def _call_tensor_method


def _concatenate_tensors(
        tensors: Sequence[Tensor],
        axis: Optional[int]
) -> Tensor:
    """Internal helper to concatenate tensors consistently."""
    tensors_tuple = tuple(_require_tensor(tensor) for tensor in tensors)
    if not tensors_tuple:
        raise ValueError("Concatenation requires at least one tensor.")
    arrays = [tensor.value for tensor in tensors_tuple]
    result = np.concatenate(arrays, axis=axis)
    return Tensor(data=np.asarray(result))
# end def _concatenate_tensors


def _as_numpy_operand(value: Union[Tensor, TensorLike, np.ndarray]) -> np.ndarray:
    """Convert Tensor or array-like input to a NumPy array."""
    if isinstance(value, Tensor):
        return value.value
    # end if
    return np.asarray(value)
# end def _as_numpy_operand


# endregion _HELPERS


#
# Shapes
#


def ts_scalar() -> Shape:
    """Return the shape of a scalar."""
    return Shape([])
# end def t_scalar


def ts_matrix(rows: int, columns: int) -> Shape:
    """Create a matrix shape."""
    return Shape.matrix(rows, columns)
# end def t_matrix


def ts_vector(size: int) -> Shape:
    """Create a vector shape."""
    return Shape.vector(size)
# end def t_vector


#
# Build
#


def tensor_from_numpy(data: np.ndarray, dtype: Optional[TypeLike] = None) -> Tensor:
    """
    Create a tensor from numpy array.

    Parameters
    ----------
    data : np.ndarray
        Input parameter.
    dtype : Optional[TypeLike]
        Input parameter.

    Returns
    -------
    'Tensor'
        Result of the operation.
    """
    return Tensor.from_numpy(data, dtype=dtype)
# end def tensor_from_numpy


def t_zeros(shape: ShapeLike, dtype: Optional[TypeLike] = None) -> Tensor:
    """
    Create a tensor of zeros.

    Parameters
    ----------
    shape : ShapeLike
        Input parameter.
    dtype : Optional[TypeLike]
        Input parameter.

    Returns
    -------
    'Tensor'
        Result of the operation.
    """
    return Tensor.zeros(shape, dtype=dtype)
# end def t_zeros


def t_full(fill_value, shape: ShapeLike, dtype: Optional[TypeLike] = None) -> Tensor:
    """Create a tensor of full.

    Parameters
    ----------
    fill_value: Any
        Value to fill the tensor with.
    shape : ShapeLike
        Input parameter.
    dtype : Optional[TypeLike]
        Input parameter.

    Returns
    -------
    'Tensor'
        Result of the operation.
    """
    return Tensor.full(fill_value, shape, dtype=dtype)
# end def t_full


def t_ones(shape: ShapeLike, dtype: Optional[TypeLike] = None) -> Tensor:
    """Create a tensor of ones.

    Parameters
    ----------
    shape : ShapeLike
        Input parameter.
    dtype : Optional[TypeLike]
        Input parameter.

    Returns
    -------
    'Tensor'
        Result of the operation.
    """
    return Tensor.ones(shape, dtype=dtype)
# end def t_ones


def t_from_list(data: List, dtype: DType) -> Tensor:
    """Create a tensor from list.

    Parameters
    ----------
    data : List
        Input parameter.
    dtype : DType
        Input parameter.

    Returns
    -------
    'Tensor'
        Result of the operation.
    """
    return Tensor.from_list(data, dtype=dtype)
# end def t_from_list


#
# Base
#

# region BASE


def t_pow(tensor: Tensor, exponent: Union['Tensor', TensorLike, np.ndarray]) -> Tensor:
    return _call_tensor_method("pow", tensor, exponent)
# end def pow


def t_square(tensor: Tensor) -> Tensor:
    return _call_tensor_method("square", tensor)
# end def square


def t_sqrt(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sqrt", tensor)
# end def sqrt


def t_cbrt(tensor: Tensor) -> Tensor:
    return _call_tensor_method("cbrt", tensor)
# end def cbrt


def t_reciprocal(tensor: Tensor) -> Tensor:
    return _call_tensor_method("reciprocal", tensor)
# end def reciprocal


def t_exp(tensor: Tensor) -> Tensor:
    return _call_tensor_method("exp", tensor)
# end def exp


def t_exp2(tensor: Tensor) -> Tensor:
    return _call_tensor_method("exp2", tensor)
# end def exp2


def t_expm1(tensor: Tensor) -> Tensor:
    return _call_tensor_method("expm1", tensor)
# end def expm1


def t_log(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log", tensor)
# end def log


def t_log2(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log2", tensor)
# end def log2


def t_log10(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log10", tensor)
# end def log10


def t_log1p(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log1p", tensor)
# end def log1p


def t_absolute(tensor: Tensor) -> Tensor:
    return _call_tensor_method("absolute", tensor)
# end def absolute


def t_abs(tensor: Tensor) -> Tensor:
    return _call_tensor_method("abs", tensor)
# end def abs


# endregion BASE


#
# Trigo
#

# region TRIGO


def t_sin(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sin", tensor)
# end def sin


def t_cos(tensor: Tensor) -> Tensor:
    return _call_tensor_method("cos", tensor)
# end def cos


def t_tan(tensor: Tensor) -> Tensor:
    return _call_tensor_method("tan", tensor)
# end def tan


def t_arcsin(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arcsin", tensor)
# end def arcsin


def t_arccos(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arccos", tensor)
# end def arccos


def t_arctan(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arctan", tensor)
# end def arctan


def t_atan2(
        tensor_y: Tensor,
        tensor_x: Union[Tensor, TensorLike, np.ndarray]
) -> Tensor:
    return tensor_y.arctan2(tensor_x)
# end def atan2


def t_sinh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sinh", tensor)
# end def sinh


def t_cosh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("cosh", tensor)
# end def cosh


def t_tanh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("tanh", tensor)
# end def tanh


def t_arcsinh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arcsinh", tensor)
# end def arcsinh


def t_arccosh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arccosh", tensor)
# end def arccosh


def t_arctanh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arctanh", tensor)
# end def arctanh

def t_deg2rad(tensor: Tensor) -> Tensor:
    return _call_tensor_method("deg2rad", tensor)
# end def deg2rad


def t_rad2deg(tensor: Tensor) -> Tensor:
    return _call_tensor_method("rad2deg", tensor)
# end def rad2deg

# endregion TRIGO


#
# Discrete
#

# region DISCRETE


def t_sign(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sign", tensor)
# end def sign


def t_floor(tensor: Tensor) -> Tensor:
    return _call_tensor_method("floor", tensor)
# end def floor


def t_ceil(tensor: Tensor) -> Tensor:
    return _call_tensor_method("ceil", tensor)
# end def ceil


def t_trunc(tensor: Tensor) -> Tensor:
    return _call_tensor_method("trunc", tensor)
# end def trunc


def t_rint(tensor: Tensor) -> Tensor:
    return _call_tensor_method("rint", tensor)
# end def rint


def t_round(tensor: Tensor, decimals: int = 0) -> Tensor:
    return _call_tensor_method("round", tensor, decimals=decimals)
# end def round


def t_clip(
        tensor: Tensor,
        min_value: Optional[Union['Tensor', TensorLike, np.ndarray]] = None,
        max_value: Optional[Union['Tensor', TensorLike, np.ndarray]] = None
) -> Tensor:
    return _call_tensor_method("clip", tensor, min_value=min_value, max_value=max_value)
# end def clip


# endregion DISCRETE


#
# Linear Algebra
#

# region LINEAR_ALGEBRA

def t_einsum(
        subscripts: str,
        *operands: Union['Tensor', TensorLike, np.ndarray],
        out: Optional[Union['Tensor', np.ndarray]] = None,
        **kwargs
) -> Tensor:
    """Tensor-based wrapper around numpy.einsum."""
    np_operands = [_as_numpy_operand(op) for op in operands]
    out_tensor: Optional[Tensor] = None
    out_array = None
    if out is not None:
        if isinstance(out, Tensor):
            out_tensor = out
            out_array = out.value
        else:
            out_array = np.asarray(out)
        # end if
    # end if
    result = np.einsum(subscripts, *np_operands, out=out_array, **kwargs)
    if out_tensor is not None:
        return out_tensor
    # end if
    return Tensor(data=np.asarray(result))
# end def einsum


def t_trace(tensor: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    return _call_tensor_method("trace", tensor, offset=offset, axis1=axis1, axis2=axis2)
# end def trace


def t_transpose(tensor: Tensor, axes: Optional[List[int]] = None) -> Tensor:
    return _call_tensor_method("transpose", tensor, axes=axes)
# end def transpose


def t_det(tensor: Tensor) -> Tensor:
    return _call_tensor_method("det", tensor)
# end def det


def t_inverse(tensor: Tensor) -> Tensor:
    return _call_tensor_method("inverse", tensor)
# end def inverse


def t_norm(tensor: Tensor, order: Union[int, float] = 2) -> Tensor:
    return _call_tensor_method("norm", tensor, ord=order)
# end def norm


# endregion LINEAR_ALGEBRA

#
# Reduction
#

# region REDUCTION


def t_mean(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("mean", tensor, axis=axis)
# end def mean


def t_std(
        tensor: Tensor,
        axis: Optional[int] = None,
        ddof: int = 0
) -> Tensor:
    return _call_tensor_method("std", tensor, axis=axis, ddof=ddof)
# end def std


def t_median(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("median", tensor, axis=axis)
# end def median


def t_q1(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("q1", tensor, axis=axis)
# end def q1


def t_q3(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("q3", tensor, axis=axis)
# end def q3


def t_max(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("max", tensor, axis=axis)
# end def max


def t_min(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("min", tensor, axis=axis)
# end def min


# endregion REDUCTION


#
# Reshape
#

# region SHAPE


def t_flatten(tensor: Tensor) -> Tensor:
    return _call_tensor_method("flatten", tensor)
# end def flatten


def t_concatenate(
        tensors: Sequence[Tensor],
        axis: Optional[int] = 0
) -> Tensor:
    """Concatenate multiple tensors using NumPy semantics."""
    return _concatenate_tensors(tensors, axis=axis)
# end def concatenate


def t_hstack(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors along axis 1."""
    return _concatenate_tensors(tensors, axis=1)
# end def hstack


def t_vstack(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors along axis 0."""
    return _concatenate_tensors(tensors, axis=0)
# end def vstack


# endregion SHAPE

#
# Boolean
#

# region BOOLEAN

def t_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("equal", tensor_a, tensor_b)
# end def equal


def t_not_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("not_equal", tensor_a, tensor_b)
# end def not_equal


def t_greater(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("greater", tensor_a, tensor_b)
# end def greater


def t_greater_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("greater_equal", tensor_a, tensor_b)
# end def greater_equal


def t_less(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("less", tensor_a, tensor_b)
# end def less


def t_less_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("less_equal", tensor_a, tensor_b)
# end def less_equal


def t_logical_not(tensor: Tensor) -> Tensor:
    return _call_tensor_method("logical_not", tensor)
# end def not


def t_any(tensor: Tensor) -> Tensor:
    return _call_tensor_method("any", tensor)
# end def any


def t_all(tensor: Tensor) -> Tensor:
    return _call_tensor_method("all", tensor)
# end def all


def t_logical_and(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_and", tensor_a, tensor_b)
# end def logical_and


def t_logical_or(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_or", tensor_a, tensor_b)
# end def logical_or


def t_logical_xor(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_xor", tensor_a, tensor_b)
# end def logical_xor


# endregion BOOLEAN
