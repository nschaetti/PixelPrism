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
from typing import Tuple, List, Union, Optional
import numpy as np
from .tensor import Tensor
from .dtype import DType, NumericType, AnyDType, DataType, ScalarType
from .shape import AnyShape, Shape



def _resolve_dtype(dtype):
    """Return a numpy dtype for helper constructors."""
    if isinstance(dtype, DType):
        return dtype.to_numpy()
    # end if
    return np.dtype(dtype)
# end def _resolve_dtype


def _normalize_shape(shape: AnyShape) -> Tuple[int, ...]:
    """Normalize user-provided shape inputs into a tuple of ints."""
    if isinstance(shape, int):
        dims = (shape,)
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


def _check_shape(
        data: np.ndarray,
        n_dims: int
):
    """Check that the shape of `data` matches the expected shape."""
    assert data.ndim == n_dims, f"shape mismatch: expected {n_dims}, got {data.shape}"
# end _check_shape


def _data_as_nparray(
        data: DataType,
        dtype: AnyDType
):
    try:
        return np.asarray(data, dtype=_resolve_dtype(dtype))
    except ValueError as e:
        raise ValueError(f"cannot convert input to NumPy array, invalid data: {e}") from e
    # end try
# end def _data_as_nparray


def tensor(
        name: str,
        data: DataType,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
    """
    Construct a tensor wrapper around an existing NumPy array.

    Parameters
    ----------
    name : str
        Human-readable identifier assigned to the tensor.
    data : numpy.ndarray
        Array buffer that will be wrapped without copying.
    dtype : AnyDType, default float
        Data type of the underlying array.
    mutable : bool, default True
        Whether subsequent operations may mutate the tensor in-place.

    Returns
    -------
    Tensor
        Tensor instance referencing ``data``.

    Examples
    --------
    >>> import numpy as np
    >>> from pixelprism.math import utils
    >>> logits = utils.tensor("logits", [[1, 2], [3, 4]])
    >>> logits.data.shape
    (2, 2)
    """
    data = _data_as_nparray(data, dtype=_resolve_dtype(dtype))
    _check_shape(data, n_dims=data.ndim)
    return Tensor(name=name, data=data, mutable=mutable)
# end def tensor


def _dim_tensor(
        name: str,
        data: DataType,
        ndim: int,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
    """Allocate a tensor with a single dimension."""
    data = _data_as_nparray(data, dtype=_resolve_dtype(dtype))
    _check_shape(data, n_dims=ndim)
    return Tensor(name=name, data=data, mutable=mutable)
# end def _dim_tensor


def scalar(
        name: str,
        value: ScalarType,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
    """
    Create a scalar tensor (zero-dimensional array) from a numeric value.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
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
    >>> import pixelprism.math as ppmath
    >>> bias = ppmath.scalar("bias", 3.5)
    >>> bias.data
    array(3.5)
    """
    return _dim_tensor(name=name, data=value, ndim=0, dtype=dtype, mutable=mutable)
# end def scalar


def vector(
        name: str,
        value: DataType,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
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
    >>> import pixelprism.math as ppmath
    >>> weights = ppmath.vector("weights", [0.2, 0.3, 0.5])
    >>> weights.data.shape
    (3,)
    """
    return _dim_tensor(name=name, data=value, ndim=1, dtype=dtype, mutable=mutable)
# end def vector


def matrix(
        name: str,
        value: DataType,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
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
    >>> import pixelprism.math as ppmath
    >>> mat = ppmath.matrix("transform", [[1, 0], [0, 1]])
    >>> mat.data
    array([[1., 0.],
           [0., 1.]])
    """
    return _dim_tensor(name=name, data=value, ndim=2, dtype=dtype, mutable=mutable)
# end def matrix


def empty(
        name: str,
        shape: AnyShape,
        dtype: AnyDType = float
) -> Tensor:
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
    >>> import pixelprism.math as ppmath
    >>> scratch = ppmath.empty("scratch", (2, 3))
    >>> scratch.data.shape
    (2, 3)
    """
    dims = _normalize_shape(shape)
    data = np.empty(dims, dtype=_resolve_dtype(dtype))
    return Tensor(name=name, data=data, mutable=True)
# end def empty


def zeros(
        name: str,
        shape: AnyShape,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
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
    >>> zeros_tensor = ppmath.zeros("seq", (2, 2))
    >>> zeros_tensor.data
    array([[0., 0.],
           [0., 0.]])
    """
    dims = _normalize_shape(shape)
    data = np.zeros(dims, dtype=_resolve_dtype(dtype))
    return Tensor(name=name, data=data, mutable=mutable)
# end def zeros


def ones(
        name: str,
        shape: AnyShape,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
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
    >>> ones_tensor = ppmath.ones("biases", 4)
    >>> ones_tensor.data
    array([1., 1., 1., 1.])
    """
    dims = _normalize_shape(shape)
    data = np.ones(dims, dtype=_resolve_dtype(dtype))
    return Tensor(name=name, data=data, mutable=mutable)
# end def ones


def full(
        name: str,
        shape: AnyShape,
        value,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
    """
    Allocate a tensor whose entries are filled with a constant value.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
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
    >>> mask = ppmath.full("mask", (2, 3), 7)
    >>> mask.data
    array([[7., 7., 7.],
           [7., 7., 7.]])
    """
    dims = _normalize_shape(shape)
    data = np.full(dims, value, dtype=_resolve_dtype(dtype))
    return Tensor(name=name, data=data, mutable=mutable)
# end def full


def nan(
        name: str,
        shape: AnyShape,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
    """
    Allocate a tensor filled with ``NaN`` sentinels.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
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
    >>> missing = ppmath.nan("missing", (2, 2))
    >>> np.isnan(missing.data).all()
    True
    """
    dims = _normalize_shape(shape)
    data = np.full(dims, np.nan, dtype=_resolve_dtype(dtype))
    return Tensor(name=name, data=data, mutable=mutable)
# end def nan


def I(
        name: str,
        n: int,
        dtype: AnyDType = float,
        mutable: bool = False
) -> Tensor:
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
    >>> import pixelprism.math as ppmath
    >>> eye = ppmath.I("eye", 3)
    >>> eye.data
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    assert isinstance(n, int) and n >= 0, "n must be a non-negative integer"
    data = np.eye(n, dtype=_resolve_dtype(dtype))
    return Tensor(name=name, data=data, mutable=mutable)
# end def I


def diag(
        name: str,
        v,
        dtype: AnyDType = float,
        mutable: bool = True
) -> Tensor:
    """
    Construct a square matrix using the provided diagonal values.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
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
    >>> diag_tensor = ppmath.diag("diag", [1, 2, 3])
    >>> diag_tensor.data
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    """
    diag_values = np.asarray(v, dtype=_resolve_dtype(dtype))
    assert diag_values.ndim == 1, "diag input must be 1-D"
    data = np.diag(diag_values)
    return Tensor(name=name, data=data, mutable=mutable)
# end def diag


def eye_like(
        name: str,
        x: Tensor | np.ndarray,
        dtype: AnyDType = None,
        mutable: bool = True
) -> Tensor:
    """
    Build an identity matrix tensor with the same shape (and optional dtype) as ``x``.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
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
    >>> import pixelprism.math as ppmath
    >>> base = np.zeros((4, 4))
    >>> eye = ppmath.eye_like("eye", base)
    >>> np.allclose(eye.data, np.eye(4))
    True
    """
    base = np.asarray(x)
    assert base.ndim == 2, "eye_like expects a 2-D input"
    rows, cols = base.shape
    assert rows == cols, "eye_like requires a square matrix input"
    dtype = _resolve_dtype(dtype) if dtype else base.dtype
    data = np.eye(rows, dtype=dtype)
    return Tensor(name=name, data=data, mutable=mutable)
# end def eye_like


def zeros_like(
        name: str,
        x: Tensor | np.ndarray,
        dtype: AnyDType = None,
        mutable: bool = True
) -> Tensor:
    """
    Allocate a zero tensor matching the shape of ``x``.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
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
    >>> zeros_clone = ppmath.zeros_like("zeros", template)
    >>> zeros_clone.data.shape
    (2, 3)
    """
    base = np.asarray(x)
    dtype = _resolve_dtype(dtype) if dtype else base.dtype
    data = np.zeros(base.shape, dtype=dtype)
    return Tensor(name=name, data=data, mutable=mutable)
# end def zeros_like


def ones_like(
        name: str,
        x: Tensor | np.ndarray,
        dtype: AnyDType = None,
        mutable: bool = True
) -> Tensor:
    """
    Allocate a tensor of ones sharing the shape of ``x``.

    Parameters
    ----------
    name : str
        Identifier assigned to the tensor.
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
    >>> ones_clone = ppmath.ones_like("ones", template)
    >>> ones_clone.data
    array([[1., 1.],
           [1., 1.]], dtype=float32)
    """
    base = np.asarray(x)
    dtype = _resolve_dtype(dtype) if dtype else base.dtype
    data = np.ones(base.shape, dtype=dtype)
    return Tensor(name=name, data=data, mutable=mutable)
# end def ones_like
