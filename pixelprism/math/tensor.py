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
import builtins
from typing import Iterable, List, Union, Any, Optional, Tuple, Callable, Sequence, Literal
import numpy as np

from .shape import Shape, ShapeLike, DimInt
from .dtype import DType, TypeLike, to_numpy, convert_numpy, from_numpy
from .typing import TensorLike, ScalarListLike

__all__ = [
    "tensor",
    "scalar",
    "Tensor",
    "TensorLike",
    "scalar_shape",
    "vector_shape",
    "matrix_shape",
    "full",
    "normal",
    "uniform",
    "randint",
    "poisson",
    "bernoulli",
    "zeros",
    "ones",
    "concatenate",
    "hstack",
    "vstack",
    "pow",
    "square",
    "sqrt",
    "cbrt",
    "reciprocal",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    "absolute",
    "abs",
    "sign",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "round",
    "clip",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_not",
    "logical_and",
    "logical_or",
    "logical_xor",
    "any",
    "all",
    "eye_like",
    "einsum",
    "transpose",
    "inverse",
    "trace",
    "matmul",
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
    assert builtins.all(isinstance(dim, int) and dim >= 0 for dim in dims), "shape must be non-negative integers"
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


def tensor(
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
    >>> from pixelprism.math import T
    >>> logits = T.tensor([[1, 2], [3, 4]])
    >>> logits.shape
    (2, 2)
    """
    return Tensor(data=data, dtype=dtype, mutable=mutable)
# end def tensor


def scalar(
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
    >>> from pixelprism.math import T
    >>> bias = T.scalar(3.5)
    >>> bias
    array(3.5)
    """
    return _dim_tensor(data=value, ndim=0, dtype=dtype, mutable=mutable)
# end def scalar


def vector(
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
    >>> from pixelprism.math import T
    >>> weights = T.vector([0.2, 0.3, 0.5])
    >>> weights.shape
    (3,)
    """
    return _dim_tensor(data=value, ndim=1, dtype=dtype, mutable=mutable)
# end def vector


def matrix(
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
    >>> from pixelprism.math as T
    >>> mat = T.matrix([[1, 0], [0, 1]])
    >>> mat
    array([[1., 0.],
           [0., 1.]])
    """
    return _dim_tensor(data=value, ndim=2, dtype=dtype, mutable=mutable)
# end def matrix


def empty(
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
    'Tensor'
        Tensor backed by ``np.empty(shape, dtype)``.

    Examples
    --------
    >>> from pixelprism.math as T
    >>> scratch = T.empty((2, 3))
    >>> scratch.shape
    (2, 3)
    """
    dims = _normalize_shape(shape)
    data = np.empty(dims, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=True)
# end def empty


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
    'Tensor'
        Tensor filled with zeros of the given ``shape``.

    Examples
    --------
    >>> from pixelprism.math as T
    >>> zeros_tensor = T.zeros((2, 2))
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
    >>> from pixelprism.math import T
    >>> ones_tensor = T.ones(4)
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
    'Tensor'
        Tensor filled entirely with ``value``.

    Examples
    --------
    >>> from pixelprism.math import T
    >>> mask = T.full((2, 3), 7)
    >>> mask
    array([[7., 7., 7.],
           [7., 7., 7.]])
    """
    dims = _normalize_shape(shape)
    data = np.full(dims, value, dtype=_resolve_dtype(dtype))
    return Tensor(data=data, mutable=mutable)
# end def full


def normal(
        shape: ShapeLike,
        loc: float = 0.0,
        scale: float = 1.0,
        dtype: Optional[TypeLike] = None,
        mutable: bool = True
) -> 'Tensor':
    """Sample a tensor from a normal (Gaussian) distribution.

    Parameters
    ----------
    shape : ShapeLike
        Output tensor shape.
    loc : float, default 0.0
        Mean of the normal distribution.
    scale : float, default 1.0
        Standard deviation of the normal distribution. Must be non-negative.
    dtype : TypeLike, optional
        Optional target dtype used to cast the sampled values.
    mutable : bool, default True
        Whether the resulting tensor is mutable.

    Returns
    -------
    Tensor
        Tensor populated with normal-distributed samples.

    Raises
    ------
    ValueError
        If ``scale`` is negative.

    Examples
    --------
    >>> from pixelprism.math import normal
    >>> t = normal((2, 3), loc=1.0, scale=0.5)
    >>> t.shape.dims
    (2, 3)
    """
    if scale < 0:
        raise ValueError("scale must be non-negative.")
    # end if
    dims = _normalize_shape(shape)
    data = np.random.normal(loc=loc, scale=scale, size=dims)
    if dtype is not None:
        data = data.astype(_resolve_dtype(dtype))
    # end if
    return Tensor(data=data, mutable=mutable)
# end def normal


def uniform(
        shape: ShapeLike,
        low: float = 0.0,
        high: float = 1.0,
        dtype: Optional[TypeLike] = None,
        mutable: bool = True
) -> 'Tensor':
    """Sample a tensor from a uniform distribution on ``[low, high)``.

    Parameters
    ----------
    shape : ShapeLike
        Output tensor shape.
    low : float, default 0.0
        Lower bound of the sampling interval.
    high : float, default 1.0
        Upper bound of the sampling interval.
    dtype : TypeLike, optional
        Optional target dtype used to cast sampled values.
    mutable : bool, default True
        Whether the resulting tensor is mutable.

    Returns
    -------
    Tensor
        Tensor populated with uniformly distributed samples.

    Raises
    ------
    ValueError
        If ``high`` is not greater than ``low``.
    """
    if high <= low:
        raise ValueError("high must be strictly greater than low.")
    # end if
    dims = _normalize_shape(shape)
    data = np.random.uniform(low=low, high=high, size=dims)
    if dtype is not None:
        data = data.astype(_resolve_dtype(dtype))
    # end if
    return Tensor(data=data, mutable=mutable)
# end def uniform


def randint(
        shape: ShapeLike,
        low: int,
        high: Optional[int] = None,
        dtype: Optional[TypeLike] = None,
        mutable: bool = True
) -> 'Tensor':
    """Sample integer tensor values from ``[low, high)``.

    Parameters
    ----------
    shape : ShapeLike
        Output tensor shape.
    low : int
        If ``high`` is provided, lower bound (inclusive). Otherwise upper bound.
    high : int, optional
        Upper bound (exclusive). When ``None``, bounds are ``[0, low)``.
    dtype : TypeLike, optional
        Optional integer dtype cast.
    mutable : bool, default True
        Whether the resulting tensor is mutable.

    Returns
    -------
    Tensor
        Tensor populated with uniformly sampled integers.
    """
    dims = _normalize_shape(shape)
    data = np.random.randint(low=low, high=high, size=dims)
    if dtype is not None:
        data = data.astype(_resolve_dtype(dtype))
    # end if
    return Tensor(data=data, mutable=mutable)
# end def randint


def poisson(
        shape: ShapeLike,
        lam: float = 1.0,
        dtype: Optional[TypeLike] = None,
        mutable: bool = True
) -> 'Tensor':
    """Sample a tensor from a Poisson distribution.

    Parameters
    ----------
    shape : ShapeLike
        Output tensor shape.
    lam : float, default 1.0
        Expected number of events (lambda). Must be non-negative.
    dtype : TypeLike, optional
        Optional target dtype cast.
    mutable : bool, default True
        Whether the resulting tensor is mutable.

    Returns
    -------
    Tensor
        Tensor populated with Poisson-distributed counts.

    Raises
    ------
    ValueError
        If ``lam`` is negative.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative.")
    # end if
    dims = _normalize_shape(shape)
    data = np.random.poisson(lam=lam, size=dims)
    if dtype is not None:
        data = data.astype(_resolve_dtype(dtype))
    # end if
    return Tensor(data=data, mutable=mutable)
# end def poisson


def bernoulli(
        shape: ShapeLike,
        p: float = 0.5,
        dtype: TypeLike = DType.Z,
        mutable: bool = True
) -> 'Tensor':
    """Sample Bernoulli trials as a tensor of 0/1 values.

    Parameters
    ----------
    shape : ShapeLike
        Output tensor shape.
    p : float, default 0.5
        Success probability for each independent trial. Must be in ``[0, 1]``.
    dtype : TypeLike, default DType.Z
        Output dtype. Defaults to integer.
    mutable : bool, default True
        Whether the resulting tensor is mutable.

    Returns
    -------
    Tensor
        Tensor filled with 0/1 samples.

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be between 0 and 1 inclusive.")
    # end if
    dims = _normalize_shape(shape)
    data = np.random.binomial(n=1, p=p, size=dims).astype(_resolve_dtype(dtype))
    return Tensor(data=data, mutable=mutable)
# end def bernoulli


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
    >>> from pixelprism.math import T
    >>> missing = T.nan((2, 2))
    >>> np.isnan(missing).logical_all()
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
    >>> from pixelprism.math import T
    >>> eye = T.I(3)
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
    >>> from pixelprism.math import T
    >>> diag_tensor = T.diag([1, 2, 3])
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
    >>> from pixelprism.math import T
    >>> base = np.zeros((4, 4))
    >>> eye = T.eye_like(base)
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
    >>> zeros_clone.shape
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


class TensorShape:
    """
    Immutable descriptor for symbolic tensor shapes.

    A :class:`TensorShape` records the axis lengths associated with a tensor.
    Each axis can be a concrete ``int``. Shapes are lightweight, hashable, and
    safe to share across nodes, ensuring downstream passes can reason about
    tensor metadata without mutating the original objects.

    Validation and normalization
    ----------------------------
    The constructor eagerly validates every dimension through ``_check_dim`` to
    guarantee negative or otherwise malformed values never propagate past the
    creation site. Internally, axes are stored as an immutable tuple, giving
    deterministic hashing and printing behavior and keeping equality semantics
    simple.

    Convenience properties
    ----------------------
    ``dims`` exposes the canonical tuple representation, ``rank``/``n_dims``
    provide fast access to the tensor arity, and ``size`` returns the product
    of all known axes (``None`` when at least one axis is symbolic).

    Compatibility helpers
    ---------------------
    Many operators need to verify that their operands can participate in
    elementwise arithmetic. ``is_elementwise_compatible`` performs the check by
    comparing ranks and allowing either matching integers. ``merge_elementwise``
    builds on top of that by returning a new :class:`TensorShape` where each
    axis is the tightened version of both operands, raising :class:`ValueError`
    when a conflict is detected. Having these utilities on the shape class
    keeps validation logic centralized and consistent across operators.

    Subclassing
    -----------
    ``TensorShape`` is intentionally concrete. Higher-level abstractions should
    wrap it rather than subclassing to avoid diverging validation paths. Should
    additional metadata (like layout or batching semantics) be required, they
    can be attached via separate objects keyed by ``TensorShape`` instances.
    """

    def __init__(self, dims: Sequence[DimInt]):
        """Initialize a TensorShape.

        Parameters
        ----------
        dims : Iterable[TensorDim]
            Iterable of dimension sizes to store in the shape.
        """
        dims_tuple = tuple(dims)
        for dim in dims_tuple:
            self._check_dim(dim)
        # end for
        self._dims: Sequence[DimInt] = dims_tuple
    # end def __init__

    # region PROPERTIES

    @property
    def dims(self) -> Sequence[DimInt]:
        """Return the dimensions' tuple.

        Returns
        -------
        TensorDims
            Tuple describing tensor dimensions.
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

    # endregion PROPERTIES

    # region PUBLIC

    def transpose(self, axes: Optional[List[int]] = None) -> "TensorShape":
        """Return the shape with axes permuted.

        Parameters
        ----------
        axes : list[int], optional
            Axis permutation. When ``None``, the axis order is reversed.

        Returns
        -------
        TensorShape
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
        return TensorShape(dims=tuple(new_shape))
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

    def drop_axis(self, axis: int) -> "TensorShape":
        """Return a new shape with the specified axis removed.

        Parameters
        ----------
        axis : int
            Axis index to drop.

        Returns
        -------
        TensorShape
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
            return TensorShape(self._dims[:axis])
        elif axis == 0:
            return TensorShape(self._dims[1:])
        else:
            return TensorShape(self._dims[:axis] + self._dims[axis + 1 :])
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

    def insert_axis(self, axis: int, size: DimInt) -> "TensorShape":
        """Return a new shape with the specified axis inserted.

        Parameters
        ----------
        axis : int
            Axis index to insert.
        size : Dim
            Size of the inserted axis.

        Returns
        -------
        TensorShape
            New shape with the axis inserted.

        Raises
        ------
        ValueError
            If the axis is out of bounds.
        """
        if axis < 0 or axis > self.rank:
            raise ValueError(f"Axis {axis} out of bounds for rank {self.rank}.")
        # end if
        return TensorShape(self._dims[:axis] + (size,) + self._dims[axis:])
    # end def insert_axis

    def insert_axis_(self, axis: int, size: DimInt) -> None:
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
        self._dims = self.insert_axis(axis, size)._dims
    # end def insert_axis_

    def as_tuple(self) -> Sequence[DimInt]:
        """Return the shape as a tuple.

        Returns
        -------
        tuple[Dim, ...]
            The shape as a tuple of dimensions.
        """
        return tuple([d for d in self._dims])
    # end def as_tuple

    def is_elementwise_compatible(self, other: "TensorShape") -> bool:
        """Check whether elementwise operations are allowed.

        Parameters
        ----------
        other : TensorShape
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

    def merge_elementwise(self, other: "TensorShape") -> "TensorShape":
        """Return the merged shape for elementwise operations.

        Parameters
        ----------
        other : TensorShape
            Shape to merge.

        Returns
        -------
        TensorShape
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
        return TensorShape(merged)
    # end def merge_elementwise

    def matmul_result(self, other: "TensorShape") -> "TensorShape":
        """Return the result shape of a matrix multiplication.

        Parameters
        ----------
        other : TensorShape
            Right-hand operand shape.

        Returns
        -------
        TensorShape
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
        batch_dims: List[DimInt] = []
        for idx in range(batch_rank):
            batch_dims.append(self._merge_dims(self._dims[idx], other._dims[idx]))
        # end for
        left_inner = self._dims[-1]
        right_inner = other._dims[-2]
        if not self._dims_equal(left_inner, right_inner):
            raise ValueError("Inner dimensions do not match for MatMul.")
        # end if
        result = tuple(batch_dims) + (self._dims[-2], other._dims[-1])
        return TensorShape(result)
    # end def matmul_result

    def concat_result(self, other: "TensorShape", axis: int) -> "TensorShape":
        """Return the result shape of concatenation along an axis.

        Parameters
        ----------
        other : TensorShape
            Shape to concatenate with.
        axis : int
            Concatenation axis.

        Returns
        -------
        TensorShape
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
        dims: List = []
        for idx, (dim_a, dim_b) in enumerate(zip(self._dims, other._dims)):
            if idx == axis_norm:
                dims.append(self._sum_dims(dim_a, dim_b))
            else:
                dims.append(self._merge_dims(dim_a, dim_b))
            # end if
        # end for
        return TensorShape(tuple(dims))
    # end def concat_result

    def can_reshape(self, new_shape: "TensorShape") -> bool:
        """Check whether reshape is symbolically valid.

        Parameters
        ----------
        new_shape : TensorShape
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

    def reshape(self, new_shape: "TensorShape") -> "TensorShape":
        """Return the symbolic shape after reshape.

        Parameters
        ----------
        new_shape : TensorShape
            Target shape.

        Returns
        -------
        TensorShape
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

    def reshape_(self, new_shape: "TensorShape") -> None:
        """
        Reshape the shape in-place.

        Parameters
        ----------
        new_shape : TensorShape
            Target shape.

        Returns
        -------
        None
            This operation updates the instance in-place.
        """
        self._dims = self.reshape(new_shape)._dims
    # end def reshape_

    def equal_or_broadcastable(self, other: "TensorShape") -> bool:
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
        for dim in self._dims:
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
    def create(shape: ShapeLike) -> "TensorShape":
        """Create a shape from a tuple, sequence, or compatible object.

        Parameters
        ----------
        shape : ShapeLike
            Shape input (tuple, sequence, scalar dimension, or Shape).

        Returns
        -------
        TensorShape
            Normalized TensorShape instance.

        Raises
        ------
        TypeError
            If the input type is unsupported.
        """
        if isinstance(shape, tuple):
            return TensorShape(shape)
        elif isinstance(shape, Sequence):
            return TensorShape(shape)
        elif isinstance(shape, int):
            return TensorShape((shape,))
        elif isinstance(shape, TensorShape):
            return shape.copy()
        elif hasattr(shape, "dims"):
            return TensorShape(getattr(shape, "dims"))
        else:
            raise TypeError(f"Unsupported shape type: {type(shape)}")
        # end if
    # end def create

    @staticmethod
    def scalar() -> "TensorShape":
        """Return a scalar (rank-0) shape.

        Returns
        -------
        TensorShape
            Shape with no dimensions.
        """
        return TensorShape(())
    # end def scalar

    @staticmethod
    def vector(n: DimInt) -> "TensorShape":
        """Return a vector shape.

        Parameters
        ----------
        n : Dim
            Length of the vector.

        Returns
        -------
        TensorShape
            Shape with a single dimension of size ``n``.
        """
        return TensorShape((n,))
    # end def vector

    @staticmethod
    def matrix(n: DimInt, m: DimInt) -> "TensorShape":
        """Return a matrix shape.

        Parameters
        ----------
        n : Dim
            Row count.
        m : Dim
            Column count.

        Returns
        -------
        TensorShape
            Shape with two dimensions ``(n, m)``.
        """
        return TensorShape((n, m))
    # end def matrix

    @staticmethod
    def stack_shape(shapes: Sequence["TensorShape"], axis: int) -> "TensorShape":
        """Return the result shape of stacking tensors.

        Parameters
        ----------
        shapes : Sequence[TensorShape]
            Shapes of tensors to stack. All shapes must be elementwise
            compatible.
        axis : int
            Axis index for the new dimension.

        Returns
        -------
        TensorShape
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
        axis_norm = TensorShape._normalize_axis(axis, base.rank, allow_new_axis=True)
        for shape in shapes[1:]:
            base = base.merge_elementwise(shape)
        # end for
        dims = list(base.dims)
        dims.insert(axis_norm, len(shapes))
        return TensorShape(tuple(dims))
    # end def stack_shape

    def copy(self):
        """Return a copy of the shape.

        Returns
        -------
        TensorShape
            Copy of the current shape.
        """
        return TensorShape(self._dims)
    # end def copy

    @staticmethod
    def _check_dim(dim: DimInt) -> None:
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
            raise ValueError("Shape dimensions cannot be None.")
        # end if
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("Shape dimensions must be non-negative integers or None.")
        # end if
    # end def _check_dim

    @staticmethod
    def _dims_equal(dim_a: DimInt, dim_b: DimInt) -> bool:
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
    def _merge_dims(dim_a: DimInt, dim_b: DimInt) -> DimInt:
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
    def _sum_dims(dim_a: DimInt, dim_b: DimInt) -> DimInt:
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

    def __contains__(self, dim: DimInt) -> bool:
        """Check whether a dimension is present in the shape.
        """
        return dim in self._dims
    # end def __contains__

    def __bool__(self) -> bool:
        """Return whether the shape is non-empty."""
        return bool(self._dims)
    # end def __bool_

    def __getitem__(self, index: int) -> DimInt:
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
        if isinstance(other, TensorShape):
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

    def __str__(self) -> str:
        """Return the str() form of the shape.

        Returns
        -------
        str
            Readable representation.
        """
        return f"{self._dims}"
    # end def __str__

    # endregion OVERRIDE

    # region NUMPY

    def __array__(self, dtype: Optional[TypeLike] = None) -> np.ndarray:
        """Convert the shape to a NumPy array.
        """
        return np.array(self._dims, dtype=to_numpy(dtype if dtype is not None else np.int32))
    # end def __array__

    # endregion NUMPY

# end class TensorShape


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
        self._shape = TensorShape(dims=self._data.shape)
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
    def shape(self) -> TensorShape:
        """Get the shape of the tensor.
        
        Returns
        -------
        TensorShape
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
        shape = TensorShape.create(shape)
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

    def tolist(self) -> ScalarListLike:
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
        return concatenate(tensors, axis=axis)
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
        return hstack(tensors)
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
        return vstack(tensors)
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
    def normal(
            shape: ShapeLike,
            loc: float = 0.0,
            scale: float = 1.0,
            dtype: Optional[TypeLike] = None,
            mutable: bool = True
    ) -> 'Tensor':
        """Sample a tensor from a normal (Gaussian) distribution.

        Parameters
        ----------
        shape : ShapeLike
            Output tensor shape.
        loc : float, default 0.0
            Mean of the distribution.
        scale : float, default 1.0
            Standard deviation of the distribution.
        dtype : Optional[TypeLike], optional
            Optional output dtype cast.
        mutable : bool, default True
            Whether the resulting tensor is mutable.

        Returns
        -------
        Tensor
            Sampled tensor.
        """
        return normal(shape=shape, loc=loc, scale=scale, dtype=dtype, mutable=mutable)
    # end def normal

    @staticmethod
    def uniform(
            shape: ShapeLike,
            low: float = 0.0,
            high: float = 1.0,
            dtype: Optional[TypeLike] = None,
            mutable: bool = True
    ) -> 'Tensor':
        """Sample a tensor from a uniform distribution on ``[low, high)``.

        Parameters
        ----------
        shape : ShapeLike
            Output tensor shape.
        low : float, default 0.0
            Lower bound of the interval.
        high : float, default 1.0
            Upper bound of the interval.
        dtype : Optional[TypeLike], optional
            Optional output dtype cast.
        mutable : bool, default True
            Whether the resulting tensor is mutable.

        Returns
        -------
        Tensor
            Sampled tensor.
        """
        return uniform(shape=shape, low=low, high=high, dtype=dtype, mutable=mutable)
    # end def uniform

    @staticmethod
    def randint(
            shape: ShapeLike,
            low: int,
            high: Optional[int] = None,
            dtype: Optional[TypeLike] = None,
            mutable: bool = True
    ) -> 'Tensor':
        """Sample integer tensor values from ``[low, high)``.

        Parameters
        ----------
        shape : ShapeLike
            Output tensor shape.
        low : int
            Lower bound (inclusive), or upper bound when ``high`` is ``None``.
        high : Optional[int], optional
            Upper bound (exclusive).
        dtype : Optional[TypeLike], optional
            Optional output dtype cast.
        mutable : bool, default True
            Whether the resulting tensor is mutable.

        Returns
        -------
        Tensor
            Sampled tensor.
        """
        return randint(shape=shape, low=low, high=high, dtype=dtype, mutable=mutable)
    # end def randint

    @staticmethod
    def poisson(
            shape: ShapeLike,
            lam: float = 1.0,
            dtype: Optional[TypeLike] = None,
            mutable: bool = True
    ) -> 'Tensor':
        """Sample a tensor from a Poisson distribution.

        Parameters
        ----------
        shape : ShapeLike
            Output tensor shape.
        lam : float, default 1.0
            Expected number of events.
        dtype : Optional[TypeLike], optional
            Optional output dtype cast.
        mutable : bool, default True
            Whether the resulting tensor is mutable.

        Returns
        -------
        Tensor
            Sampled tensor.
        """
        return poisson(shape=shape, lam=lam, dtype=dtype, mutable=mutable)
    # end def poisson

    @staticmethod
    def bernoulli(
            shape: ShapeLike,
            p: float = 0.5,
            dtype: TypeLike = DType.Z,
            mutable: bool = True
    ) -> 'Tensor':
        """Sample Bernoulli trials as a tensor of 0/1 values.

        Parameters
        ----------
        shape : ShapeLike
            Output tensor shape.
        p : float, default 0.5
            Success probability in ``[0, 1]``.
        dtype : TypeLike, default DType.Z
            Output dtype (integer by default).
        mutable : bool, default True
            Whether the resulting tensor is mutable.

        Returns
        -------
        Tensor
            Sampled tensor.
        """
        return bernoulli(shape=shape, p=p, dtype=dtype, mutable=mutable)
    # end def bernoulli

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


def scalar_shape() -> TensorShape:
    """Return the shape of a scalar."""
    return TensorShape([])
# end def t_scalar


def matrix_shape(rows: int, columns: int) -> TensorShape:
    """Create a matrix shape."""
    return TensorShape.matrix(rows, columns)
# end def t_matrix


def vector_shape(size: int) -> TensorShape:
    """Create a vector shape."""
    return TensorShape.vector(size)
# end def t_vector


#
# Build
#


def tensor_from_numpy(data: np.ndarray, dtype: Optional[TypeLike] = None) -> Tensor:
    """
    Create a tensor from a numpy array.

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


def from_list(data: List, dtype: DType) -> Tensor:
    """Create a tensor from a list.

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
# end def from_list


#
# Base
#

# region BASE


def pow(
        t: Tensor,
        exponent: Union['Tensor', TensorLike, np.ndarray]
) -> Tensor:
    return _call_tensor_method("pow", t, exponent)
# end def pow


def square(t: Tensor) -> Tensor:
    return _call_tensor_method("square", t)
# end def square


def sqrt(t: Tensor) -> Tensor:
    return _call_tensor_method("sqrt", t)
# end def sqrt


def cbrt(t: Tensor) -> Tensor:
    return _call_tensor_method("cbrt", t)
# end def cbrt


def reciprocal(t: Tensor) -> Tensor:
    return _call_tensor_method("reciprocal", t)
# end def reciprocal


def exp(t: Tensor) -> Tensor:
    return _call_tensor_method("exp", t)
# end def exp


def exp2(t: Tensor) -> Tensor:
    return _call_tensor_method("exp2", t)
# end def exp2


def expm1(t: Tensor) -> Tensor:
    return _call_tensor_method("expm1", t)
# end def expm1


def log(t: Tensor) -> Tensor:
    return _call_tensor_method("log", t)
# end def log


def log2(t: Tensor) -> Tensor:
    return _call_tensor_method("log2", t)
# end def log2


def log10(t: Tensor) -> Tensor:
    return _call_tensor_method("log10", t)
# end def log10


def log1p(t: Tensor) -> Tensor:
    return _call_tensor_method("log1p", t)
# end def log1p


def absolute(t: Tensor) -> Tensor:
    return _call_tensor_method("absolute", t)
# end def absolute


def abs(t: Tensor) -> Tensor:
    return _call_tensor_method("abs", t)
# end def abs


# endregion BASE


#
# Trigo
#

# region TRIGO


def sin(t: Tensor) -> Tensor:
    return _call_tensor_method("sin", t)
# end def sin


def cos(t: Tensor) -> Tensor:
    return _call_tensor_method("cos", t)
# end def cos


def tan(t: Tensor) -> Tensor:
    return _call_tensor_method("tan", t)
# end def tan


def arcsin(t: Tensor) -> Tensor:
    return _call_tensor_method("arcsin", t)
# end def arcsin


def arccos(t: Tensor) -> Tensor:
    return _call_tensor_method("arccos", t)
# end def arccos


def arctan(t: Tensor) -> Tensor:
    return _call_tensor_method("arctan", t)
# end def arctan


def atan2(
        tensor_y: Tensor,
        tensor_x: Union[Tensor, TensorLike, np.ndarray]
) -> Tensor:
    return tensor_y.arctan2(tensor_x)
# end def atan2


def sinh(t: Tensor) -> Tensor:
    return _call_tensor_method("sinh", t)
# end def sinh


def cosh(t: Tensor) -> Tensor:
    return _call_tensor_method("cosh", t)
# end def cosh


def tanh(t: Tensor) -> Tensor:
    return _call_tensor_method("tanh", t)
# end def tanh


def arcsinh(t: Tensor) -> Tensor:
    return _call_tensor_method("arcsinh", t)
# end def arcsinh


def arccosh(t: Tensor) -> Tensor:
    return _call_tensor_method("arccosh", t)
# end def arccosh


def arctanh(t: Tensor) -> Tensor:
    return _call_tensor_method("arctanh", t)
# end def arctanh

def deg2rad(t: Tensor) -> Tensor:
    return _call_tensor_method("deg2rad", t)
# end def deg2rad


def rad2deg(t: Tensor) -> Tensor:
    return _call_tensor_method("rad2deg", t)
# end def rad2deg

# endregion TRIGO


#
# Discrete
#

# region DISCRETE


def sign(t: Tensor) -> Tensor:
    return _call_tensor_method("sign", t)
# end def sign


def floor(t: Tensor) -> Tensor:
    return _call_tensor_method("floor", t)
# end def floor


def ceil(t: Tensor) -> Tensor:
    return _call_tensor_method("ceil", t)
# end def ceil


def trunc(t: Tensor) -> Tensor:
    return _call_tensor_method("trunc", t)
# end def trunc


def rint(t: Tensor) -> Tensor:
    return _call_tensor_method("rint", t)
# end def rint


def round(t: Tensor, decimals: int = 0) -> Tensor:
    return _call_tensor_method("round", t, decimals=decimals)
# end def round


def clip(
        t: Tensor,
        min_value: Optional[Union['Tensor', TensorLike, np.ndarray]] = None,
        max_value: Optional[Union['Tensor', TensorLike, np.ndarray]] = None
) -> Tensor:
    return _call_tensor_method("clip", t, min_value=min_value, max_value=max_value)
# end def clip


# endregion DISCRETE


#
# Linear Algebra
#

# region LINEAR_ALGEBRA

def einsum(
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


def matmul(t: Tensor, other: Union['Tensor', TensorLike, np.ndarray]) -> Tensor:
    """Matrix multiplication."""
    return _call_tensor_method("matmul", t, other)
# end def matmul


def trace(t: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    return _call_tensor_method("trace", t, offset=offset, axis1=axis1, axis2=axis2)
# end def trace


def transpose(t: Tensor, axes: Optional[List[int]] = None) -> Tensor:
    return _call_tensor_method("transpose", t, axes=axes)
# end def transpose


def det(t: Tensor) -> Tensor:
    return _call_tensor_method("det", t)
# end def det


def inverse(t: Tensor) -> Tensor:
    return _call_tensor_method("inverse", t)
# end def inverse


def norm(t: Tensor, order: Union[int, float] = 2) -> Tensor:
    return _call_tensor_method("norm", t, ord=order)
# end def norm


# endregion LINEAR_ALGEBRA

#
# Reduction
#

# region REDUCTION


def mean(
        t: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("mean", t, axis=axis)
# end def mean


def std(
        t: Tensor,
        axis: Optional[int] = None,
        ddof: int = 0
) -> Tensor:
    return _call_tensor_method("std", t, axis=axis, ddof=ddof)
# end def std


def median(
        t: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("median", t, axis=axis)
# end def median


def q1(
        t: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("q1", t, axis=axis)
# end def q1


def q3(
        t: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("q3", t, axis=axis)
# end def q3


def max(
        t: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("max", t, axis=axis)
# end def max


def min(
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


def flatten(t: Tensor) -> Tensor:
    return _call_tensor_method("flatten", t)
# end def flatten


def concatenate(
        tensors: Sequence[Tensor],
        axis: Optional[int] = 0
) -> Tensor:
    """Concatenate multiple tensors using NumPy semantics."""
    return _concatenate_tensors(tensors, axis=axis)
# end def concatenate


def hstack(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors along axis 1."""
    return _concatenate_tensors(tensors, axis=1)
# end def hstack


def vstack(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors along axis 0."""
    return _concatenate_tensors(tensors, axis=0)
# end def vstack


# endregion SHAPE

#
# Boolean
#

# region BOOLEAN

def equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("equal", tensor_a, tensor_b)
# end def equal


def not_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("not_equal", tensor_a, tensor_b)
# end def not_equal


def greater(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("greater", tensor_a, tensor_b)
# end def greater


def greater_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("greater_equal", tensor_a, tensor_b)
# end def greater_equal


def less(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("less", tensor_a, tensor_b)
# end def less


def less_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("less_equal", tensor_a, tensor_b)
# end def less_equal


def logical_not(tensor: Tensor) -> Tensor:
    return _call_tensor_method("logical_not", tensor)
# end def not


def any(tensor: Tensor) -> Tensor:
    return _call_tensor_method("any", tensor)
# end def any


def all(tensor: Tensor) -> Tensor:
    return _call_tensor_method("all", tensor)
# end def all


def logical_and(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_and", tensor_a, tensor_b)
# end def logical_and


def logical_or(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_or", tensor_a, tensor_b)
# end def logical_or


def logical_xor(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_xor", tensor_a, tensor_b)
# end def logical_xor


# endregion BOOLEAN
