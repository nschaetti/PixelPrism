"""Utility helpers used across the symbolic math core."""

from __future__ import annotations

from typing import Any, Iterator, List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .value import Value
# end if


def is_sequence_like(value: Any) -> bool:
    """Return True when an object supports len() and iteration.

    Args:
        value: Object to check.

    Returns:
        bool: True when the object is sequence-like.
    """
    return hasattr(value, "__iter__") and hasattr(value, "__len__")
# end def is_sequence_like


def as_sequence(value: Any) -> List[Any]:
    """Convert an iterable object to a list.

    Args:
        value: Iterable to convert.

    Returns:
        list[Any]: Converted sequence.

    Raises:
        TypeError: If the value is not iterable.
    """
    if isinstance(value, list):
        return value
    # end if
    if isinstance(value, tuple):
        return list(value)
    # end if
    if is_sequence_like(value):
        return list(value)
    # end if
    raise TypeError("Value is not sequence-like and cannot be expanded.")
# end def as_sequence


def infer_dims_from_data(data: Any) -> Tuple[int, ...]:
    """Infer concrete tensor dimensions from nested python data.

    Args:
        data: Input data.

    Returns:
        tuple[int, ...]: Concrete dimensions inferred from the value.
    """
    if not is_sequence_like(data):
        return ()
    # end if
    seq = as_sequence(data)
    if not seq:
        return (0,)
    # end if
    child_dims = infer_dims_from_data(seq[0])
    return (len(seq),) + child_dims
# end def infer_dims_from_data


def num_elements(dims: Sequence[int | None]) -> int | None:
    """Compute the product of symbolic dimensions when possible.

    Args:
        dims: Sequence of tensor dimensions.

    Returns:
        Optional[int]: Number of elements or None when unknown.
    """
    total = 1
    for dim in dims:
        if dim is None:
            return None
        # end if
        total *= dim
    # end for
    return total
# end def num_elements


def flatten_simple(data: Any) -> List[Any]:
    """Flatten nested python sequences without shape metadata.

    Args:
        data: Nested data.

    Returns:
        list[Any]: Flattened list of elements.
    """
    if not is_sequence_like(data):
        return [data]
    # end if
    flat: List[Any] = []
    for item in as_sequence(data):
        flat.extend(flatten_simple(item))
    # end for
    return flat
# end def flatten_simple


def build_from_flat(iterator: Iterator[Any], dims: Sequence[int]) -> Any:
    """Rebuild nested data from a flat iterator.

    Args:
        iterator: Iterator over flat data.
        dims: Concrete dimension sizes.

    Returns:
        Any: Nested data shaped according to dims.
    """
    if not dims:
        return next(iterator)
    # end if
    length = dims[0]
    built: List[Any] = []
    for _ in range(length):
        built.append(build_from_flat(iterator, dims[1:]))
    # end for
    return built
# end def build_from_flat


def reshape_python(data: Any, new_dims: Sequence[int | None]) -> Any:
    """Reshape python sequences without backend help.

    Args:
        data: Input nested data.
        new_dims: Target dimension sizes.

    Returns:
        Any: Reshaped nested data.

    Raises:
        RuntimeError: If the target size is unknown.
        ValueError: If element counts do not match.
    """
    if any(dim is None for dim in new_dims):
        raise RuntimeError("Cannot reshape with unknown dimensions without backend support.")
    # end if
    flat = flatten_simple(data)
    target = num_elements(new_dims)
    if target is None:
        raise RuntimeError("Reshape target size is unknown.")
    # end if
    if len(flat) != target:
        raise ValueError(
            f"Cannot reshape data of size {len(flat)} into target with {target} elements."
        )
    # end if
    iterator = iter(flat)
    concrete_dims = tuple(int(dim) for dim in new_dims)
    return build_from_flat(iterator, concrete_dims)
# end def reshape_python


def concat_python(values: Sequence[Any], axis: int) -> Any:
    """Concatenate nested python sequences along an axis.

    Args:
        values: Sequence of nested data to concatenate.
        axis: Axis index.

    Returns:
        Any: Concatenated data.
    """
    if axis == 0:
        result: List[Any] = []
        for value in values:
            result.extend(as_sequence(value))
        # end for
        return result
    # end if
    heads = [as_sequence(value) for value in values]
    if not heads:
        return []
    # end if
    length = len(heads[0])
    for head in heads:
        if len(head) != length:
            raise ValueError("Cannot concatenate sequences with mismatched lengths.")
        # end if
    # end for
    concatenated: List[Any] = []
    for idx in range(length):
        concatenated.append(concat_python([head[idx] for head in heads], axis - 1))
    # end for
    return concatenated
# end def concat_python


def stack_python(values: Sequence[Any], axis: int) -> Any:
    """Stack nested python sequences along a new axis.

    Args:
        values: Sequence of nested data to stack.
        axis: Axis index for the new dimension.

    Returns:
        Any: Stacked nested data.
    """
    if axis == 0:
        return [value for value in values]
    # end if
    heads = [as_sequence(value) for value in values]
    if not heads:
        return []
    # end if
    length = len(heads[0])
    for head in heads:
        if len(head) != length:
            raise ValueError("Cannot stack sequences with mismatched lengths.")
        # end if
    # end for
    stacked: List[Any] = []
    for idx in range(length):
        stacked.append(stack_python([head[idx] for head in heads], axis - 1))
    # end for
    return stacked
# end def stack_python


def unravel_index(index: int, dims: Sequence[int]) -> Tuple[int, ...]:
    """Convert a flat index into a coordinate tuple.

    Args:
        index: Flat index.
        dims: Dimension sizes.

    Returns:
        tuple[int, ...]: Coordinate tuple.
    """
    if not dims:
        return ()
    # end if
    coords: List[int] = []
    remainder = index
    for size in reversed(dims):
        if size == 0:
            coords.append(0)
            continue
        # end if
        coords.append(remainder % size)
        remainder //= size
    # end for
    return tuple(reversed(coords))
# end def unravel_index


def ravel_index(coords: Sequence[int], dims: Sequence[int]) -> int:
    """Convert a coordinate tuple into a flat index.

    Args:
        coords: Coordinate tuple.
        dims: Dimension sizes.

    Returns:
        int: Flat index.
    """
    flat = 0
    for coord, size in zip(coords, dims):
        flat *= size
        flat += coord
    # end for
    return flat
# end def ravel_index


def transpose_python(data: Any, perm: Sequence[int]) -> Any:
    """Transpose nested python data following a permutation.

    Args:
        data: Nested python data.
        perm: Axis permutation.

    Returns:
        Any: Transposed data.
    """
    dims = infer_dims_from_data(data)
    if not dims:
        return data
    # end if
    if any(dim == 0 for dim in dims):
        target = tuple(dims[idx] for idx in perm)
        return reshape_python([], target)
    # end if
    concrete = tuple(int(dim) for dim in dims)
    flat = flatten_simple(data)
    total = len(flat)
    permuted_dims = tuple(concrete[idx] for idx in perm)
    new_flat: List[Any] = [None] * total
    for index in range(total):
        coords = unravel_index(index, concrete)
        permuted = tuple(coords[idx] for idx in perm)
        target_index = ravel_index(permuted, permuted_dims)
        new_flat[target_index] = flat[index]
    # end for
    iterator = iter(new_flat)
    return build_from_flat(iterator, permuted_dims)
# end def transpose_python


def select_ops(values: Sequence["Value"]) -> Any | None:
    """Return the first backend helper registered in the given values.

    Args:
        values: Runtime values participating in an operation.

    Returns:
        Any | None: Backend helper when available.
    """
    for value in values:
        ops = getattr(value, "_ops", None)
        if ops is not None:
            return ops
        # end if
    # end for
    return None
# end def select_ops

