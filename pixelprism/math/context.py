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
"""Evaluation contexts for symbolic math expressions.

This module defines :class:`Context`, a hierarchical container used to override
the values of tensor symbols while evaluating symbolic math expressions.
Contexts can be entered as context managers which temporarily push them onto a
process-wide stack. Helper functions operate on the currently active context to
create variables, override tensors, and query values without having to store a
reference to the context object explicitly.

Attributes
----------
root_context : Context
    Process-wide context instance that is eagerly constructed when the module
    loads so user code can start interacting with the stack immediately.

Functions
---------
context
    Return the currently active context (or create it if missing).
new_context
    Create a standalone context that can later be pushed to the stack.
push_context / pop_context
    Convenience helpers mirroring ``Context.__enter__`` and ``Context.__exit__``.
set_value / get_value / lookup
    Accessors that manage tensor overrides bound to the active context.
create_variable / remove_variable / remove_deep
    Helpers that expose a higher level vocabulary for managing overrides.
snapshot_context_stack / restore_context_stack
    Serialize or reload the active context stack for persistence.
dump_context_tree / trace_variable
    Diagnostics helpers for inspecting the context stack and lookup paths.

Notes
-----
Values pushed through :func:`set_value` and :meth:`Context.set` are normalized
with :func:`numpy.asarray` to ensure downstream code receives arrays compatible
with the active :class:`~pixelprism.math.dtype.DType`. All values flow through
:meth:`Context.set`, so the conversion is centralized regardless of which
public helper is used.

Examples
--------
>>> import numpy as np
>>> import pixelprism.math as pm
>>> with pm.new_context() as scope:
...     scope.set("w", np.zeros((2, 2)))
...     pm.set_value("b", np.ones((2,)))
...     pm.get_value("w").input_shape
(2, 2)
>>> pm.lookup("tensor:w") is None
True
"""


# Imports
from __future__ import annotations
from typing import Any, ClassVar, Dict, Optional, List
import numpy as np

from .dtype import DType, to_numpy
from .tensor import Tensor, TensorLike


__all__ = [
    "ContextError",
    "ContextStateError",
    "ContextAlreadyActiveError",
    "ContextNotActiveError",
    "ContextValueNotFoundError",
    "ContextInitializationError",
    "Context",
    "snapshot_context_stack",
    "restore_context_stack",
    "dump_context_tree",
    "trace_variable",
    "root_context",
    "context",
    "new_context",
    "push_context",
    "pop_context",
    "set_value",
    "get_value",
    "lookup",
    "root",
    "create_variable",
    "remove_variable",
    "remove_deep"
]


class ContextError(RuntimeError):
    """Base class for context-related errors."""
    pass
# end class ContextError


class ContextStateError(ContextError):
    """Error raised when a context is used in an invalid state."""
    pass
# end class ContextStateError


class ContextAlreadyActiveError(ContextStateError):
    """Raised when attempting to activate an already active context."""
    pass
# end class ContextAlreadyActiveError


class ContextNotActiveError(ContextStateError):
    """Raised when attempting to use a context that is not active."""
    pass
# end class ContextNotActiveError


class ContextValueNotFoundError(KeyError, ContextError):
    """Raised when an override lookup fails."""

    def __init__(self, name: str) -> None:
        message = f"No value bound for tensor {name!r}."
        super().__init__(message)
        self.name = name
    # end def __init__

# end class ContextValueNotFoundError


class ContextInitializationError(ContextError):
    """Raised when the root context cannot be initialized."""
    pass
# end class ContextInitializationError


class Context:
    """Hierarchical evaluation context for Tensor overrides.

    Parameters
    ----------
    parent : Context, optional
        Explicit parent context to inherit from. If omitted, the currently
        active context becomes the parent.

    Attributes
    ----------
    _parent : Context or None
        Parent context in the stack, or ``None`` for the root.
    _values : dict of str to numpy.ndarray
        Mapping of tensor identifiers to their override values.
    _previous_active : Context or None
        Context that was active before entering this one.
    _is_active : bool
        Flag indicating whether the context is currently entered.
    _root : ClassVar[Context or None]
        Singleton root context shared by the process.
    _current : ClassVar[Context or None]
        Context that is currently active.
    """

    __slots__ = ("_parent", "_values", "_previous_active", "_is_active")

    _root: ClassVar[Optional["Context"]] = None
    _current: ClassVar[Optional["Context"]] = None

    def __init__(self, parent: Optional["Context"] = None) -> None:
        """Create a context that optionally inherits from ``parent``.

        Parameters
        ----------
        parent : Context, optional
            Parent context to inherit stored tensor values from. Defaults to
            the context returned by :meth:`Context.current`.
        """
        self.initialize_root_context()
        if parent is None:
            parent = Context.current()
        # end if
        self._parent = parent
        self._values: Dict[str, Optional[Tensor]] = {}
        self._previous_active: Optional["Context"] = None
        self._is_active: bool = False
    # end def __init__

    # region PROPERTIES

    @property
    def parent(self) -> Optional["Context"]:
        """Return the parent context of this one, if any."""
        return self._parent
    # end def parent

    @property
    def is_active(self) -> bool:
        """Return ``True`` if this context is currently active."""
        return self._is_active
    # end def is_active

    # endregion PROPERTIES

    # region CLASS_ACCESSORS

    @classmethod
    def root(cls) -> "Context":
        """Return the singleton root context.

        Returns
        -------
        Context
            Root context that permanently exists in the process.
        """
        if cls._root is not None:
            return cls._root
        # end if
        cls.initialize_root_context()
        if cls._root is None:
            raise ContextInitializationError("Failed to initialize root context.")
        # end if
        return cls._root
    # end def root

    @classmethod
    def current(cls) -> "Context":
        """Return the context that is currently active.

        Returns
        -------
        Context
            Context instance that is active within the current thread.
        """
        if cls._current is not None:
            return cls._current
        # end if
        cls.initialize_root_context()
        if cls._current is None:
            raise ContextInitializationError(
                "Failed to resolve the current context after initialization."
            )
        # end if
        return cls._current
    # end def current

    # endregion CLASS_ACCESSORS

    # region CONTEXT_MANAGER

    def __enter__(self) -> "Context":
        """Activate the context as a context manager.

        Returns
        -------
        Context
            The context itself so it can be bound to ``as`` targets.

        Raises
        ------
        ContextAlreadyActiveError
            If the context is already active.
        """
        if self._is_active:
            raise ContextAlreadyActiveError("Context is already active.")
        # end if
        self._previous_active = Context.current()
        Context._current = self
        self._is_active = True
        return self
    # end def __enter__

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Deactivate the context and restore the previous one.

        Parameters
        ----------
        exc_type : type or None
            Exception type propagated by the ``with`` statement, if any.
        exc_val : BaseException or None
            Exception instance propagated by the ``with`` statement, if any.
        exc_tb : TracebackType or None
            Traceback information for the propagating exception.

        Raises
        ------
        ContextNotActiveError
            If the context is not active.
        """
        if not self._is_active:
            raise ContextNotActiveError("Context is not active.")
        # end if
        Context._current = self._previous_active or Context.root()
        self._previous_active = None
        self._is_active = False
    # end def __exit__

    # endregion CONTEXT_MANAGER

    # region VALUE_MANAGEMENT

    def lookup(self, name: str) -> Optional[Tensor]:
        """Retrieve an override without raising for missing names.

        Parameters
        ----------__float__
        name : str
            Identifier of the tensor to look up.

        Returns
        -------
        Tensor or None
            Override value if found in this context stack, otherwise ``None``.
        """
        if name in self._values:
            return self._values[name]
        # end if
        if self._parent is not None:
            return self._parent.lookup(name)
        # end if
        return None
    # end def lookup

    def get(self, name: str) -> Tensor:
        """Return an override and raise an error when it is missing.

        Parameters
        ----------
        name : str
            Identifier of the tensor to retrieve.

        Returns
        -------
        Tensor
            Override associated with ``name``.

        Raises
        ------
        ContextValueNotFoundError
            If the tensor is not bound in this context stack.
        """
        value = self.lookup(name)
        if value is None:
            raise ContextValueNotFoundError(name)
        # end if
        return value
    # end def get

    def set(self, name: str, value: Optional[TensorLike]) -> None:
        """Bind a tensor identifier to a value within this context.

        Parameters
        ----------
        name : str
            Identifier of the tensor to override.
        value : DataType or None
            Value to store. It is converted to a Tensor before storage.
        """
        self._values[name] = value
    # end def set

    def clear(self) -> None:
        """Clear all overrides in this context."""
        self._values.clear()
    # end def clear

    def items(self):
        """Return an iterator over the (name, value) pairs in this context."""
        merged = {}
        if self._parent is not None:
            merged.update(dict(self._parent.items()))
        # end if
        merged.update(self._values)
        return merged.items()
    # end def items

    def remove(self, name: str) -> None:
        """Remove an override from this context."""
        if name in self._values:
            self._values.pop(name)
        # end if
    # end def remove

    def remove_deep(self, name: str) -> None:
        """Remove an override from this context and all child contexts."""
        self.remove(name)
        if self._parent is not None:
            self._parent.remove_deep(name)
        # end if
    # end def remove_deep

    def _to_tensor(self, data: TensorLike) -> Tensor:
        """Convert a value to a Tensor.

        Parameters
        ----------
        data : DataType
            The value to convert.
        """


    # endregion VALUE_MANAGEMENT

    # region DIAGNOSTICS

    def trace(self, name: str) -> Dict[str, Any]:
        """Return metadata describing how ``name`` resolves in this stack."""
        chain = []
        node: Optional["Context"] = self
        depth = 0
        found_context: Optional["Context"] = None
        located_value: Optional[Tensor] = None
        while node is not None:
            has_value = name in node._values
            entry = {
                "depth": depth,
                "context_id": id(node),
                "is_active": node._is_active,
                "has_value": has_value,
                "value_metadata": (
                    self._describe_value(node._values[name]) if has_value else None
                ),
            }
            chain.append(entry)
            if has_value and found_context is None:
                found_context = node
                located_value = node._values[name]
            # end if
            node = node._parent
            depth += 1
        # end while
        return {
            "name": name,
            "found": found_context is not None,
            "context_id": id(found_context) if found_context is not None else None,
            "value": located_value,
            "value_metadata": (
                self._describe_value(located_value) if located_value is not None else None
            ),
            "chain": chain,
        }
    # end def trace

    def dump_tree(self, include_values: bool = True) -> str:
        """Return a string representation of this context and its parents."""
        lines = []
        node: Optional["Context"] = self
        depth = 0
        while node is not None:
            indent = "  " * depth
            status = "active" if node._is_active else "inactive"
            lines.append(f"{indent}- Context(id={id(node)}, {status})")
            if include_values:
                if node._values:
                    for key in sorted(node._values):
                        summary = self._describe_value(node._values[key])
                        fragments = ", ".join(f"{k}={v}" for k, v in summary.items())
                        lines.append(f"{indent}    {key}: {fragments}")
                else:
                    lines.append(f"{indent}    <empty>")
            # end if
            node = node._parent
            depth += 1
        # end while
        return "\n".join(lines)
    # end def dump_tree

    @staticmethod
    def _describe_value(value: Optional[Any]) -> Dict[str, Any]:
        """Return summary metadata for ``value``."""
        if value is None:
            return {"type": "None"}
        if isinstance(value, Tensor):
            summary: Dict[str, Any] = {
                "type": "Tensor",
                "dtype": getattr(value.dtype, "name", str(value.dtype)),
                "shape": tuple(value.shape.dims),
                "mutable": value.mutable,
            }
            return summary
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "dtype": str(value.dtype),
                "shape": tuple(value.shape),
            }
        return {"type": type(value).__name__, "repr": repr(value)}
    # end def _describe_value

    @staticmethod
    def _serialize_value(value: Optional[Any]) -> Dict[str, Any]:
        """Serialize a value stored in the context."""
        if value is None:
            return {"kind": "none"}
        if isinstance(value, Tensor):
            return {
                "kind": "tensor",
                "dtype": getattr(value.dtype, "name", None),
                "mutable": value.mutable,
                "data": value.value.tolist(),
            }
        if isinstance(value, np.ndarray):
            return {
                "kind": "ndarray",
                "dtype": str(value.dtype),
                "data": value.tolist(),
            }
        raise ContextError(f"Cannot serialize value of type {type(value)!r}.")
    # end def _serialize_value

    @staticmethod
    def _deserialize_value(payload: Any) -> Any:
        """Deserialize the payload produced by :meth:`_serialize_value`."""
        if payload is None:
            return None
        if not isinstance(payload, dict):
            if isinstance(payload, np.ndarray):
                return payload.copy()
            return payload
        kind = payload.get("kind")
        if kind == "none":
            return None
        if kind == "tensor":
            dtype_name = payload.get("dtype")
            dtype = DType[dtype_name] if dtype_name is not None else None
            array = np.array(payload.get("data"))
            return Tensor(
                data=array,
                dtype=dtype,
                mutable=bool(payload.get("mutable", True)),
            )
        if kind == "ndarray":
            dtype = payload.get("dtype")
            array = np.array(payload.get("data"))
            if dtype is not None:
                array = array.astype(np.dtype(dtype))
            return array
        raise ContextError(f"Cannot deserialize value payload: {payload!r}")
    # end def _deserialize_value

    # endregion DIAGNOSTICS

    # region SERIALIZATION

    def to_dict(self, include_parent: bool = True) -> Dict[str, Any]:
        """Serialize this context into a dictionary.

        Parameters
        ----------
        include_parent : bool, optional
            If ``True`` (default), recurse into the parent chain so the entire
            stack can be reconstructed from the resulting dictionary.

        Returns
        -------
        dict
            Serializable mapping containing the stored overrides, activation
            state, and optional parent information.
        """
        parent_state = None
        if include_parent and self._parent is not None:
            parent_state = self._parent.to_dict(include_parent=True)
        # end if
        values = {name: self._serialize_value(value) for name, value in self._values.items()}
        return {
            "values": values,
            "is_active": self._is_active,
            "parent": parent_state,
        }
    # end def to_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Context":
        """Deserialize a context (and optionally its parents) from ``data``."""
        parent_state = data.get("parent")
        parent = cls.from_dict(parent_state) if parent_state is not None else None
        restored = cls._allocate(parent)
        raw_values = data.get("values", {})
        for name, value in raw_values.items():
            restored._values[name] = cls._deserialize_value(value)
        restored._is_active = bool(data.get("is_active", False))
        restored._previous_active = parent if (restored._is_active and parent is not None) else None
        return restored
    # end def from_dict

    # endregion SERIALIZATION

    # region INTERNALS

    @classmethod
    def _allocate(cls, parent: Optional["Context"]) -> "Context":
        """Allocate a context instance without invoking ``__init__``."""
        ccontext = super().__new__(cls)
        ccontext._parent = parent
        ccontext._values = {}
        ccontext._previous_active = None
        ccontext._is_active = False
        return ccontext
    # end def _allocate

    @classmethod
    def initialize_root_context(cls) -> "Context":
        """Ensure that the process-wide root context exists.

        Returns
        -------
        Context
            Root context instance that must always exist.
        """
        if cls._root is None:
            try:
                c_root = cls._allocate(None)
            except Exception as exc:  # pragma: no cover - defensive
                raise ContextInitializationError(
                    "Failed to allocate the root context."
                ) from exc
            c_root._is_active = True
            cls._root = c_root
            cls._current = c_root
            return c_root
        # end if
        return cls._root
    # end initialize_root_context

    # endregion INTERNALS

# end class Context


def snapshot_context_stack(ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Return a serializable snapshot of the active context stack."""
    target = ctx if ctx is not None else Context.current()
    return {"current": target.to_dict(include_parent=True)}
# end def snapshot_context_stack


def restore_context_stack(state: Dict[str, Any]) -> Context:
    """Restore the context stack from :func:`snapshot_context_stack` output."""
    current_state = state.get("current")
    if current_state is None:
        raise ContextError("Snapshot missing 'current' context data.")
    # end if
    restored = Context.from_dict(current_state)
    active = _first_active_context(restored)
    if active is None:
        active = restored
        active._is_active = True
    # end if
    c_root = _walk_to_root(restored)
    c_root._is_active = True
    Context._root = c_root
    Context._current = active
    global root_context
    root_context = c_root
    return active
# end def restore_context_stack


def _walk_to_root(ctx: Context) -> Context:
    """Return the root ancestor for ``ctx``."""
    node = ctx
    while node._parent is not None:
        node = node._parent
    # end while
    return node
# end def _walk_to_root


def _first_active_context(ctx: Context) -> Optional[Context]:
    """Return the first active context when traversing up the parent chain."""
    node: Optional[Context] = ctx
    while node is not None:
        if node._is_active:
            return node
        # end if
        node = node._parent
    # end while
    return None
# end def _first_active_context


def dump_context_tree(ctx: Optional[Context] = None, *, include_values: bool = True) -> str:
    """Return :meth:`Context.dump_tree` for ``ctx`` or the current context."""
    target = ctx if ctx is not None else Context.current()
    return target.dump_tree(include_values=include_values)
# end def dump_context_tree


def trace_variable(name: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Return :meth:`Context.trace` for ``name`` from ``ctx`` or the current context."""
    target = ctx if ctx is not None else Context.current()
    return target.trace(name)
# end def trace_variable


def context() -> Context:
    """Create a new Context"""
    return Context().current()
# end def context


def root() -> Context:
    """Create a new Context"""
    return Context.root()
# end def root


def new_context() -> Context:
    """Create a new Context"""
    return Context()
# end def env


def push_context() -> Context:
    """Push a new context onto the stack"""
    c = new_context()
    c.__enter__()
    return c
# end def push_context


def pop_context(c: Context) -> None:
    """Pop a context from the stack"""
    c.__exit__(None, None, None)
# end def pop_context


def _get_list_of_list_first_element(lst: List[Any]) -> Optional[Any]:
    """Return the first element of the first list in ``lst`` or ``None`` if empty."""
    if len(lst) == 0:
        return None
    # end if

    first = lst[0]

    if isinstance(first, list):
        return _get_list_of_list_first_element(first)
    else:
        return first
    # end if
# end def _get_list_of_list_first_element


def set_value(name: str, value: Optional[TensorLike | Tensor]) -> None:
    """
    Set a value in the current context.

    Parameters
    ----------
    name: str
        Name of the variable to set.
    value: int, float, bool, Tensor, list, tuple, tensor, np.ndarray, None
        Value to set (or None if empty).
    """
    if isinstance(value, Tensor):
        value = value.copy(copy_data=False)
    elif isinstance(value, np.ndarray):
        value = Tensor.from_numpy(value)
    elif isinstance(value, int):
        value = Tensor.from_numpy(np.array(value, dtype=to_numpy(DType.Z)))
    elif isinstance(value, float):
        value = Tensor.from_numpy(np.array(value, dtype=to_numpy(DType.R)))
    elif isinstance(value, complex):
        value = Tensor.from_numpy(np.array(value, dtype=to_numpy(DType.C)))
    elif isinstance(value, bool):
        value = Tensor.from_numpy(np.array(value, dtype=np.bool_))
    elif isinstance(value, list | tuple):
        if len(value) == 0:
            value = Tensor.from_list([], dtype=DType.R)
        else:
            first = _get_list_of_list_first_element(value)
            if isinstance(first, int):
                value = Tensor.from_list(value, dtype=DType.Z)
            elif isinstance(first, float):
                value = Tensor.from_list(value, dtype=DType.R)
            elif isinstance(first, complex):
                value = Tensor.from_list(value, dtype=DType.C)
            else:
                raise TypeError(f"Cannot convert list of type {type(first)!r}.")
            # end if
        # end if
    else:
        raise TypeError(f"Cannot set value of type {type(value)!r}.")
    # end if

    context().set(name, value)
# end def set_value


def get_value(name: str) -> Tensor:
    """Get a value from the current context"""
    return context().get(name)
# end def get_value


def get_value_dtype(name: str) -> Optional[DType]:
    """Get the dtype of a value from the current context"""
    value = lookup(name)
    if value is None:
        return None
    else:
        return value.dtype
    # end if
# end def get_value_dtype


def lookup(name: str) -> Optional[Tensor]:
    """Lookup a value in the current context"""
    return context().lookup(name)
# end def lookup


def create_variable(name: str, value: Optional[Tensor] = None) -> None:
    """Create a new variable in the current context"""
    set_value(name, value)
# end def create_variable


def remove_variable(name: str) -> None:
    """Remove a variable from the current context"""
    context().remove(name)
# end if


def remove_deep(name: str) -> None:
    """Remove a variable from the current context and all child contexts"""
    context().remove_deep(name)
# end def remove_deep


# Ensure the root context exists as soon as the module is imported.
root_context: Context = Context.initialize_root_context()
"""Context: Module-level reference ensuring the root context exists eagerly."""
