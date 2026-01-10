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
"""Evaluation contexts for symbolic math expressions."""


from __future__ import annotations
from typing import ClassVar, Dict, Optional
import numpy as np


__all__ = [
    "Context",
    "root_context"
]


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
        self._values: Dict[str, np.ndarray] = {}
        self._previous_active: Optional["Context"] = None
        self._is_active: bool = False
    # end def __init__

    # region CLASS_ACCESSORS

    @classmethod
    def root(cls) -> "Context":
        """Return the singleton root context.

        Returns
        -------
        Context
            Root context that permanently exists in the process.
        """
        cls.initialize_root_context()
        assert cls._root is not None
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
        cls.initialize_root_context()
        assert cls._current is not None
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
        RuntimeError
            If the context is already active.
        """
        if self._is_active:
            raise RuntimeError("Context is already active.")
        # end if
        self._previous_active = Context.current()
        Context._current = self
        self._is_active = True
        return self

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
        RuntimeError
            If the context is not active.
        """
        if not self._is_active:
            raise RuntimeError("Context is not active.")
        # end if
        Context._current = self._previous_active or Context.root()
        self._previous_active = None
        self._is_active = False
    # end def __exit__

    # endregion CONTEXT_MANAGER

    # region VALUE_MANAGEMENT

    def lookup(self, name: str) -> Optional[np.ndarray]:
        """Retrieve an override without raising for missing names.

        Parameters
        ----------
        name : str
            Identifier of the tensor to look up.

        Returns
        -------
        numpy.ndarray or None
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

    def get(self, name: str) -> np.ndarray:
        """Return an override and raise an error when it is missing.

        Parameters
        ----------
        name : str
            Identifier of the tensor to retrieve.

        Returns
        -------
        numpy.ndarray
            Override associated with ``name``.

        Raises
        ------
        KeyError
            If the tensor is not bound in this context stack.
        """
        value = self.lookup(name)
        if value is None:
            raise KeyError(f"No value bound for tensor {name!r}.")
        # end if
        return value
    # end def get

    def set(self, name: str, value: np.ndarray) -> None:
        """Bind a tensor identifier to a value within this context.

        Parameters
        ----------
        name : str
            Identifier of the tensor to override.
        value : numpy.ndarray
            Value to store. It is converted to a NumPy array before storage.
        """
        self._values[name] = np.asarray(value)
    # end def set

    # endregion VALUE_MANAGEMENT

    # region INTERNALS

    @classmethod
    def initialize_root_context(cls) -> "Context":
        """Ensure that the process-wide root context exists.

        Returns
        -------
        Context
            Root context instance that must always exist.
        """
        if cls._root is None:
            root = super().__new__(cls)
            root._parent = None
            root._values = {}
            root._previous_active = None
            root._is_active = True
            cls._root = root
            cls._current = root
            return root
        # end if
        return Context.root()
    # end initialize_root_context

    # endregion INTERNALS

# end class Context


# Ensure the root context exists as soon as the module is imported.
root_context: Context = Context.initialize_root_context()
"""Context: Module-level reference ensuring the root context exists eagerly."""
