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
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, List
from enum import Enum, auto

# Local imports
from pixelprism.data import Event



class MathEvent(Event):
    """
    Represents an event for mathematical expressions.
    """
    pass
# end class MathEvent


class MathExpr(ABC):
    """
    Abstract base class for mathematical expressions.

    Example:
        def on_change_handler():
            print("Value changed!")

        expr = SomeMathExprSubclass(on_change=on_change_handler)
        expr.value = 42

    Args:
        on_change (Optional[Callable[[], None]]): Function to call when the value changes.
        readonly (bool): If True, the value cannot be changed.
    """

    # Expression type
    expr_type = None

    # Return type
    return_type = None

    def __init__(
        self,
        on_change: Optional[Callable[[], None]] = None,
        readonly: bool = False
    ) -> None:
        self._data_locked: bool = readonly

        # Set on change
        self._on_change: MathEvent = MathEvent()
        if on_change is not None:
            self.add_on_change_listener(on_change)
        # end if

        # Check expression type
        if self.__class__.expr_type is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define class variable 'expr_type'")
        # end if

        # Must set the return type
        if self.__class__.return_type is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define class variable 'return_type'")
        # end if
    # end __init__

    @property
    def value(self) -> Any:
        """
        Get the scalar value.

        Returns:
            Any: The current value.
        """
        return self._get()
    # end value getter

    @value.setter
    def value(self, value: Any) -> None:
        """
        Set the scalar value.

        Args:
            value (Any): The value to set.

        Raises:
            RuntimeError: If the object is read-only.
        """
        self._check_locked()
        if isinstance(value, MathExpr):
            value = value._get()
        # end if
        self._set(value)
    # end value setter

    @property
    def on_change(self) -> MathEvent:
        """
        Get the on change event.

        Returns:
            MathEvent: The event object.
        """
        return self._on_change
    # end on_change

    @abstractmethod
    def copy(self) -> 'MathExpr':
        """
        Return a copy of the data.

        Returns:
            MathExpr: A copy of the current object.
        """
        pass
    # end copy

    def add_on_change_listener(self, listener: Callable[[], None]) -> None:
        """
        Add a listener to the on change event.

        Args:
            listener (Callable[[], None]): Listener function.

        Raises:
            TypeError: If listener is not callable.
        """
        if not callable(listener):
            raise TypeError("Listener must be callable")
        # end if
        self._on_change += listener
    # end add_on_change_listener

    def unregister_event(self, listener: Callable[[], None]) -> None:
        """
        Remove a listener from the on change event.

        Args:
            listener (Callable[[], None]): Listener function to remove.

        Raises:
            TypeError: If listener is not callable.
        """
        if not callable(listener):
            raise TypeError("Listener must be callable")
        # end if
        self._on_change -= listener
    # end unregister_event

    @abstractmethod
    def to_list(self) -> List[Any]:
        """
        Convert the scalar to a list.

        Returns:
            List[Any]: The value as a list.
        """
        pass
    # end to_list

    def _check_locked(self) -> None:
        """
        Check if the data is locked.

        Raises:
            RuntimeError: If the data is locked.
        """
        if self._data_locked:
            raise RuntimeError(f"{self.__class__.__name__} is read only !")
        # end if
    # end _check_locked

    @abstractmethod
    def _set(self, value: Any) -> None:
        """
        Set the scalar value.

        Args:
            value (Any): Value to set.
        """
        pass
    # end _set

    @abstractmethod
    def _get(self) -> Any:
        """
        Get the scalar value.

        Returns:
            Any: The current value.
        """
        pass
    # end _get

# end class MathExpr


# Node Mathematical Expression
class NodeMathExpr(MathExpr, ABC):
    """
    Node Mathematical Expression
    """

    def __init__(
            self,
            children: List[MathExpr],
            on_change: Optional[Callable[[], None]] = None,
            readonly: bool = False
    ) -> None:
        """
        Constructor
        """
        # Init
        super().__init__(
            on_change=on_change,
            readonly=readonly
        )

        # Children
        self._children: List[MathExpr] = children
    # end __init__

    # region PUBLIC

    def add_children(self, children: List[MathExpr]) -> None:
        """
        Add children
        """
        self._children += children
    # end add_children

    def add_child(self, child_node: MathExpr) -> None:
        """
        Add child
        """
        self._check_locked()
        child_node.add_on_change_listener(self._children_updated)
        self._children.append(child_node)
    # end add_child

    # endregion PUBLIC

    # region PRIVATE

    def _children_updated(self) -> None:
        """

        """

    # endregion PRIVATE

# end NodeMathExpr



