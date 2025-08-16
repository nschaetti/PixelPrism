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

    This class extends the base Event class to provide specialized event handling
    for mathematical expressions. It is used to notify listeners when values change
    in mathematical expressions.

    Example:
        ```python
        # Create a math_old event
        event = MathEvent()

        # Add a listener
        def on_math_change(data):
            print(f"Value changed from {data.past_value} to {data.value}")

        event += on_math_change

        # Trigger the event
        event.trigger(data=MathEventData(past_value=5, value=10, direct=True, source=None))
        ```
    """
    pass
# end class MathEvent


# Event data
class MathEventData:
    """
    Container for data passed with mathematical expression events.

    This class holds information about value changes in mathematical expressions,
    including the previous value, new value, whether the change was direct or indirect,
    and the source of the change.

    Attributes:
        past_value (Any): The previous value before the change.
        value (Any): The new value after the change.
        direct (bool): True if the change was made directly to this object, False if propagated.
        source (object): The object that originated the change.
    """

    # Constructor
    def __init__(
            self,
            past_value: Any,
            value: Any,
            direct: bool,
            source: object
    ) -> None:
        """
        Initialize a new MathEventData instance.

        Args:
            past_value (Any): The previous value before the change.
            value (Any): The new value after the change.
            direct (bool): True if the change was made directly to this object, 
                          False if propagated from another object.
            source (object): The object that originated the change.
        """
        self.past_value = past_value
        self.value = value
        self.direct = direct
        self.source = source
    # end __init_-

# end MathEventData


class MathExpr(ABC):
    """
    Abstract base class for mathematical expressions.

    This class serves as the foundation for all mathematical expressions in the PixelPrism library.
    It provides common functionality for handling values, events, and operations.
    Subclasses must implement the abstract methods to provide specific behavior.

    Example:
        ```python
        def on_change_handler(event_data):
            print(f"Value changed from {event_data.past_value} to {event_data.value}!")

        # Create an instance of a concrete subclass
        expr = SomeMathExprSubclass(on_change=on_change_handler)

        # Set a new value, which will trigger the on_change_handler
        expr.value = 42

        # Get the current value
        current_value = expr.value
        ```

    Args:
        on_change (Optional[Callable[[MathEventData], None]]): Function to call when the value changes.
            The function should accept a MathEventData parameter.
        readonly (bool): If True, the value cannot be changed after initialization.

    Attributes:
        expr_type (str): The type of expression (must be defined by subclasses).
        return_type (str): The return type of the expression (must be defined by subclasses).
        arity (int): The number of operands this expression takes (must be defined by subclasses).
    """

    # Expression type - identifies the category of mathematical expression
    expr_type = None

    # Return type - specifies the type of value returned by this expression
    return_type = None

    # Arity - number of operands (e.g., 1 for unary, 2 for binary operations)
    arity = None

    def __init__(
        self,
        on_change: Optional[Callable[[MathEventData], None]] = None,
        readonly: bool = False
    ) -> None:
        """
        Initialize a new mathematical expression.

        This constructor sets up the basic properties of a mathematical expression,
        including its read-only status and event handling. It also validates that
        subclasses have properly defined required class variables.

        Args:
            on_change (Optional[Callable[[MathEventData], None]]): Function to call when the value changes.
                The function should accept a MathEventData parameter containing information about the change.
            readonly (bool): If True, the value cannot be changed after initialization.
                Attempts to modify a read-only expression will raise a RuntimeError.

        Raises:
            NotImplementedError: If the subclass does not define the required class variables
                (expr_type, return_type, arity).
        """
        self._readonly: bool = readonly

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

        # Check that arity has been set
        if self.__class__.arity is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define class variable 'arity'")
        # end if
    # end __init__

    # region PROPERTIES

    @property
    def value(self) -> Any:
        """
        Get the current value of this mathematical expression.

        This property provides access to the underlying value of the expression.
        For leaf nodes, this is the actual value; for operator nodes, this is the
        result of evaluating the expression.

        Returns:
            Any: The current value of the expression. The specific type depends on
                the implementation in the subclass, as indicated by the return_type
                class variable.

        Example:
            ```python
            # Get the value of an expression
            expr = SomeExpression(42)
            current_value = expr.value  # Returns 42
            ```
        """
        return self._get()
    # end value getter

    @value.setter
    def value(self, value: Any) -> None:
        """
        Set the value of this mathematical expression.

        This property allows changing the value of the expression. When the value
        is changed, the on_change event is triggered, notifying all registered listeners.
        If the expression is read-only, attempting to set the value will raise an error.

        If the provided value is another mathematical expression, its value will be
        extracted and used instead of the expression object itself.

        Args:
            value (Any): The new value to set. This can be a primitive value or another
                MathExpr instance, in which case its value will be extracted.

        Raises:
            RuntimeError: If the object is read-only (readonly=True was passed to the constructor).

        Example:
            ```python
            # Set the value of an expression
            expr = SomeExpression()
            expr.value = 42  # Sets the value to 42 and triggers on_change event

            # Using another expression as the value
            expr1 = SomeExpression(10)
            expr2 = SomeExpression()
            expr2.value = expr1  # Sets expr2's value to 10 (not the expr1 object)
            ```
        """
        self._check_locked()
        if isinstance(value, MathExpr):
            value = value._get()
        # end if
        self._set(value)
    # end value setter

    # endregion PROPERTIES

    # region PUBLIC

    @property
    def on_change(self) -> MathEvent:
        """
        Get the on_change event object for this expression.

        This property provides access to the event object that is triggered when the
        expression's value changes. You can use this to add or remove listeners directly,
        although the add_on_change_listener and unregister_event methods are preferred.

        Returns:
            MathEvent: The event object that is triggered when the value changes.

        Example:
            ```python
            # Access the event object directly
            expr = SomeExpression()
            event = expr.on_change

            # Add a listener using the event object
            def on_change_handler(data):
                print(f"Value changed to {data.value}")

            event += on_change_handler
            ```
        """
        return self._on_change
    # end on_change

    @abstractmethod
    def copy(self) -> 'MathExpr':
        """
        Create a deep copy of this mathematical expression.

        This method creates a new instance with the same value and type as the current
        instance, but without sharing any mutable state. Changes to the copy will not
        affect the original, and vice versa.

        This is an abstract method that must be implemented by subclasses.

        Returns:
            MathExpr: A new instance with the same value as the current object.

        Example:
            ```python
            # Create a copy of an expression
            expr1 = SomeExpression(42)
            expr2 = expr1.copy()  # expr2 has value 42 but is a different object

            expr2.value = 100  # Changes expr2 but not expr1
            print(expr1.value)  # Still 42
            ```
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement 'copy'")
    # end copy

    def add_on_change_listener(self, listener: Callable[[MathEventData], None]) -> None:
        """
        Register a function to be called when the expression's value changes.

        This method adds a listener function that will be called whenever the value
        of this expression changes. The listener will receive a MathEventData object
        containing information about the change.

        Multiple listeners can be registered, and they will be called in the order
        they were added.

        Args:
            listener (Callable[[MathEventData], None]): Function to call when the value changes.
                The function must accept a single parameter of type MathEventData.

        Raises:
            TypeError: If the listener is not a callable function.

        Example:
            ```python
            def on_change_handler(data):
                print(f"Value changed from {data.past_value} to {data.value}")

            expr = SomeExpression()
            expr.add_on_change_listener(on_change_handler)
            expr.value = 42  # Triggers on_change_handler
            ```
        """
        if not callable(listener):
            raise TypeError("Listener must be callable")
        # end if
        self._on_change += listener
    # end add_on_change_listener

    def unregister_event(self, listener: Callable[[MathEventData], None]) -> None:
        """
        Remove a previously registered listener from the on_change event.

        This method removes a listener that was previously added with add_on_change_listener.
        If the listener was not previously registered, this method has no effect.

        Args:
            listener (Callable[[MathEventData], None]): The listener function to remove.
                This must be the same function object that was passed to add_on_change_listener.

        Raises:
            TypeError: If the listener is not a callable function.

        Example:
            ```python
            def on_change_handler(data):
                print(f"Value changed to {data.value}")

            expr = SomeExpression()
            expr.add_on_change_listener(on_change_handler)

            # Later, when you no longer want to receive notifications
            expr.unregister_event(on_change_handler)
            ```
        """
        if not callable(listener):
            raise TypeError("Listener must be callable")
        # end if
        self._on_change -= listener
    # end unregister_event

    @abstractmethod
    def to_list(self) -> List[Any]:
        """
        Convert the expression's value to a list representation.

        This method provides a way to convert the expression's value to a list format,
        which can be useful for serialization or for operations that require list input.

        This is an abstract method that must be implemented by subclasses.

        Returns:
            List[Any]: The value represented as a list. The specific format depends
                on the subclass implementation.

        Example:
            ```python
            # Convert an expression to a list
            expr = SomeExpression(42)
            value_list = expr.to_list()  # Might return [42] or another list format
            ```
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_list")
    # end to_list

    # endregion PUBLIC

    # region PRIVATE

    # Trigger on change event
    def _trigger_on_change(self, event_data: MathEventData) -> None:
        """
        Trigger the on_change event with the provided event data.

        This internal method is used to notify all registered listeners that the
        expression's value has changed. It passes the provided event data to all
        listeners.

        This method is typically called by subclasses when they modify their value.

        Args:
            event_data (MathEventData): Object containing information about the change,
                including the previous value, new value, whether the change was direct,
                and the source of the change.
        """
        self._on_change.trigger(data=event_data)
    # end _trigger_on_change

    def _check_locked(self) -> None:
        """
        Check if this expression is read-only and raise an error if it is.

        This internal method is used before modifying the expression's value to ensure
        that the expression is not read-only. If the expression is read-only, a
        RuntimeError is raised.

        This method is typically called by the value setter and other methods that
        modify the expression's value.

        Raises:
            RuntimeError: If the expression was created with readonly=True.
        """
        if self._readonly:
            raise RuntimeError(f"{self.__class__.__name__} is read only !")
        # end if
    # end _check_locked

    @abstractmethod
    def _set(self, value: Any) -> None:
        """
        Set the internal value of this expression.

        This is an abstract method that must be implemented by subclasses to provide
        specific behavior for setting the value. Implementations should update the
        internal state and trigger the on_change event.

        This method is typically called by the value setter after checking that the
        expression is not read-only.

        Args:
            value (Any): The new value to set. The specific type depends on the
                subclass implementation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _set")
    # end _set

    @abstractmethod
    def _get(self) -> Any:
        """
        Get the internal value of this expression.

        This is an abstract method that must be implemented by subclasses to provide
        specific behavior for getting the value. Implementations should return the
        current value of the expression.

        This method is typically called by the value getter.

        Returns:
            Any: The current value of the expression. The specific type depends on
                the subclass implementation, as indicated by the return_type class variable.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '_get'")
    # end _get

    # endregion PRIVATE

    # region OVERRIDE

    # Override the integer conversion
    def __int__(self):
        """
        Return the integer representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__int__'")
    # end __int__

    # Override the float conversion
    def __float__(self):
        """
        Return the float representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__float__'")
    # end __float__

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__str__'")
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__repr__'")
    # end __repr__

    # Operator overloading
    def __add__(self, other):
        """
        Add the scalar value to another scalar or value.

        Args:
            other (any): Scalar or value to add
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__add__'")
    # end __add__

    def __radd__(self, other):
        """
        Add the scalar value to another scalar or value.

        Args:
            other (any): Scalar or value to add
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__radd__'")
    # end __radd__

    def __sub__(self, other):
        """
        Subtract the scalar value from another scalar or value.

        Args:
            other (any): Scalar or value to subtract
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__sub__'")
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract the scalar value from another scalar or value.

        Args:
            other (any): Scalar or value to subtract
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__rsub__'")
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to multiply
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__mul__'")
    # end __mul__

    def __rmul__(self, other):
        """
        Multiply the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to multiply
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__rmul__'")
    # end __rmul__

    def __truediv__(self, other):
        """
        Divide the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to divide by
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__truediv__'")
    # end __truediv__

    def __rtruediv__(self, other):
        """
        Divide the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to divide by
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__rtruediv__'")
    # end __rtruediv__

    def __eq__(self, other):
        """
        Check if the scalar value is equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__eq__'")
    # end __eq__

    def __ne__(self, other):
        """
        Check if the scalar value is not equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__ne__'")
    # end __ne__

    # Override less
    def __lt__(self, other):
        """
        Check if the scalar value is less than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__lt__'")
    # end __lt__

    # Override less or equal
    def __le__(self, other):
        """
        Check if the scalar value is less than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__le__'")
    # end __le__

    # Override greater
    def __gt__(self, other):
        """
        Check if the scalar value is greater than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__gt__'")
    # end __gt__

    # Override greater or equal
    def __ge__(self, other):
        """
        Check if the scalar value is greater than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement '__ge__'")
    # end __ge__

    # endregion OVERRIDE

# end class MathExpr


# A mathematical expression which combines several
# children expressions.
class MathOperator(MathExpr, ABC):
    """
    Abstract base class for mathematical operators that combine multiple expressions.

    MathOperator represents a node in an expression tree that performs an operation
    on one or more child expressions. Examples include arithmetic operations (add, subtract),
    logical operations (and, or), and other operations that combine multiple values.

    The value of a MathOperator is computed based on its children's values. When a child's
    value changes, the operator's value is automatically updated, and the change is
    propagated to any listeners.

    Subclasses must define:
    - expr_type: The type of expression (e.g., "Add", "Multiply")
    - return_type: The type of value returned (e.g., "Scalar", "Point2D")
    - arity: The number of operands this operator takes (e.g., 2 for binary operations)
    - _get(): Method to compute the value based on children's values

    Example:
        ```python
        # Example of a binary operator (like Add)
        class Add(MathOperator):
            expr_type = "Add"
            return_type = "Scalar"
            arity = 2

            def _get(self):
                return self._children[0].value + self._children[1].value

        # Using the operator
        a = Scalar(5)
        b = Scalar(10)
        sum_expr = Add([a, b])  # Creates an expression that computes a + b
        print(sum_expr.value)  # 15

        a.value = 7  # Automatically updates sum_expr
        print(sum_expr.value)  # 17
        ```

    Attributes:
        _children (List[MathExpr]): The child expressions this operator operates on.
        _value: The cached result of the operation.
    """

    # Expression type - identifies the category of mathematical operator
    expr_type = None

    # Return type - specifies the type of value returned by this operator
    return_type = None

    # Arity - number of operands this operator takes (e.g., 2 for binary operations)
    arity = 0

    # Constructor
    def __init__(
            self,
            children: List[MathExpr],
            on_change: Optional[Callable[[MathEventData], None]] = None,
            readonly: bool = False
    ) -> None:
        """
        Initialize a new mathematical operator with the specified children.

        This constructor sets up the operator with its child expressions and registers
        listeners to detect when children's values change. When a child's value changes,
        the operator's value is automatically updated.

        Args:
            children (List[MathExpr]): The child expressions this operator operates on.
                The number of children must match the operator's arity.
            on_change (Optional[Callable[[MathEventData], None]]): Function to call when the value changes.
                The function should accept a MathEventData parameter.
            readonly (bool): If True, the value cannot be changed after initialization.

        Raises:
            NotImplementedError: If the subclass does not define the required class variables.
            AssertionError: If the number of children does not match the operator's arity.

        Example:
            ```python
            # Create two scalar expressions
            a = Scalar(5)
            b = Scalar(10)

            # Create an operator that combines them
            add_op = Add([a, b])  # Add has arity=2, so it needs exactly 2 children
            ```
        """
        # Init
        super().__init__(
            on_change=on_change,
            readonly=readonly
        )

        # Value of the node
        self._value = None

        # Check that arity has been set
        if self.__class__.arity > 0:
            raise NotImplementedError(f"{self.__class__.__name__} must define class variable 'arity'")
        # end if

        # Check that the number of children is equal to arity
        assert len(children) == self.arity, f"{self.__class__.__name__} children must have length {self.arity}"

        # Children
        self._children: List[MathExpr] = list()
        self._add_children(children)
    # end __init__

    # region PRIVATE

    # Update value
    def _update(self) -> None:
        """
        Update the cached value of this operator by recomputing it from children.

        This internal method recalculates the operator's value based on the current
        values of its child expressions. It calls the _get() method to perform the
        actual computation and stores the result in the _value attribute.

        This method is typically called when a child's value changes or when the
        operator's value is accessed for the first time.
        """
        self._value = self._get()
    # end _update

    def _add_children(self, children: List[MathExpr]) -> None:
        """
        Add multiple child expressions to this operator.

        This internal method adds the provided child expressions to the operator's
        list of children and registers listeners to detect when their values change.

        Args:
            children (List[MathExpr]): The child expressions to add to this operator.
                These expressions will be used to compute the operator's value.
        """
        for child in children:
            # Register to listen for changes in the child
            child.add_on_change_listener(self._children_updated)
            self._children.append(child)
        # end for
    # end _add_children

    def _add_child(self, child_node: MathExpr) -> None:
        """
        Add a single child expression to this operator.

        This internal method adds the provided child expression to the operator's
        list of children and registers a listener to detect when its value changes.

        This method checks if the operator is read-only before adding the child.

        Args:
            child_node (MathExpr): The child expression to add to this operator.
                This expression will be used to compute the operator's value.

        Raises:
            RuntimeError: If the operator is read-only.
        """
        self._check_locked()
        child_node.add_on_change_listener(self._children_updated)
        self._children.append(child_node)
    # end _add_child

    def _children_updated(self, event_data: MathEventData) -> None:
        """
        Handle updates to child expressions.

        This internal method is called when a child expression's value changes.
        It updates the operator's value based on the new child values and notifies
        listeners of the change.

        The method creates a new event data object that includes the operator's
        previous value, new value, and information about the source of the change.

        Args:
            event_data (MathEventData): Information about the change in the child expression,
                including the previous value, new value, and source of the change.
        """
        # Store the past value before updating
        past_value = self._value

        # Compute new value based on updated children
        self._update()

        # Create new event data for this operator's change
        send_event_data = MathEventData(
            past_value=past_value,
            value=self._value,
            direct=False,  # This is an indirect change (propagated from a child)
            source=event_data.source  # Preserve the original source of the change
        )

        # Notify listeners of the change
        self._on_change.trigger(data=send_event_data)
    # end _children_updated

    def _set(self, value: Any) -> None:
        """
        Set the value of this operator (not supported).

        Operators compute their values based on their children, so their values
        cannot be set directly. This method raises an error if called.

        To change an operator's value, you should modify its child expressions instead.

        Args:
            value (Any): The value that was attempted to be set.

        Raises:
            ValueError: Always, because operator values cannot be set directly.

        Example:
            ```python
            # This will raise an error:
            add_op = Add([Scalar(5), Scalar(10)])
            add_op.value = 20  # ValueError: You cannot set the value of an operator

            # Instead, modify the children:
            add_op._children[0].value = 10  # Now add_op.value will be 20
            ```
        """
        raise ValueError(f"You cannot set the value of an operator on a {self}")
    # end _set

    # endregion PRIVATE

# end MathOperator


# An expression which does not contain sub-expressions
class MathLeaf(MathExpr, ABC):
    """
    Abstract base class for leaf nodes in the expression tree.

    MathLeaf represents a terminal node in an expression tree that holds a single value.
    Unlike MathOperator, which computes its value based on child expressions, a MathLeaf
    stores its value directly. Examples include constants, variables, and other primitive values.

    Leaf nodes are the building blocks of more complex expressions. They can be used
    directly or combined with operators to form expression trees.

    Subclasses must define:
    - expr_type: The type of expression (e.g., "Scalar", "Point2D")
    - return_type: The type of value returned (usually the same as expr_type)
    - Implementations of abstract methods (_set, _get, etc.)

    Example:
        ```python
        # Example of a leaf node (like Scalar)
        class IntegerValue(MathLeaf):
            expr_type = "Integer"
            return_type = "Integer"
            arity = 0

            def _set(self, value):
                past_value = self._value
                self._value = int(value)  # Ensure the value is an integer
                self._on_change.trigger(data=MathEventData(
                    past_value=past_value,
                    value=self._value,
                    direct=True,
                    source=self
                ))

            def _get(self):
                return self._value

        # Using the leaf node
        x = IntegerValue(5)
        print(x.value)  # 5

        x.value = 10
        print(x.value)  # 10

        x.value = 3.14  # Will be converted to 3 by _set
        print(x.value)  # 3
        ```

    Attributes:
        _value (Any): The actual value stored in this leaf node.
    """

    # Expression type - identifies the category of mathematical leaf
    expr_type = None

    # Return type - specifies the type of value returned by this leaf
    return_type = None

    # Arity - always 0 for leaf nodes as they don't have child expressions
    arity = 0

    def __init__(
            self,
            value: Any,
            on_change: Optional[Callable[[MathEventData], None]] = None,
            readonly: bool = False
    ):
        """
        Initialize a new leaf node with the specified value.

        This constructor sets up the leaf node with its initial value and configures
        event handling for value changes.

        Args:
            value (Any): The initial value to store in this leaf node.
                The specific type depends on the subclass implementation.
            on_change (Optional[Callable[[MathEventData], None]]): Function to call when the value changes.
                The function should accept a MathEventData parameter.
            readonly (bool): If True, the value cannot be changed after initialization.

        Example:
            ```python
            # Create a leaf node with an initial value
            x = IntegerValue(42)

            # Create a leaf node with a change listener
            def on_change(data):
                print(f"Value changed from {data.past_value} to {data.value}")

            y = IntegerValue(10, on_change=on_change)

            # Create a read-only leaf node
            z = IntegerValue(100, readonly=True)
            # z.value = 200  # This would raise a RuntimeError
            ```
        """
        # Init
        super().__init__(
            on_change=on_change,
            readonly=readonly
        )

        # Value
        self._value: Any = value
    # end __init__

    # region PRIVATE

    @abstractmethod
    def _set(self, value: Any) -> None:
        """
        Set the internal value of this leaf node.

        This is an abstract method that must be implemented by subclasses to provide
        specific behavior for setting the value. The implementation should:
        1. Store the previous value for event notification
        2. Update the internal value (with any necessary type conversion)
        3. Trigger the on_change event with appropriate event data

        The default implementation provided here shows the basic pattern that
        subclasses should follow, but they may need to add type checking or conversion.

        Args:
            value (Any): The new value to set. The specific type depends on the
                subclass implementation.

        Example implementation:
            ```python
            def _set(self, value):
                past_value = self._value
                self._value = self._convert_value(value)  # Apply any type conversion
                self._on_change.trigger(data=MathEventData(
                    past_value=past_value,
                    value=self._value,
                    direct=True,
                    source=self
                ))
            ```
        """
        past_value = self._value
        self._value = value
        self._on_change.trigger(data=MathEventData(past_value=past_value, value=value, direct=True, source=self))
    # end _set

    @abstractmethod
    def _get(self) -> Any:
        """
        Get the internal value of this leaf node.

        This is an abstract method that must be implemented by subclasses to provide
        specific behavior for getting the value. The implementation should return
        the current value, possibly with type conversion if needed.

        The default implementation provided here simply returns the internal value,
        but subclasses may need to add type conversion or other processing.

        Returns:
            Any: The current value of the leaf node. The specific type depends on
                the subclass implementation, as indicated by the return_type class variable.

        Example implementation:
            ```python
            def _get(self):
                return self._value  # Return the value, possibly with conversion
            ```
        """
        return self._value
    # end _get

    # endregion PRIVATE

# end MathLeaf
