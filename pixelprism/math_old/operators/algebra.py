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
from abc import ABC
from typing import Callable, List, Any, Optional
from pixelprism.math.math_base import MathOperator, MathEvent, MathEventData
from pixelprism.math.scalar import Scalar


class ScalarToScalarAddition(MathOperator, ABC):
    """
    A mathematical operator that adds two scalar values.

    This class represents an addition operation between two Scalar objects.
    It computes the sum of the two scalars and updates automatically when
    either of the operands changes.

    Example:
        ```python
        # Create two scalar values
        a = Scalar(5)
        b = Scalar(10)

        # Create an addition operator
        sum_op = ScalarToScalarAddition(a, b)

        # Get the result
        print(sum_op.value)  # 15

        # When an operand changes, the result updates automatically
        a.value = 7
        print(sum_op.value)  # 17
        ```

    Attributes:
        expr_type (str): The type of expression ("Addition").
        return_type (str): The type of value returned ("Scalar").
        arity (int): The number of operands (2 for binary addition).
    """

    # Expression type - identifies the category of mathematical operator
    expr_type = "Addition"

    # Return type - specifies the type of value returned by this operator
    return_type = "Scalar"

    # Arity - number of operands this operator takes (2 for binary addition)
    arity = 2

    def __init__(
            self,
            scalar1: Scalar,
            scalar2: Scalar,
            on_change: Callable[[MathEventData], None] = None,
    ) -> None:
        """
        Initialize a new ScalarToScalarAddition operator.

        This constructor sets up the addition operator with its two scalar operands
        and registers listeners to detect when their values change.

        Args:
            scalar1 (Scalar): The first scalar operand.
            scalar2 (Scalar): The second scalar operand.
            on_change (Optional[Callable[[MathEventData], None]]): Function to call when the result changes.
                The function should accept a MathEventData parameter.

        Example:
            ```python
            # Create two scalar values
            a = Scalar(5)
            b = Scalar(10)

            # Create an addition operator
            def on_change(data):
                print(f"Sum changed from {data.past_value} to {data.value}")

            sum_op = ScalarToScalarAddition(a, b, on_change=on_change)
            ```
        """
        MathOperator.__init__(
            self,
            children=[scalar1, scalar2],
            on_change=on_change,
        )

        # Check that child one is a Scalar
        assert isinstance(scalar1, Scalar), f"Scalar 1 {scalar1} must be a Scalar"
        assert isinstance(scalar2, Scalar), f"Scalar 2 {scalar2} must be a Scalar"

        # Keep direct link to scalar
        self.scalar1 = scalar1
        self.scalar2 = scalar2
    # end __init__

    def _get(self) -> float:
        """
        Compute the sum of the two scalar operands.

        This method is called internally to calculate the current value of the addition
        operation based on the current values of the operands.

        Returns:
            float: The sum of the two scalar operands.

        Example:
            ```python
            # This method is called automatically when accessing the value property
            sum_op = ScalarToScalarAddition(Scalar(5), Scalar(10))
            result = sum_op.value  # Internally calls _get() to compute 5 + 10
            ```
        """
        return self._children[0].value + self._children[1].value
    # end _get

    def to_list(self) -> List[Any]:
        """
        Convert the addition operation to a list representation.

        This method provides a way to convert the addition operation to a list format,
        which can be useful for serialization or for operations that require list input.

        Returns:
            List[Any]: A list containing the addition operator and its operands.

        Example:
            ```python
            # Convert an addition operation to a list
            sum_op = ScalarToScalarAddition(Scalar(5), Scalar(10))
            list_repr = sum_op.to_list()  # Might return ["Addition", 5, 10]
            ```
        """
        return ["Addition", self._children[0].to_list(), self._children[1].to_list()]
    # end to_list
