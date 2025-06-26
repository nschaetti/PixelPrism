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
import pixelprism.math as m


def add(
        operand1: m.MathExpr,
        operand2: m.MathExpr
) -> m.MathOperator:
    """
    Add two math expressions.

    Args:
        operand1: First math expression.
        operand2: Second math expression.

    Returns:
        A new math expression.
    """
    # Get class names of the two operands
    operand1_class_name = operand1.__class__.__name__
    operand2_class_name = operand2.__class__.__name__

    # Compute name of the operator
    operator_class_name = f"{operand1_class_name}To{operand2_class_name}Addition"

    # Search for this class in pixelprism.math (m)
    # ...
# end add

