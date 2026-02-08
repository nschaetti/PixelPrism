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

"""
TODO: Add documentation.
"""


__all__ = [
    "SymbolicMathError",
    "SymbolicMathOperatorError",
    "SymbolicMathValidationError",
    "SymbolicMathLookupError",
    "SymbolicMathNotImplementedError",
    "SymbolicMathRuntimeError",
    "SymbolicMathTypeError",
    "SymbolicMathShapeError",
]


class SymbolicMathError(RuntimeError):
    """
    Base exception for symbolic math expression failures.

    Subclasses capture finer grain failure modes (validation, lookup, etc.)
    to keep error handling explicit without exposing the internals of this
    module to callers.
    """


class SymbolicMathOperatorError(SymbolicMathError):
    """Raised when an operator declaration does not match a node definition."""


class SymbolicMathValidationError(SymbolicMathError):
    """Raised when runtime metadata (dtype, shape, etc.) violates invariants."""


class SymbolicMathLookupError(SymbolicMathError):
    """Raised when an expression, variable, or constant cannot be located."""


class SymbolicMathNotImplementedError(SymbolicMathError):
    """Raised when a subclass fails to implement an abstract requirement."""
# end class MathExprNotImplementedError


class SymbolicMathRuntimeError(SymbolicMathError):
    """Raised when an internal invariant is violated."""
# end class MathExprRuntimeError


class SymbolicMathTypeError(SymbolicMathError):
    """Raised when an operation is applied to an invalid operand type."""
# end class MathExprTypeError


class SymbolicMathShapeError(SymbolicMathError):
    """Raised when an operation is applied to operands with mismatched shapes."""
# end class MathExprShapeError
