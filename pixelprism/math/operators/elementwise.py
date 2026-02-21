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
Elementwise operator implementations.
"""
from abc import ABC
from typing import Sequence
from ..tensor import Tensor
from ..dtype import DType, promote
from ..shape import Shape
from ..math_node import MathNode
from ..math_leaves import Variable, Constant
from ..typing import MathExpr, LeafKind, SimplifyOptions, OpSimplifyResult, SimplifyRule, OpAssociativity, AlgebraicExpr
from .base import Operands, OperatorBase, operator_registry, ParametricOperator


__all__ = [
    "ElementwiseOperator",
    "UnaryElementwiseOperator",
    "UnaryElementwiseParametricOperator",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Exp",
    "Exp2",
    "Expm1",
    "Log",
    "Log1p",
    "Log2",
    "Log10",
    "Sqrt",
    "Square",
    "Cbrt",
    "Reciprocal",
    "Deg2rad",
    "Rad2deg",
    "Absolute",
    "Abs",
    "Neg",
]


class ElementwiseOperator(OperatorBase, ABC):
    """
    Element-wise operator.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # end def __init__

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return True
    # end def check_operands

    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return False
    # end def contains

    def print(self, operands: Operands, **kwargs) -> str:
        """Return a human-readable representation of the operator.

        Parameters
        ----------
        operands : Operands
            The operands of the operator.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        str : A human-readable representation of the operator.
        """
        need_left_paren = self._needs_parentheses(child=operands[0], is_right_child=True)
        need_right_paren = self._needs_parentheses(child=operands[1], is_right_child=False)
        left_str = f"({operands[0]})" if need_left_paren else operands[0].__str__()
        right_str = f"({operands[1]})" if need_right_paren else operands[1].__str__()
        return f"{left_str} {self.SYMBOL} {right_str}"
    # end def print

    def __str__(self) -> str:
        """Return a concise human-readable identifier."""
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        return f"{self.NAME}(arity={self.ARITY})"
    # end def __repr__

    # region STATIC

    def check_shapes(self, operands: Operands) -> bool:
        """
        Check that both operands have identical shapes or support scalar broadcasting.
        """
        a, b = operands
        if a.rank != b.rank:
            if a.rank != 0 and b.rank != 0:
                raise ValueError(
                    f"{self.NAME} requires operands with identical ranks if not scalar, "
                    f"got {a.rank} and {b.rank}."
                )
            # end if
        else:
            # Same rank, but require same shape
            if not a.shape.equals(b.shape):
                raise ValueError(
                    f"{self.NAME} requires operands with identical shapes, "
                    f"got {a.shape} and {b.shape}."
                )
            # end if
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        """
        Output shape is identical to operand shapes.
        """
        a, b = operands
        if a.rank == b.rank:
            return operands[0].shape
        elif a.rank > b.rank:
            return operands[0].shape
        else:
            return operands[1].shape
        # end if
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, b = operands
        return promote(a.dtype, b.dtype)
    # end def infer_dtype

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        pass
    # end def check_shapes

    # endregion STATIC

# end class ElementwiseOperator


class Add(ElementwiseOperator):
    """
    Element-wise addition operator.
    """

    ARITY = 2
    IS_VARIADIC = False
    IS_DIFF = True
    COMMUTATIVE = True
    ASSOCIATIVE = True
    PRECEDENCE = 10
    ASSOCIATIVITY = OpAssociativity.LEFT
    NAME = "add"
    SYMBOL = "+"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """
        Evaluate element-wise addition.
        """
        a, b = operands
        return a.eval() + b.eval()
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        a, b = operands
        return a.diff(wrt) + b.diff(wrt)
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        a: MathExpr = operands[0]
        b: MathExpr = operands[1]

        # Replace expression
        new_operands = [a, b]
        repr_expr = None

        # Merge constants
        if self._apply_rule(SimplifyRule.MERGE_CONSTANTS, options):
            # Replaces with constant when both operands are constant
            if isinstance(a, Constant) and isinstance(b, Constant):
                repr_expr = Constant.new(a.eval() + b.eval())
                new_operands = []
            # end if
        # end if

        # Add zero
        if self._apply_rule(SimplifyRule.ADD_ZERO, options):
            if isinstance(a, Constant) and a.eval().is_null():
                repr_expr = b
                new_operands = []
            elif isinstance(b, Constant) and b.eval().is_null():
                repr_expr = a
                new_operands = []
            # end if
        # end if

        return OpSimplifyResult(new_operands, repr_expr)
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

# end class Add


class Sub(ElementwiseOperator):
    """
    Element-wise subtraction operator.
    """

    ARITY = 2
    NAME = "sub"
    IS_VARIADIC = False
    IS_DIFF = True
    COMMUTATIVE = False
    ASSOCIATIVE = False
    PRECEDENCE = 10
    ASSOCIATIVITY = OpAssociativity.LEFT
    SYMBOL = "-"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise subtraction."""
        a, b = operands
        return a.eval() - b.eval()
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        a, b = operands
        return a.diff(wrt) - b.diff(wrt)
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        a: MathExpr = operands[0]
        b: MathExpr = operands[1]

        # Replace expression
        new_operands = [a, b]
        repr_expr = None

        # a - b = c
        if self._apply_rule(SimplifyRule.MERGE_CONSTANTS, options):
            # Replaces with constant when both operands are constant
            if isinstance(a, Constant) and isinstance(b, Constant):
                repr_expr = Constant.new(a.eval() - b.eval())
                new_operands = []
            # end if
        # end if

        # x - 0 = x
        # 0 - x = -x
        if self._apply_rule(SimplifyRule.SUB_ZERO, options):
            if isinstance(a, Constant) and a.eval().is_null():
                repr_expr = -b
                new_operands = []
            elif isinstance(b, Constant) and b.eval().is_null():
                repr_expr = a
                new_operands = []
            # end if
        # end if

        return OpSimplifyResult(new_operands, repr_expr)
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Sub


class Mul(ElementwiseOperator):
    """
    Element-wise multiplication operator.
    """

    ARITY = 2
    IS_VARIADIC = False
    IS_DIFF = True
    COMMUTATIVE = True
    ASSOCIATIVE = True
    NAME = "mul"
    SYMBOL = "*"
    PRECEDENCE = 20
    ASSOCIATIVITY = OpAssociativity.LEFT

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise multiplication."""
        a, b = operands
        return a.eval() * b.eval()
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        a, b = operands
        return a.diff(wrt) * b + b.diff(wrt) * a
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        a: MathExpr = operands[0]
        b: MathExpr = operands[1]

        # Replace expression
        new_operands = [a, b]
        repr_expr = None

        # a * b = c
        if self._apply_rule(SimplifyRule.MERGE_CONSTANTS, options):
            # Replaces with constant when both operands are constant
            if isinstance(a, Constant) and isinstance(b, Constant):
                repr_expr = Constant.new(a.eval() * b.eval())
                new_operands = []
            # end if
        # end if

        # x * 1 = x
        if self._apply_rule(SimplifyRule.MUL_ONE, options):
            if isinstance(a, Constant) and a.eval().is_full(1):
                repr_expr = b
                new_operands = []
            elif isinstance(b, Constant) and b.eval().is_full(1):
                repr_expr = a
                new_operands = []
            # end if
        # end if

        # x * 0 = 0
        if self._apply_rule(SimplifyRule.MUL_ZERO, options):
            if isinstance(a, Constant) and a.eval().is_null():
                repr_expr = Constant.new(0)
                new_operands = []
            elif isinstance(b, Constant) and b.eval().is_null():
                repr_expr = Constant.new(0)
                new_operands = []
            # end if
        # end if

        return OpSimplifyResult(new_operands, repr_expr)
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Mul


class Div(ElementwiseOperator):
    """
    Element-wise division operator.
    """

    ARITY = 2
    IS_VARIADIC = False
    IS_DIFF = True
    COMMUTATIVE = False
    ASSOCIATIVE = False
    NAME = "div"
    SYMBOL = "/"
    PRECEDENCE = 20
    ASSOCIATIVITY = OpAssociativity.LEFT

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise division."""
        a, b = operands
        return a.eval() / b.eval()
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        a, b = operands
        num = a.diff(wrt) * b - b.diff(wrt) * a
        denom = b * b
        return num / denom
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        a: MathExpr = operands[0]
        b: MathExpr = operands[1]

        # Replace expression
        new_operands = [a, b]
        repr_expr = None

        # a / b = c
        if self._apply_rule(SimplifyRule.MERGE_CONSTANTS, options):
            # Replaces with constant when both operands are constant
            if isinstance(a, Constant) and isinstance(b, Constant):
                repr_expr = Constant.new(a.eval() / b.eval())
                new_operands = []
            # end if
        # end if

        # x / 1 = x
        if self._apply_rule(SimplifyRule.DIV_ONE, options):
            if isinstance(b, Constant) and b.eval().is_full(1):
                repr_expr = a
                new_operands = [a]
            # end if
        # end if

        # 0 / x = 0
        if self._apply_rule(SimplifyRule.ZERO_DIV, options):
            if isinstance(a, Constant) and a.eval().is_null():
                repr_expr = Constant.new(0)
                new_operands = []
            # end if
        # end if

        return OpSimplifyResult(new_operands, repr_expr)
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Div



class UnaryElementwiseOperator(OperatorBase, ABC):
    """
    Base class for unary element-wise operators.
    """

    ARITY = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # end def __init__

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return True
    # end def check_operands

    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return False
    # end def contains

    def __str__(self) -> str:
        """Return a concise human-readable identifier."""
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        return f"{self.__class__.__name__}(arity={self.ARITY})"
    # end def __repr__

    # region STATIC

    def check_shapes(self, operands: Operands) -> bool:
        """Unary operators always match operand shape."""
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        """Shape matches operand."""
        return operands[0].shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """Return floating dtype for results."""
        operand_dtype = operands[0].dtype
        if operand_dtype in {DType.R, DType.C}:
            return operand_dtype
        return DType.R
    # end def infer_dtype

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        pass
    # end def check_shapes

    # endregion STATIC

# end class UnaryElementwiseOperator


class UnaryElementwiseParametricOperator(UnaryElementwiseOperator, ParametricOperator, ABC):
    pass
# end class UnaryElementwiseParametricOperator


class Neg(UnaryElementwiseOperator):
    """
    Element-wise negation operator.
    """

    ARITY = 1
    IS_VARIADIC = False
    IS_DIFF = True
    NAME = "neg"
    SYMBOL = "-"
    PRECEDENCE = 40
    ASSOCIATIVITY = OpAssociativity.NONE

    def __init__(self):
        super().__init__()
    # end def __init__

    def print(self, operands: Operands, **kwargs) -> str:
        """Return a human-readable representation of the operator.

        Parameters
        ----------
        operands : Operands
            The operands of the operator.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        str : A human-readable representation of the operator.
        """
        need_paren = isinstance(operands[0], MathNode) and operands[0].op.arity > 1
        child_str = f"({operands[0]})" if need_paren else operands[0].__str__()
        return f"{self.SYMBOL}{child_str}"
    # end def print

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """Neg accepts any operand shape."""
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """Shape matches operand."""
        return operands[0].shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """Return operand dtype."""
        return operands[0].dtype
    # end def infer_dtype

    # endregion STATIC

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise negation."""
        (value,) = operands
        return -value.eval()
    # end def _eval

    # TODO: That issue with MathNode -> diff should return MathNode
    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        x = operands[0]
        return -(x.diff(wrt))
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        a: MathExpr = operands[0]

        # Replace expression
        new_operands = [a]
        repr_expr = None

        # --x = x
        if hasattr(a, "op") and isinstance(a, MathNode) and a.op.name == Neg.NAME:
            if self._apply_rule(SimplifyRule.NEGATE_NEGATE, options):
                repr_expr = a.children[0]
                new_operands = []
            # end if
        # end if

        return OpSimplifyResult(new_operands, repr_expr)
    # end def _simplify

    def _canonicalize(self, operands: Sequence[AlgebraicExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Neg


class Pow(ElementwiseOperator):
    """
    Element-wise power operator.
    """

    ARITY = 2
    IS_VARIADIC = False
    IS_DIFF = True
    COMMUTATIVE = False
    ASSOCIATIVE = False
    NAME = "pow"
    SYMBOL = "^"
    PRECEDENCE = 30
    ASSOCIATIVITY = OpAssociativity.NONE

    def print(self, operands: Operands, **kwargs) -> str:
        """Return a human-readable representation of the operator.

        Parameters
        ----------
        operands : Operands
            The operands of the operator.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        str : A human-readable representation of the operator.
        """
        left_need_paren = isinstance(operands[0], MathNode) and operands[0].op.arity > 1
        right_need_paren = isinstance(operands[1], MathNode) and operands[1].op.arity > 1
        left_child_str = f"({operands[0]})" if left_need_paren else operands[0].__str__()
        right_child_str = f"({operands[1]})" if right_need_paren else operands[1].__str__()
        return f"{left_child_str}{self.SYMBOL}{right_child_str}"
    # end def print

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise exponentiation."""
        base, exponent = operands
        return Tensor.pow(base.eval(), exponent.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        base, exp = operands
        if exp.is_constant():
            c1 = Constant.new(exp.eval(), dtype=exp.dtype)
            if exp.eval() == 0:
                return Constant.new(0)
            elif exp.eval() > 0:
                c2 = Constant.new(exp.eval() - 1, dtype=exp.dtype)
                return c1 * Pow.create_node(operands=(base, c2))  * base.diff(wrt)
            else:
                c2 = Constant.new(-(exp.eval() - 1), dtype=exp.dtype)
                return c1 * 1 / Pow.create_node(operands=(base, c2)) * base.diff(wrt)
            # end if
        else:
            pow_ab = Pow.create_node(operands=(base, exp))
            bd_loga = exp.diff(wrt) * Log.create_node(operands=(base,))
            b_ad_on_a = exp * (base.diff(wrt) / base)
            return pow_ab * (bd_loga + b_ad_on_a)
        # end if
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Pow


class Exp(UnaryElementwiseOperator):
    """
    Element-wise exponential operator.
    """

    NAME = "exp"
    IS_VARIADIC = False
    IS_DIFF = True

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.exp(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (value,) = operands
        return Exp.create_node(operands=(value,)) * value.diff(wrt)
        # end if
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Exp


class Exp2(UnaryElementwiseOperator):
    """
    Element-wise base-2 exponential operator.
    """

    NAME = "exp2"
    IS_VARIADIC = False
    IS_DIFF = True

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.exp2(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (value,) = operands
        return Pow.create_node(operands=(Constant.new(2), value)) * Log.create_node(operands=(Constant.new(2),))
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Exp2


class Expm1(UnaryElementwiseOperator):
    """
    Element-wise exp(x) - 1 operator.
    """

    NAME = "expm1"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.expm1(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Expm1 does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Expm1


class Log(UnaryElementwiseOperator):
    """
    Element-wise natural logarithm operator.
    """

    NAME = "log"
    IS_VARIADIC = False
    IS_DIFF = True

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (value,) = operands
        return (Constant.new(1) / value) * value.diff(wrt)
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Log


class Log1p(UnaryElementwiseOperator):
    """
    Element-wise log(1 + x) operator.
    """

    NAME = "log1p"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log1p(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Log1p does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Log1p


class Sqrt(UnaryElementwiseOperator):
    """
    Element-wise square root operator.
    """

    NAME = "sqrt"
    IS_VARIADIC = False
    IS_DIFF = True

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.sqrt(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (value,) = operands
        return (Constant.new(1) / (Constant.new(2) * Sqrt.create_node(operands=(value,)))) * value.diff(wrt)
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Sqrt


class Square(UnaryElementwiseOperator):
    """
    Element-wise square operator.
    """

    NAME = "square"
    IS_VARIADIC = False
    IS_DIFF = True

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.square(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (value,) = operands
        return Constant.new(2) * value * value.diff(wrt)
    # end def _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Square


class Cbrt(UnaryElementwiseOperator):
    """
    Element-wise cubic root operator.
    """

    NAME = "cbrt"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.cbrt(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Cbrt does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Cbrt


class Reciprocal(UnaryElementwiseOperator):
    """
    Element-wise reciprocal operator.
    """

    NAME = "reciprocal"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.reciprocal(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Reciprocal does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Reciprocal


class Log2(UnaryElementwiseOperator):
    """
    Element-wise base-2 logarithm operator.
    """

    NAME = "log2"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log2(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Log2 does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Log2


class Log10(UnaryElementwiseOperator):
    """
    Element-wise base-10 logarithm operator.
    """

    NAME = "log10"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log10(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Log10 does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Log10


class Deg2rad(UnaryElementwiseOperator):
    """
    Convert degrees to radians.
    """

    NAME = "deg2rad"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.deg2rad(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Deg2rad does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Deg2rad


class Rad2deg(UnaryElementwiseOperator):
    """
    Convert radians to degrees.
    """

    NAME = "rad2deg"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.rad2deg(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Rad2deg does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Rad2deg


class Absolute(UnaryElementwiseOperator):
    """
    Element-wise absolute value operator.
    """

    NAME = "absolute"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.absolute(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Absolute does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Absolute


class Abs(UnaryElementwiseOperator):
    """
    Element-wise absolute value alias operator.
    """

    NAME = "abs"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.abs(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: MathNode,
            node: MathNode,
    ) -> Sequence[MathNode]:
        raise NotImplementedError("Abs does not support backward.")
    # end def _backward

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        pass
    # end def _simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]:
        pass
    # end def _canonicalize

    # endregion PRIVATE

# end class Abs


operator_registry.register(Add)
operator_registry.register(Sub)
operator_registry.register(Mul)
operator_registry.register(Div)
operator_registry.register(Pow)
operator_registry.register(Exp)
operator_registry.register(Exp2)
operator_registry.register(Expm1)
operator_registry.register(Log)
operator_registry.register(Log1p)
operator_registry.register(Sqrt)
operator_registry.register(Square)
operator_registry.register(Cbrt)
operator_registry.register(Reciprocal)
operator_registry.register(Log2)
operator_registry.register(Log10)
operator_registry.register(Deg2rad)
operator_registry.register(Rad2deg)
operator_registry.register(Absolute)
operator_registry.register(Abs)
operator_registry.register(Neg)
