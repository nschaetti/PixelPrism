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

from typing import Sequence
from ..operators.base import operator_registry
from ..math_node import MathNode
from ..typing import MathExpr, Operator, Operand


def apply_operator(
        op_name: str,
        operands: Sequence["Operand"],
        display_name: str,
        **kwargs
) -> MathExpr:
    """
    Build a MathNode by applying a registered operator to operands.

    Parameters
    ----------
    op_name : str
        Name of the operator registered in :class:`OperatorRegistry`.
    operands : tuple[MathNode, ...]
        Operands to apply the operator to.
    display_name : str
        Human-readable name assigned to the resulting expression.
    kwargs: Dict[str, Any]
        Operator-specific keyword arguments.

    Returns
    -------
    MathExpr
        A new operator node wrapping the operands.

    Raises
    ------
    KeyError
        If `op_name` is not registered.
    TypeError
        If the number of operands does not match the operator arity.
    ValueError
        If operand shapes are incompatible for the operator.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math import operator_registry
    >>> a = pm.const("a", 1.0)
    >>> b = pm.const("b", 2.0)
    >>> expr = apply_operator("add", (a, b), "a + b")
    >>> expr.eval()
    array(3.)
    """
    # Get operator class
    op_cls = operator_registry.get(op_name)

    # Instantiate operator
    op_result = op_cls.construct(operands=operands, **kwargs)

    if isinstance(op_result.expr, MathExpr):
        return op_result.expr
    elif isinstance(op_result.expr, Operator):
        op = op_result.expr
        operands = op_result.operands
        return MathNode(
            name=display_name,
            op=op,
            children=operands,
            dtype=op.infer_dtype(operands),
            shape=op.infer_shape(operands),
        )
    else:
        raise TypeError(f"Unexpected operator type: {type(op_result.expr)}")
    # end if
# end def apply_operator
