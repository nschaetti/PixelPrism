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


from pixelprism.math.math_expr import MathExpr
from pixelprism.math.operators import operator_registry


__all__ = [
    "add",
    "sub",
    "mul",
    "div"
]


def add(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    op = operator_registry.get("add")
    operands = (op1, op2)
    if not op.check_arity(operands):
        raise TypeError(f"Operator {op.name}({op.arity}) expected {op.arity} operands, got {len(operands)}")
    # end if
    if not op.check_shapes(operands):
        raise TypeError(f"Incompatible shape for operator {op.name}, got {op1.shape} + {op2.shape}")
    # end if
    node_shape = op.infer_shape(operands)
    node_dtype = op.infer_dtype(operands)
    return MathExpr(
        name=f"{op1.name} + {op2.name}",
        op=op,
        children=operands,
        dtype=node_dtype,
        shape=node_shape
    )
# end def add


def sub(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """Substraction operator
    """
    op = operator_registry.get("sub")
    operands = (op1, op2)
    if not op.check_arity(operands):
        raise TypeError(
            f"Operator {op.name}({op.arity}) expected {op.arity} operands, got {len(operands)}"
        )
    # end if
    if not op.check_shapes(operands):
        raise TypeError(f"Incompatible shape for operator {op.name}, got {op1.shape} - {op2.shape}")
    # end if
    node_shape = op.infer_shape(operands)
    node_dtype = op.infer_dtype(operands)
    return MathExpr(
        name=f"{op1.name} - {op2.name}",
        op=op,
        children=operands,
        dtype=node_dtype,
        shape=node_shape
    )
# end def sub


def mul(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    op = operator_registry.get("mul")
    operands = (op1, op2)
    if not op.check_arity(operands):
        raise TypeError(
            f"Operator {op.name}({op.arity}) expected {op.arity} operands, got {len(operands)}"
        )
    # end if
    if not op.check_shapes(operands):
        raise TypeError(f"Incompatible shape for operator {op.name}, got {op1.shape} * {op2.shape}")
    # end if
    node_shape = op.infer_shape(operands)
    node_dtype = op.infer_dtype(operands)
    return MathExpr(
        name=f"{op1.name} * {op2.name}",
        op=op,
        children=operands,
        dtype=node_dtype,
        shape=node_shape
    )
# end def mul


def div(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    op = operator_registry.get("div")
    operands = (op1, op2)
    if not op.check_arity(operands):
        raise TypeError(
            f"Operator {op.name}({op.arity}) expected {op.arity} operands, got {len(operands)}"
        )
    # end if
    if not op.check_shapes(operands):
        raise TypeError(f"Incompatible shape for operator {op.name}, got {op1.shape} / {op2.shape}")
    # end if
    node_shape = op.infer_shape(operands)
    node_dtype = op.infer_dtype(operands)
    return MathExpr(
        name=f"{op1.name} / {op2.name}",
        op=op,
        children=operands,
        dtype=node_dtype,
        shape=node_shape
    )
# end def div


