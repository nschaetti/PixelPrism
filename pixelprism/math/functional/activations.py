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

from ..build import as_expr
from ..math_node import MathNode
from .helpers import apply_operator

__all__ = [
    "relu",
    "leaky_relu",
    "sigmoid",
    "softplus",
    "gelu",
]


def relu(tensor: MathNode) -> MathNode:
    tensor = as_expr(tensor)
    return apply_operator(
        op_name="relu",
        operands=(tensor,),
        display_name=f"relu({tensor.name})",
    )
# end def relu


def leaky_relu(tensor: MathNode, alpha: float = 0.01) -> MathNode:
    tensor = as_expr(tensor)
    return apply_operator(
        op_name="leaky_relu",
        operands=(tensor,),
        display_name=f"leaky_relu({tensor.name})",
        alpha=float(alpha),
    )
# end def leaky_relu


def sigmoid(tensor: MathNode) -> MathNode:
    tensor = as_expr(tensor)
    return apply_operator(
        op_name="sigmoid",
        operands=(tensor,),
        display_name=f"sigmoid({tensor.name})",
    )
# end def sigmoid


def softplus(tensor: MathNode) -> MathNode:
    tensor = as_expr(tensor)
    return apply_operator(
        op_name="softplus",
        operands=(tensor,),
        display_name=f"softplus({tensor.name})",
    )
# end def softplus


def gelu(tensor: MathNode) -> MathNode:
    tensor = as_expr(tensor)
    return apply_operator(
        op_name="gelu",
        operands=(tensor,),
        display_name=f"gelu({tensor.name})",
    )
# end def gelu
