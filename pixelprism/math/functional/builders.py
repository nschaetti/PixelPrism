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

# Imports
from typing import Sequence, Union, Optional

from ..math_expr import MathNode
from ..build import as_expr
from ..shape import Shape
from .helpers import apply_operator


__all__ = [
    "build_tensor",
]


def build_tensor(
        elements: Sequence[Union[MathNode, int, float]],
        shape: Optional[Union[Sequence[int], Shape]] = None,
) -> MathNode:
    """
    Assemble a tensor from scalar math expressions.

    Parameters
    ----------
    elements:
        Flat sequence of scalar expressions, Python numbers, or NumPy scalars.
    shape:
        Optional target shape. When omitted the operator returns a 1-D tensor
        whose length matches ``len(elements)``.
    """
    operands = tuple(as_expr(o) for o in elements)
    tensor_shape = Shape.create(shape) if shape is not None else None
    element_names = ", ".join(op.name for op in operands)
    display_shape = f" @ {tensor_shape.dims}" if tensor_shape is not None else ""
    return apply_operator(
        op_name="build_tensor",
        operands=operands,
        display_name=f"tensor([{element_names}]){display_shape}",
        input_shape=tensor_shape
    )
# end def build_tensor

