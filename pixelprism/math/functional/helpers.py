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

from pixelprism.math import MathExpr, tensor, operator_registry


Operands = tuple[MathExpr, ...]


def apply_operator(
        op_name: str,
        operands: Operands,
        display_name: str,
        **kwargs
) -> MathExpr:
    """
    Build a MathExpr by applying a registered operator to operands.

    Parameters
    ----------
    op_name : str
        Name of the operator registered in :class:`OperatorRegistry`.
    operands : tuple[MathExpr, ...]
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
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math import operator_registry
    >>> a = tensor("a", 1.0)
    >>> b = tensor("b", 2.0)
    >>> expr = apply_operator("add", (a, b), "a + b")
    >>> expr.eval()
    array(3.)
    """
    # Get operator class
    op_cls = operator_registry.get(op_name)

    # We check that operator arity is respected
    if not op_cls.check_arity(operands):
        raise TypeError(
            f"Operator {op_cls.NAME}({op_cls.ARITY}) expected {op_cls.ARITY} operands, "
            f"got {len(operands)}"
        )
    # end if

    # Instantiate operator
    op = op_cls(**kwargs)

    # We check that shapes of the operands are compatible
    if not op.check_shapes(operands):
        shapes = ", ".join(str(o.shape) for o in operands)
        raise TypeError(
            f"Incompatible shapes for operator {op_cls.NAME}: {shapes}"
        )
    # end if

    # We check that the operator approves the operand(s)
    if not op.check_operands(operands):
        raise ValueError(f"Invalid parameters for operator {op.name}: {kwargs}")
    # end if

    return MathExpr(
        name=display_name,
        op=op,
        children=operands,
        dtype=op.infer_dtype(operands),
        shape=op.infer_shape(operands),
    )
# end def apply_operator
