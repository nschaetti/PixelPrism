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


from __future__ import annotations


# Backward-compatible typing facade.
#
# Keep historical `pixelprism.math.typing` import surface while definitions
# live in focused modules:
# - `typing_data`: scalar/tensor/index aliases
# - `typing_rules`: simplification rules and options
# - `typing_expr`: expression/operator protocols and related structures
from .typing_data import (
    ScalarLike,
    ScalarListLike,
    TensorLike,
    Index,
)
from .typing_rules import (
    SimplifyRule,
    SimplifyOptions,
    SimplifyRuleType,
    RuleSpec,
)
from .typing_expr import (
    ExprPattern,
    NodePattern,
    VariablePattern,
    ConstantPattern,
    AnyPattern,
    p_node,
    p_var,
    p_const,
    p_any,
    node_match,
    MatchResult,
    MathExpr,
    Operand,
    Operands,
    Operator,
    LeafKind,
    OpAssociativity,
    OpSimplifyResult,
    OpConstruct,
    AlgebraicExpr,
    ExprLike,
    ExprDomain,
    ExprKind,
    OperatorSpec,
    AritySpec,
)


__all__ = [
    "ScalarLike",
    "ScalarListLike",
    "Index",
    "ExprPattern",
    "NodePattern",
    "VariablePattern",
    "ConstantPattern",
    "AnyPattern",
    "p_node",
    "p_var",
    "p_const",
    "p_any",
    "node_match",
    "MathExpr",
    "TensorLike",
    "Operand",
    "Operands",
    "Operator",
    "SimplifyRule",
    "SimplifyOptions",
    "SimplifyRuleType",
    "RuleSpec",
    "LeafKind",
    "OpAssociativity",
    "OpSimplifyResult",
    "OpConstruct",
    "AlgebraicExpr",
    "ExprLike",
    "ExprDomain",
    "ExprKind",
    "OperatorSpec",
    "AritySpec"
]


# Keep stable module identity for legacy introspection/pickle expectations.
for _name in (
    "ExprPattern",
    "NodePattern",
    "VariablePattern",
    "ConstantPattern",
    "AnyPattern",
    "MatchResult",
    "SimplifyRule",
    "SimplifyRuleType",
    "SimplifyOptions",
    "RuleSpec",
    "LeafKind",
    "OpAssociativity",
    "OpSimplifyResult",
    "OpConstruct",
    "MathExpr",
    "Operator",
    "AlgebraicExpr",
):
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
    # end if
# end for
