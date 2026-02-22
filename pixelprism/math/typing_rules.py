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

from dataclasses import dataclass
from enum import Enum, auto
from typing import FrozenSet


__all__ = [
    "SimplifyRule",
    "SimplifyOptions",
    "SimplifyRuleType",
    "RuleSpec",
]


# Symbolic rewrite rules available to `simplify`.
# - Algebraic identities reduce expression size (e.g. x + 0 -> x).
# - Canonicalization rules normalize a tree form for stable comparisons.
class SimplifyRule(Enum):
    # Operator agnostic rules
    MERGE_CONSTANTS = auto()  # a + b -> c, a * b -> c, etc

    SUM_CONSTANTS = auto()
    CONSTANT_FIRST = auto()

    # Add (+) rules
    ADD_FLATTEN = auto()        # (a+b)+c -> a+b+c
    ADD_GROUP_ALIKE = auto()    # a * x + b * x -> (a+b)*x
    ADD_REMOVE_ZEROS = auto()   # x + 0 -> x ; x + -0 -> x

    # Sub (-) rules
    SUB_FLATTEN = auto()
    SUB_FIRST_ZERO = auto()
    SUB_GROUP_ALIKE = auto()
    SUB_REMOVE_ZEROS = auto()

    # Mul (*) rules
    MUL_ONE = auto()            # x * 1 -> x ; 1 * x -> x
    MUL_ZERO = auto()           # x * 0 -> 0 ; 0 * x -> 0
    MUL_BY_CONSTANT = auto()    # x * c -> c * x
    MUL_ITSELF = auto()         # x * x -> x^2
    MUL_NEG = auto()            # x * -y -> -x * y
    MUL_BY_NEG_ONE = auto()     # x * -1 -> -x
    MUL_BY_INV = auto()         # x * 1/y -> x/y
    MUL_BY_INV_NEG = auto()     # x * -1/y -> -x/y
    MUL_BY_NEG_ITSELF = auto()  # x * -x -> -x^2

    DIV_ONE = auto()           # x / 1 -> x
    ZERO_DIV = auto()          # 0 / x -> 0

    # Remove double negation
    NEGATE_NEGATE = auto()

    # Constant folding
    CONST_FOLD = auto()        # combine constant-only subexpressions

    # Canonicalization (normal form)
    FLATTEN_ASSOC = auto()     # (a+b)+c -> a+b+c ; (a*b)*c -> a*b*c
    SORT_COMMUTATIVE = auto()  # stable operand ordering for + and *
# end class SimplifyRule


class SimplifyRuleType(Enum):
    SIMPLIFICATION = auto()
    CANONICALIZATION = auto()
    BOTH = auto()
# end class SimplifyRuleType


# Rule selection for one simplify pass.
# - enabled=None means "all rules enabled by default".
# - disabled always takes precedence over enabled.
@dataclass(frozen=True)
class SimplifyOptions:
    enabled: FrozenSet[SimplifyRule] | None = None
    disabled: FrozenSet[SimplifyRule] = frozenset()
# end class SimplifyOptions


@dataclass(frozen=True)
class RuleSpec:
    """
    Specify a rewrite rule for `simplify`.
    """
    flag: SimplifyRule
    rule_type: SimplifyRuleType
    priority: int = 100
# end class RuleSpec
