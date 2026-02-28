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

from functools import wraps
from typing import Any, Callable, Sequence

from .typing_expr import OpSimplifyResult
from .typing_rules import SimplifyRule, RuleSpec, SimplifyRuleType


__all__ = [
    "rule",
    "rule_when",
    "rule_needs_constants",
    "rule_needs_variables",
    "rule_returns_operands",
    "rule_no_op_if_unchanged",
    "rule_finalize_result",
]


def rule(flag: SimplifyRule, rule_type: SimplifyRuleType, priority: int = 100):
    """
    Set a method's rule specification.

    Parameters
    ----------
    flag: SimplifyRule
        The simplification rule flag.
    rule_type: SimplifyRuleType
        The type of rule: simplification, canonicalization, or both.
    priority: int, optional
        The priority of the rule, default is 100.

    Returns
    -------
    callable
        The decorator function.
    """
    def deco(fn):
        fn.__rule_spec__ = RuleSpec(flag, rule_type, priority)
        return fn
    # end def
    return deco
# end def rule


def rule_when(*predicates: Callable[[Any, Sequence[Any]], bool]):
    """
    Apply a rule only when all predicates return ``True``.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(self, operands, *args, **kwargs):
            for predicate in predicates:
                if not predicate(self, operands):
                    return None
                # end if
            # end for
            return fn(self, operands, *args, **kwargs)
        # end def wrapper
        return wrapper
    # end def deco
    return deco
# end def rule_when


def rule_needs_constants(min_count: int = 1):
    """
    Skip rule when fewer than ``min_count`` operands are constant.
    """
    def _predicate(_self, operands: Sequence[Any]) -> bool:
        return sum(1 for op in operands if op.is_constant()) >= min_count
    # end def _predicate
    return rule_when(_predicate)
# end def rule_needs_constants


def rule_needs_variables(min_count: int = 1):
    """
    Skip rule when fewer than ``min_count`` operands are variable-containing.
    """
    def _predicate(_self, operands: Sequence[Any]) -> bool:
        return sum(1 for op in operands if op.is_variable()) >= min_count
    # end def _predicate
    return rule_when(_predicate)
# end def rule_needs_variables


def rule_returns_operands(fn):
    """
    Allow rules to return operands directly instead of ``OpSimplifyResult``.
    """
    @wraps(fn)
    def wrapper(self, operands, *args, **kwargs):
        out = fn(self, operands, *args, **kwargs)
        if out is None:
            return None
        # end if
        if isinstance(out, OpSimplifyResult):
            return out
        # end if
        return OpSimplifyResult(operands=list(out), replacement=None)
    # end def wrapper
    return wrapper
# end def rule_returns_operands


def rule_no_op_if_unchanged(fn):
    """
    Convert unchanged outputs to ``None`` (rule not applied).
    """
    @wraps(fn)
    def wrapper(self, operands, *args, **kwargs):
        out = fn(self, operands, *args, **kwargs)
        if out is None:
            return None
        # end if
        if out.replacement is not None:
            return out
        # end if
        if out.operands is None:
            return out
        # end if

        old_operands = list(operands)
        new_operands = list(out.operands)
        unchanged = (
            len(old_operands) == len(new_operands)
            and all(old is new for old, new in zip(old_operands, new_operands))
        )
        if unchanged:
            return None
        # end if
        return OpSimplifyResult(operands=new_operands, replacement=None)
    # end def wrapper
    return wrapper
# end def rule_no_op_if_unchanged


def rule_finalize_result(
        *,
        empty: Any | Callable[[Any, Sequence[Any]], Any] | None = None,
        collapse_single: bool = True,
        min_out: int | None = None,
):
    """
    Finalize a rule output by enforcing cardinality and replacement policy.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(self, operands, *args, **kwargs):
            out = fn(self, operands, *args, **kwargs)
            if out is None:
                return None
            # end if
            if out.replacement is not None:
                return out
            # end if
            if out.operands is None:
                return out
            # end if

            new_operands = list(out.operands)

            if min_out is not None and len(new_operands) < min_out:
                raise RuntimeError(
                    f"Rule produced too few operands: {len(new_operands)} < {min_out} "
                    f"for {self.__class__.__name__}."
                )
            # end if

            if len(new_operands) == 0:
                if empty is None:
                    raise RuntimeError(
                        f"Rule produced empty operands for {self.__class__.__name__}."
                    )
                # end if
                replacement = empty(self, operands) if callable(empty) else empty
                return OpSimplifyResult(operands=None, replacement=replacement)
            # end if

            if collapse_single and len(new_operands) == 1:
                return OpSimplifyResult(operands=None, replacement=new_operands[0])
            # end if

            return OpSimplifyResult(operands=new_operands, replacement=None)
        # end def wrapper
        return wrapper
    # end def deco
    return deco
# end def rule_finalize_result
