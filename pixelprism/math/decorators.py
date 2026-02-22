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


from .typing_rules import SimplifyRule, RuleSpec, SimplifyRuleType


__all__ = [
    "rule",
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
