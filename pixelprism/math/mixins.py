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
TODO: Add documentation.
"""

# Imports
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, List


__all__ = [
    "EvaluableMixin",
    "DifferentiableMixin",
    "PredicateMixin",
]


class EvaluableMixin(ABC):
    """
    TODO: Add documentation.
    """

    @abstractmethod
    def eval(self) -> 'Tensor':
        """
        Evaluate this expression in the active math context.

        Returns
        -------
        Tensor
            Result of executing ``self.op`` with evaluated children.
        """
    # end def eval

# end class EvaluableMixin


class DifferentiableMixin(ABC):
    """
    TODO: Add documentation.
    """

    @abstractmethod
    def diff(self, wrt: "Variable") -> 'MathBase':
        """
        Compute the derivative of this expression with respect to ``wrt``.

        Parameters
        ----------
        wrt : Variable
            The variable to differentiate with respect to.

        Returns
        -------
        MathBase
            The derivative of ``self`` with respect to ``wrt``.
        """
    # end def diff

# end class Differentiable


class PredicateMixin(ABC):
    """
    TODO: Add documentation.
    """

    @abstractmethod
    def variables(self) -> List['Variable']:
        """
        Enumerate variable leaves reachable from this node.

        Returns
        -------
        list['MathNode']
            List of :class:`Variable` instances (duplicates possible).
        """
    # end def variables

    def constants(self) -> List['Constant']:
        """
        Enumerate constant leaves reachable from this node.

        Returns
        -------
        list['MathNode']
            List of :class:`Constant` instances (duplicates possible).
        """
    # end def constants

    @abstractmethod
    def contains(
            self,
            leaf: Union[str, 'PredicateMixin'],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        """
        Test whether ``var`` appears in the expression tree.

        Parameters
        ----------
        leaf : str or MathNode
            Reference searched for within the tree.  Strings are matched
            against node names; ``MathExpr`` instances are matched either by
            identity or by their ``name``.
        by_ref : bool, default False
            When ``True`` the search compares identities instead of names.
        check_operator : bool, default True
            When ``True`` the search also queries the operator to determine if
            it captures ``var`` internally.
        look_for : Optional[str], default None
            Can be None, "var", or "const"

        Returns
        -------
        bool
            ``True`` when ``var`` was located in ``self`` or any child.
        """

    # end def contains

    @abstractmethod
    def contains_variable(
            self,
            variable: Union[str, 'Variable'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a variable `variable`"""

    # end def contains_variable

    @abstractmethod
    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a constant `constant`"""

    # end def contains_constant

    @abstractmethod
    def replace(self, old_m: 'PredicateMixin', new_m: 'PredicateMixin'):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
    # end def replace

    @abstractmethod
    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
    # end rename

    @abstractmethod
    def is_constant(self) -> bool:
        """Does the expression contain only constant values?"""
    # end def is_constant

    @abstractmethod
    def is_variable(self) -> bool:
        """Does the expression contain a variable?"""
    # end def is_variable

# end class PredicateMixin

