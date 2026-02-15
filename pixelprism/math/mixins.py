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
from typing import Optional, Union, Dict, List, TYPE_CHECKING, Mapping, cast

from .typing import LeafKind, MathExpr


if TYPE_CHECKING:
    from .tensor import Tensor
    from .math_base import MathBase
    from .math_leaves import Variable, Constant
# end if


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
        List variable leaves reachable from this node.

        Returns
        -------
        list['MathNode']
            List of :class:`Variable` instances (duplicates possible).
        """
    # end def variables

    @abstractmethod
    def constants(self) -> List['Constant']:
        """
        List constant leaves reachable from this node.

        Returns
        -------
        list['MathNode']
            List of :class:`Constant` instances (duplicates possible).
        """
    # end def constants

    @abstractmethod
    def contains(
            self,
            leaf: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
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
        look_for : LeafKind, default LeafKind.ANY
            Restrict lookup to variables/constants, or keep full lookup.

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
        """
        Return True if the expression contains a variable `variable`
        """

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

    def substitute(
            self,
            mapping: Mapping[MathExpr, MathExpr],
            *,
            by_ref: bool = True
    ) -> MathExpr:
        """Return an expression with substitutions applied.

        Parameters
        ----------
        mapping : Mapping[MathExpr, MathExpr]
            Mapping from old subexpressions to replacement expressions.
        by_ref : bool, default True
            ``True`` for identity matching, ``False`` for symbolic matching.
        """
        if type(self).replace is not PredicateMixin.replace:
            if not by_ref:
                raise NotImplementedError(
                    "Legacy replace(...) bridge supports only by_ref=True."
                )
            # end if
            for old_m, new_m in mapping.items():
                self.replace(old_m, new_m)
            # end for
            return cast(MathExpr, cast(object, self))
        # end if

        raise NotImplementedError("substitute(...) must be implemented by subclasses.")
    # end def substitute

    def renamed(self, old_name: str, new_name: str) -> MathExpr:
        """Return an expression where ``old_name`` is replaced by ``new_name``.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        if type(self).rename is not PredicateMixin.rename:
            self.rename(old_name, new_name)
            return cast(MathExpr, cast(object, self))
        # end if

        raise NotImplementedError("renamed(...) must be implemented by subclasses.")
    # end def renamed

    def replace(self, old_m: MathExpr, new_m: MathExpr):
        """Compatibility shim for mutable replacement APIs."""
        return self.substitute({old_m: new_m}, by_ref=True)
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Compatibility shim for mutable rename APIs."""
        self.renamed(old_name, new_name)
        return {old_name: new_name}
    # end def rename

    @abstractmethod
    def is_constant(self) -> bool:
        """Does the expression contain only constant values?"""
    # end def is_constant

    @abstractmethod
    def is_variable(self) -> bool:
        """Does the expression contain a variable?"""
    # end def is_variable

# end class PredicateMixin
