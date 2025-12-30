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
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Symbolic variable expression."""

# Imports
from __future__ import annotations
from typing import Any, Dict, Mapping
from .math_expr import MathExpr
from .shape import Shape
from .value import Value


__all__ = ["Var"]


class Var(MathExpr):
    """Symbolic variable that resolves to a runtime Value."""

    def __init__(
            self,
            name: str,
            shape: Shape,
            dtype: Any
    ):
        """Initialize a Var.

        Args:
            name: Display name of the variable.
            shape: Variable shape.
            dtype: Runtime dtype metadata.
        """
        super().__init__(shape, dtype, children=())
        self._name = name
    # end def __init__

    @property
    def name(self) -> str:
        """Return the variable's display name.

        Returns:
            str: Variable name.
        """
        return self._name
    # end def name

    def evaluate(
            self,
            env: Mapping["Var", Value]
    ) -> Value:
        """Evaluate the variable using the provided environment.

        Args:
            env: Mapping from Var to Value.

        Returns:
            Value: Runtime value bound to the variable.

        Raises:
            KeyError: If the variable is missing from the environment.
            TypeError: If the mapped object is not a Value.
        """
        if self not in env:
            raise KeyError(f"Value for variable '{self._name}' is missing from the environment.")
        # end if
        value = env[self]
        if not isinstance(value, Value):
            raise TypeError("Evaluation environment must map variables to Value instances.")
        # end if
        return value
    # end def evaluate

    def _graph_params(self) -> Dict[str, Any]:
        """Return graph parameters describing the variable.

        Returns:
            Dict[str, Any]: Node metadata containing the name.
        """
        return {"name": self._name}
    # end def _graph_params
# end class Var
