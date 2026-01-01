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
"""Symbolic dimension representation."""

# Imports
from dataclasses import dataclass


__all__ = ["SymbolicDim"]


@dataclass(frozen=True)
class SymbolicDim:
    """
    A symbolic dimension (e.g., N, M) usable in shapes.

    Example
    -------
    >>> N = SymbolicDim("N", 2)
    >>> N.name
    'N'
    >>> N.size
    2
    """

    # Name of the dimension
    name: str

    # Size of the dimension
    size: int

# end class SymbolicDim

