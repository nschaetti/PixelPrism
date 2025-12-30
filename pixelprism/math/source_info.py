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
"""Source provenance representation."""

from dataclasses import dataclass
from typing import Optional

__all__ = ["SourceInfo"]


@dataclass(frozen=True)
class SourceInfo:
    """
    Provenance for a node.

    Attributes
    ----------
    file : str | None
        Path-like location. May be None for synthetic nodes.
    line : int | None
        1-based line number if available.
    module : str | None
        Importable module path if available.
    """

    file: Optional[str] = None
    line: Optional[int] = None
    module: Optional[str] = None

