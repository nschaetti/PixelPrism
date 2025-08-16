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

# Imports
from typing import Any, Iterable, Mapping, Optional, Tuple, FrozenSet, Protocol


class Op(Protocol):
    """
    Operator descriptor protocol (stateless, reusable).

    MathExpr holds a reference to an Op when it represents an operator
    application (i.e., an OpNode). Implementations live in
    `pixelprism.math.ops`.

    Only the surface necessary for typing here is declared; engines and
    autodiff use richer interfaces in their own modules.
    """

    @property
    def name(self) -> str: ...

    @property
    def arity(self) -> int: ...

# end Op
