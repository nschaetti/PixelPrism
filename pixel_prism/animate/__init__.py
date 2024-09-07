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

#
# Description: This file is used to import all classes from the following modules
#


# Interpolation
from .interpolate import (
    Interpolator,
    LinearInterpolator,
    EaseInOutInterpolator
)

# Transition animation
from .animate import (
    Animate,
    Scale,
    Rotate,
    Build,
    Destroy
)

from .animator import (
    Animator
)

# Build

# Changes
from .changes import (
    CallableMixin,
    Call
)

# Decorators
from .decorators import (
    animeclass,
    animeattr
)

# Fade
from .fade import (
    FadeableMixin,
    FadeIn,
    FadeOut
)

# Move
from .move import (
    MovableMixin,
    Move
)

# Range
from .range import (
    Range,
    RangeableMixin
)


# ALL
__all__ = [
    # Interpolate
    "Interpolator",
    "LinearInterpolator",
    "EaseInOutInterpolator",
    # Mixin
    "FadeableMixin",
    "RangeableMixin",
    "MovableMixin",
    "CallableMixin",
    # Decorators
    "animeclass",
    "animeattr",
    # Animate
    "Animate",
    "Animator",
    "Move",
    "Scale",
    "Rotate",
    "FadeIn",
    "FadeOut",
    "Range",
    "Call",
    "Build",
    "Destroy"
]

