"""
Pixel Prism Animate - Animation Framework
=======================================

This subpackage provides a comprehensive framework for creating and managing animations
in the Pixel Prism library. It includes classes for interpolation, transitions, and
animation effects that can be applied to various objects.

Main Components
--------------
- Interpolators: Classes for interpolating values between keyframes
  - :class:`~pixelprism.animate.interpolate.Interpolator`: Base class for interpolators
  - :class:`~pixelprism.animate.interpolate.LinearInterpolator`: Linear interpolation
  - :class:`~pixelprism.animate.interpolate.EaseInOutInterpolator`: Smooth easing interpolation

- Animation Mixins: Interfaces for objects that can be animated
  - :class:`~pixelprism.animate.fade.FadeableMixin`: Interface for objects that can fade in/out
  - :class:`~pixelprism.animate.range.RangeableMixin`: Interface for objects with range animations
  - :class:`~pixelprism.animate.move.MovableMixin`: Interface for objects that can be moved
  - :class:`~pixelprism.animate.changes.CallableMixin`: Interface for objects that can trigger callbacks

- Animation Types: Specific animation implementations
  - :class:`~pixelprism.animate.animate.Animate`: Base class for animations
  - :class:`~pixelprism.animate.move.Move`: Animation for moving objects
  - :class:`~pixelprism.animate.animate.Scale`: Animation for scaling objects
  - :class:`~pixelprism.animate.animate.Rotate`: Animation for rotating objects
  - :class:`~pixelprism.animate.fade.FadeIn`: Animation for fading in objects
  - :class:`~pixelprism.animate.fade.FadeOut`: Animation for fading out objects
  - :class:`~pixelprism.animate.range.Range`: Animation for range-based animations
  - :class:`~pixelprism.animate.changes.Call`: Animation for triggering callbacks
  - :class:`~pixelprism.animate.animate.Build`: Animation for building objects
  - :class:`~pixelprism.animate.animate.Destroy`: Animation for destroying objects

- Animation Management:
  - :class:`~pixelprism.animate.animator.Animator`: Class for managing multiple animations

- Decorators:
  - :func:`~pixelprism.animate.decorators.animeattr`: Decorator for marking attributes as animatable
"""


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
