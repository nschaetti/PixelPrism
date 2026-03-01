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
# Copyright (C) 2026 Pixel Prism
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

"""Easing and interpolation helpers for timeline animation.

This module contains:

- scalar easing callables (`u in [0, 1] -> v in [0, 1]`),
- easing factories for parameterized curves,
- a generic interpolation helper to combine `lerp` and easing.
"""

# Imports
from __future__ import annotations
import math
from typing import Callable, TypeVar, Protocol, Any, Dict, List, NamedTuple

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class EasingFn(Protocol):
    """Protocol for easing functions."""
    def __call__(self, t: float, **kwargs: Any) -> float: ...
# end class EasingFn


class InterpolateFn(Protocol[T_in, T_out]):
    def __call__(
            self,
            a: T_in,
            b: T_in,
            t: float,
            easing: EasingFn,
            **kwargs: Any,
    ) -> T_out: ...
# end class InterpolateFn


Interpolation = Callable[[float], float]


def _clamp01(u: float) -> float:
    """Clamp a scalar to the ``[0, 1]`` interval.

    Parameters
    ----------
    u : float
        Input progression value.

    Returns
    -------
    float
        Clamped progression value.
    """
    if u <= 0.0:
        return 0.0
    # end if
    if u >= 1.0:
        return 1.0
    # end if
    return u
# end def _clamp01


# region EASING FUNCTIONS


def linear(t: float) -> float:
    """Compute linear easing.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Clamped progression without curve distortion.
    """
    return _clamp01(t)
# end def linear


def ease_in_quad(t: float) -> float:
    """Compute quadratic ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression favoring slower starts.
    """
    t = _clamp01(t)
    return t * t
# end def ease_in_quad


def ease_out_quad(t: float) -> float:
    """Compute quadratic ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression favoring slower ends.
    """
    t = _clamp01(t)
    return 1.0 - (1.0 - t) * (1.0 - t)
# end def ease_out_quad


def ease_in_out_quad(t: float) -> float:
    """Compute quadratic ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression.
    """
    t = _clamp01(t)
    if t < 0.5:
        return 2.0 * t * t
    # end if
    return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
# end def ease_in_out_quad


def ease_in_cubic(t: float) -> float:
    """Compute cubic ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with stronger acceleration.
    """
    t = _clamp01(t)
    return t * t * t
# end def ease_in_cubic


def ease_out_cubic(t: float) -> float:
    """Compute cubic ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with stronger deceleration.
    """
    t = _clamp01(t)
    return 1.0 - (1.0 - t) ** 3
# end def ease_out_cubic


def ease_in_out_cubic(t: float) -> float:
    """Compute cubic ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression with cubic profile.
    """
    t = _clamp01(t)
    if t < 0.5:
        return 4.0 * t * t * t
    # end if
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0
# end def ease_in_out_cubic


def ease_in_quart(t: float) -> float:
    """Compute quartic ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with high acceleration.
    """
    t = _clamp01(t)
    return t ** 4
# end def ease_in_quart


def ease_out_quart(t: float) -> float:
    """Compute quartic ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with high deceleration.
    """
    t = _clamp01(t)
    return 1.0 - (1.0 - t) ** 4
# end def ease_out_quart


def ease_in_out_quart(t: float) -> float:
    """Compute quartic ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression with quartic profile.
    """
    t = _clamp01(t)
    if t < 0.5:
        return 8.0 * (t ** 4)
    # end if
    return 1.0 - ((-2.0 * t + 2.0) ** 4) / 2.0
# end def ease_in_out_quart


def ease_in_quint(t: float) -> float:
    """Compute quintic ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with very strong acceleration.
    """
    t = _clamp01(t)
    return t ** 5
# end def ease_in_quint


def ease_out_quint(t: float) -> float:
    """Compute quintic ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with very strong deceleration.
    """
    t = _clamp01(t)
    return 1.0 - (1.0 - t) ** 5
# end def ease_out_quint


def ease_in_out_quint(t: float) -> float:
    """Compute quintic ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression with quintic profile.
    """
    t = _clamp01(t)
    if t < 0.5:
        return 16.0 * (t ** 5)
    # end if
    return 1.0 - ((-2.0 * t + 2.0) ** 5) / 2.0
# end def ease_in_out_quint


def ease_in_sine(t: float) -> float:
    """Compute sine ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression based on a cosine phase.
    """
    t = _clamp01(t)
    return 1.0 - math.cos((t * math.pi) / 2.0)
# end def ease_in_sine


def ease_out_sine(t: float) -> float:
    """Compute sine ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression based on a sine phase.
    """
    t = _clamp01(t)
    return math.sin((t * math.pi) / 2.0)
# end def ease_out_sine


def ease_in_out_sine(t: float) -> float:
    """Compute sine ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Smooth symmetric trigonometric easing.
    """
    t = _clamp01(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0
# end def ease_in_out_sine


def ease_in_expo(t: float) -> float:
    """Compute exponential ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with exponential growth.
    """
    t = _clamp01(t)
    if t == 0.0:
        return 0.0
    # end if
    return 2.0 ** (10.0 * t - 10.0)
# end def ease_in_expo


def ease_out_expo(t: float) -> float:
    """Compute exponential ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with exponential decay.
    """
    t = _clamp01(t)
    if t == 1.0:
        return 1.0
    # end if
    return 1.0 - (2.0 ** (-10.0 * t))
# end def ease_out_expo


def ease_in_out_expo(t: float) -> float:
    """Compute exponential ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric exponential easing.
    """
    t = _clamp01(t)
    if t == 0.0:
        return 0.0
    # end if
    if t == 1.0:
        return 1.0
    # end if
    if t < 0.5:
        return (2.0 ** (20.0 * t - 10.0)) / 2.0
    # end if
    return (2.0 - (2.0 ** (-20.0 * t + 10.0))) / 2.0
# end def ease_in_out_expo


def ease_in_circ(t: float) -> float:
    """Compute circular ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression following a circular arc.
    """
    t = _clamp01(t)
    return 1.0 - math.sqrt(1.0 - t * t)
# end def ease_in_circ


def ease_out_circ(t: float) -> float:
    """Compute circular ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression decelerating along a circular arc.
    """
    t = _clamp01(t)
    return math.sqrt(1.0 - (t - 1.0) ** 2)
# end def ease_out_circ


def ease_in_out_circ(t: float) -> float:
    """Compute circular ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric circular easing.
    """
    t = _clamp01(t)
    if t < 0.5:
        return (1.0 - math.sqrt(1.0 - (2.0 * t) ** 2)) / 2.0
    # end if
    return (math.sqrt(1.0 - (-2.0 * t + 2.0) ** 2) + 1.0) / 2.0
# end def ease_in_out_circ


def ease_in_back(t: float, overshoot: float = 1.70158) -> float:
    """Compute back ease-in with default overshoot.

    Parameters
    ----------
    t : float
        Normalized progression.
    overshoot :
    float, default=1.70158

    Returns
    -------
    float
        Eased progression with initial negative overshoot.
    """
    u2 = _clamp01(t)
    c3 = overshoot + 1.0
    return c3 * u2 * u2 * u2 - overshoot * u2 * u2
# end def ease_in_back


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Compute back ease-out with default overshoot.

    Parameters
    ----------
    t : float
        Normalized progression.
    overshoot :
        float, default=1.70158

    Returns
    -------
    float
        Eased progression with final positive overshoot.
    """
    u2 = _clamp01(t)
    c3 = overshoot + 1.0
    return 1.0 + c3 * ((u2 - 1.0) ** 3) + overshoot * ((u2 - 1.0) ** 2)
# end def ease_out_back


def ease_in_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Compute back ease-in-out with default overshoot.

    Parameters
    ----------
    t : float
        Normalized progression.
    overshoot :
        float, default=1.70158

    Returns
    -------
    float
        Symmetric back easing with overshoot around both ends.
    """
    u2 = _clamp01(t)
    c2 = overshoot * 1.525
    if u2 < 0.5:
        return ((2.0 * u2) ** 2 * ((c2 + 1.0) * 2.0 * u2 - c2)) / 2.0
    # end if
    return (((2.0 * u2 - 2.0) ** 2) * ((c2 + 1.0) * (u2 * 2.0 - 2.0) + c2) + 2.0) / 2.0
# end def ease_in_out_back


def _ease_out_bounce_core(u: float) -> float:
    """Compute the piecewise bounce-out primitive.

    Parameters
    ----------
    u : float
        Assumed normalized progression.

    Returns
    -------
    float
        Bounce-out progression.
    """
    n1 = 7.5625
    d1 = 2.75
    if u < 1.0 / d1:
        return n1 * u * u
    # end if
    if u < 2.0 / d1:
        u2 = u - (1.5 / d1)
        return n1 * u2 * u2 + 0.75
    # end if
    if u < 2.5 / d1:
        u2 = u - (2.25 / d1)
        return n1 * u2 * u2 + 0.9375
    # end if
    u2 = u - (2.625 / d1)
    return n1 * u2 * u2 + 0.984375
# end def _ease_out_bounce_core


def ease_out_bounce(t: float) -> float:
    """Compute bounce ease-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Progression with decaying bounces near the end.
    """
    return _ease_out_bounce_core(_clamp01(t))
# end def ease_out_bounce


def ease_in_bounce(t: float) -> float:
    """Compute bounce ease-in.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Progression with bounces near the start.
    """
    u2 = _clamp01(t)
    return 1.0 - _ease_out_bounce_core(1.0 - u2)
# end def ease_in_bounce


def ease_in_out_bounce(t: float) -> float:
    """Compute bounce ease-in-out.

    Parameters
    ----------
    t : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric bounce easing.
    """
    u2 = _clamp01(t)
    if u2 < 0.5:
        return (1.0 - _ease_out_bounce_core(1.0 - 2.0 * u2)) / 2.0
    # end if
    return (1.0 + _ease_out_bounce_core(2.0 * u2 - 1.0)) / 2.0
# end def ease_in_out_bounce


def ease_in_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
    """Compute elastic ease-in with default parameters.

    Parameters
    ----------
    t : float
        Normalized progression.
    amplitude : float, default=1.0
    period : float, default=0.3

    Returns
    -------
    float
        Oscillatory progression near the start.
    """
    u2 = _clamp01(t)
    if u2 == 0.0:
        return 0.0
    # end if
    if u2 == 1.0:
        return 1.0
    # end if
    c4 = (2.0 * math.pi) / period
    return -(amplitude * (2.0 ** (10.0 * u2 - 10.0)) * math.sin((u2 * 10.0 - 10.75) * c4))
# end def ease_in_elastic


def ease_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
    """Compute elastic ease-out with default parameters.

    Parameters
    ----------
    t : float
        Normalized progression.
    amplitude : float, default=1.0
        Amplitude of the oscillation.
    period : float, default=0.3
        Period of the oscillation.

    Returns
    -------
    float
        Oscillatory progression near the end.
    """
    u2 = _clamp01(t)
    if u2 == 0.0:
        return 0.0
    # end if
    if u2 == 1.0:
        return 1.0
    # end if
    c4 = (2.0 * math.pi) / period
    return amplitude * (2.0 ** (-10.0 * u2)) * math.sin((u2 * 10.0 - 0.75) * c4) + 1.0
# end def ease_out_elastic


def ease_in_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.45) -> float:
    """Compute elastic ease-in-out with default parameters.

    Parameters
    ----------
    t : float
        Normalized progression.
    amplitude : float, default=1.0
        Amplitude of the oscillation.
    period : float, default=0.45
        Period of the oscillation.

    Returns
    -------
    float
        Symmetric oscillatory easing.
    """
    u2 = _clamp01(t)
    if u2 == 0.0:
        return 0.0
    # end if
    if u2 == 1.0:
        return 1.0
    # end if
    c5 = (2.0 * math.pi) / period
    if u2 < 0.5:
        return -(
                amplitude * (2.0 ** (20.0 * u2 - 10.0)) * math.sin((20.0 * u2 - 11.125) * c5)
        ) / 2.0
    # end if
    return (
            amplitude * (2.0 ** (-20.0 * u2 + 10.0)) * math.sin((20.0 * u2 - 11.125) * c5)
    ) / 2.0 + 1.0
# end def ease_in_out_elastic


def ease_staircase(t: float, steps: int) -> float:
    """Compute step easing.

    Parameters
    ----------
    t: float
        Normalized progression.
    steps: int
        Number of steps.

    Returns
    -------
    float
        Step easing value.
    """
    u2 = _clamp01(t)
    if steps <= 0:
        raise ValueError("steps must be positive")
    # end if
    step_size = 1.0 / steps
    return min(1.0, math.floor(u2 / step_size) * step_size)
# end def ease_staircase


def ease_step(t: float, pos: float) -> float:
    """Compute discrete step easing."""
    u2 = _clamp01(t)
    if t <= pos:
        return 0.0
    # end if
    return 1.0
# end def ease_step


def ease_flat(t: float, top: bool = False) -> float:
    """Compute flat easing."""
    if top:
        return 1.0
    # end if
    return 0.0
# end def ease_flat


# endregion EASING FUNCTIONS


def interpolate(
        start: float,
        end: float,
        t: float,
        easing: EasingFn,
        **kwargs: Any,
) -> float:
    """Interpolate between two values using a specific easing function.

    Parameters
    ----------
    start : float
        Starting value.
    end : float
        Ending value.
    t : float
        Linear progression in ``[0, 1]``.
    easing : EasingFn
        Easing function applied to ``t``.

    Returns
    -------
    float
        Interpolated value.
    """
    v = easing(t, **kwargs)
    return (end - start) * v + start
# end def interpolate


def interpolate_step(
        start: float,
        end: float,
        t: float,
        easing: EasingFn,
        steps: int,
        **kwargs: Any,
) -> float:
    """
    Interpolate between two values using a discrete step easing function.

    Parameters
    ----------
    start: float
        Starting value.
    end: float
        Ending value.
    t : float
        Linear progression in ``[0, 1]``.
    easing : EasingFn
        Easing function applied to ``t``.
    steps : int
        Number of steps in the step function.

    Returns
    -------
    float
        Interpolated value.
    """
    v = easing(t, **kwargs)
    if steps <= 0:
        raise ValueError(
            "steps must be positive, got {}".format(steps)
        )
    # end if
    vi = (end - start) * v
    step_size = (end - start) / steps
    disc = math.floor(vi / step_size) * step_size
    return start + disc
# end def interpolate_step


def interpolate_integer(
        start: int,
        end: int,
        t: float,
        easing: EasingFn,
        **kwargs: Any,
) -> int:
    """
    Interpolate between two integer values using a specific easing function.
    """
    v = easing(t, **kwargs)
    return int(math.floor(start + (end - start) * v))
# end def interpolate_integer


def create_interpolate(
        start: float,
        end: float,
        interpolate_fn: InterpolateFn,
        easing_fn: EasingFn,
        **kwargs: Any,
) -> Interpolation:
    """Create an interpolation function."""
    # return functools.partial(interpolate_fn, start=start, end=end, easing=easing_fn)
    def _interpolate(t: float) -> float:
        return interpolate_fn(start, end, t, easing_fn, **kwargs)
    # end def _interpolate
    return _interpolate
# end def create_interpolate


class Easings(NamedTuple):
    linear: EasingFn = linear
    ease_in_quad: EasingFn = ease_in_quad
    ease_out_quad: EasingFn = ease_out_quad
    ease_in_out_quad: EasingFn = ease_in_out_quad
    ease_in_cubic: EasingFn = ease_in_cubic
    ease_out_cubic: EasingFn = ease_out_cubic
    ease_in_out_cubic: EasingFn = ease_in_out_cubic
    ease_in_quart: EasingFn = ease_in_quart
    ease_out_quart: EasingFn = ease_out_quart
    ease_in_out_quart: EasingFn = ease_in_out_quart
    ease_in_quint: EasingFn = ease_in_quint
    ease_out_quint: EasingFn = ease_out_quint
    ease_in_out_quint: EasingFn = ease_in_out_quint
    ease_in_sine: EasingFn = ease_in_sine
    ease_out_sine: EasingFn = ease_out_sine
    ease_in_out_sine: EasingFn = ease_in_out_sine
    ease_in_expo: EasingFn = ease_in_expo
    ease_out_expo: EasingFn = ease_out_expo
    ease_in_out_expo: EasingFn = ease_in_out_expo
    ease_in_circ: EasingFn = ease_in_circ
    ease_out_circ: EasingFn = ease_out_circ
    ease_in_out_circ: EasingFn = ease_in_out_circ
    ease_in_back: EasingFn = ease_in_back
    ease_out_back: EasingFn = ease_out_back
    ease_in_out_back: EasingFn = ease_in_out_back
    ease_in_elastic: EasingFn = ease_in_elastic
    ease_out_elastic: EasingFn = ease_out_elastic
    ease_in_out_elastic: EasingFn = ease_in_out_elastic
    ease_in_bounce: EasingFn = ease_in_bounce
    ease_out_bounce: EasingFn = ease_out_bounce
    ease_in_out_bounce: EasingFn = ease_in_out_bounce
    ease_staircase: EasingFn = ease_staircase
    step: EasingFn = ease_step
    flat: EasingFn = ease_flat
# end class Easings


EASINGS = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_quart": ease_in_quart,
    "ease_out_quart": ease_out_quart,
    "ease_in_out_quart": ease_in_out_quart,
    "ease_in_quint": ease_in_quint,
    "ease_out_quint": ease_out_quint,
    "ease_in_out_quint": ease_in_out_quint,
    "ease_in_sine": ease_in_sine,
    "ease_out_sine": ease_out_sine,
    "ease_in_out_sine": ease_in_out_sine,
    "ease_in_expo": ease_in_expo,
    "ease_out_expo": ease_out_expo,
    "ease_in_out_expo": ease_in_out_expo,
    "ease_in_circ": ease_in_circ,
    "ease_out_circ": ease_out_circ,
    "ease_in_out_circ": ease_in_out_circ,
    "ease_in_back": ease_in_back,
    "ease_out_back": ease_out_back,
    "ease_in_out_back": ease_in_out_back,
    "ease_in_elastic": ease_in_elastic,
    "ease_out_elastic": ease_out_elastic,
    "ease_in_out_elastic": ease_in_out_elastic,
    "ease_in_bounce": ease_in_bounce,
    "ease_out_bounce": ease_out_bounce,
    "ease_in_out_bounce": ease_in_out_bounce,
    "ease_staircase": ease_staircase,
    "ease_step": ease_step,
    "ease_flat": ease_flat,
}


class Interpolates(NamedTuple):
    interpolate: InterpolateFn = interpolate
    interpolate_step: InterpolateFn = interpolate_step
    interpolate_integer: InterpolateFn = interpolate_integer
# end class Interpolates


INTERPOLATES = {
    "interpolate": interpolate,
    "step": interpolate_step,
    "integer": interpolate_integer,
}


def get_easing(name: str) -> EasingFn:
    """Get an easing function by name.

    Parameters
    ----------
    name : str
        Easing identifier.

    Returns
    -------
    EasingFn
        Easing callable.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in EASINGS:
        available = ", ".join(sorted(EASINGS.keys()))
        raise KeyError(f"Unknown easing '{name}'. Available: {available}")
    # end if
    return EASINGS[name]
# end def get_easing


def list_easings() -> list[str]:
    """List all registered easing names.

    Returns
    -------
    list[str]
        Sorted list of easing identifiers.
    """
    return sorted(EASINGS.keys())
# end def list_easings


__all__ = [
    # Types
    "T_in",
    "T_out",
    "EasingFn",
    "InterpolateFn",
    "Interpolation",
    # Easing functions
    "linear",
    "ease_in_quad",
    "ease_out_quad",
    "ease_in_out_quad",
    "ease_in_cubic",
    "ease_out_cubic",
    "ease_in_out_cubic",
    "ease_in_quart",
    "ease_out_quart",
    "ease_in_out_quart",
    "ease_in_quint",
    "ease_out_quint",
    "ease_in_out_quint",
    "ease_in_sine",
    "ease_out_sine",
    "ease_in_out_sine",
    "ease_in_expo",
    "ease_out_expo",
    "ease_in_out_expo",
    "ease_in_circ",
    "ease_out_circ",
    "ease_in_out_circ",
    "ease_in_back",
    "ease_out_back",
    "ease_in_out_back",
    "ease_in_elastic",
    "ease_out_elastic",
    "ease_in_out_elastic",
    "ease_in_bounce",
    "ease_out_bounce",
    "ease_in_out_bounce",
    # Interpolation functions
    "interpolate",
    "interpolate_step",
    "interpolate_integer",
    # Others
    "Interpolates",
    "Easings",
    "create_interpolate",
    "EASINGS",
    "INTERPOLATES",
    "get_easing",
    "list_easings",
]
