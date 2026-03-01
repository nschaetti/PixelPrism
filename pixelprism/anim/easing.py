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
from typing import Callable, TypeVar


T = TypeVar("T")

# Types
EasingFn = Callable[[float], float]
InterpolateFn = Callable[[T, T, float], T]


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


def linear(u: float) -> float:
    """Compute linear easing.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Clamped progression without curve distortion.
    """
    return _clamp01(u)
# end def linear


def ease_in_quad(u: float) -> float:
    """Compute quadratic ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression favoring slower starts.
    """
    u = _clamp01(u)
    return u * u
# end def ease_in_quad


def ease_out_quad(u: float) -> float:
    """Compute quadratic ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression favoring slower ends.
    """
    u = _clamp01(u)
    return 1.0 - (1.0 - u) * (1.0 - u)
# end def ease_out_quad


def ease_in_out_quad(u: float) -> float:
    """Compute quadratic ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression.
    """
    u = _clamp01(u)
    if u < 0.5:
        return 2.0 * u * u
    # end if
    return 1.0 - ((-2.0 * u + 2.0) ** 2) / 2.0
# end def ease_in_out_quad


def ease_in_cubic(u: float) -> float:
    """Compute cubic ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with stronger acceleration.
    """
    u = _clamp01(u)
    return u * u * u
# end def ease_in_cubic


def ease_out_cubic(u: float) -> float:
    """Compute cubic ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with stronger deceleration.
    """
    u = _clamp01(u)
    return 1.0 - (1.0 - u) ** 3
# end def ease_out_cubic


def ease_in_out_cubic(u: float) -> float:
    """Compute cubic ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression with cubic profile.
    """
    u = _clamp01(u)
    if u < 0.5:
        return 4.0 * u * u * u
    # end if
    return 1.0 - ((-2.0 * u + 2.0) ** 3) / 2.0
# end def ease_in_out_cubic


def ease_in_quart(u: float) -> float:
    """Compute quartic ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with high acceleration.
    """
    u = _clamp01(u)
    return u ** 4
# end def ease_in_quart


def ease_out_quart(u: float) -> float:
    """Compute quartic ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with high deceleration.
    """
    u = _clamp01(u)
    return 1.0 - (1.0 - u) ** 4
# end def ease_out_quart


def ease_in_out_quart(u: float) -> float:
    """Compute quartic ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression with quartic profile.
    """
    u = _clamp01(u)
    if u < 0.5:
        return 8.0 * (u ** 4)
    # end if
    return 1.0 - ((-2.0 * u + 2.0) ** 4) / 2.0
# end def ease_in_out_quart


def ease_in_quint(u: float) -> float:
    """Compute quintic ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with very strong acceleration.
    """
    u = _clamp01(u)
    return u ** 5
# end def ease_in_quint


def ease_out_quint(u: float) -> float:
    """Compute quintic ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with very strong deceleration.
    """
    u = _clamp01(u)
    return 1.0 - (1.0 - u) ** 5
# end def ease_out_quint


def ease_in_out_quint(u: float) -> float:
    """Compute quintic ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric eased progression with quintic profile.
    """
    u = _clamp01(u)
    if u < 0.5:
        return 16.0 * (u ** 5)
    # end if
    return 1.0 - ((-2.0 * u + 2.0) ** 5) / 2.0
# end def ease_in_out_quint


def ease_in_sine(u: float) -> float:
    """Compute sine ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression based on a cosine phase.
    """
    u = _clamp01(u)
    return 1.0 - math.cos((u * math.pi) / 2.0)
# end def ease_in_sine


def ease_out_sine(u: float) -> float:
    """Compute sine ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression based on a sine phase.
    """
    u = _clamp01(u)
    return math.sin((u * math.pi) / 2.0)
# end def ease_out_sine


def ease_in_out_sine(u: float) -> float:
    """Compute sine ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Smooth symmetric trigonometric easing.
    """
    u = _clamp01(u)
    return -(math.cos(math.pi * u) - 1.0) / 2.0
# end def ease_in_out_sine


def ease_in_expo(u: float) -> float:
    """Compute exponential ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with exponential growth.
    """
    u = _clamp01(u)
    if u == 0.0:
        return 0.0
    # end if
    return 2.0 ** (10.0 * u - 10.0)
# end def ease_in_expo


def ease_out_expo(u: float) -> float:
    """Compute exponential ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with exponential decay.
    """
    u = _clamp01(u)
    if u == 1.0:
        return 1.0
    # end if
    return 1.0 - (2.0 ** (-10.0 * u))
# end def ease_out_expo


def ease_in_out_expo(u: float) -> float:
    """Compute exponential ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric exponential easing.
    """
    u = _clamp01(u)
    if u == 0.0:
        return 0.0
    # end if
    if u == 1.0:
        return 1.0
    # end if
    if u < 0.5:
        return (2.0 ** (20.0 * u - 10.0)) / 2.0
    # end if
    return (2.0 - (2.0 ** (-20.0 * u + 10.0))) / 2.0
# end def ease_in_out_expo


def ease_in_circ(u: float) -> float:
    """Compute circular ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression following a circular arc.
    """
    u = _clamp01(u)
    return 1.0 - math.sqrt(1.0 - u * u)
# end def ease_in_circ


def ease_out_circ(u: float) -> float:
    """Compute circular ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression decelerating along a circular arc.
    """
    u = _clamp01(u)
    return math.sqrt(1.0 - (u - 1.0) ** 2)
# end def ease_out_circ


def ease_in_out_circ(u: float) -> float:
    """Compute circular ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric circular easing.
    """
    u = _clamp01(u)
    if u < 0.5:
        return (1.0 - math.sqrt(1.0 - (2.0 * u) ** 2)) / 2.0
    # end if
    return (math.sqrt(1.0 - (-2.0 * u + 2.0) ** 2) + 1.0) / 2.0
# end def ease_in_out_circ


def make_ease_in_back(overshoot: float = 1.70158) -> EasingFn:
    """Create an ease-in-back callable.

    Parameters
    ----------
    overshoot : float, default=1.70158
        Overshoot factor controlling how far the curve goes below zero before
        accelerating toward one.

    Returns
    -------
    EasingFn
        Parameterized back ease-in function.
    """
    def _ease(u: float) -> float:
        u2 = _clamp01(u)
        c3 = overshoot + 1.0
        return c3 * u2 * u2 * u2 - overshoot * u2 * u2
    # end def _ease
    return _ease
# end def make_ease_in_back


def make_ease_out_back(overshoot: float = 1.70158) -> EasingFn:
    """Create an ease-out-back callable.

    Parameters
    ----------
    overshoot : float, default=1.70158
        Overshoot factor controlling how far the curve goes above one before
        settling.

    Returns
    -------
    EasingFn
        Parameterized back ease-out function.
    """
    def _ease(u: float) -> float:
        u2 = _clamp01(u)
        c3 = overshoot + 1.0
        return 1.0 + c3 * ((u2 - 1.0) ** 3) + overshoot * ((u2 - 1.0) ** 2)
    # end def _ease
    return _ease
# end def make_ease_out_back


def make_ease_in_out_back(overshoot: float = 1.70158) -> EasingFn:
    """Create an ease-in-out-back callable.

    Parameters
    ----------
    overshoot : float, default=1.70158
        Overshoot factor used on both halves of the curve.

    Returns
    -------
    EasingFn
        Parameterized back ease-in-out function.
    """
    def _ease(u: float) -> float:
        u2 = _clamp01(u)
        c2 = overshoot * 1.525
        if u2 < 0.5:
            return ((2.0 * u2) ** 2 * ((c2 + 1.0) * 2.0 * u2 - c2)) / 2.0
        # end if
        return (((2.0 * u2 - 2.0) ** 2) * ((c2 + 1.0) * (u2 * 2.0 - 2.0) + c2) + 2.0) / 2.0
    # end def _ease
    return _ease
# end def make_ease_in_out_back


def ease_in_back(u: float) -> float:
    """Compute back ease-in with default overshoot.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with initial negative overshoot.
    """
    return make_ease_in_back()(u)
# end def ease_in_back


def ease_out_back(u: float) -> float:
    """Compute back ease-out with default overshoot.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Eased progression with final positive overshoot.
    """
    return make_ease_out_back()(u)
# end def ease_out_back


def ease_in_out_back(u: float) -> float:
    """Compute back ease-in-out with default overshoot.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric back easing with overshoot around both ends.
    """
    return make_ease_in_out_back()(u)
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


def ease_out_bounce(u: float) -> float:
    """Compute bounce ease-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Progression with decaying bounces near the end.
    """
    return _ease_out_bounce_core(_clamp01(u))
# end def ease_out_bounce


def ease_in_bounce(u: float) -> float:
    """Compute bounce ease-in.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Progression with bounces near the start.
    """
    u2 = _clamp01(u)
    return 1.0 - _ease_out_bounce_core(1.0 - u2)
# end def ease_in_bounce


def ease_in_out_bounce(u: float) -> float:
    """Compute bounce ease-in-out.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric bounce easing.
    """
    u2 = _clamp01(u)
    if u2 < 0.5:
        return (1.0 - _ease_out_bounce_core(1.0 - 2.0 * u2)) / 2.0
    # end if
    return (1.0 + _ease_out_bounce_core(2.0 * u2 - 1.0)) / 2.0
# end def ease_in_out_bounce


def make_ease_in_elastic(amplitude: float = 1.0, period: float = 0.3) -> EasingFn:
    """Create an ease-in-elastic callable.

    Parameters
    ----------
    amplitude : float, default=1.0
        Elastic oscillation amplitude.
    period : float, default=0.3
        Oscillation period.

    Returns
    -------
    EasingFn
        Parameterized elastic ease-in function.
    """
    def _ease(u: float) -> float:
        u2 = _clamp01(u)
        if u2 == 0.0:
            return 0.0
        # end if
        if u2 == 1.0:
            return 1.0
        # end if
        c4 = (2.0 * math.pi) / period
        return -(amplitude * (2.0 ** (10.0 * u2 - 10.0)) * math.sin((u2 * 10.0 - 10.75) * c4))
    # end def _ease
    return _ease
# end def make_ease_in_elastic


def make_ease_out_elastic(amplitude: float = 1.0, period: float = 0.3) -> EasingFn:
    """Create an ease-out-elastic callable.

    Parameters
    ----------
    amplitude : float, default=1.0
        Elastic oscillation amplitude.
    period : float, default=0.3
        Oscillation period.

    Returns
    -------
    EasingFn
        Parameterized elastic ease-out function.
    """
    def _ease(u: float) -> float:
        u2 = _clamp01(u)
        if u2 == 0.0:
            return 0.0
        # end if
        if u2 == 1.0:
            return 1.0
        # end if
        c4 = (2.0 * math.pi) / period
        return amplitude * (2.0 ** (-10.0 * u2)) * math.sin((u2 * 10.0 - 0.75) * c4) + 1.0
    # end def _ease
    return _ease
# end def make_ease_out_elastic


def make_ease_in_out_elastic(amplitude: float = 1.0, period: float = 0.45) -> EasingFn:
    """Create an ease-in-out-elastic callable.

    Parameters
    ----------
    amplitude : float, default=1.0
        Elastic oscillation amplitude.
    period : float, default=0.45
        Oscillation period.

    Returns
    -------
    EasingFn
        Parameterized elastic ease-in-out function.
    """
    def _ease(u: float) -> float:
        u2 = _clamp01(u)
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
    # end def _ease
    return _ease
# end def make_ease_in_out_elastic


def ease_in_elastic(u: float) -> float:
    """Compute elastic ease-in with default parameters.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Oscillatory progression near the start.
    """
    return make_ease_in_elastic()(u)
# end def ease_in_elastic


def ease_out_elastic(u: float) -> float:
    """Compute elastic ease-out with default parameters.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Oscillatory progression near the end.
    """
    return make_ease_out_elastic()(u)
# end def ease_out_elastic


def ease_in_out_elastic(u: float) -> float:
    """Compute elastic ease-in-out with default parameters.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        Symmetric oscillatory easing.
    """
    return make_ease_in_out_elastic()(u)
# end def ease_in_out_elastic


def step_start(u: float) -> float:
    """Compute step-start easing.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        ``0`` at the exact start and ``1`` for any positive progression.
    """
    u2 = _clamp01(u)
    if u2 <= 0.0:
        return 0.0
    # end if
    return 1.0
# end def step_start


def step_end(u: float) -> float:
    """Compute step-end easing.

    Parameters
    ----------
    u : float
        Normalized progression.

    Returns
    -------
    float
        ``0`` until completion and ``1`` only at the end.
    """
    u2 = _clamp01(u)
    if u2 < 1.0:
        return 0.0
    # end if
    return 1.0
# end def step_end


def steps(count: int, mode: str = "end") -> EasingFn:
    """Create a discrete step easing function.

    Parameters
    ----------
    count : int
        Number of intervals.
    mode : str, default="end"
        Either ``"start"`` or ``"end"``.

    Returns
    -------
    EasingFn
        Step easing callable.
    """
    if count <= 0:
        raise ValueError("count must be > 0.")
    # end if
    if mode not in ("start", "end"):
        raise ValueError("mode must be either 'start' or 'end'.")
    # end if

    def _ease(u: float) -> float:
        u2 = _clamp01(u)
        if mode == "start":
            return min(1.0, math.ceil(u2 * count) / count)
        # end if
        return min(1.0, math.floor(u2 * count) / count)
    # end def _ease

    return _ease
# end def steps


def interpolate_value(
    start: T,
    end: T,
    u: float,
    lerp: InterpolateFn[T],
    easing: EasingFn = linear,
) -> T:
    """Interpolate between two values using a specific easing function.

    Parameters
    ----------
    start : T
        Starting value.
    end : T
        Ending value.
    u : float
        Linear progression in ``[0, 1]``.
    lerp : InterpolateFn[T]
        Interpolation function for the target type.
    easing : EasingFn, default=linear
        Easing function applied to ``u``.

    Returns
    -------
    T
        Interpolated value.
    """
    v = easing(_clamp01(u))
    return lerp(start, end, v)
# end def interpolate_value


EASINGS: dict[str, EasingFn] = {
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
    "step_start": step_start,
    "step_end": step_end,
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
    "T",
    "EasingFn",
    "InterpolateFn",
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
    "make_ease_in_back",
    "make_ease_out_back",
    "make_ease_in_out_back",
    "ease_in_back",
    "ease_out_back",
    "ease_in_out_back",
    "make_ease_in_elastic",
    "make_ease_out_elastic",
    "make_ease_in_out_elastic",
    "ease_in_elastic",
    "ease_out_elastic",
    "ease_in_out_elastic",
    "ease_in_bounce",
    "ease_out_bounce",
    "ease_in_out_bounce",
    "step_start",
    "step_end",
    "steps",
    "interpolate_value",
    "EASINGS",
    "get_easing",
    "list_easings",
]
