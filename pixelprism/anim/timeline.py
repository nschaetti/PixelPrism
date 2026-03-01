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
# Copyright (C) 2024 Pixel Prism
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


from __future__ import annotations
from typing import Callable, Generic, TypeVar, Optional, Any
from dataclasses import dataclass


T = TypeVar("T")
EaseFn = Callable[[float], float]              # u in [0,1] -> v in [0,1]
LerpFn = Callable[[T, T, float], T]            # interpolate(a, b, v)


@dataclass(frozen=True)
class Keyframe(Generic[T]):
    time: float                                # Time position of this keyframe
    value: T                                   # Value at this keyframe
    easing_out: Optional[EaseFn] = None        # Interpolation method between keyframes
    hold: bool = False                         # Step interpolation (no easing)
# end class Keyframe


class Track(Generic[T]):
    """
    Animate a single variable (named with an astr) over time (ex: 'line.start.x', 'line.end.y').
    """

    def __init__(
        self,
        var_name: str,
        lerp: LerpFn[T],
        default_ease: EaseFn | None = None,
    ) -> None:
        pass
    # end def __init__

    def at(
        self,
        time: float,
        value: T,
        easing_out: EaseFn | None = None,
        hold: bool = False,
    ) -> "Track[T]":
        pass
    # end def at

    def value_at(self, t: float) -> T:
        pass
    # end def value_at

    def is_active_at(self, t: float) -> bool:
        pass
    # end def is_active_at

    def validate(self) -> None:
        pass
    # end def validate

# end class Track


class Timeline:
    """
    Collection of tracks + evaluation at a time t.
    """

    def __init__(self, duration: float, fps: int = 60) -> None:
        pass
    # end def __init__

    def add_track(self, track: Track[Any]) -> None:
        pass
    # end def add_track

    def track(self, var_name: str, lerp: LerpFn[T]) -> Track[T]:
        pass
    # end def track

    def evaluate(self, t: float) -> dict[str, Any]:
        pass
    # end def evaluate

    def frame_time(self, frame_idx: int) -> float:
        pass
    # end def frame_time

    def validate(self) -> None: ...

# end class Timeline
