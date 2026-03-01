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
from typing import Callable, Generic, TypeVar, Optional, Any, List
from dataclasses import dataclass

from .easing import EasingFn, create_interpolate, Interpolates, Easings, Interpolation, InterpolateFn

T_var = TypeVar("T_var")


@dataclass(frozen=True)
class Keyframe(Generic[T_var]):
    time: float                                         # Time position of this keyframe
    value: T_var                                        # Value at this keyframe
    hold: bool = False                                  # Step interpolation (no easing)
# end class Keyframe


@dataclass
class Segment(Generic[T_var]):
    start: Keyframe[T_var]
    end: Keyframe[T_var]
    easing: EasingFn
    interpolation: InterpolateFn = Interpolates.interpolate

    def interpolate(self, t: float) -> T_var:
        fn = create_interpolate(
            start=self.start.value,
            end=self.end.value,
            interpolate_fn=self.interpolation,
            easing_fn=self.easing,
        )
        t_rel = (t - self.start.time) / (self.end.time - self.start.time)
        return fn(t=t_rel)
    # end def interpolate

# end class Segment


class Track(Generic[T_var]):
    """
    Animate a single variable (named with an astr) over time (ex: 'line.start.x', 'line.end.y').
    """

    def __init__(
            self,
            var_name: str,
            value: T_var,
    ) -> None:
        """
        Constructor.

        Args:
            var_name:
        """
        self._var_name = var_name
        self._keyframes: list[Keyframe[T_var]] = [
            Keyframe(0.0, value),
        ]
        self._segments: list[Segment[T_var]] = []
        self._length: float = 0.0
    # end def __init__

    # region PROPERTIES

    @property
    def var_name(self) -> str:
        return self._var_name
    # end def var_name

    @property
    def keyframes(self) -> List[Keyframe[T_var]]:
        return self._keyframes
    # end def keyframes

    @property
    def segments(self) -> List[Segment[T_var]]:
        return self._segments
    # end def segments

    @property
    def length(self) -> float:
        return self._length
    # end def length

    @property
    def duration(self) -> float:
        return self._length
    # end def duration

    # endregion PROPERTIES

    # region PUBLIC

    def add_keyframe(self, keyframe: Keyframe[T_var]) -> None:
        """
        Add a keyframe to the track.

        Parameters
        ----------
        keyframe : Keyframe[T_var]
            Keyframe to add.
        """
        self._keyframes.append(keyframe)
        self._length = max(self._length, keyframe.time)
        previous_keyframe = self._previous_keyframe(keyframe.time)
        next_keyframe = self._next_keyframe(keyframe.time)
        current_segment = self._segment_at(keyframe.time)
        if current_segment is not None:
            self.remove_segment(current_segment)
        # end if
        if previous_keyframe.hold:
            self._add_hold_segment(previous_keyframe, keyframe)
        else:
            self._add_linear_segment(previous_keyframe, keyframe)
        # end if
        if next_keyframe is not None:
            if keyframe.hold:
                self._add_hold_segment(keyframe, next_keyframe)
            else:
                self._add_linear_segment(keyframe, next_keyframe)
            # end if
        # end if
    # end def add_keyframe

    def remove_keyframe(self, keyframe: Keyframe[T_var]) -> None:
        """
        Remove a keyframe from the track.
        """
        self._keyframes.remove(keyframe)
    # end def remove_keyframe

    def add_segment(self, segment: Segment[T_var]) -> None:
        self._segments.append(segment)
    # end def add_segment

    def remove_segment(self, segment: Segment[T_var]) -> None:
        self._segments.remove(segment)
    # end def remove_segment

    def set_interpolation(self, t: float, easing: EasingFn, interpolation: InterpolateFn[T_var]) -> None:
        current_segment = self._segment_at(t)
        if current_segment is not None:
            current_segment.easing = easing
            current_segment.interpolation = interpolation
        else:
            previous_keyframe = self._previous_keyframe(t)
            next_keyframe = self._next_keyframe(t)
            if previous_keyframe is not None and next_keyframe is not None:
                self._add_linear_segment(previous_keyframe, next_keyframe)
            else:
                raise ValueError("No keyframes found")
            # end if
        # end if
    # end def set_interpolation

    def at(
        self,
        time: float,
        value: T_var,
        easing_out: EasingFn | None = None,
        hold: bool = False,
    ) -> "Track[T_var]":
        pass
    # end def at

    def value_at(self, t: float) -> T_var:
        if t < 0.0:
            raise ValueError("t must be positive")
        # end if
        pos_segment = self._segment_at(t)
        return pos_segment.interpolate(t)
    # end def value_at

    def is_active_at(self, t: float) -> bool:
        if self._segments:
            return self._segments[0].start.time <= t <= self._segments[-1].end.time
        else:
            return False
        # end if
    # end def is_active_at

    def validate(self) -> None:
        pass
    # end def validate

    # endregion PUBLIC

    # region PRIVATE

    def _add_linear_segment(self, key1: Keyframe[T_var], key2: Keyframe[T_var]) -> None:
        """Adds a linear segment between two keyframes.
        """
        new_segment = Segment[T_var](
            start=key1,
            end=key2,
            easing=Easings.linear,
            interpolation=Interpolates.interpolate,
        )
        self.add_segment(new_segment)
    # end def _add_linear_segment

    def _add_hold_segment(self, key1: Keyframe[T_var], key2: Keyframe[T_var]) -> None:
        """Adds a hold segment between two keyframes.

        A hold segment keeps the value constant between the two keyframes.
        """
        new_segment = Segment[T_var](
            start=key1,
            end=key2,
            easing=Easings.flat,
            interpolation=Interpolates.interpolate,
        )
        self.add_segment(new_segment)
    # end def _add_hold_segment

    def _segment_at(self, t: float) -> Optional[Segment[T_var]]:
        for seg in self._segments:
            if seg.start.time <= t <= seg.end.time:
                return seg
            # end if
        # end for
        return None
    # end def _segment_at

    def _previous_keyframe(self, t: float) -> Optional[Keyframe[T_var]]:
        for keyframe in reversed(self._keyframes):
            if keyframe.time <= t:
                return keyframe
            # end if
        # end for
        return None
    # end def _previous_keyframe

    def _next_keyframe(self, t: float) -> Optional[Keyframe[T_var]]:
        for keyframe in self._keyframes:
            if keyframe.time >= t:
                return keyframe
            # end if
        # end for
        return None
    # end def _next_keyframe

    # endregion PRIVATE

    # region OVERRIDE

    def __getitem__(self, t: float) -> T_var:
        return self.value_at(t)
    # end def __getitem__

    # endregion OVERRIDE

# end class Track


class Timeline:
    """
    Collection of tracks + evaluation at a time t.
    """

    def __init__(
            self
    ) -> None:
        self._tracks: dict[str, Track[Any]] = {}
    # end def __init__

    def add_track(self, track: Track[Any]) -> None:
        self._tracks[track.var_name] = track
    # end def add_track

    def track(self, var_name: str, value: Any) -> Track[Any]:
        self._tracks[var_name] = Track(var_name, value)
        return self._tracks[var_name]
    # end def track

    def evaluate(self, t: float) -> dict[str, Any]:
        return {track.var_name: track[t] for track in self._tracks.values()}
    # end def evaluate

# end class Timeline
