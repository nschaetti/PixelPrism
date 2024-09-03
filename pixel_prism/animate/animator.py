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
from typing import Any

# Imports
from . import LinearInterpolator, Interpolator, EaseInOutInterpolator
from .move import Move
from .fade import FadeIn, FadeOut
from .range import Range


# Animator
class Animator:
    """
    Animator class
    """

    # Constructor
    def __init__(self, target_object):
        """
        Constructor
        """
        # Properties
        self._target_object = target_object
        self._animations = list()
        self._last_start_time = 0.0
    # end __init__

    # region PROPERTIES

    @property
    def target_object(self):
        """
        Target object
        """
        return self._target_object
    # end target_object

    @property
    def animations(self):
        """
        Animation
        """
        return self._animations
    # end animation

    @property
    def last_start_time(self):
        """
        Last start time
        """
        return self._last_start_time
    # end last_start_time

    @last_start_time.setter
    def last_start_time(self, value):
        """
        Last start time
        """
        self._last_start_time = value
    # end last_start_time

    @property
    def animated(self):
        """
        Animated
        """
        return self._target_object
    # end animated

    # endregion PROPERTIES

    # region PUBLIC

    # Add
    def add(self, animation):
        """
        Add an animation
        """
        self._animations.append(animation)
        return self
    # end add

    # Pause
    def pause(
            self,
            duration
    ):
        """
        Pause the animation

        Args:
            duration (float): Duration of the pause
        """
        self._last_start_time += duration
    # end pause

    # endregion PUBLIC

    # region CREATE_ANIMATIONS

    # Range
    def range(
            self,
            duration: float,
            target_value: Any,
            start_time: float = None,
            interpolator: Interpolator = EaseInOutInterpolator(),
            name: str = ""
    ):
        """
        Create a range animation.

        Args:
            duration (float): Duration of the animation
            target_value (Any): Target value of the animation
            start_time (float): Start time of the animation
            interpolator (Interpolator): Interpolator
            name (str): Name of the animation
        """
        # Start time
        start_time = start_time if start_time is not None else self._last_start_time

        # Create the range animation
        range_animation = Range(
            obj=self.animated,
            start_time=start_time,
            end_time=start_time+duration,
            target_value=target_value,
            interpolator=interpolator,
            name=name
        )
        self._animations.append(range_animation)

        # Update the last start time
        self._last_start_time = start_time + duration

        return self
    # end range

    # Move
    def move(
            self,
            duration,
            target_value,
            start_time=None,
            interpolator=LinearInterpolator(),
            name: str = "",
            relative: bool = False
    ):
        """
        Move the target object

        Args:
            start_time (float): Start time of the animation
            duration (float): Duration of the animation
            target_value (Any): Target value of the animation
            interpolator (Interpolator): Interpolator
            name (str): Name of the animation
            relative (bool): Relative or absolute move
        """
        # Start time
        start_time = start_time if start_time is not None else self._last_start_time

        # Create the move animation
        move_animation = Move(
            obj=self.animated,
            start_time=start_time,
            end_time=start_time+duration,
            target_value=target_value,
            interpolator=interpolator,
            name=name,
            relative=relative
        )
        self._animations.append(move_animation)
        return self
    # end move

    # Fade in
    def fadein(
            self,
            duration,
            start_time=None,
            interpolator=LinearInterpolator(),
            name: str = ""
    ):
        """
        Fade in the target object

        Args:
            start_time (float): Start time of the animation
            duration (float): Duration of the animation
            interpolator (Interpolator): Interpolator
            name (str): Name of the animation
        """
        # Start time
        start_time = start_time if start_time is not None else self._last_start_time

        # Create the fade-in animation
        fadein_animation = FadeIn(
            obj=self.animated,
            start_time=start_time,
            end_time=start_time+duration,
            interpolator=interpolator,
            name=name
        )
        self._animations.append(fadein_animation)
        return self
    # end fadein

    # Fade out
    def fadeout(
            self,
            duration,
            start_time=None,
            interpolator=LinearInterpolator(),
            name: str = ""
    ):
        """
        Fade out the target object

        Args:
            start_time (float): Start time of the animation
            duration (float): Duration of the animation
            interpolator (Interpolator): Interpolator
            name (str): Name of the animation
        """
        # Start time
        start_time = start_time if start_time is not None else self._last_start_time

        fadeout_animation = FadeOut(
            obj=self.animated,
            start_time=start_time,
            end_time=start_time+duration,
            interpolator=interpolator,
            name=name
        )
        self._animations.append(fadeout_animation)
        return self
    # end fadeout

    # endregion CREATE_ANIMATIONS

    # region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        """
        return len(self._animations)
    # end __len__

    # Get item
    def __getitem__(self, index):
        """
        Get item
        """
        return self._animations[index]
    # end __getitem__

    # endregion OVERRIDE

# end Animator



