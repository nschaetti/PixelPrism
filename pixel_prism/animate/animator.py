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
from . import LinearInterpolator
# Imports
from .animate import Move
from .fade import FadeIn, FadeOut


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
        self._animation = list()
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
    def animation(self):
        """
        Animation
        """
        return self._animation
    # end animation

    # endregion PROPERTIES

    # region PUBLIC

    # Add
    def add(self, animation):
        """
        Add an animation
        """
        self._animation.append(animation)
        return self
    # end add

    # Move
    def move(
            self,
            start_time,
            duration,
            target_value,
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
        move_animation = Move(
            obj=self._target_object,
            start_time=start_time,
            end_time=start_time+duration,
            target_value=target_value,
            interpolator=interpolator,
            name=name,
            relative=relative
        )
        self._animation.append(move_animation)
        return self
    # end move

    # Fade in
    def fadein(
            self,
            start_time,
            duration,
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
        # Create the fade-in animation
        fadein_animation = FadeIn(
            obj=self._target_object,
            start_time=start_time,
            end_time=start_time+duration,
            interpolator=interpolator,
            name=name
        )
        self._animation.append(fadein_animation)
        return self
    # end fadein

    # Fade out
    def fadeout(
            self,
            start_time,
            duration,
            interpolator=LinearInterpolator(),
            name: str = ""
    ):
        """
        Fade out the target object
        """
        fadeout_animation = FadeOut(
            obj=self._target_object,
            start_time=start_time,
            end_time=start_time+duration,
            interpolator=interpolator,
            name=name
        )
        self._animation.append(fadeout_animation)
        return self
    # end fadeout

    # endregion PUBLIC

    # region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        """
        return len(self._animation)
    # end __len__

    # Get item
    def __getitem__(self, index):
        """
        Get item
        """
        return self._animation[index]
    # end __getitem__

    # endregion OVERRIDE

# end Animator



