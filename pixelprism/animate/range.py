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
from typing import Any
from . import LinearInterpolator, EaseInOutInterpolator
from .able import AnimableMixin
from .animate import Animate


class RangeableMixin(AnimableMixin):
    """
    Interface class for ranging values
    """

    # Initialize
    def __init__(
            self,
            rangeable_animated_attribute: str
    ):
        """
        Initialize the movable object.
        """
        AnimableMixin.__init__(self)
        self.rangeable_animated_attribute = rangeable_animated_attribute
        self.start_position = None
    # end __init__

    # Create range animation
    def range(
            self,
            duration: float,
            target_value: Any,
            start_time: float = None,
            interpolator=EaseInOutInterpolator()
    ):
        """
        Create a range animation.

        Args:
            start_time (float): Start time
            duration (float): End time
            target_value (any): Target value
            interpolator (Interpolator): Interpolator
        """
        from .animator import Animator

        # Create the animator
        animator = Animator(self)

        # Create the range animation
        animator.range(
            duration=duration,
            target_value=target_value,
            start_time=start_time,
            interpolator=interpolator
        )

        # Add to animation
        self.animable_registry.add(animator)

        # Add to animation
        return animator
    # end create_range

    # Initialize animation
    def init_range(self):
        """
        Initialize the range animation.
        """
        pass
    # end init

    # Start animation
    def start_range(
            self,
            start_value: Any
    ):
        """
        Start the range animation.

        Args:
            start_value (any): The start position of the object
        """
        self.start_position = getattr(self, self.rangeable_animated_attribute)
    # end start_move

    def animate_range(
            self,
            t,
            duration,
            interpolated_t,
            end_value
    ):
        """
        Perform the move animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The end position of the object
        """
        new_value = self.start_position * (1 - interpolated_t) + end_value * interpolated_t
        setattr(self, self.rangeable_animated_attribute, new_value)
    # end animate_range

    # Stop animation
    def end_range(
            self,
            end_value: Any
    ):
        """
        Stop the range animation.

        Args:
            end_value (any): The end value of the object
        """
        pass
    # end end_range

    # Finish animation
    def finish_range(self):
        """
        Finish the range animation.
        """
        pass
    # end finish_range

# end RangeAble


class Range(Animate):
    """
    A transition that changes a range of values over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            target_value,
            interpolator=LinearInterpolator(),
            name=""
    ):
        """
        Initialize the range transition.

        Args:
            name (str): Name of the transition
            obj (any): Object to range
            start_time (float): Start time
            end_time (float): End time
            target_value (any): End value
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, RangeableMixin), "Object must be an instance of RangeAble"
        super().__init__(
            obj,
            start_time,
            end_time,
            None,
            target_value,
            interpolator=interpolator,
            name=name
        )
    # end __init__

# end Range

