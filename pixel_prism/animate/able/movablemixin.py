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
# Description: Movable interface class
#

# Imports
from typing import Any
import numpy as np
from .animablemixin import AnimableMixin


class MovableMixin(AnimableMixin):
    """
    Interface class for movable objects
    """

    # Initialize
    def __init__(self):
        """
        Initialize the movable object.
        """
        super().__init__()
    # end __init__

    # region PROPERTIES

    @property
    def movable_position(self) -> Any:
        """
        Get the position of the object.

        Returns:
            any: Position of the object
        """
        raise NotImplementedError("Property 'movable_position' must be implemented in the derived class.")
    # end movable_position

    @movable_position.setter
    def movable_position(self, value: Any):
        """
        Set the position of the object.

        Args:
            value (any): Position of the object
        """
        raise NotImplementedError("Property 'movable_position' must be implemented in the derived class.")
    # end movable_position

    # end PROPERTIES

    # region PUBLIC

    # Initialize position
    def init_move(self):
        """
        Initialize the move animation.
        """
        raise NotImplementedError("Method 'init_move' must be implemented in the derived class.")
    # end init_move

    # Start animation
    def start_move(
            self,
            start_value: Any
    ):
        """
        Start the move animation.

        Args:
            start_value (any): The start position of the object
        """
        raise NotImplementedError("Method 'start_move' must be implemented in the derived class.")
    # end start_move

    def animate_move(
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
        # self.pos = self.start_position * (1 - interpolated_t) + end_value.pos * interpolated_t
        raise NotImplementedError("Method 'animate_move' must be implemented in the derived class.")
    # end animate_move

    # Stop animation
    def end_move(
            self,
            end_value: Any
    ):
        """
        Stop the move animation.

        Args:
            end_value (any): The end position of the object
        """
        raise NotImplementedError("Method 'end_move' must be implemented in the derived class.")
    # end end_move

    # Finish animation
    def finish_move(self):
        """
        Finish the move animation.
        """
        raise NotImplementedError("Method 'finish_move' must be implemented in the derived class.")
    # end finish_move

    # endregion PUBLIC

# end MovAble


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
        super().__init__()
        self.rangeable_animated_attribute = rangeable_animated_attribute
        self.start_position = None
    # end __init__

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
