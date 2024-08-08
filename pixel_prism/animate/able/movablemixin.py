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
        self.pos = None
        self.start_position = None
    # end __init__

    # Get position
    def get_position(self):
        """
        Get the position of the point.
        """
        return self.pos
    # end get_position

    # Initialize position
    def init_move(self):
        """
        Initialize the move animation.
        """
        pass
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
        self.start_position = self.pos.copy()
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
        self.pos = self.start_position * (1 - interpolated_t) + end_value.pos * interpolated_t
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
        pass
    # end end_move

    # Finish animation
    def finish_move(self):
        """
        Finish the move animation.
        """
        pass
    # end finish_move

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
