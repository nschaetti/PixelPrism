#
# Description: Movable interface class
#

# Imports
from typing import Any
import numpy as np
from .able import Able


class MovAble(Able):
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
        self.start_position = self.pos
    # end start_move

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
        self.pos = self.start_position * (1 - interpolated_t) + np.array(end_value) * interpolated_t
    # end animate_move

# end MovAble
