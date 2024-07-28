#
# Description: Movable interface class
#

# Imports
from .able import Able


class MovAble(Able):
    """
    Interface class for movable objects
    """

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
        raise NotImplementedError("Subclasses should implement this method")
    # end animate_move

# end MovAble
