#
# Description: This file contains the Able class, which is a subclass of the Drawable class.
#

class Able:
    """
    Abstract class for animatable objects.
    """

    def animate(
            self,
            t,
            duration,
            interpolated_t,
            end_value
    ):
        """
        Method to be implemented by subclasses to perform animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The target value of the animation
        """
        raise NotImplementedError("Subclasses should implement this method")
    # end animate

# end Able

