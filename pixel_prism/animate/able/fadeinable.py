
# Imports
from .able import Able


class FadeInAble(Able):
    """
    Interface class for fade-in animations
    """

    def animate_fadein(
            self,
            t,
            duration,
            interpolated_t
    ):
        """
        Perform the fade-in animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
        """
        raise NotImplementedError("Subclasses should implement this method")
    # end animate_fadein

# end FadeInAble


