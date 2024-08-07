
# Imports
from typing import Any
from .animablemixin import AnimableMixin


class FadeInableMixin(AnimableMixin):
    """
    Interface class for fade-in animations
    """

    # Constructor
    def __init__(self):
        """
        Initialize the object.
        """
        super().__init__()
        self.opacity = None
    # end __init__

    # Initialize fade-in animation
    def init_fadein(self):
        """
        Initialize the fade-in animation.
        """
        pass
    # end init_fadein

    # Start fade-in animation
    def start_fadein(self, start_value: Any):
        """
        Start the fade-in animation.
        """
        pass
    # end start_fadein

    def animate_fadein(self, t, duration, interpolated_t, target_value):
        """
        Animate the fade-in effect.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            target_value (any): The target value of the animation
        """
        self.opacity = interpolated_t
    # end animate_fadein

    def end_fadein(self, end_value: Any):
        """
        End the fade-in animation.
        """
        pass
    # end end_fadein

    # Finish fade-in animation
    def finish_fadein(self):
        """
        Finish the fade-in animation.
        """
        pass
    # end finish_fadein

# end FadeInAble


