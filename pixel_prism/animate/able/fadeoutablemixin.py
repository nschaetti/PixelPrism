
# Imports
from typing import Any
from .animablemixin import AnimableMixin


class FadeOutableMixin(AnimableMixin):
    """
    Interface class for fade-out animations
    """

    # Constructor
    def __init__(self):
        """
        Initialize the object.
        """
        super().__init__()
        self.opacity = None
    # end __init__

    # Start fade-out animation
    def start_fadeout(self, start_value: Any):
        """
        Start the fade-out animation.
        """
        pass
    # end start_fadeout

    def animate_fadeout(self, t, duration, interpolated_t, target_value):
        """
        Animate the fade-out effect.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            target_value (any): The target value of the animation
        """
        self.opacity = 1.0 - interpolated_t
    # end animate_fadeout

    def end_fadeout(self, end_value: Any):
        """
        End the fade-out animation.
        """
        pass
    # end end_fadeout

# end FadeOutAble


