

# Imports
import numpy as np


class Interpolator:
    """
    An interpolator that interpolates between two values.
    """

    def interpolate(
            self,
            start_value,
            end_value,
            progress
    ):
        """
        Interpolate between two values.

        Args:
            start_value (any): Start value
            end_value (any): End value
            progress (float): Progress
        """
        raise NotImplementedError("Interpolate method must be implemented by subclasses")
    # end interpolate

# end Interpolator


class LinearInterpolator(Interpolator):
    """
    Linear interpolator
    """

    def interpolate(
            self,
            start_value,
            end_value,
            progress
    ):
        """
        Interpolate between two values linearly.

        Args:
            start_value (any): Start value
            end_value (any): End value
            progress (float): Progress
        """
        return start_value + progress * (end_value - start_value)
    # end interpolate

# end LinearInterpolator


class EaseInOutInterpolator(Interpolator):
    """
    Ease in out interpolator
    """

    def interpolate(
            self,
            start_value,
            end_value,
            progress
    ):
        """
        Interpolate between two values with ease in out.

        Args:
            start_value (any): Start value
            end_value (any): End value
            progress (float): Progress
        """
        return start_value + self.ease_in_out(progress) * (end_value - start_value)
    # end interpolate

    @staticmethod
    def ease_in_out(
            progress
    ):
        """
        Ease in out function.

        Args:
            progress (float): Progress
        """
        return -0.5 * (np.cos(np.pi * progress) - 1)
    # end ease

# end EaseInOutInterpolator

