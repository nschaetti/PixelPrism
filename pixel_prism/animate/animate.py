

# Imports
import numpy as np
from .interpolate import Interpolator, LinearInterpolator


class Animate:
    """
    A transition that changes a property of an object over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            start_value,
            end_value,
            interpolator=LinearInterpolator(),
            **kwargs
    ):
        """
        Initialize the transition.

        Args:
            obj (object): Object to transition
            property_name (str): Name of the property to transition
            start_time (float): Start time
            end_time (float): End time
            start_value (float): Start value
            end_value (float): End value
            interpolator (Interpolator): Interpolator
            kwargs: Additional keyword arguments
        """
        self.obj = obj
        self.start_time = start_time
        self.end_time = end_time
        self.start_value = start_value
        self.end_value = end_value
        self.interpolator = interpolator
        self.kwargs = kwargs
    # end __init__

    def update(
            self,
            t
    ):
        """
        Update the object property at time t.

        Args:
            t (float): Time
        """
        raise NotImplementedError("Update method must be implemented by subclasses")
    # end update

# end Animate


class Move(Animate):
    """
    A transition that moves an object over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            start_value,
            end_value,
            interpolator=LinearInterpolator()
    ):
        """
        Initialize the move transition.

        Args:
            obj (object): Object to move
            start_time (float): Start time
            end_time (float): End time
            start_value (any): Start position
            end_value (any): End position
            interpolator (Interpolator): Interpolator
        """
        super().__init__(
            obj,
            start_time,
            end_time,
            start_value,
            end_value,
            interpolator
        )
    # end __init__

    def update(
            self,
            t
    ):
        """
        Update the object position at time t.

        Args:
            t (float): Time
        """
        if t < self.start_time:
            value = np.array(self.start_value)
        elif t > self.end_time:
            value = np.array(self.end_value)
        else:
            progress = (t - self.start_time) / (self.end_time - self.start_time)
            value = self.interpolator.interpolate(
                start_value=np.array(self.start_value),
                end_value=np.array(self.end_value),
                progress=progress
            )
        # end if

        print("Move.update")
        print(self.obj)
        print(self.obj.pos)

        # Update the object position
        self.obj.pos = value
    # end update

# end Move
