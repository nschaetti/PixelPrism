

# Imports
import numpy as np
from .able import MovAble, FadeInAble
from .interpolate import Interpolator, LinearInterpolator


class Animate:
    """
    A transition that changes a property of an object over time.
    """

    def __init__(
            self,
            name,
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
            name (str): Name of the transition
            obj (object): Object to transition
            property_name (str): Name of the property to transition
            start_time (float): Start time
            end_time (float): End time
            start_value (float): Start value
            end_value (float): End value
            interpolator (Interpolator): Interpolator
            kwargs: Additional keyword arguments
        """
        self.name = name
        self.obj = obj
        self.start_time = start_time
        self.end_time = end_time
        self.start_value = start_value
        self.end_value = end_value
        self.interpolator = interpolator
        self.kwargs = kwargs
        self.running = False
    # end __init__

    def start(self):
        """
        Start the animation.
        """
        self.running = True
    # end start

    def stop(self):
        """
        Stop the animation.
        """
        self.running = False
    # end stop

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
            name,
            obj,
            start_time,
            end_time,
            end_value,
            interpolator=LinearInterpolator()
    ):
        """
        Initialize the move transition.

        Args:
            name (str): Name of the transition
            obj (any): Object to move
            start_time (float): Start time
            end_time (float): End time
            end_value (any): End position
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, MovAble), "Object must be an instance of MovAble"
        super().__init__(
            name,
            obj,
            start_time,
            end_time,
            None,
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
            self.stop()
            return
        elif t > self.end_time:
            self.stop()
            return
        else:
            self.start()
            relative_t = (t - self.start_time) / (self.end_time - self.start_time)
        # end if

        # if running
        if self.running:
            # Interpolate time
            interpolated_t = self.interpolator.interpolate(0, 1, relative_t)

            # Perform the move animation
            print(f"Transition {self.name}, Time {t}")
            self.obj.animate_move(relative_t, self.end_time - self.start_time, interpolated_t, self.end_value)
        # end if
    # end update

# end Move


# Fade in animation
class FadeIn(Animate):
    """
    A transition that fades in an object over time.
    """

    def __init__(
            self,
            name,
            obj,
            start_time,
            end_time,
            interpolator=LinearInterpolator()
    ):
        """
        Initialize the fade-in transition.

        Args:
            obj (any): Object to fade in
            start_time (float): Start time
            end_time (float): End time
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, FadeInAble), "Object must be an instance of FadeInAble"
        super().__init__(name, obj, start_time, end_time, 0, 1, interpolator)
    # end __init__

    def update(self, t):
        """
        Update the object opacity at time t.
        """
        if t < self.start_time:
            self.stop()
            return
        elif t > self.end_time:
            self.stop()
            return
        else:
            self.start()
            relative_t = (t - self.start_time) / (self.end_time - self.start_time)
        # end if

        # if running
        if self.running:
            # Interpolate time
            interpolated_t = self.interpolator.interpolate(0, 1, relative_t)

            # Perform the fade-in animation
            self.obj.animate_fadein(relative_t, self.end_time - self.start_time, interpolated_t)
        # end if
    # end update

# end FadeIn
