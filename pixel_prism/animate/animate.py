

# Imports
import numpy as np
from enum import Enum
from .able import (
    MovableMixin,
    FadeInableMixin,
    FadeOutableMixin,
    RangeableMixin,
    BuildableMixin,
    DestroyableMixin
)
from .interpolate import Interpolator, LinearInterpolator


class AnimationState(Enum):
    """
    Animation state
    """
    WAITING_START = 0
    RUNNING = 1
    FINISHED = 2
# end AnimationState


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
            target_value,
            interpolator=LinearInterpolator(),
            name="",
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
            start_value (any): Start value
            target_value (any): End value
            interpolator (Interpolator): Interpolator
            kwargs: Additional keyword arguments
        """
        self.name = name
        self.obj = obj
        self.start_time = start_time
        self.end_time = end_time
        self.start_value = start_value
        self.target_value = target_value
        self.interpolator = interpolator
        self.kwargs = kwargs
        self.state = AnimationState.WAITING_START

        # Animation
        animation_name = self.__class__.__name__.lower()
        self.init_method = f"init_{animation_name}"
        self.start_method = f"start_{animation_name}"
        self.animate_method = f"animate_{animation_name}"
        self.end_method = f"end_{animation_name}"
        self.finish_method = f"finish_{animation_name}"
    # end __init__

    def start(self):
        """
        Start the animation.
        """
        if self.state == AnimationState.WAITING_START:
            # Enter running state
            self.state = AnimationState.RUNNING

            # Get the init method
            if hasattr(self.obj, self.init_method):
                init_method = getattr(self.obj, self.init_method)
                init_method()
            else:
                raise NotImplementedError(f"{self.init_method} not implemented for {self.obj.__class__.__name__}")
            # end if

            # Get the start method
            if hasattr(self.obj, self.start_method):
                start_method = getattr(self.obj, self.start_method)
                start_method(self.start_value)
            else:
                raise NotImplementedError(f"{self.start_method} not implemented for {self.obj.__class__.__name__}")
            # end if
        # end if
    # end start

    def stop(self):
        """
        Stop the animation.
        """
        if self.state == AnimationState.RUNNING:
            # Entering the finished state
            self.state = AnimationState.FINISHED

            # Get the end method
            if hasattr(self.obj, self.end_method):
                end_method = getattr(self.obj, self.end_method)
                end_method(self.target_value)
            else:
                raise NotImplementedError(f"{self.end_method} not implemented for {self.obj.__class__.__name__}")
            # end if

            # Get the finish method
            if hasattr(self.obj, self.finish_method):
                finish_method = getattr(self.obj, self.finish_method)
                finish_method()
            else:
                raise NotImplementedError(f"{self.finish_method} not implemented for {self.obj.__class__.__name__}")
            # end if
        # end if
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
        if self.state == AnimationState.WAITING_START and t >= self.start_time:
            self.start()
        elif self.state == AnimationState.RUNNING and t > self.end_time:
            self.stop()
            return
        elif self.state in [AnimationState.WAITING_START, AnimationState.FINISHED]:
            return
        # end if

        # Relative time
        relative_t = (t - self.start_time) / (self.end_time - self.start_time)

        # Interpolate time
        interpolated_t = self.interpolator.interpolate(0, 1, relative_t)

        # Get the animate method
        if hasattr(self.obj, self.animate_method):
            animate_method = getattr(self.obj, self.animate_method)
            animate_method(relative_t, self.end_time - self.start_time, interpolated_t, self.target_value)
        else:
            raise NotImplementedError(f"{self.animate_method} not implemented for {self.obj.__class__.__name__}")
        # end if
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
            target_value,
            interpolator=LinearInterpolator(),
            name=""
    ):
        """
        Initialize the move transition.

        Args:
            name (str): Name of the transition
            obj (any): Object to move
            start_time (float): Start time
            end_time (float): End time
            target_value (any): End position
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, MovableMixin), "Object must be an instance of MovAble"
        super().__init__(
            obj,
            start_time,
            end_time,
            None,
            target_value,
            name=name,
            interpolator=interpolator
        )
    # end __init__

# end Move


class Range(Animate):
    """
    A transition that changes a range of values over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            target_value,
            interpolator=LinearInterpolator(),
            name=""
    ):
        """
        Initialize the range transition.

        Args:
            name (str): Name of the transition
            obj (any): Object to range
            start_time (float): Start time
            end_time (float): End time
            target_value (any): End value
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, RangeableMixin), "Object must be an instance of RangeAble"
        super().__init__(
            obj,
            start_time,
            end_time,
            None,
            target_value,
            interpolator=interpolator,
            name=name
        )
    # end __init__

# end Range


# Fade in animation
class FadeIn(Animate):
    """
    A transition that fades in an object over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            interpolator=LinearInterpolator(),
            name="",
    ):
        """
        Initialize the fade-in transition.

        Args:
            obj (any): Object to fade in
            start_time (float): Start time
            end_time (float): End time
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, FadeInableMixin), "Object must be an instance of FadeInAble"
        super().__init__(obj, start_time, end_time, 0, 1, interpolator, name=name)
    # end __init__

# end FadeIn


# Fade out animation
class FadeOut(Animate):
    """
    A transition that fades in an object over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            interpolator=LinearInterpolator(),
            name="",
    ):
        """
        Initialize the fade-out transition.

        Args:
            obj (any): Object to fade in
            start_time (float): Start time
            end_time (float): End time
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, FadeOutableMixin), "Object must be an instance of FadeInAble"
        super().__init__(obj, start_time, end_time, 0, 1, interpolator, name=name)
    # end __init__

# end FadeOut


# Build animation
class Build(Animate):
    """
    A transition that fades in an object over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            interpolator=LinearInterpolator(),
            name="",
    ):
        """
        Initialize the build transition.

        Args:
            obj (any): Object to fade in
            start_time (float): Start time
            end_time (float): End time
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, BuildableMixin), "Object must be an instance of BuildableMixin"
        super().__init__(obj, start_time, end_time, 0, 1, interpolator, name=name)
    # end __init__

# end Build


# Destroy animation
class Destroy(Animate):
    """
    A transition that fades in an object over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            interpolator=LinearInterpolator(),
            name="",
    ):
        """
        Initialize the destroy transition.

        Args:
            obj (any): Object to fade in
            start_time (float): Start time
            end_time (float): End time
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, DestroyableMixin), f"Object must be an instance of DestroyableMixin (type {type(obj)})"
        super().__init__(obj, start_time, end_time, 1, 0, interpolator, name=name)
    # end __init__

# end Destroy

