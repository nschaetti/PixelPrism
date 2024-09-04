#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
from enum import Enum
from typing import Optional, Any

from .able import (
    RotableMixin,
    ScalableMixin,
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
            start_time: Optional[float] = None,
            end_time: Optional[float] = None,
            start_value: Optional[Any] = None,
            target_value: Optional[Any] = None,
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
        self._name = name
        self._obj = obj
        self._start_time = start_time
        self._end_time = end_time
        self._start_value = start_value
        self._target_value = target_value
        self._interpolator = interpolator
        self._kwargs = kwargs
        self._state = AnimationState.WAITING_START

        # Animation
        animation_name = self.__class__.__name__.lower()
        self._init_method = f"init_{animation_name}"
        self._start_method = f"start_{animation_name}"
        self._animate_method = f"animate_{animation_name}"
        self._end_method = f"end_{animation_name}"
        self._finish_method = f"finish_{animation_name}"
    # end __init__

    # region PROPERTIES

    @property
    def name(self):
        """
        Get the name of the transition.
        """
        return self._name
    # end name

    @property
    def obj(self):
        """
        Get the object of the transition.
        """
        return self._obj
    # end obj

    @property
    def start_time(self):
        """
        Get the start time of the transition.
        """
        return self._start_time
    # end start_time

    @property
    def end_time(self):
        """
        Get the end time of the transition.
        """
        return self._end_time
    # end end_time

    @property
    def start_value(self):
        """
        Get the start value of the transition.
        """
        return self._start_value
    # end start_value

    @property
    def target_value(self):
        """
        Get the target value of the transition.
        """
        return self._target_value
    # end target_value

    @property
    def interpolator(self):
        """
        Get the interpolator of the transition.
        """
        return self._interpolator
    # end interpolator

    @property
    def kwargs(self):
        """
        Get the keyword arguments of the transition.
        """
        return self._kwargs
    # end kwargs

    @property
    def state(self):
        """
        Get the state of the transition.
        """
        return self._state
    # end state

    @state.setter
    def state(self, value):
        """
        Set the state of the transition.
        """
        self._state = value
    # end state

    @property
    def init_method(self):
        """
        Get the init method of the transition.
        """
        return self._init_method
    # end init_method

    @property
    def start_method(self):
        """
        Get the start method of the transition.
        """
        return self._start_method
    # end start_method

    @property
    def animate_method(self):
        """
        Get the animate method of the transition.
        """
        return self._animate_method
    # end animate_method

    @property
    def end_method(self):
        """
        Get the end method of the transition.
        """
        return self._end_method
    # end end_method

    @property
    def finish_method(self):
        """
        Get the finish method of the transition.
        """
        return self._finish_method
    # end finish_method

    # endregion PROPERTIES

    # region PUBLIC

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
                init_method(**self.kwargs)
            else:
                raise NotImplementedError(f"{self.init_method} not implemented for {self.obj.__class__.__name__}")
            # end if

            # Get the start method
            if hasattr(self.obj, self.start_method):
                start_method = getattr(self.obj, self.start_method)
                start_method(self.start_value, **self.kwargs)
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
                end_method(self.target_value, **self.kwargs)
            else:
                raise NotImplementedError(f"{self.end_method} not implemented for {self.obj.__class__.__name__}")
            # end if

            # Get the finish method
            if hasattr(self.obj, self.finish_method):
                finish_method = getattr(self.obj, self.finish_method)
                finish_method(**self.kwargs)
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
            animate_method(
                t=relative_t,
                duration=self.end_time - self.start_time,
                interpolated_t=interpolated_t,
                end_value=self.target_value,
                **self.kwargs
            )
        else:
            raise NotImplementedError(f"{self.animate_method} not implemented for {self.obj.__class__.__name__}")
        # end if
    # end update

    # endregion PUBLIC

    # region OVERRIDE

    def __str__(self):
        """
        Return a string representation of the transition.
        """
        return (
            f"{self.__class__.__name__}(name={self.name}, obj={self.obj}, "
            f"start_time={self.start_time}, end_time={self.end_time}, start_value={self.start_value}, "
            f"target_value={self.target_value}, interpolator={self.interpolator})"
        )
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the transition.
        """
        return (
            f"{self.__class__.__name__}(name={self.name}, obj={self.obj}, start_time={self.start_time}, "
            f"end_time={self.end_time}, start_value={self.start_value}, target_value={self.target_value}, "
            f"interpolator={self.interpolator})"
        )
    # end __repr__

    # Get item in kwargs
    def __getitem__(self, key):
        """
        Get an item from the keyword arguments.
        """
        return self.kwargs[key]
    # end __getitem__

    # Set item in kwargs
    def __setitem__(self, key, value):
        """
        Set an item in the keyword arguments.
        """
        self.kwargs[key] = value
    # end __setitem__

    # endregion OVERRIDE

# end Animate



class Rotate(Animate):
    """
    A transition that rotates an object over time.
    """

    # Constructor
    def __init__(
            self,
            obj,
            start_time,
            end_time,
            target_value,
            interpolator=LinearInterpolator(),
            name="",
            **kwargs
    ):
        """
        Initialize the rotation transition.

        Args:
            name (str): Name of the transition
            obj (any): Object to move
            start_time (float): Start time
            end_time (float): End time
            target_value (any): End position
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, RotableMixin), "Object must be an instance of RotableMixin"
        super().__init__(
            obj,
            start_time,
            end_time,
            None,
            target_value,
            name=name,
            interpolator=interpolator,
            **kwargs
        )
    # end __init__

    # region PUBLIC

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
            animate_method(
                t=relative_t,
                duration=self.end_time - self.start_time,
                interpolated_t=interpolated_t,
                end_value=self.target_value,
                **self.kwargs
            )
        else:
            raise NotImplementedError(f"{self.animate_method} not implemented for {self.obj.__class__.__name__}")
        # end if
    # end update

    # endregion PUBLIC

# end Rotate


# Scale
class Scale(Animate):
    """
    A transition that scales an object over time.
    """

    def __init__(
            self,
            obj,
            start_time,
            end_time,
            target_value,
            interpolator=LinearInterpolator(),
            name="",
            **kwargs
    ):
        """
        Initialize the scale transition.

        Args:
            name (str): Name of the transition
            obj (any): Object to scale
            start_time (float): Start time
            end_time (float): End time
            target_value (any): End scale
            interpolator (Interpolator): Interpolator
        """
        assert isinstance(obj, ScalableMixin), "Object must be an instance of ScalableMixin"
        super().__init__(
            obj,
            start_time,
            end_time,
            None,
            target_value,
            interpolator=interpolator,
            name=name,
            **kwargs
        )
    # end __init__

    # region PUBLIC

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
            animate_method(
                t=relative_t,
                duration=self.end_time - self.start_time,
                interpolated_t=interpolated_t,
                end_value=self.target_value,
                **self.kwargs
            )
        else:
            raise NotImplementedError(f"{self.animate_method} not implemented for {self.obj.__class__.__name__}")
        # end if
    # end update

# end Scale


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

