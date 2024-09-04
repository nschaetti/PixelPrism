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

#
# Description: Movable interface class
#

# Imports
from typing import Any
from .interpolate import Interpolator, LinearInterpolator, EaseInOutInterpolator
from .able import AnimableMixin
from .animate import Animate, AnimationState



class MovableMixin(AnimableMixin):
    """
    Interface class for movable objects
    """

    # Initialize
    def __init__(self):
        """
        Initialize the movable object.
        """
        super().__init__()
        self.movablemixin_state = AnimableMixin.AnimationRegister()
        self.movablemixin_state.start_position = None
        self.movablemixin_state.last_position = None
    # end __init__

    # region PROPERTIES

    @property
    def movable_position(self) -> Any:
        """
        Get the position of the object.

        Returns:
            any: Position of the object
        """
        raise NotImplementedError("Property 'movable_position' must be implemented in the derived class.")
    # end movable_position

    @movable_position.setter
    def movable_position(self, value: Any):
        """
        Set the position of the object.

        Args:
            value (any): Position of the object
        """
        raise NotImplementedError("Property 'movable_position' must be implemented in the derived class.")
    # end movable_position

    # end PROPERTIES

    # region PUBLIC

    # Create new move animation
    def move(
            self,
            duration: float,
            target_value: Any,
            start_time: float = None,
            interpolator=EaseInOutInterpolator(),
            relative: bool = False
    ):
        """
        Create a move animation.

        Args:
            start_time (float): Start time
            duration (float): End time
            target_value (any): Target value
            interpolator (Interpolator): Interpolator
            relative (bool): Relative movement
        """
        from .animator import Animator
        animator = Animator(self)
        animator.move(
            duration=duration,
            target_value=target_value,
            start_time=start_time,
            interpolator=interpolator,
            relative=relative
        )
        return animator
    # end move

    # Initialize position
    def init_move(self, *args, **kwargs):
        """
        Initialize the move animation.
        """
        self.movablemixin_state.start_position = None
    # end init_move

    # Start animation
    def start_move(
            self,
            start_value: Any,
            *args,
            **kwargs
    ):
        """
        Start the move animation.

        Args:
            start_value (any): The start position of the object
        """

        self.movablemixin_state.start_position = self.movable_position.copy()
    # end start_move

    def animate_move(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            *args,
            **kwargs
    ):
        """
        Perform the move animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The end position of the object
        """
        start_position = self.movablemixin_state.start_position
        self.movable_position = start_position * (1 - interpolated_t) + end_value * interpolated_t
        self.movablemixin_state.last_position = self.movable_position
    # end animate_move

    # Stop animation
    def end_move(
            self,
            end_value: Any,
            *args,
            **kwargs
    ):
        """
        Stop the move animation.

        Args:
            end_value (any): The end position of the object
        """
        pass
    # end end_move

    # Finish animation
    def finish_move(self, *args, **kwargs):
        """
        Finish the move animation.
        """
        pass
    # end finish_move

    # endregion PUBLIC

# end MovAble


class ScalableMixin(AnimableMixin):
    """
    Interface class for ranging values
    """

    # Initialize
    def __init__(
            self
    ):
        """
        Initialize the movable object.
        """
        super().__init__()
        self.scalablemixin_state = AnimableMixin.AnimationRegister()
        self.scalablemixin_state.scale = None
    # end __init__

    # Initialize animation
    def init_scale(
            self,
            *args,
            **kwargs
    ):
        """
        Initialize the range animation.
        """
        raise NotImplementedError(f"Method 'init_scale' must be implemented in the {self.__class__.__name__} class.")
    # end init_scale

    # Start animation
    def start_scale(
            self,
            start_value: Any,
            center: Any,
            *args,
            **kwargs
    ):
        """
        Start the range animation.

        Args:
            start_value (any): The start position of the object
            center (Point2D): Center of the scaling
        """
        raise NotImplementedError(f"Method 'start_scale' must be implemented in the {self.__class__.__name__} class.")
    # end start_scale

    def animate_scale(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            center,
            *args,
            **kwargs
    ):
        """
        Perform the move animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The end position of the object
            center (Point2D): Center of the scaling
        """
        raise NotImplementedError(f"Method 'animate_scale' must be implemented in the {self.__class__.__name__} class.")
    # end animate_scale

    # Stop animation
    def end_scale(
            self,
            end_value: Any,
            *args,
            **kwargs
    ):
        """
        Stop the range animation.

        Args:
            end_value (any): The end value of the object
        """
        raise NotImplementedError(f"Method 'end_scale' must be implemented in the {self.__class__.__name__} class.")
    # end end_scale

    # Finish animation
    def finish_scale(
            self,
            *args,
            **kwargs
    ):
        """
        Finish the range animation.
        """
        raise NotImplementedError(f"Method 'finish_scale' must be implemented in the {self.__class__.__name__} class.")
    # end finish_scale

# end ScalableMixin





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
            name="",
            relative: bool = False,
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
            relative (bool): Relative movement
        """
        assert isinstance(obj, MovableMixin), "Object must be an instance of MovAble"
        super().__init__(
            obj,
            start_time,
            end_time,
            None,
            target_value,
            name=name,
            interpolator=interpolator,
            relative=relative
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
                relative=self.kwargs.get("relative", False)
            )
        else:
            raise NotImplementedError(f"{self.animate_method} not implemented for {self.obj.__class__.__name__}")
        # end if
    # end update

    # endregion PUBLIC

# end Move


