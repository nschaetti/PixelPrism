# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
from typing import Callable, List, Any
from pixelprism.animate.able import AnimableMixin
from .animate import Animate, AnimationState


# CallableMixin
class CallableMixin(AnimableMixin):
    """
    Interface class for callable objects.
    """

    # Initialize
    def __init__(self):
        """
        Initialize the callable object.
        """
        super().__init__()
        self.callablemixin_state = AnimableMixin.AnimationRegister()
    # end __init__

    # region PUBLIC

    # Create new call animation
    def call(
            self,
            times: List[float],
            func: str,
            values: List[Any]
    ):
        """
        Create a new call animation.

        Args:
            times (list): List of times to call the function
            func (callable): Function to call
            values (list): List of values to pass to the function
        """
        from .animator import Animator

        # Create the animator
        animator = Animator(self)

        # Call the function
        animator.call(
            func=func,
            times=times,
            values=values
        )

        # Add to animation
        self.animable_registry.add(animator)

        # Return the animator
        return animator
    # end call

    # Initialize call
    def init_call(self):
        """
        Initialize the call.
        """
        raise NotImplementedError("Method 'init_call' must be implemented in the derived class.")
    # end init_call

    # Start call
    def start_call(self, *args, **kwargs):
        """
        Start the call.
        """
        raise NotImplementedError("Method 'start_call' must be implemented in the derived class.")
    # end start_call

    # Animate call
    def animate_call(
            self,
            t,
            duration,
            interpolated_t,
            end_value
    ):
        """
        Animate the call.
        """
        raise NotImplementedError("Method 'animate_call' must be implemented in the derived class.")
    # end animate_call

    # End call
    def end_call(self):
        """
        End the call.
        """
        raise NotImplementedError("Method 'end_call' must be implemented in the derived class.")
    # end end_call

    # Finish call
    def finish_call(self):
        """
        Finish the call.
        """
        raise NotImplementedError("Method 'finish_call' must be implemented in the derived class.")
    # end finish_call

# end CallableMixin


# A Call animation
class Call(Animate):
    """
    A transition that calls a function at specific times.
    """

    def __init__(
            self,
            obj,
            func: Callable,
            start_time: float,
            target_value: Any,
    ):
        """
        Initialize the call transition.

        Args:
            obj (any): Object to call
            func (callable): Function to call
            start_time (float): Start time
            target_value (any): Target value
        """
        assert isinstance(obj, CallableMixin), "Object must be an instance of CallableMixin"
        super().__init__(
            obj,
            start_time=start_time,
            target_value=target_value
        )

        # Properties
        self._func = func

        # List of boolean to keep states
        self._exec_state = False
    # end __init__

    # region PROPERTIES

    # Exec. states
    @property
    def exec_state(self) -> bool:
        """
        Get the execution state.

        Returns:
            bool: Whether the function has been executed
        """
        return self._exec_state
    # end exec_state

    @exec_state.setter
    def exec_state(self, value):
        """
        Set the execution states.

        Args:
            value (list): List of execution states
        """
        self._exec_state = value
    # end exec_state

    @property
    def func(self) -> Callable:
        """
        Get the function to call.

        Returns:
            callable: Function to call
        """
        return self._func
    # end func

    # endregion PROPERTIES

    # region PUBLIC

    # Update
    def update(
            self,
            t
    ):
        """
        Update the function call.

        Args:
            t (float): Time of the update
        """
        if t >= self.start_time and not self.exec_state:
            # It's a list
            if isinstance(self.target_value, list):
                self.func(*self.target_value)
            elif isinstance(self.target_value, dict):
                self.func(**self.target_value)
            else:
                self.func(self.target_value)
            # end if
            self.exec_state = True
        # end if
    # end update

    # end PUBLIC

# end Call
