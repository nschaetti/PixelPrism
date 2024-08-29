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
from typing import Any
from . import LinearInterpolator
from .able import AnimableMixin
from .animate import Animate


# Interface class for fade-in support objects
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
        self.fadeinablemixin_state = AnimableMixin.AnimationRegister()
        self.fadeinablemixin_state.opacity = None
    # end __init__

    # region PUBLIC

    # Create a fade-in effect
    def fadein(self, start_time: float, duration: float, interpolator=LinearInterpolator()):
        """
        Create a fade-in effect.

        Args:
            start_time (float): Start time of the effect
            duration (float): Duration of the effect
            interpolator (Interpolator): Interpolator
        """
        from .animator import Animator

        # Put into an animator
        fadein_animator = Animator(self)

        # Create the fade-in animation
        fadein_animator.fadein(start_time, duration, interpolator)
    # end fadein

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
        self.fadeinablemixin_state.opacity = interpolated_t
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

    # endregion PUBLIC

# end FadeInableMixin


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

    # Initialize fade-out animation
    def init_fadeout(self):
        """
        Initialize the fade-out animation.
        """
        pass
    # end init_fadeout

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

    # Finish fade-out animation
    def finish_fadeout(self):
        """
        Finish the fade-out animation.
        """
        pass
    # end finish_fadeout

# end FadeOutAble



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

