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
from typing import Any, Optional
from . import LinearInterpolator
from .able import AnimableMixin
from .animate import Animate


# Interface class for fade-in support objects
class FadeableMixin(AnimableMixin):
    """
    Interface class for fade-in animations
    """

    # Constructor
    def __init__(self):
        """
        Initialize the object.
        """
        super().__init__()
        self.fadablemixin_state = AnimableMixin.AnimationRegister()
        self.fadablemixin_state.opacity = None
    # end __init__

    # region PUBLIC

    # Create a fade-in effect
    def fadein(
            self,
            duration: float,
            start_time: Optional[float] = None,
            interpolator=LinearInterpolator()
    ):
        """
        Create a fade-in effect.

        Args:
            start_time (float): Start time of the effect
            duration (float): Duration of the effect
            interpolator (Interpolator): Interpolator
        """
        from .animator import Animator

        # Put into an animator
        animator = Animator(self)

        # Create the fade-in animation
        animator.fadein(
            duration=duration,
            start_time=start_time,
            interpolator=interpolator
        )

        # Add to animation
        self.animable_registry.add(animator)

        return animator
    # end fadein

    # Create a fade-out effect
    def fadeout(
            self,
            duration: float,
            start_time: Optional[float] = None,
            interpolator=LinearInterpolator()
    ):
        """
        Create a fade-in effect.

        Args:
            start_time (float): Start time of the effect
            duration (float): Duration of the effect
            interpolator (Interpolator): Interpolator
        """
        from .animator import Animator

        # Put into an animator
        animator = Animator(self)

        # Create the fade-in animation
        animator.fadeout(
            duration=duration,
            start_time=start_time,
            interpolator=interpolator
        )

        # Add to animation
        self.animable_registry.add(animator)

        return animator
    # end fadeout

    # Initialize fade-in animation
    def init_fadein(self, *args, **kwargs):
        """
        Initialize the fade-in animation.
        """
        pass
    # end init_fadein

    # Start fade-in animation
    def start_fadein(self, start_value: Any, *args, **kwargs):
        """
        Start the fade-in animation.
        """
        pass
    # end start_fadein

    def animate_fadein(
            self,
            t,
            duration,
            interpolated_t,
            *args,
            **kwargs
    ):
        """
        Animate the fade-in effect.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
        """
        self.fadablemixin_state.opacity = interpolated_t
    # end animate_fadein

    def animate_fadeout(
            self,
            t,
            duration,
            interpolated_t,
            *args,
            **kwargs
    ):
        """
        Animate the fade-out effect.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
        """
        self.fadablemixin_state.opacity = 1.0 - interpolated_t
    # end animate_fadeout

    def end_fadein(self, end_value: Any, *args, **kwargs):
        """
        End the fade-in animation.
        """
        pass
    # end end_fadein

    # Finish fade-in animation
    def finish_fadein(self, *args, **kwargs):
        """
        Finish the fade-in animation.
        """
        pass
    # end finish_fadein

    # Initialize fade-out animation
    def init_fadeout(self, *args, **kwargs):
        """
        Initialize the fade-out animation.
        """
        pass
    # end init_fadeout

    # Start fade-out animation
    def start_fadeout(self, start_value: Any, *args, **kwargs):
        """
        Start the fade-out animation.
        """
        pass
    # end start_fadeout

    def end_fadeout(self, end_value: Any, *args, **kwargs):
        """
        End the fade-out animation.
        """
        pass

    # end end_fadeout

    # Finish fade-out animation
    def finish_fadeout(self, *args, **kwargs):
        """
        Finish the fade-out animation.
        """
        pass
    # end finish_fadeout

    # endregion PUBLIC

# end FadeableMixin


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
        assert isinstance(obj, FadeableMixin), "Object must be an instance of FadeInAble"
        super().__init__(
            obj,
            start_time,
            end_time,
            0,
            1,
            interpolator,
            name=name
        )
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
        assert isinstance(obj, FadeableMixin), "Object must be an instance of FadeInAble"
        super().__init__(
            obj,
            start_time,
            end_time,
            0,
            1,
            interpolator,
            name=name
        )
    # end __init__

# end FadeOut

