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
from typing import Any
from .animablemixin import AnimableMixin


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
    def fadein(self, start_time: float, duration: float):
        """
        Create a fade-in effect.

        Args:
            start_time (float): Start time of the effect
            duration (float): Duration of the effect
        """
        # Create a fadein animation
        pass
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


