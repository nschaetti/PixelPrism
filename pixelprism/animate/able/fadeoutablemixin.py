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

# Imports
from typing import Any
from .animablemixin import AnimableMixin


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


