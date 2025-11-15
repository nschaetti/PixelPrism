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
import numpy as np


class Interpolator:
    """
    An interpolator that interpolates between two values.
    """

    def interpolate(
            self,
            start_value,
            end_value,
            progress
    ):
        """
        Interpolate between two values.

        Args:
            start_value (any): Start value
            end_value (any): End value
            progress (float): Progress
        """
        raise NotImplementedError("Interpolate method must be implemented by subclasses")
    # end interpolate

# end Interpolator


class LinearInterpolator(Interpolator):
    """
    Linear interpolator
    """

    def interpolate(
            self,
            start_value,
            end_value,
            progress
    ):
        """
        Interpolate between two values linearly.

        Args:
            start_value (any): Start value
            end_value (any): End value
            progress (float): Progress
        """
        return start_value + progress * (end_value - start_value)
    # end interpolate

# end LinearInterpolator


class EaseInOutInterpolator(Interpolator):
    """
    Ease in out interpolator
    """

    def interpolate(
            self,
            start_value,
            end_value,
            progress
    ):
        """
        Interpolate between two values with ease in out.

        Args:
            start_value (any): Start value
            end_value (any): End value
            progress (float): Progress
        """
        return start_value + self.ease_in_out(progress) * (end_value - start_value)
    # end interpolate

    @staticmethod
    def ease_in_out(
            progress
    ):
        """
        Ease in out function.

        Args:
            progress (float): Progress
        """
        return -0.5 * (np.cos(np.pi * progress) - 1)
    # end ease

# end EaseInOutInterpolator

