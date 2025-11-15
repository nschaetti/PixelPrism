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

#
# Rotable objects and elements
#

# Imports
from typing import Any
import numpy as np
from .animablemixin import AnimableMixin


class RotableMixin(AnimableMixin):
    """
    Interface class for rotatable objects.
    """

    # Initialize
    def __init__(self):
        """
        Initialize the rotatable object.
        """
        super().__init__()
        self.rotablemixin_state = AnimableMixin.AnimationRegister()
        self.rotablemixin_state.angle = None
    # end __init__

    # region PUBLIC

    # Initialize the rotation
    def init_rotate(self):
        """
        Initialize the rotation.
        """
        raise NotImplementedError("Method 'init_rotation' must be implemented in the derived class.")
    # end init_rotate

    # Start animation
    def start_rotate(self, *args, **kwargs):
        """
        Start the rotation animation.
        """
        raise NotImplementedError("Method 'start_rotation' must be implemented in the derived class.")
    # end start_rotate

    # Animate rotation
    def animate_rotate(
            self,
            t,
            duration,
            interpolated_t,
            end_value
    ):
        """
        Animate the rotation.
        """
        raise NotImplementedError("Method 'animate_rotation' must be implemented in the derived class.")
    # end animate_rotate

    # End rotate
    def end_rotate(self):
        """
        End the rotation.
        """
        raise NotImplementedError("Method 'end_rotation' must be implemented in the derived class.")
    # end end_rotate

    # Finish rotate
    def finish_rotate(self):
        """
        Finish the rotation.
        """
        raise NotImplementedError("Method 'finish_rotation' must be implemented in the derived class.")
    # end finish_rotate

    # endregion PUBLIC

# end RotableMixin

