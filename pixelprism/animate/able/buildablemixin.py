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
# Description: Mixin class for buildable objects.
#

# Imports
from typing import Any
from .animablemixin import AnimableMixin


class BuildableMixin(AnimableMixin):
    """
    Interface class for buildable objects
    """

    # Initialize
    def __init__(
            self,
            is_built: bool = False,
            build_ratio: float = 0.0,
    ):
        """
        Initialize the buildable object.

        Args:
            is_built (bool): Is the object built
            build_ratio (float): Build ratio
        """
        super().__init__()
        self.is_built = is_built
        self.build_ratio = 1.0 if is_built else build_ratio
    # end __init__

    # Initialize building
    def init_build(self):
        """
        Initialize building the object.
        """
        self.build_ratio = 0.0
        self.is_built = False
    # end init_build

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the object.
        """
        self.build_ratio = 0.0
        self.is_built = False
    # end start_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the object.
        """
        self.build_ratio = 1.0
        self.is_built = True
    # end end_build

    # Animate building
    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the object.
        """
        self.build_ratio = interpolated_t
    # end animate_build

    # Finish building
    def finish_build(self):
        """
        Finish building the object.
        """
        self.build_ratio = 1.0
        self.is_built = True
    # end finish_build

# end BuildableMixin


# DestroyableMixin
class DestroyableMixin(AnimableMixin):
    """
    Interface class for destroyable objects
    """

    # Initialize
    def __init__(
            self,
            is_built: bool = True,
            build_ratio: float = 1.0,
    ):
        """
        Initialize the buildable object.

        Args:
            is_built (bool): Is the object built
            build_ratio (float): Build ratio
        """
        super().__init__()
        self.is_built = is_built
        self.build_ratio = 1.0 if is_built else build_ratio
    # end __init__

    # Initialize building
    def init_destroy(self):
        """
        Initialize building the object.
        """
        self.build_ratio = 1.0
        self.is_built = True
    # end init_build

    # Start building
    def start_destroy(self, start_value: Any):
        """
        Start building the object.
        """
        self.build_ratio = 1.0
        self.is_built = False
    # end start_build

    # Animate building
    def animate_destroy(self, t, duration, interpolated_t, env_value):
        """
        Animate building the object.
        """
        self.build_ratio = (1 - interpolated_t)
        # print(f"{self.__class__.__name__} DestroyableMixin.animate_destroy: build_ratio={self.build_ratio}")
    # end animate_build

    # End building
    def end_destroy(self, end_value: Any):
        """
        End building the object.
        """
        self.build_ratio = 0.0
        self.is_built = False
    # end end_build

    # Finish building
    def finish_destroy(self):
        """
        Finish destroying the object.
        """
        self.build_ratio = 0.0
        self.is_built = False
    # end finish_destroy

# end BuildableMixin

