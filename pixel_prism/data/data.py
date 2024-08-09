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


class Data(object):
    """
    A data class that holds data.
    """

    def set(self, *args, **kwargs):
        """
        Set the scalar value.

        Args:
            value (any): Value to set
        """
        raise NotImplementedError(f"{self.__class__.__name__}.set method must be implemented in subclass.")
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.get method must be implemented in subclass.")
    # end get

    def copy(self):
        """
        Return a copy of the data.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.copy method must be implemented in subclass.")
    # end copy

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.__str__ method must be implemented in subclass.")
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.__repr__ method must be implemented in subclass.")
    # end __repr__

# end Data
