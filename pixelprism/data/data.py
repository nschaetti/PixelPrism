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


class Data:
    """
    A data class that holds data.
    """

    def __init__(self, readonly: bool = False):
        """
        Initialize the data.
        """
        # Data closed ?
        self._data_closed = readonly
    # end __init__

    # region PROPERTIES

    @property
    def data_closed(self) -> bool:
        """
        Data closed property.
        """
        return self._data_closed
    # end data_closed

    @data_closed.setter
    def data_closed(self, value: bool):
        """
        Data closed property.
        """
        self._data_closed = value
    # end data_closed

    # endregion PROPERTIES

    # region PUBLIC

    # Check closed
    def check_closed(self):
        """
        Check if the data is closed.

        Args:
            must_be_closed (bool): Must be closed
        """
        if self.data_closed:
            raise ValueError(f"{self.__class__.__name__} is read only !")
        # end if
    # end check_closed

    # Close data
    def readonly(self):
        """
        Close the data.
        """
        self.data_closed = True
    # end data_close

    # Open data
    def readwrite(self):
        """
        Open the data.
        """
        self.data_closed = False
    # end data_open

    # open

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

    def copy(self, *args, **kwargs):
        """
        Return a copy of the data.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.copy method must be implemented in subclass.")
    # end copy

    # endregion PUBLIC

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
