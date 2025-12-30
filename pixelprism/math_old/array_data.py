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
from .data import Data


class ArrayData(Data):

    def __init__(self, data, on_change=None, dtype=np.float32):
        """
        Initialize the array math_old.

        Args:
            data (np.array or list): Initial math_old
            on_change (callable, optional): Function to call on change
            dtype (math_old-type, optional): Desired math_old type for the array
        """
        Data.__init__(self)
        self._data = np.array(data, dtype=dtype)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    # region PROPERTIES

    @property
    def data(self):
        """
        Get the array math_old.
        """
        return self._data
    # end math_old

    @data.setter
    def data(self, value):
        """
        Set the array math_old.
        """
        self._data = np.array(value, dtype=self._data.dtype)
        self.dispatch_event("on_change", self._data)
    # end math_old

    # endregion PROPERTIES

    # region PUBLIC

    def dispatch_event(self, event, *args, **kwargs):
        """
        Dispatch an event to all listeners.

        Args:
            event (str): Event to dispatch
        """
        if event in self.event_listeners:
            for listener in self.event_listeners[event]:
                listener(*args, **kwargs)
            # end for
        # end if
    # end dispatch_event

    def copy(self):
        """
        Return a copy of the array math_old.
        """
        new_copy = ArrayData(self._data.copy())
        return new_copy
    # end copy

    # endregion PUBLIC

    # region OVERRIDE

    def __setitem__(self, key, value):
        """
        Set an item in the array math_old.

        Args:
            key (int): Index of the item
            value (any): Value to set
        """
        self._data[key] = value
        self.dispatch_event("on_change", self._data)
    # end __setitem__

    def __getitem__(self, key):
        """
        Get an item from the array math_old.

        Args:
            key (int): Index of the item
        """
        return self._data[key]
    # end __getitem__

    def __len__(self):
        """
        Return the length of the array math_old.
        """
        return len(self._data)
    # end __len__

    def __str__(self):
        """
        Return a string representation of the array math_old.
        """
        return str(self._data)
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the array math_old.
        """
        return f"ArrayData(math_old={self._data})"
    # end __repr__

    def __add__(self, other):
        """
        Addition operator.
        """
        if isinstance(other, ArrayData):
            return ArrayData(self._data + other._data)
        elif isinstance(other, list):
            return ArrayData(self._data + np.array(other, dtype=self._data.dtype))
        else:
            return ArrayData(self._data + other)
        # end if
    # end __add__

    def __radd__(self, other):
        """
        Reverse addition operator.
        """
        return self.__add__(other)
    # end __radd__

    def __sub__(self, other):
        """
        Subtraction operator.
        """
        if isinstance(other, ArrayData):
            return ArrayData(self._data - other._data)
        elif isinstance(other, list):
            return ArrayData(self._data - np.array(other, dtype=self._data.dtype))
        else:
            return ArrayData(self._data - other)
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Reverse subtraction operator.
        """
        if isinstance(other, ArrayData):
            return ArrayData(other._data - self._data)
        elif isinstance(other, list):
            return ArrayData(np.array(other, dtype=self._data.dtype) - self._data)
        else:
            return ArrayData(other - self._data)
        # end if
    # end __rsub__

    def __mul__(self, other):
        """
        Multiplication operator.
        """
        if isinstance(other, ArrayData):
            return ArrayData(self._data * other._data)
        elif isinstance(other, list):
            return ArrayData(self._data * np.array(other, dtype=self._data.dtype))
        else:
            return ArrayData(self._data * other)
        # end if
    # end __mul__

    def __rmul__(self, other):
        """
        Reverse multiplication operator.
        """
        return self.__mul__(other)
    # end __rmul__

    def __truediv__(self, other):
        """
        Division operator.
        """
        if isinstance(other, ArrayData):
            return ArrayData(self._data / other._data)
        elif isinstance(other, list):
            return ArrayData(self._data / np.array(other, dtype=self._data.dtype))
        else:
            return ArrayData(self._data / other)
        # end if
    # end __truediv__

    def __rtruediv__(self, other):
        """
        Reverse division operator.
        """
        if isinstance(other, ArrayData):
            return ArrayData(other._data / self._data)
        elif isinstance(other, list):
            return ArrayData(np.array(other, dtype=self._data.dtype) / self._data)
        else:
            return ArrayData(other / self._data)
        # end if
    # end __rtruediv__

    def __eq__(self, other):
        """
        Equality operator.
        """
        if isinstance(other, ArrayData):
            return np.array_equal(self._data, other._data)
        elif isinstance(other, list):
            return np.array_equal(self._data, np.array(other, dtype=self._data.dtype))
        else:
            return np.array_equal(self._data, other)
        # end if
    # end __eq__

    def __ne__(self, other):
        """
        Not equal operator.
        """
        return not self.__eq__(other)
    # end __ne__

    # endregion OVERRIDE

# end ArrayData

