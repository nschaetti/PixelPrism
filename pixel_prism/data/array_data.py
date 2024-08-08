
# Imports
import numpy as np
from .data import Data


class ArrayData(Data):

    def __init__(self, data, on_change=None, dtype=np.float32):
        """
        Initialize the array data.

        Args:
            data (np.array or list): Initial data
            on_change (callable, optional): Function to call on change
            dtype (data-type, optional): Desired data type for the array
        """
        super().__init__()
        self._data = np.array(data, dtype=dtype)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    # region PROPERTIES

    @property
    def data(self):
        """
        Get the array data.
        """
        return self._data
    # end data

    @data.setter
    def data(self, value):
        """
        Set the array data.
        """
        self._data = np.array(value, dtype=self._data.dtype)
        self.dispatch_event("on_change", self._data)
    # end data

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
        Return a copy of the array data.
        """
        new_copy = ArrayData(self._data.copy())
        return new_copy
    # end copy

    # endregion PUBLIC

    # region OVERRIDE

    def __setitem__(self, key, value):
        """
        Set an item in the array data.

        Args:
            key (int): Index of the item
            value (any): Value to set
        """
        self._data[key] = value
        self.dispatch_event("on_change", self._data)
    # end __setitem__

    def __getitem__(self, key):
        """
        Get an item from the array data.

        Args:
            key (int): Index of the item
        """
        return self._data[key]
    # end __getitem__

    def __len__(self):
        """
        Return the length of the array data.
        """
        return len(self._data)
    # end __len__

    def __str__(self):
        """
        Return a string representation of the array data.
        """
        return str(self._data)
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the array data.
        """
        return f"ArrayData(data={self._data})"
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

