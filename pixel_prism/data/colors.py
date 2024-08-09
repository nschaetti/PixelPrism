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
#
#

# Imports
import numpy as np
from pixel_prism.animate.able import RangeableMixin
from .data import Data


class Color(Data, RangeableMixin):
    """
    A class to represent a scalar value
    """

    def __init__(
            self,
            red=0,
            green=0,
            blue=0,
            alpha=1.0,
            on_change=None
    ):
        """
        Initialize the scalar value.

        Args:
            red (float): Red value
            green (float): Green value
            blue (float): Blue value
            alpha (float): Alpha value
        """
        super().__init__()
        self._value = np.array([red, green, blue, alpha])

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    @property
    def value(self):
        """
        Get the scalar value.
        """
        return self.get()
    # end value

    @value.setter
    def value(self, value):
        """
        Set the scalar value.
        """
        self.set(value)
    # end value

    @property
    def red(self):
        return self._value[0]
    # end red

    @red.setter
    def red(self, value):
        self._value[0] = value
        self.dispatch_event("on_change", value)
    # end red

    @property
    def green(self):
        return self._value[1]
    # end green

    @green.setter
    def green(self, value):
        self._value[1] = value
        self.dispatch_event("on_change", value)
    # end green

    @property
    def blue(self):
        return self._value[2]
    # end blue

    @blue.setter
    def blue(self, value):
        self._value[2] = value
        self.dispatch_event("on_change", value)
    # end blue

    @property
    def alpha(self):
        return self._value[3]
    # end alpha

    @alpha.setter
    def alpha(self, value):
        self._value[3] = value
        self.dispatch_event("on_change", value)
    # end alpha

    @property
    def opacity(self):
        return self.alpha
    # end opacity

    def set(self, value):
        """
        Set the scalar value.

        Args:
            value (Any): Value to set
        """
        if isinstance(value, Color):
            value = value.get()
        elif isinstance(value, list):
            value = np.array(value)
        # end if
        self._value = value
        self.dispatch_event("on_change", value)
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        return self._value[0], self._value[1], self._value[2], self._value[3]
    # end get

    def copy(self):
        """
        Return a copy of the data.
        """
        return Color(self.red, self.green, self.blue, self.alpha)
    # end copy

    # Change alpha
    def change_alpha(self, alpha):
        """
        Change the alpha value of the color.

        Args:
            alpha (int): Alpha value to change to
        """
        self.alpha = alpha
        return self
    # end change_alpha

    # Add color
    def __add__(self, other):
        if isinstance(other, Color):
            return Color(*(self._value + other.value))
        return Color(*(self._value + other))
    # end __add__

    def __sub__(self, other):
        if isinstance(other, Color):
            return Color(*(self._value - other.value))
        return Color(*(self._value - other))
    # end __sub__

    def __mul__(self, other):
        if isinstance(other, Color):
            return Color(*(self._value * other.value))
        return Color(*(self._value * other))
    # end __mul__

    def __truediv__(self, other):
        if isinstance(other, Color):
            return Color(*(self._value / other.value))
        return Color(*(self._value / other))
    # end __truediv__

    def __iadd__(self, other):
        if isinstance(other, Color):
            self._value += other.value
        else:
            self._value += other
        self.dispatch_event("on_change", self._value)
        return self
    # end __iadd__

    def __isub__(self, other):
        if isinstance(other, Color):
            self._value -= other.value
        else:
            self._value -= other
        self.dispatch_event("on_change", self._value)
        return self
    # end __isub__

    def __imul__(self, other):
        if isinstance(other, Color):
            self._value *= other.value
        else:
            self._value *= other
        self.dispatch_event("on_change", self._value)
        return self
    # end __imul__

    def __itruediv__(self, other):
        if isinstance(other, Color):
            self._value /= other.value
        else:
            self._value /= other
        self.dispatch_event("on_change", self._value)
        return self
    # end __itruediv__

    # Comparison operators
    def __eq__(self, other):
        if not isinstance(other, Color):
            return False
        return np.array_equal(self._value, other.value)
    # end __eq__

    def __ne__(self, other):
        return not self.__eq__(other)
    # end __ne__

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        return f"Color(red={self.red}, green={self.green}, blue={self.blue}, alpha={self.alpha})"
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        return self.__str__()
    # end __repr__

# end Color
