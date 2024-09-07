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
from typing import Union
from pixel_prism.animate import animeclass, animeattr
from .data import Data
from .scalar import Scalar
from .eventmixin import EventMixin


@animeattr("red")
@animeattr("green")
@animeattr("blue")
@animeattr("alpha")
@animeclass
class Color(Data, EventMixin):
    """
    A class to represent a scalar value
    """

    def __init__(
            self,
            red: Scalar = 0,
            green: Scalar = 0,
            blue: Scalar = 0,
            alpha: Scalar = 1.0,
            on_change=None,
            readonly: bool = False
    ):
        """
        Initialize the scalar value.

        Args:
            red (Scalar): Red value
            green (Scalar): Green value
            blue (Scalar): Blue value
            alpha (Scalar): Alpha value
            on_change (function): On change event
            readonly (bool): Read-only flag
        """
        Data.__init__(self, readonly=readonly)
        EventMixin.__init__(self)

        # Properties
        self._red = red
        self._green = green
        self._blue = blue
        self._alpha = alpha

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    # region PROPERTIES

    @property
    def red(self):
        return self._red
    # end red

    @red.setter
    def red(self, value):
        self.check_closed()
        if isinstance(value, Scalar):
            value = value.value
        # end if
        self._red.value = value
        self.dispatch_event("on_change", value)
    # end red

    @property
    def green(self):
        return self._green
    # end green

    @green.setter
    def green(self, value):
        self.check_closed()
        if isinstance(value, Scalar):
            value = value.value
        # end if
        self._green.value = value
        self.dispatch_event("on_change", value)
    # end green

    @property
    def blue(self):
        return self._blue
    # end blue

    @blue.setter
    def blue(self, value):
        self.check_closed()
        if isinstance(value, Scalar):
            value = value.value
        # end if
        self._blue.value = value
        self.dispatch_event("on_change", value)
    # end blue

    @property
    def alpha(self):
        return self._alpha
    # end alpha

    @alpha.setter
    def alpha(self, value):
        self.check_closed()
        if isinstance(value, Scalar):
            value = value.value
        # end if
        self._alpha.value = value
        self.dispatch_event("on_change", value)
    # end alpha

    @property
    def opacity(self):
        return self.alpha
    # end opacity

    # endregion PROPERTIES

    # region PUBLIC

    def transparent(self):
        """
        Make the color transparent.
        """
        self.check_closed()
        self._alpha.value = 0.0
    # end transparent

    def opaque(self):
        """
        Make the color opaque.
        """
        self.check_closed()
        self._alpha.value = 1.0
    # end opaque

    def set(self, value):
        """
        Set the scalar value.

        Args:
            value (Any): Value to set
        """
        self.check_closed()
        if isinstance(value, Color):
            r, g, b, a = value.get()
        elif isinstance(value, list):
            r, g, b, a = value
        # end if
        self._red = r
        self._green = g
        self._blue = b
        self._alpha = a
        self.dispatch_event("on_change", value)
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        return self.red, self.green, self.blue, self.alpha
    # end get

    def copy(self):
        """
        Return a copy of the data.
        """
        return Color.from_objects(
            self.red.copy(),
            self.green.copy(),
            self.blue.copy(),
            self.alpha.copy()
        )
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

    # endregion PUBLIC

    # region OVERRIDE

    # Addition
    def __add__(self, other):
        """
        Addition operator override for the scalar class.

        Args:
            other (Any): Other value
        """
        if isinstance(other, Color):
            return Color(
                self.red + other.red,
                self.green + other.green,
                self.blue + other.blue,
                self.alpha + other.alpha
            )
        # end if
        return Color(
            self.red + other,
            self.green + other,
            self.blue + other,
            self.alpha
        )

    # end __add__

    # Right-side addition
    def __radd__(self, other):
        return self.__add__(other)
    # end __radd__

    # Subtraction
    def __sub__(self, other):
        """
        Subtraction operator override for the scalar class.

        Args:
            other (Any): Other value
        """
        if isinstance(other, Color):
            return Color(
                self.red - other.red,
                self.green - other.green,
                self.blue - other.blue,
                self.alpha - other.alpha
            )
        # end if
        return Color(
            self.red - other,
            self.green - other,
            self.blue - other,
            self.alpha
        )
    # end __sub__

    # Right-side subtraction
    def __rsub__(self, other):
        # Reverse subtraction: (other - self) => -(self - other)
        return Color(
            other - self.red,
            other - self.green,
            other - self.blue,
            1.0 if isinstance(other, (int, float)) else self.alpha
        )
    # end __rsub__

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, Color):
            return Color(
                self.red * other.red,
                self.green * other.green,
                self.blue * other.blue,
                self.alpha * other.alpha
            )
        # end if
        return Color(
            self.red * other,
            self.green * other,
            self.blue * other,
            self.alpha
        )
    # end __mul__

    # Right-side multiplication
    def __rmul__(self, other):
        return self.__mul__(other)
    # end __rmul__

    # Division
    def __truediv__(self, other):
        if isinstance(other, Color):
            return Color(
                self.red / other.red,
                self.green / other.green,
                self.blue / other.blue,
                self.alpha / other.alpha
            )
        # end if
        return Color(
            self.red / other,
            self.green / other,
            self.blue / other,
            self.alpha
        )
    # end __truediv__

    # Right-side true division
    def __rtruediv__(self, other):
        # Reverse division: (other / self)
        return Color(
            other / self.red,
            other / self.green,
            other / self.blue,
            1.0 if isinstance(other, (int, float)) else self.alpha
        )
    # end __rtruediv__

    # In-place addition
    def __iadd__(self, other):
        """
        In-place addition

        Args:
            other (Any): Other value
        """
        if isinstance(other, Color):
            self.red += other.red
            self.green += other.green
            self.blue += other.blue
            self.alpha += other.alpha
        else:
            self.red += other
            self.green += other
            self.blue += other
        # end if
        self.dispatch_event("on_change", self.get())
        return self
    # end __iadd__

    # In-place subtraction
    def __isub__(self, other):
        if isinstance(other, Color):
            self.red -= other.red
            self.green -= other.green
            self.blue -= other.blue
            self.alpha -= other.alpha
        else:
            self.red -= other
            self.green -= other
            self.blue -= other
        self.dispatch_event("on_change", self.get())
        return self

    # end __isub__

    # In-place multiplication
    def __imul__(self, other):
        if isinstance(other, Color):
            self.red *= other.red
            self.green *= other.green
            self.blue *= other.blue
            self.alpha *= other.alpha
        else:
            self.red *= other
            self.green *= other
            self.blue *= other
        self.dispatch_event("on_change", self.get())
        return self

    # end __imul__

    # In-place division
    def __itruediv__(self, other):
        if isinstance(other, Color):
            self.red /= other.red
            self.green /= other.green
            self.blue /= other.blue
            self.alpha /= other.alpha
        else:
            self.red /= other
            self.green /= other
            self.blue /= other
        self.dispatch_event("on_change", self.get())
        return self

    # end __itruediv__

    # Comparison operators
    def __eq__(self, other):
        if not isinstance(other, Color):
            return False
        return (
                self.red == other.red and
                self.green == other.green and
                self.blue == other.blue and
                self.alpha == other.alpha
        )

    # end __eq__

    def __ne__(self, other):
        return not self.__eq__(other)

    # end __ne__

    # String representation
    def __str__(self):
        return f"Color(red={self.red}, green={self.green}, blue={self.blue}, alpha={self.alpha})"

    # end __str__

    def __repr__(self):
        return self.__str__()
    # end __repr__

    # endregion OVERRIDE

    # region CLASS_METHODS

    # Create a color from a hex string
    @classmethod
    def from_hex(
            cls,
            hex_string: str,
            alpha: float = 1.0
    ):
        """
        Get a color from a hexadecimal string.

        Args:
            hex_string (str): Hexadecimal string
            alpha (float): Alpha value

        Returns:
            Color: Color
        """
        hex_string = hex_string.lstrip("#")
        red = int(hex_string[0:2], 16)
        green = int(hex_string[2:4], 16)
        blue = int(hex_string[4:6], 16)
        return cls(
            red / 255.0,
            green / 255.0,
            blue / 255.0,
            alpha
        )
    # end from_hex

    # From objects
    @classmethod
    def from_objects(
            cls,
            red: Union[float, Scalar],
            green: Union[float, Scalar],
            blue: Union[float, Scalar],
            alpha: Union[float, Scalar] = Scalar(1.0),
            on_change=None,
            readonly: bool = False
    ):
        """
        Create a color from objects.

        Args:
            red (Union[float, Scalar]): Red value
            green (Union[float, Scalar]): Green value
            blue (Union[float, Scalar]): Blue value
            alpha (Union[float, Scalar]): Alpha value
            on_change (function): On change event
            readonly (bool): Read-only flag

        Returns:
            Color: Color
        """
        return cls(red, green, blue, alpha, on_change, readonly)
    # end from_objects

    # From value
    @classmethod
    def from_value(
            cls,
            red: float,
            green: float,
            blue: float,
            alpha: float = 1.0,
            on_change=None,
            readonly: bool = False
    ):
        """
        Create a color from a value.

        Args:
            red (float): Red value
            green (float): Green value
            blue (float): Blue value
            alpha (float): Alpha value
            on_change (function): On change event
            readonly (bool): Read-only flag

        Returns:
            Color: Color
        """
        return cls(Scalar(red), Scalar(green), Scalar(blue), Scalar(alpha), on_change, readonly)
    # end from_value

    # endregion CLASS_METHODS

# end Color
