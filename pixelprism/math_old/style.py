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
from typing import Optional, Union
import cairo

from pixelprism.animate import animeattr, FadeableMixin
# from pixelprism.math_old import (
from pixelprism.data import (
    Color,
    Scalar,
    Event,
    EventType
)


# Style
# represents the style of the drawable object.
@animeattr("fill_color")
@animeattr("line_color")
@animeattr("line_width")
@animeattr("line_cap")
@animeattr("line_join")
@animeattr("line_dash")
class Style(FadeableMixin):

    # Init
    def __init__(
            self,
            fill_color: Color = Color.from_value(0, 0, 0),
            line_color: Color = Color.from_value(0, 0, 0),
            line_width: Scalar = Scalar(1.0),
            line_cap: str = cairo.LineCap.BUTT,
            line_join: str = "round",
            line_dash: list = None
    ):
        """
        Initialize the style.
        """
        # Init
        FadeableMixin.__init__(self)

        # Properties
        self._fill_color = fill_color
        self._line_color = line_color
        self._line_width = line_width
        self._line_cap = line_cap
        self._line_join = line_join
        self._line_dash = line_dash

        # Event
        self._on_change = Event()
    # end __init__

    # region PROPERTIES

    # Properties

    @property
    def fill_color(self) -> Color:
        return self._fill_color
    # end fill_color

    @fill_color.setter
    def fill_color(self, value: Color):
        """
        Set the fill color.

        Args:
            value: Fill color
        """
        self._fill_color.red = value.red
        self._fill_color.green = value.green
        self._fill_color.blue = value.blue
        self._fill_color.alpha = value.alpha
        self.on_change.trigger(
            self,
            event_type=EventType.FILL_COLOR_CHANGED,
            red=value.red,
            green=value.green,
            blue=value.blue,
            alpha=value.alpha
        )
    # end fill_color

    @property
    def line_color(self) -> Color:
        """
        Get the line color.
        """
        return self._line_color
    # end line_color

    @line_color.setter
    def line_color(self, value: Color):
        """
        Set the line color.

        Args:
            value: Line color
        """
        self._line_color.red = value.red
        self._line_color.green = value.green
        self._line_color.blue = value.blue
        self._line_color.alpha = value.alpha
        self.on_change.trigger(
            self,
            event_type=EventType.LINE_COLOR_CHANGED,
            red=value.red,
            green=value.green,
            blue=value.blue,
            alpha=value.alpha
        )
    # end line_color

    @property
    def line_width(self) -> Scalar:
        """
        Get the line width.
        """
        return self._line_width
    # end line_width

    @line_width.setter
    def line_width(self, value: Optional[Union[Scalar, float]]):
        """
        Set the line width.

        Args:
            value: Line width
        """
        if isinstance(value, float):
            self._line_width.value = value
        else:
            self._line_width.value = value.value
        # end if
        self.on_change.trigger(
            self,
            event_type=EventType.LINE_WIDTH_CHANGED,
            width=self._line_width.value
        )
    # end line_width

    @property
    def line_cap(self) -> str:
        return self._line_cap
    # end line_cap

    @line_cap.setter
    def line_cap(self, value: str):
        self._line_cap = value
        self.on_change.trigger(self, event_type=EventType.LINE_CAP_CHANGED, cap=value)
    # end line_cap

    @property
    def line_join(self) -> str:
        return self._line_join
    # end line_join

    @line_join.setter
    def line_join(self, value: str):
        self._line_join = value
        self.on_change.trigger(self, event_type=EventType.LINE_JOIN_CHANGED, join=value)
    # end line_join

    @property
    def line_dash(self) -> list:
        return self._line_dash
    # end line_dash

    @line_dash.setter
    def line_dash(self, value: list):
        self._line_dash = value
        self.on_change.trigger(self, event_type=EventType.LINE_DASH_CHANGED, dash=value)
    # end line_dash

    @property
    def on_change(self) -> Event:
        return self._on_change
    # end on_change

    # endregion PROPERTIES

    # region PUBLIC

    # Copy
    def copy(self) -> 'Style':
        """
        Copy the style.
        """
        return Style(
            fill_color=self.fill_color.copy(),
            line_color=self.line_color.copy(),
            line_width=self.line_width.copy(),
            line_cap=self.line_cap,
            line_join=self.line_join,
            line_dash=self.line_dash
        )
    # end copy

    # endregion PUBLIC

# end Style

