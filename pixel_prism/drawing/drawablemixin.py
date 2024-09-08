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
from typing import Optional
import math
from pixel_prism import utils, p2, s
from pixel_prism.base import Context
from pixel_prism.data import (
    Color,
    Scalar,
    Point2D,
    Event,
    EventType,
    call_after,
    Transform
)


# Drawable mixin
class DrawableMixin:

    # Style
    # represents the style of the drawable object.
    class Style:

        # Init
        def __init__(
                self,
                fill_color: Color = Color.from_value(0, 0, 0),
                line_color: Color = Color.from_value(0, 0, 0),
                line_width: Scalar = s(1.0),
                line_cap: str = "round",
                line_join: str = "round",
                line_dash: list = None
        ):
            """
            Initialize the style.
            """
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

        # Properties

        @property
        def fill_color(self) -> Color:
            return self._fill_color
        # end fill_color

        @fill_color.setter
        def fill_color(self, value: Color):
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
            return self._line_color
        # end line_color

        @line_color.setter
        def line_color(self, value: Color):
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
            return self._line_width
        # end line_width

        @line_width.setter
        def line_width(self, value: Optional[Scalar, float]):
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

    # end Style

    # region INIT

    # Init
    def __init__(
            self,
            transform: Transform,
            style: Style
    ):
        """
        Initialize the drawable mixin.

        Args:
            transform (Transform): Transform
            style (Style): Style
        """
        # EventMixin.__init__()
        super().__init__()

        # Properties
        self._transform = transform
        self._style = style

        # Events
        self._on_change = Event()

        # Subscribe to transform and style events
        self._transform.on_change.subscribe(self._on_transform_change)
        self._style.on_change.subscribe(self._on_style_change)
    # end __init__

    # endregion INIT

    # region PROPERTIES

    # Transform
    @property
    def transform(self) -> Transform:
        """
        Transform property.
        """
        return self._transform
    # end transform

    @property
    def style(self) -> Style:
        """
        Style property.
        """
        return self._style
    # end style

    @property
    def length(self) -> Scalar:
        """
        Length property.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.length property must be implemented in subclass.")
    # end length

    @property
    def on_change(self) -> Event:
        """
        On change event.
        """
        return self._on_change
    # end on_change

    # endregion PROPERTIES

    # region PUBLIC

    # Apply transformation from context
    def apply_context(self, context: Context):
        """
        Apply transformation from context.

        Args:
            context (Context): Context
        """
        self.transform.apply_context(context)
    # end apply_context

    # Update object data
    def update_data(
            self
    ):
        """
        Update the data of the drawable.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.update_data method must be implemented in subclass.")
    # end update_data

    # Translate object
    def translate(
            self,
            *args,
            **kwargs
    ):
        """
        Translate the object.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        self._translate_object(*args, **kwargs)
    # end translate

    # Scale object
    def scale(
            self,
            *args,
            **kwargs
    ):
        """
        Scale the object.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        self._scale_object(*args, **kwargs)
    # end scale

    # Rotate object
    def rotate(
            self,
            *args,
            **kwargs
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate
        """
        self._rotate_object(*args, **kwargs)
    # end rotate

    def draw(
            self,
            *args,
            **kwargs
    ):
        """
        Draw the point to the context.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError(f"{self.__class__.__name__}.draw method must be implemented in subclass.")
    # end draw

    # endregion PUBLIC

    # region PRIVATE

    # Scale object (to override)
    def _scale_object(
            self,
            *args,
            **kwargs
    ):
        """
        Scale the object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._scale_object method must be implemented in subclass."
        )
    # end _scale_object

    # Translate object (to override)
    def _translate_object(
            self,
            *args,
            **kwargs
    ):
        """
        Translate the object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._translate_object method must be implemented in subclass."
        )
    # _translate_object

    # Rotate object (to override)
    def _rotate_object(
            self,
            *args,
            **kwargs
    ):
        """
        Rotate the object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._rotate_object method must be implemented in subclass.")
    # end _rotate_object

    # endregion PRIVATE

    # region EVENTS

    @call_after('update_data')
    def _on_transform_change(self, sender, event_type, **kwargs):
        """
        Handle transform change event.
        """
        self.on_change.trigger(self, event_type=event_type, **kwargs)
    # end _on_transform_change

    @call_after('update_data')
    def _on_style_change(self, sender, event_type, **kwargs):
        """
        Handle style change event.
        """
        self.on_change.trigger(self, event_type=event_type, **kwargs)
    # end _on_style_change

    # endregion EVENTS

# end DrawableMixin


