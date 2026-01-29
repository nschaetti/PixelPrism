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

# Imports
from typing import Optional, Union
import math
from pixelprism import utils, p2, s
from pixelprism.animate import animeattr
from pixelprism.base import Context
from pixelprism.data import (
    Color,
    Scalar,
    Point2D,
    Event,
    EventType,
    call_after,
    Transform,
    Style
)


# Drawable mixin
class DrawableMixin:

    # region INIT

    # Init
    def __init__(
            self,
            style: Style = None,
            transform: Transform = None,
    ):
        """
        Initialize the drawable mixin.

        Args:
            transform (Transform): Transform
            style (Style): Style
        """
        # Properties
        self._transform = transform
        self._style = style

        # Events
        self._on_change = Event()

        # Subscribe to transform and style events
        if transform: self._transform.on_change.subscribe(self._on_transform_change)
        if style: self._style.on_change.subscribe(self._on_style_change)
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


