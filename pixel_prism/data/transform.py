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
import math
from typing import Optional
from .points import Point2D
from .scalar import Scalar
from .events import Event, EventType
from ..base import Context


# Transform
# represents the transformation of the drawable object.
class Transform:

    # Init
    def __init__(
            self,
            position: Point2D,
            scale: Point2D = Point2D(1.0, 1.0),
            rotation: Scalar = Scalar(0.0)
    ):
        """
        Initialize the context.
        """
        # Properties
        self._position = position
        self._scale = scale
        self._rotation = rotation

        # Event
        self._on_change = Event()
    # end __init__

    # region PROPERTIES

    # Properties
    @property
    def position(self) -> Point2D:
        return self._position
    # end position

    @position.setter
    def position(self, value: Point2D):
        self._position.x = value.x
        self._position.y = value.y
        self.on_change.trigger(self, event_type=EventType.POSITION_CHANGED, x=value.x, y=value.y)
    # end position

    @property
    def scale(self) -> Point2D:
        return self._scale
    # end scale

    @scale.setter
    def scale(self, value: Point2D):
        self._scale.x = value
        self._scale.y = value
        self.on_change.trigger(self, event_type=EventType.SCALE_CHANGED, sx=value.x, sy=value.y)
    # end scale

    @property
    def rotation(self) -> Scalar:
        return self._rotation
    # end rotation

    @rotation.setter
    def rotation(self, value: Optional[Scalar, float]):
        if isinstance(value, float):
            self._rotation.value = value
        else:
            self._rotation.value = value.value
        # end if
        self.on_change.trigger(self, event_type=EventType.ROTATION_CHANGED, angle=self._rotation.value)
    # end rotation

    @property
    def on_change(self) -> Event:
        return self._on_change
    # end on_change

    # endregion PROPERTIES

    # region PUBLIC

    # Apply transformation
    def apply(self, relative_point: Point2D) -> Point2D:
        """
        Apply transformation to a point.

        Args:
            relative_point (Point2D): Relative point
        """
        return self._apply(relative_point)
    # end apply

    # Reverse transformation
    def reverse(self, absolute_point: Point2D) -> Point2D:
        """
        Reverse transformation to a point.

        Args:
            absolute_point (Point2D): Absolute point
        """
        return self._reverse(absolute_point)
    # end reverse

    # Apply transformation (in-place)
    def apply_(self, relative_point: Point2D):
        """
        Apply transformation to a point in-place.
        """
        # Transform
        point = self._apply(relative_point)
        relative_point.x = point.x
        relative_point.y = point.y
    # end apply_

    # Reverse transformation (in-place)
    def reverse_(self, absolute_point: Point2D):
        """
        Reverse transformation to a point in-place.
        """
        # Transform
        point = self._reverse(absolute_point)
        absolute_point.x = point.x
        absolute_point.y = point.y
    # end reverse_

    # Apply transformation from context
    def apply_context(self, context: Context):
        """
        Apply transformation from context.

        Args:
            context (Context): Context
        """
        context.translate(self.position)
        context.rotate(self.rotation)
        context.scale(self.scale)
    # end apply_context

    # endregion PUBLIC

    # region PRIVATE

    # Apply transformation
    def _apply(self, relative_point: Point2D) -> Point2D:
        """
        Apply transformation to a point.

        Args:
            relative_point (Point2D): Relative point
        """
        # Apply scale
        abs_x = relative_point.x * self.scale.x
        abs_y = relative_point.y * self.scale.y

        # Apply rotation
        angle = self.rotation.value
        if angle != 0.0:
            rot_abs_x = abs_x * math.cos(angle) - abs_y * math.sin(angle)
            rot_abs_y = abs_x * math.sin(angle) + abs_y * math.cos(angle)
        else:
            rot_abs_x = abs_x
            rot_abs_y = abs_y
        # end if

        # Apply translation
        new_x = self.position.x + rot_abs_x
        new_y = self.position.y + rot_abs_y

        return Point2D(new_x, new_y)
    # end _apply

    # Reverse transformation
    def _reverse(self, absolute_point: Point2D) -> Point2D:
        """
        Reverse transformation to a point.

        Args:
            absolute_point (Point2D): Absolute point
        """
        # Apply translation
        rel_x = absolute_point.x - self.position.x
        rel_y = absolute_point.y - self.position.y

        # Apply rotation
        angle = -self.rotation.value
        if angle != 0.0:
            rot_rel_x = rel_x * math.cos(angle) - rel_y * math.sin(angle)
            rot_rel_y = rel_x * math.sin(angle) + rel_y * math.cos(angle)
        else:
            rot_rel_x = rel_x
            rot_rel_y = rel_y
        # end if

        # Apply scale
        new_x = rot_rel_x / self.scale.x
        new_y = rot_rel_y / self.scale.y

        return Point2D(new_x, new_y)
    # end _reverse

    # endregion PRIVATE

# end Transform

