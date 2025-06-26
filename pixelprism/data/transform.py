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

from pixelprism.animate import animeattr, MovableMixin
from .points import Point2D
from .scalar import Scalar
from .events import Event, EventType


# Transform
# represents the cumulative transformations of the drawable object.
@animeattr("position")
@animeattr("scale")
@animeattr("rotation")
@animeattr("parent")
class Transform(MovableMixin):
    """
    A class representing a series of transformations with position, scale, and rotation,
    which can be part of a hierarchy of transformations.
    """

    def __init__(
            self,
            position: Point2D = Point2D(0.0, 0.0),
            scale: Point2D = Point2D(1.0, 1.0),
            rotation: Scalar = Scalar(0.0),
            parent: 'Transform' = None
    ):
        """
        Initialize the transform with optional parent transform.

        Args:
            position (Point2D): Position of the transform.
            scale (Point2D): Scale of the transform.
            rotation (Scalar): Rotation of the transform.
            parent (Transform): Parent transform in the hierarchy.
        """
        # Init
        MovableMixin.__init__(self)

        # Properties
        self._position = position
        self._scale = scale
        self._rotation = rotation
        self._parent = parent  # Parent transform (can be None)

        # Object changed
        self._on_change = Event()

        # Subscribe to changes in parent transform
        if self._parent:
            self._parent.on_change.subscribe(self._on_parent_changed)
        # end if
    # end __init__

    # region PROPERTIES

    @property
    def position(self) -> Point2D:
        """
        Get the position of the transform.
        """
        return self._position
    # end position

    @position.setter
    def position(self, value: Point2D):
        """
        Set the position of the transform.
        """
        self._position = value
        self._on_change.trigger(self)
    # end position

    @property
    def scale(self) -> Point2D:
        """
        Get the scale of the transform.
        """
        return self._scale
    # end scale

    @scale.setter
    def scale(self, value: Point2D):
        """
        Set the scale of the transform.
        """
        self._scale = value
        self._on_change.trigger(self)
    # end scale

    @property
    def rotation(self) -> Scalar:
        """
        Get the rotation of the transform.
        """
        return self._rotation
    # end rotation

    @rotation.setter
    def rotation(self, value: Scalar):
        """
        Set rotation of the transform.
        """
        self._rotation = value
        self._on_change.trigger(self)
    # end rotation

    @property
    def on_change(self) -> Event:
        """
        On object changed.
        """
        return self._on_change
    # end on_change

    @property
    def parent(self) -> 'Transform':
        """
        Get the parent transform.
        """
        return self._parent
    # end parent

    # endregion PROPERTIES

    # region PUBLIC

    # Copy
    def copy(self, deep: bool = False) -> 'Transform':
        """
        Copy the transform.

        Args:
            deep (bool): Copy the parent transform as well.
        """
        if deep:
            return Transform(
                position=self.position.copy(),
                scale=self.scale.copy(),
                rotation=self.rotation.copy(),
                parent=self.parent.copy() if self.parent else None
            )
        else:
            return Transform(
                position=self.position.copy(),
                scale=self.scale.copy(),
                rotation=self.rotation.copy(),
                parent=self.parent
            )
        # end if
    # end copy

    # Apply cumulative transformation
    def forward(self, relative_point: Point2D) -> Point2D:
        """
        Apply the cumulative transformation (including parent's transform) to a point.
        """
        if self._parent:
            # Apply parent's transformation first
            relative_point = self._parent.forward(relative_point)
        # end if

        # Then apply the current transformation
        return self._forward_local(relative_point)
    # end apply

    def backward(self, absolute_point: Point2D) -> Point2D:
        """
        Reverse the cumulative transformation to a point.

        Args:
            absolute_point (Point2D): Absolute point.

        Returns:
            Point2D: Reversed point.
        """
        # Apply the local transformation in reverse
        point = self._backward_local(absolute_point)

        if self._parent:
            # Reverse the parent's transformation
            return self._parent.backward(point)
        # end if

        return point
    # end reverse

    # endregion PUBLIC

    # region EVENTS

    def _on_parent_changed(self, _):
        """
        Triggered when the parent transform changes.
        """
        self._on_change.trigger(self)
    # end _on_parent_changed

    # endregion EVENTS

    # region PRIVATE

    # Local apply function (for this transform only)
    def _forward_local(self, relative_point: Point2D) -> Point2D:
        """
        Apply the current transform (position, scale, rotation) to a point.
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
    # end _forward_local

    def _backward_local(self, absolute_point: Point2D) -> Point2D:
        """
        Reverse the current transform (position, scale, rotation) on a point.
        """
        # Apply translation
        rel_x = absolute_point.x - self.position.x
        rel_y = absolute_point.y - self.position.y

        # Apply rotation in reverse
        angle = -self.rotation.value
        if angle != 0.0:
            rot_rel_x = rel_x * math.cos(angle) - rel_y * math.sin(angle)
            rot_rel_y = rel_x * math.sin(angle) + rel_y * math.cos(angle)
        else:
            rot_rel_x = rel_x
            rot_rel_y = rel_y
        # end if

        # Apply scale in reverse
        new_x = rot_rel_x / self.scale.x
        new_y = rot_rel_y / self.scale.y

        return Point2D(new_x, new_y)
    # end _backward_local

    # endregion PRIVATE

# end Transform


# class AffinePoint:
#     """
#     A class that synchronizes a relative point and an absolute point
#     using a given Transform object, while avoiding recursive updates.
#     """
#
#     def __init__(
#             self,
#             relative_point: Point2D,
#             absolute_point: Point2D,
#             transform: Optional[Transform] = None
#     ):
#         """
#         Initialize the affine point.
#         """
#         self._relative_point = relative_point
#         self._absolute_point = absolute_point
#         self._transform = transform
#         self._lock = False  # Lock to prevent recursive updates
#
#         # Initial synchronization
#         self._update_absolute_from_relative()
#
#         # Subscribe to changes in relative and absolute points
#         self._relative_point.on_change.subscribe(self._on_relative_point_changed)
#         self._absolute_point.on_change.subscribe(self._on_absolute_point_changed)
#     # end __init__
#
#     # region PROPERTIES
#
#     @property
#     def relative(self):
#         return self._relative_point
#     # end relative
#
#     @property
#     def absolute(self):
#         return self._absolute_point
#     # end absolute
#
#     # endregion PROPERTIES
#
#     # region PRIVATE
#
#     def _update_absolute_from_relative(self):
#         """
#         Update the absolute point based on the relative point and the transform.
#         """
#         if not self._lock:
#             self._lock = True
#             transformed_point = self._transform.forward(self._relative_point)
#             self._absolute_point.x = transformed_point.x
#             self._absolute_point.y = transformed_point.y
#             self._lock = False
#         # end if
#     # end _update_absolute_from_relative
#
#     def _update_relative_from_absolute(self):
#         """
#         Update the relative point based on the absolute point and the inverse transform.
#         """
#         if not self._lock:
#             self._lock = True
#             relative_x = (self._absolute_point.x - self._transform.position.x) / self._transform.scale.x
#             relative_y = (self._absolute_point.y - self._transform.position.y) / self._transform.scale.y
#             self._relative_point.x = relative_x
#             self._relative_point.y = relative_y
#             self._lock = False
#         # end if
#     # end _update_relative_from_absolute
#
#     # endregion PRIVATE
#
#     # region EVENT
#
#     def _on_relative_point_changed(self, point):
#         """
#         Triggered when the relative point changes, updates the absolute point.
#         """
#         self._update_absolute_from_relative()
#     # end _on_relative_point_changed
#
#     def _on_absolute_point_changed(self, point):
#         """
#         Triggered when the absolute point changes, updates the relative point.
#         """
#         self._update_relative_from_absolute()
#     # end _on_absolute_point_changed
#
#     # endregion EVENT
#
# # end AffinePoint

