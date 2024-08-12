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


from typing import List, Tuple, Sequence
import cairo
from pixel_prism.data import Color, Point2D
from .coordsystem import CoordSystem


class Context:
    """
    Context class for the Pixel Prism library.
    """

    # Initialize
    def __init__(
            self,
            image,
            coord_system: CoordSystem,
    ):
        """
        Initialize the context.

        Args:
            image (Image): Image
            coord_system (CoordSystem): Coordinate system
        """
        # Properties
        self._image = image
        self._coord_system = coord_system
        self._surface = self.create_surface()
        self._context = self.create_context()
        self._image_width = self._image.width
        self._image_height = self._image.height
    # end __init__

    # region PROPERTIES

    @property
    def image(self):
        """
        Get the image.
        """
        return self._image
    # end image

    @property
    def surface(self):
        """
        Get the surface.
        """
        return self._surface
    # end surface

    @property
    def image_width(self):
        """
        Get the width.
        """
        return self._image_width
    # end image_width

    @property
    def image_height(self):
        """
        Get the height.
        """
        return self._image_height
    # end image_height

    @property
    def size(self):
        """
        Get the size.
        """
        return self._coord_system.size
    # end size

    @property
    def width(self):
        """
        Get the width.
        """
        return self._coord_system.width
    # end width

    @property
    def height(self):
        """
        Get the height.
        """
        return self._coord_system.height
    # end height

    @property
    def x_range(self):
        """
        Get the x range.
        """
        return self._coord_system.x_range
    # end x_range

    @property
    def y_range(self):
        """
        Get the y range.
        """
        return self._coord_system.y_range
    # end y_range

    @property
    def context(self):
        """
        Get the context.
        """
        return self._context
    # end context

    @property
    def coord_system(self):
        """
        Get the coordinate system.
        """
        return self._coord_system
    # end coord_system

    # endregion PROPERTIES

    # region PUBLIC

    # Arc
    def arc(self, xc: float, yc: float, radius: float, angle1: float, angle2: float) -> None:
        """
        Add an arc to the context.

        Args:
            xc (float): X center
            yc (float): Y center
            radius (float): Radius
            angle1 (float): Start angle
            angle2 (float): End angle
        """
        self.context.arc(xc, yc, radius, angle1, angle2)
    # end arc

    # Arc negative
    def arc_negative(self, xc: float, yc: float, radius: float, angle1: float, angle2: float) -> None:
        """
        Add a negative arc to the context.

        Args:
            xc (float): X center
            yc (float): Y center
            radius (float): Radius
            angle1 (float): Start angle
            angle2 (float): End angle
        """
        self.context.arc_negative(xc, yc, radius, angle1, angle2)
    # end arc_negative

    # Curve to
    def curve_to(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> None:
        """
        Add a curve to the context.

        Args:
            x1 (float): X1 coordinate
            y1 (float): Y1 coordinate
            x2 (float): X2 coordinate
            y2 (float): Y2 coordinate
            x3 (float): X3 coordinate
            y3 (float): Y3 coordinate
        """
        self.context.curve_to(x1, y1, x2, y2, x3, y3)
    # end curve_to

    def fill(self) -> None:
        """
        Fill the context.
        """
        self.context.fill()
    # end fill

    def fill_extents(self) -> Tuple[float, float, float, float]:
        """
        Fill the extents.

        Returns:
            tuple: Fill extents
        """
        return self.context.fill_extents()
    # end fill_extents

    def fill_preserve(self) -> None:
        """
        Fill and preserve the context.
        """
        self.context.fill_preserve()
    # end fill_preserve

    def get_line_width(self) -> float:
        """
        Get the line width.

        Returns:
            float: Line width
        """
        return self.context.get_line_width()
    # end get_line_width

    def line_to(self, x: float, y: float) -> None:
        """
        Add a line to the context.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.context.line_to(x, y)
    # end line_to

    def move_to(self, x: float, y: float) -> None:
        """
        Move to a point in the context.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.context.move_to(x, y)
    # end move_to

    def rectangle(self, x: float, y: float, width: float, height: float) -> None:
        """
        Add a rectangle to the context.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
            width (float): Width
            height (float): Height
        """
        self.context.rectangle(x, y, width, height)
    # end rectangle

    def rel_curve_to(self, dx1: float, dy1: float, dx2: float, dy2: float, dx3: float, dy3: float) -> None:
        """
        Add a relative curve to the context.

        Args:
            dx1 (float): X1 distance
            dy1 (float): Y1 distance
            dx2 (float): X2 distance
            dy2 (float): Y2 distance
            dx3 (float): X3 distance
            dy3 (float): Y3 distance
        """
        self.context.rel_curve_to(dx1, dy1, dx2, dy2, dx3, dy3)
    # end rel_curve_to

    def rel_line_to(self, dx: float, dy: float) -> None:
        """
        Add a relative line to the context.

        Args:
            dx (float): X distance
            dy (float): Y distance
        """
        self.context.rel_line_to(dx, dy)
    # end rel_line_to

    def rel_move_to(self, dx: float, dy: float) -> None:
        """
        Move to a relative point in the context.

        Args:
            dx (float): X distance
            dy (float): Y distance
        """
        self.context.rel_move_to(dx, dy)
    # end rel_move_to

    def restore(self) -> None:
        """
        Restore the context.
        """
        self.context.restore()
    # end restore

    def rotate(self, angle: float) -> None:
        """
        Rotate the context.

        Args:
            angle (float): Angle
        """
        self.context.rotate(angle)
    # end rotate

    def save(self) -> None:
        """
        Save the context.
        """
        self.context.save()
    # end save

    def scale(self, sx: float, sy: float) -> None:
        """
        Scale the context.

        Args:
            sx (float): X scale
            sy (float): Y scale
        """
        self.context.scale(sx, sy)
    # end scale

    def set_font_size(self, size: float) -> None:
        """
        Set the font size.

        Args:
            size (float): Font size
        """
        scale = self.image_width / self.width
        self.context.set_font_size(size * scale)
    # end set_font_size

    def set_line_width(self, width: float) -> None:
        """
        Set the line width.

        Args:
            width (float): Line width
        """
        self.context.set_line_width(width)
    # end set_line_width

    def set_source_rgb(self, color: Color) -> None:
        """
        Set the source RGB.

        Args:
            color (Color): Color
        """
        self.context.set_source_rgb(color.red, color.green, color.blue)
    # end set_source_rgb

    def set_source_rgba(self, color: Color) -> None:
        """
        Set the source RGBA.

        Args:
            color (Color): Color
        """
        self.context.set_source_rgba(color.red, color.green, color.blue, color.alpha)
    # end set_source_rgba

    # Setup context
    def setup_context(self):
        """
        Setup the context.
        """
        self.coord_system.setup(self.context)
    # end setup_context

    # endregion PUBLIC

    # region PRIVATE

    # Create surface
    def create_surface(self):
        """
        Create a Cairo surface from the image data.
        """
        # Create context
        return cairo.ImageSurface.create_for_data(
            self.image.data,
            cairo.FORMAT_ARGB32,
            self.image.width,
            self.image.height
        )
    # end create_surface

    # Create context
    def create_context(self):
        """
        Create a Cairo context from the surface.
        """
        return cairo.Context(self.surface)
    # end create_context

    # endregion PRIVATE

    # region OVERRIDE

    def __getattr__(self, item):
        """
        Redirect attribute access to the Cairo context if the attribute is not found in RelativeContext.

        Args:
            item (str): The attribute name.

        Returns:
            Any: The attribute from the Cairo context.
        """
        if hasattr(self._context, item):
            return getattr(self._context, item)
        else:
            raise AttributeError(f"'Context' object has no attribute '{item}'")
        # end
    # end __getattr__

    # endregion OVERRIDE

    # region CLASS_METHODS

    # From image to context
    @classmethod
    def from_image(cls, image, *args, **kwargs):
        """
        Get the context from an image.

        Args:
            image (Image): Image to get the context from

        Returns:
            Context: Context object
        """
        # Create context
        context = cls(image, *args, **kwargs)

        return context
    # end from_image

    # endregion CLASS_METHODS

# end Context

