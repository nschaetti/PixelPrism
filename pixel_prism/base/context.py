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


from typing import List, Tuple, Sequence, Union
import cairo
from pixel_prism.data import Color, Point2D, Scalar
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
    def arc(self, pc: Point2D, radius: float, angle1: float, angle2: float) -> None:
        """
        Add an arc to the context.

        Args:
            pc (Point2D): Center point
            radius (float): Radius
            angle1 (float): Start angle
            angle2 (float): End angle
        """
        self.context.arc(pc.x, pc.y, radius, angle1, angle2)
    # end arc

    # Arc negative
    def arc_negative(self, pc: Point2D, radius: float, angle1: float, angle2: float) -> None:
        """
        Add a negative arc to the context.

        Args:
            pc (Point2D): Center point
            radius (float): Radius
            angle1 (float): Start angle
            angle2 (float): End angle
        """
        self.context.arc_negative(pc.x, pc.y, radius, angle1, angle2)
    # end arc_negative

    # Curve to
    def curve_to(self, p1: Point2D, p2: Point2D, p3: Point2D) -> None:
        """
        Add a curve to the context.

        Args:
            p1 (Point2D): First control point
            p2 (Point2D): Second control point
            p3 (Point2D): End point
        """
        self.context.curve_to(
            p1.x,
            p1.y,
            p2.x,
            p2.y,
            p3.x,
            p3.y
        )
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

    def line_to(self, position: Point2D) -> None:
        """
        Add a line to the context.

        Args:
            position (Point2D): Position to line to.
        """
        self.context.line_to(position.x, position.y)
    # end line_to

    def move_to(self, position: Point2D) -> None:
        """
        Move to a point in the context.

        Args:
            position (Point2D): Position to move to.
        """
        self.context.move_to(position.x, position.y)
    # end move_to

    def rectangle(self, position: Point2D, width: float, height: float) -> None:
        """
        Add a rectangle to the context.

        Args:
            position (Point2D): Position
            width (float): Width
            height (float): Height
        """
        self.context.rectangle(position.x, position.y, width, height)
    # end rectangle

    def rel_curve_to(self, dp1: Point2D, dp2: Point2D, dp3: Point2D) -> None:
        """
        Add a relative curve to the context.

        Args:
            dp1 (Point2D): First control point
            dp2 (Point2D): Second control point
            dp3 (Point2D): End point
        """
        self.context.rel_curve_to(dp1.x, dp1.y, dp2.x, dp2.y, dp3.x, dp3.y)
    # end rel_curve_to

    def rel_line_to(self, dp: Point2D) -> None:
        """
        Add a relative line to the context.

        Args:
            dp (Point2D): Relative point
        """
        self.context.rel_line_to(dp.x, dp.y)
    # end rel_line_to

    def rel_move_to(self, dp: Point2D) -> None:
        """
        Move to a relative point in the context.

        Args:
            dp (Point2D): Relative point
        """
        self.context.rel_move_to(dp.x, dp.y)
    # end rel_move_to

    def restore(self) -> None:
        """
        Restore the context.
        """
        self.context.restore()
    # end restore

    def save(self) -> None:
        """
        Save the context.
        """
        self.context.save()
    # end save

    def translate(self, position: Point2D) -> None:
        """
        Translate the context.

        Args:
            position (Point2D): Position (x, y)
        """
        self.context.translate(position.x, position.y)
    # end translate

    def scale(self, sp: Point2D) -> None:
        """
        Scale the context.

        Args:
            sp (Point2D): Scale point (x, y)
        """
        self.context.scale(sp.x, sp.y)
    # end scale

    def rotate(self, angle: Union[float, Scalar]) -> None:
        """
        Rotate the context.

        Args:
            angle (float): Angle
        """
        if isinstance(angle, Scalar):
            angle = angle.value
        # end if
        self.context.rotate(angle)
    # end rotate

    def set_font_size(self, size: float) -> None:
        """
        Set the font size.

        Args:
            size (float): Font size
        """
        self.context.set_font_matrix(cairo.Matrix(size, 0, 0, -size, 0, 0))
    # end set_font_size

    def set_line_width(self, width: Union[float, Scalar]) -> None:
        """
        Set the line width.

        Args:
            width (float): Line width
        """
        if isinstance(width, Scalar):
            width = width.value
        # end
        self.context.set_line_width(width)
    # end set_line_width

    def set_source_rgb(self, color: Color) -> None:
        """
        Set the source RGB.

        Args:
            color (Color): Color
        """
        self.context.set_source_rgb(
            color.red.value,
            color.green.value,
            color.blue.value
        )
    # end set_source_rgb

    def set_source_rgba(self, color: Color) -> None:
        """
        Set the source RGBA.

        Args:
            color (Color): Color
        """
        self.context.set_source_rgba(
            color.red.value,
            color.green.value,
            color.blue.value,
            color.alpha.value
        )
    # end set_source_rgba

    def set_source_rgb_alpha(self, color: Color, alpha: Union[float, Scalar]) -> None:
        """
        Set the source RGB with alpha.

        Args:
            color (Color): Color
            alpha (float): Alpha
        """
        self.context.set_source_rgba(
            color.red.value,
            color.green.value,
            color.blue.value,
            alpha if isinstance(alpha, float) else alpha.value
        )
    # end set_source_rgb_alpha

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

