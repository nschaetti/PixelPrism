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
import math
from typing import Tuple, Union, Any
import cairo
from pixelprism.data import Color, Point2D, Scalar, Transform, Style
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
        self._transform = None
        self._image_width = self._image.width
        self._image_height = self._image.height

        # List of saved transforms
        self._saved_transforms = list()
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
    def transform(self):
        """
        Get the transform.
        """
        return self._transform
    # end transform

    @property
    def coord_system(self):
        """
        Get the coordinate system.
        """
        return self._coord_system
    # end coord_system

    # endregion PROPERTIES

    # region PUBLIC

    # Draw an arc using Bézier curves
    def arc(self, pc: Point2D, radius: float, start_angle: float, end_angle: float):
        """
        Draw an arc using Bézier curves.

        Args:
            pc (Point2D): Centre de l'arc
            radius (float): Rayon de l'arc
            start_angle (float): Angle de départ en radians
            end_angle (float): Angle de fin en radians
        """
        # Points of the Bézier curve
        points = []

        # How many segments to divide the arc into for approximation
        num_segments = 4

        # Angle step for each segment
        angle_step = (end_angle - start_angle) / num_segments

        # Create the Bézier curve points
        for i in range(num_segments):
            # Calculate the angles for the start and end of the segment
            theta1 = start_angle + i * angle_step
            theta2 = start_angle + (i + 1) * angle_step

            # Calculate the points for the segment
            p1 = Point2D(pc.x + radius * math.cos(theta1), pc.y + radius * math.sin(theta1))
            p2 = Point2D(pc.x + radius * math.cos(theta2), pc.y + radius * math.sin(theta2))

            # Calculate the control points for the segment
            control_points = self._get_bezier_control_points_for_arc(pc, radius, theta1, theta2)
            points.append((p1, control_points[0], control_points[1], p2))
        # end for

        # Apply the transform
        if self.transform:
            for i, (p1, cp1, cp2, p2) in enumerate(points):
                p1 = self.transform.forward(p1)
                cp1 = self.transform.forward(cp1)
                cp2 = self.transform.forward(cp2)
                p2 = self.transform.forward(p2)
                points[i] = (p1, cp1, cp2, p2)
            # end for
        # end if

        # Draw the Bézier curve
        for p1, cp1, cp2, p2 in points:
            self.context.curve_to(cp1.x, cp1.y, cp2.x, cp2.y, p2.x, p2.y)
        # end for
    # end bezier_arc

    # Draw an arc in the negative direction using Bézier curves
    def arc_negative(self, pc: Point2D, radius: float, start_angle: float, end_angle: float):
        """
        Draw an arc in the negative direction using Bézier curves.

        Args:
            pc (Point2D): Centre of the arc
            radius (float): Radius of the arc
            start_angle (float): Starting angle in radians
            end_angle (float): Ending angle in radians
        """
        # Points of the Bézier curve
        points = []

        # How many segments to divide the arc into for approximation
        num_segments = 4

        # Angle step for each segment (negative for arc_negative)
        angle_step = (end_angle - start_angle) / num_segments

        # Create the Bézier curve points
        for i in range(num_segments):
            # Calculate the angles for the start and end of the segment (in reverse)
            theta1 = start_angle + i * angle_step
            theta2 = start_angle + (i + 1) * angle_step

            # Calculate the points for the segment
            p1 = Point2D(pc.x + radius * math.cos(theta1), pc.y + radius * math.sin(theta1))
            p2 = Point2D(pc.x + radius * math.cos(theta2), pc.y + radius * math.sin(theta2))

            # Calculate the control points for the segment
            control_points = self._get_bezier_control_points_for_arc(pc, radius, theta1, theta2)
            points.append((p1, control_points[0], control_points[1], p2))
        # end for

        # Apply the transform if set
        if self.transform:
            for i, (p1, cp1, cp2, p2) in enumerate(points):
                p1 = self.transform.forward(p1)
                cp1 = self.transform.forward(cp1)
                cp2 = self.transform.forward(cp2)
                p2 = self.transform.forward(p2)
                points[i] = (p1, cp1, cp2, p2)
            # end for
        # end if

        # Draw the Bézier curve in reverse
        for p1, cp1, cp2, p2 in reversed(points):
            self.context.curve_to(cp2.x, cp2.y, cp1.x, cp1.y, p1.x, p1.y)
        # end for
    # end arc_negative

    # Cairo arc method
    def cairo_arc(self, pc: Point2D, radius: float, angle1: float, angle2: float) -> None:
        """
        Add an arc to the context.

        Args:
            pc (Point2D): Center point
            radius (float): Radius
            angle1 (float): Start angle
            angle2 (float): End angle
        """
        self.context.arc(pc.x, pc.y, radius, angle1, angle2)
    # end cairo_arc

    # Cairo arc negative method
    def cairo_arc_negative(self, pc: Point2D, radius: float, angle1: float, angle2: float) -> None:
        """
        Add a negative arc to the context.

        Args:
            pc (Point2D): Center point
            radius (float): Radius
            angle1 (float): Start angle
            angle2 (float): End angle
        """
        self.context.arc_negative(pc.x, pc.y, radius, angle1, angle2)
    # end cairo_arc_negative

    # Curve to
    def curve_to(self, p1: Point2D, p2: Point2D, p3: Point2D) -> None:
        """
        Add a curve to the context.

        Args:
            p1 (Point2D): First control point
            p2 (Point2D): Second control point
            p3 (Point2D): End point
        """
        # Apply the transform
        if self.transform:
            p1 = self.transform.forward(p1)
            p2 = self.transform.forward(p2)
            p3 = self.transform.forward(p3)
        # end if

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
        if self.transform:
            position = self.transform.forward(position)
        # end if
        self.context.line_to(position.x, position.y)
    # end line_to

    def cairo_line_to(self, position: Point2D) -> None:
        """
        Add a line to the context.

        Args:
            position (Point2D): Position to line to.
        """
        self.context.line_to(position.x, position.y)
    # end cairo_line_to

    def move_to(self, position: Point2D) -> None:
        """
        Move to a point in the context.

        Args:
            position (Point2D): Position to move to.
        """
        if self.transform:
            position = self.transform.forward(position)
        # end if
        self.context.move_to(position.x, position.y)
    # end move_to

    def cairo_move_to(self, position: Point2D) -> None:
        """
        Move to a point in the context.

        Args:
            position (Point2D): Position to move to.
        """
        self.context.move_to(position.x, position.y)
    # end cairo_move_to

    def rectangle(self, top_left: Point2D, width: float, height: float):
        """
        Draw a rectangle using Bézier curves.

        Args:
            top_left (Point2D): Upper-left corner
            width (float): Width
            height (float): Height
        """
        # Create the points of the rectangle
        p1 = top_left
        p2 = Point2D(p1.x + width, p1.y)
        p3 = Point2D(p1.x + width, p1.y + height)
        p4 = Point2D(p1.x, p1.y + height)

        # Apply the transform
        if self.transform:
            p1 = self.transform.forward(p1)
            p2 = self.transform.forward(p2)
            p3 = self.transform.forward(p3)
            p4 = self.transform.forward(p4)
        # end if

        # Draw the Bézier curve
        self.context.move_to(p1.x, p1.y)
        self.context.line_to(p2.x, p2.y)
        self.context.line_to(p3.x, p3.y)
        self.context.line_to(p4.x, p4.y)
        self.context.line_to(p1.x, p1.y)
        self.context.close_path()
    # end rectangle

    def cairo_rectangle(self, position: Point2D, width: float, height: float) -> None:
        """
        Add a rectangle to the context.

        Args:
            position (Point2D): Position
            width (float): Width
            height (float): Height
        """
        self.context.rectangle(position.x, position.y, width, height)
    # end cairo_rectangle

    def cairo_rel_curve_to(self, dp1: Point2D, dp2: Point2D, dp3: Point2D) -> None:
        """
        Add a relative curve to the context.

        Args:
            dp1 (Point2D): First control point
            dp2 (Point2D): Second control point
            dp3 (Point2D): End point
        """
        self.context.rel_curve_to(dp1.x, dp1.y, dp2.x, dp2.y, dp3.x, dp3.y)
    # end cairo_rel_curve_to

    def rel_curve_to(self, dp1: Point2D, dp2: Point2D, dp3: Point2D) -> None:
        """
        Add a relative curve to the context, applying the transform if set.

        Args:
            dp1 (Point2D): First control point relative to the current point
            dp2 (Point2D): Second control point relative to the current point
            dp3 (Point2D): End point relative to the current point
        """
        # Get the current point
        current_x, current_y = self.context.get_current_point()

        # Create the absolute points
        abs_p1 = Point2D(current_x + dp1.x, current_y + dp1.y)
        abs_p2 = Point2D(current_x + dp2.x, current_y + dp2.y)
        abs_p3 = Point2D(current_x + dp3.x, current_y + dp3.y)

        # Apply the transform
        if self.transform:
            abs_p1 = self.transform.forward(abs_p1)
            abs_p2 = self.transform.forward(abs_p2)
            abs_p3 = self.transform.forward(abs_p3)
        # end if

        # Compute the relative positions
        rel_cp1_x, rel_cp1_y = abs_p1.x - current_x, abs_p1.y - current_y
        rel_cp2_x, rel_cp2_y = abs_p2.x - current_x, abs_p2.y - current_y
        rel_p3_x, rel_p3_y = abs_p3.x - current_x, abs_p3.y - current_y

        # Draw the curve
        self.context.rel_curve_to(rel_cp1_x, rel_cp1_y, rel_cp2_x, rel_cp2_y, rel_p3_x, rel_p3_y)
    # end rel_curve_to

    def cairo_rel_line_to(self, dp: Point2D) -> None:
        """
        Add a relative line to the context.

        Args:
            dp (Point2D): Relative point
        """
        self.context.rel_line_to(dp.x, dp.y)
    # end cairo_rel_line_to

    def rel_line_to(self, dp: Point2D) -> None:
        """
        Add a relative line to the context, applying the transform if set.

        Args:
            dp (Point2D): Point to draw relative to the current point
        """
        # Get the current point
        current_x, current_y = self.context.get_current_point()

        # Create the absolute point
        abs_p = Point2D(current_x + dp.x, current_y + dp.y)

        # Apply the transform
        if self.transform:
            abs_p = self.transform.forward(abs_p)
        # end if

        # Compute the relative positions
        rel_p_x, rel_p_y = abs_p.x - current_x, abs_p.y - current_y

        # Draw the line
        self.context.rel_line_to(rel_p_x, rel_p_y)
    # end rel_line_to

    def cairo_rel_move_to(self, dp: Point2D) -> None:
        """
        Move to a relative point in the context.

        Args:
            dp (Point2D): Relative point
        """
        self.context.rel_move_to(dp.x, dp.y)
    # end cairo_rel_move_to

    def rel_move_to(self, dp: Point2D) -> None:
        """
        Move to a relative point in the context, applying the transform if set.

        Args:
            dp (Point2D): Relative point to move to
        """
        # Get the current point
        current_x, current_y = self.context.get_current_point()

        # Create the absolute point
        abs_p = Point2D(current_x + dp.x, current_y + dp.y)

        # Apply the transform
        if self.transform:
            abs_p = self.transform.forward(abs_p)
        # end if

        # Compute the relative positions
        rel_p_x, rel_p_y = abs_p.x - current_x, abs_p.y - current_y

        # Do the move to the relative point
        self.context.rel_move_to(rel_p_x, rel_p_y)
    # end rel_move_to

    def restore(self) -> None:
        """
        Restore the context.
        """
        # Unstack transform
        self._unstack_transform()

        # Restore Cairo context
        self.context.restore()
    # end restore

    # Set transform
    def set_transform(self, transform: Transform):
        """
        Set the transform.

        Args:
            transform (Transform): Transform
        """
        self._transform  = transform
    # end set_transform

    def save(self) -> None:
        """
        Save the context.
        """
        # If transform is set, save it
        if self.transform is not None:
            self._stack_transform()
        # end if

        # Save Cairo context
        self.context.save()
    # end save

    def translate(self, position: Point2D) -> None:
        """
        Translate the context.

        Args:
            position (Point2D): Position (x, y)
        """
        # self.context.translate(position.x, position.y)
        # Add a transformation
        self._transform = Transform(
            position,
            scale=Point2D(1.0, 1.0),
            rotation=Scalar(1.0),
            parent=self.transform if self.transform else None
        )
    # end translate

    def scale(self, sp: Point2D) -> None:
        """
        Scale the context.

        Args:
            sp (Point2D): Scale point (x, y)
        """
        # self.context.scale(sp.x, sp.y)
        # Add a transformation
        self._transform = Transform(
            position=Point2D(0.0, 0.0),
            scale=sp,
            rotation=Scalar(0.0),
            parent=self.transform if self.transform else None
        )
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
        # self.context.rotate(angle)
        # Add a transformation
        self._transform = Transform(
            position=Point2D(0.0, 0.0),
            scale=Point2D(1.0, 1.0),
            rotation=Scalar(angle),
            parent=self.transform if self.transform else None
        )
    # end rotate

    def set_font_size(self, size: float) -> None:
        """
        Set the font size.

        Args:
            size (float): Font size
        """
        self.context.set_font_matrix(cairo.Matrix(size, 0, 0, -size, 0, 0))
    # end set_font_size

    # Set style
    def set_style(self, style: Style):
        """
        Set the style.

        Args:
            style (Style): Style
        """
        # Set line width
        self.set_line_width(style.line_width)

        # If dash
        if style.line_dash is not None:
            self.set_dash(self.style.line_dash)
        # end if

        # Set cap
        self.set_line_cap(style.line_cap)
    # end set_style

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

    def set_source_rgb_a(self, color: Color, alpha: Union[float, Scalar]) -> None:
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
    # end set_source_rgb_a

    # Setup context
    def setup_context(self):
        """
        Setup the context.
        """
        self.coord_system.setup(self.context)
    # end setup_context

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

    # endregion PUBLIC

    # region PRIVATE

    # Stack transform
    def _stack_transform(self):
        """
        Stack the current transform.
        """
        if self.transform:
            self._saved_transforms.append(self.transform)
        # end if
    # end _stack_transform

    # Unstack transform
    def _unstack_transform(self):
        """
        Unstack the current transform.
        """
        if len(self._saved_transforms) > 0:
            self._transform = self._saved_transforms.pop()
        # end if
    # end _unstack_transform

    def _get_bezier_control_points_for_arc(self, center: Point2D, radius: float, theta1: float, theta2: float):
        """
        Compute the control points for a Bézier curve representing an arc.

        Args:
            center (Point2D): Center of the arc
            radius (float): Radius of the arc
            theta1 (float): Start angle in radians
            theta2 (float): End angle in radians

        Returns:
            Tuple[Point2D, Point2D]: Control points for the Bézier curve
        """
        # Compute the angle between the two points
        alpha = (4 / 3) * math.tan((theta2 - theta1) / 4)

        # Start and end points of the arc
        p1 = Point2D(center.x + radius * math.cos(theta1), center.y + radius * math.sin(theta1))
        p2 = Point2D(center.x + radius * math.cos(theta2), center.y + radius * math.sin(theta2))

        # Control points
        cp1 = Point2D(
            p1.x - alpha * radius * math.sin(theta1),
            p1.y + alpha * radius * math.cos(theta1)
        )
        cp2 = Point2D(
            p2.x + alpha * radius * math.sin(theta2),
            p2.y - alpha * radius * math.cos(theta2)
        )

        return cp1, cp2
    # end _get_bezier_control_points_for_arc

    # endregion PRIVATE

    # region OVERRIDE

    def __getattr__(self, item) -> Any:
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
    def from_image(cls, image, *args, **kwargs) -> 'Context':
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
