#
#
#
import math

# Imports
from pixel_prism.animate.able import MovableMixin
from pixel_prism.data import Point2D, Scalar, Color

from .rectangles import Rectangle
from .drawablemixin import DrawableMixin
from .. import utils


# An arc
class Arc(DrawableMixin, MovableMixin):
    """
    A class to represent a cubic Bezier curve in 2D space.
    """

    def __init__(
            self,
            cx: float,
            cy: float,
            radius: float,
            start_angle: float,
            end_angle: float,
            scale: float = 1.0,
            rotation: float = 0.0,
            line_width: float = 0.0,
            line_color: Color = utils.WHITE,
            fill_color: Color = None,
            bbox_border_width: float = 0.5,
            bbox_border_color: Color = utils.RED.copy()
    ):
        """
        Initialize the arc with its center, radius, start angle, and end angle.

        Args:
            cx (float): X-coordinate of the center
            cy (float): Y-coordinate of the center
            radius (float): Radius of the arc
            start_angle (float): Start angle of the arc
            end_angle (float): End angle of the arc
            scale (float): Scale of the arc
            rotation (float): Rotation of the arc
            line_width (float): Width of the line
            line_color (Color): Color of the line
            fill_color (Color): Color to fill the arc
            bbox_border_width (float): Width of the bounding box border
            bbox_border_color (Color): Color of the bounding box border
        """
        # Properties
        self._center = Point2D(cx, cy)
        self._radius = Scalar(radius)
        self._start_angle = Scalar(start_angle)
        self._end_angle = Scalar(end_angle)
        self._line_width = Scalar(line_width)
        self._line_color = line_color
        self._fill_color = fill_color

        # Beginning, end, and control points
        self._start_point = Point2D(0, 0)
        self._end_point = Point2D(0, 0)
        self._middle_point = Point2D(0, 0)

        # Init
        DrawableMixin.__init__(
            self,
            scale=scale,
            rotation=rotation,
            has_bbox=True,
            bbox_border_width=bbox_border_width,
            bbox_border_color=bbox_border_color
        )
        MovableMixin.__init__(self)

        # Update points
        self.update_points()

        # Set events
        self._center.add_event_listener("on_change", self._on_center_changed)
        self._radius.add_event_listener("on_change", self._on_radius_changed)
        self._start_angle.add_event_listener("on_change", self._on_start_angle_changed)
        self._end_angle.add_event_listener("on_change", self._on_end_angle_changed)
    # end __init__

    # region PROPERTIES

    # Get center
    @property
    def center(self):
        """
        Get the center of the arc.
        """
        return self._center
    # end center

    @property
    def cx(self):
        """
        Get the X-coordinate of the center.
        """
        return self._center.x
    # end cx

    @cx.setter
    def cx(self, value):
        """
        Set the X-coordinate of the center.
        """
        self._center.x = value
    # end cx

    @property
    def cy(self):
        """
        Get the Y-coordinate of the center.
        """
        return self._center.y
    # end cy

    @cy.setter
    def cy(self, value):
        """
        Set the Y-coordinate of the center.
        """
        self._center.y = value
    # end cy

    # Get radius
    @property
    def radius(self):
        """
        Get the radius of the arc.
        """
        return self._radius
    # end radius

    @radius.setter
    def radius(self, value):
        """
        Set the radius of the arc.
        """
        self._radius.set(value)
    # end radius

    # Get start angle
    @property
    def start_angle(self):
        """
        Get the start angle of the arc.
        """
        return self._start_angle
    # end start_angle

    @start_angle.setter
    def start_angle(self, value):
        """
        Set the start angle of the arc.
        """
        self._start_angle.set(value)
    # end start_angle

    # Get end angle
    @property
    def end_angle(self):
        """
        Get the end angle of the arc.
        """
        return self._end_angle
    # end end_angle

    @end_angle.setter
    def end_angle(self, value):
        """
        Set the end angle of the arc.
        """
        self._end_angle.set(value)
    # end end_angle

    # Get the width of the arc
    @property
    def width(self):
        """
        Get the width of the arc.
        """
        if self.bounding_box is None:
            return None
        # end if
        return self.bounding_box.width
    # end width

    # Get the height of the arc
    @property
    def height(self):
        """
        Get the height of the arc.
        """
        if self.bounding_box is None:
            return None
        # end if
        return self.bounding_box.height
    # end height

    # Start point
    @property
    def start_point(self):
        """
        Get the start point of the arc.
        """
        return self._start_point
    # end start_point

    # End point
    @property
    def end_point(self):
        """
        Get the end point of the arc.
        """
        return self._end_point
    # end end_point

    # Middle point
    @property
    def middle_point(self):
        """
        Get the middle point of the arc.
        """
        return self._middle_point
    # end middle_point

    # Line width
    @property
    def line_width(self):
        """
        Get the line width of the arc.
        """
        return self._line_width
    # end line_width

    # Line color
    @property
    def line_color(self):
        """
        Get the line color of the arc.
        """
        return self._line_color
    # end line_color

    # Fill color
    @property
    def fill_color(self):
        """
        Get the fill color of the arc.
        """
        return self._fill_color
    # end fill_color

    # endregion PROPERTIES

    # region PUBLIC

    # Set center
    def set_center(self, x: float, y: float):
        """
        Set the center of the arc.

        Args:
            x (float): X-coordinate of the center
            y (float): Y-coordinate of the center
        """
        self._center.x = x
        self._center.y = y
    # end set_center

    # Set radius
    def set_radius(self, radius: float):
        """
        Set the radius of the arc.

        Args:
            radius (float): Radius of the arc
        """
        self._radius.set(radius)
    # end set_radius

    # Set start angle
    def set_start_angle(self, angle: float):
        """
        Set the start angle of the arc.

        Args:
            angle (float): Start angle of the arc
        """
        self._start_angle.set(angle)
    # end set_start_angle

    # Set end angle
    def set_end_angle(self, angle: float):
        """
        Set the end angle of the arc.

        Args:
            angle (float): End angle of the arc
        """
        self._end_angle.set(angle)
    # end set_end_angle

    # Translating
    def translating(self, dp: Point2D):
        """
        Move the rectangle by a delta.

        Args:
            dp (Point2D): Delta to move the rectangle by
        """
        self._center.x += dp.x
        self._center.y += dp.y
    # end translate

    # Update data
    def update_data(
            self
    ):
        """
        Update the data of the arc.
        """
        self.update_points()
        self.update_bbox()
    # end update_data

    # Update points
    def update_points(
            self
    ):
        """
        Update the start, end, and middle points.
        """
        # Update points
        self._start_point.x = self.center.x + self.radius.value * self.scale.value * math.cos(self.start_angle.value)
        self._start_point.y = self.center.y + self.radius.value * self.scale.value * math.sin(self.start_angle.value)
        self._end_point.x = self.center.x + self.radius.value * self.scale.value * math.cos(self.end_angle.value)
        self._end_point.y = self.center.y + self.radius.value * self.scale.value * math.sin(self.end_angle.value)
        self._middle_point.x = self.center.x + self.radius.value * self.scale.value * math.cos(
            self.start_angle.value + (self.end_angle.value - self.start_angle.value) / 2.0)
        self._middle_point.y = self.center.y + self.radius.value * self.scale.value * math.sin(
            self.start_angle.value + (self.end_angle.value - self.start_angle.value) / 2.0)
    # end update_points

    # Update bounding box
    def update_bbox(
            self
    ):
        """
        Update the bounding box of the arc.
        """
        # Get the bounding box
        bbox = self._create_bbox()

        # Update the bounding box
        self._bounding_box.upper_left.x = bbox.upper_left.x
        self._bounding_box.upper_left.y = bbox.upper_left.y
        self._bounding_box.width = bbox.width
        self._bounding_box.height = bbox.height
    # end update_bbox

    # Realize
    def realize(self, context):
        """
        Realize the arc.
        """
        context.arc(
            self.center.x,
            self.center.y,
            self.radius.value * self.scale.value,
            self.start_angle.value,
            self.end_angle.value
        )
    # end realize

    # Draw points
    def draw_points(
            self,
            context
    ):
        """
        Draw the points of the arc.

        Args:
            context (cairo.Context): Context to draw the points to
        """
        # Save context
        context.save()

        # Set fill color
        self.set_source_rgba(
            context,
            utils.RED
        )

        # Draw starting point
        context.arc(
            self._start_point.x,
            self._start_point.y,
            10,
            0,
            2 * math.pi
        )
        context.stroke()

        # Draw ending point
        context.arc(
            self._end_point.x,
            self._end_point.y,
            10,
            0,
            2 * math.pi
        )
        context.stroke()

        # Draw middle point
        context.arc(
            self._middle_point.x,
            self._middle_point.y,
            10,
            0,
            2 * math.pi
        )
        context.stroke()

        # Draw center
        context.arc(
            self.center.x,
            self.center.y,
            10,
            0,
            2 * math.pi
        )
        context.stroke()

        # Restore context
        context.restore()
    # end draw_points

    # Draw the element
    def draw(
            self,
            context,
            draw_bboxes: bool = False,
            draw_reference_point: bool = False,
            draw_points: bool = False,
            *args,
            **kwargs
    ):
        """
        Draw the arc to the context.
        """
        # Save context
        context.save()

        # Draw path
        self.realize(context)

        # Fill color is set
        if self.fill_color is not None:
            # Set fill color
            self.set_source_rgba(
                context,
                self.fill_color
            )

            # Stroke or not
            if self.line_width.value == 0:
                context.fill()
            else:
                context.fill_preserve()
            # end if
        # end if

        # Stroke
        if self.line_width.value > 0:
            # Set line color
            self.set_source_rgba(
                context,
                self.line_color
            )

            # Set line width
            context.set_line_width(self.line_width.value)
            context.stroke()
        # end if

        # Draw bounding box
        if draw_bboxes:
            self.bounding_box.draw(context)
        # end if

        # Draw reference point
        if draw_reference_point:
            self.draw_bbox_anchors(context)
        # end if

        # Draw points
        if draw_points:
            self.draw_points(context)
        # end if

        # Restore context
        context.restore()
    # end draw

    # endregion PUBLIC

    # region EVENTS

    # Center changed
    def _on_center_changed(
            self,
            x,
            y
    ):
        """
        Event handler for the center changing.
        """
        self.update_data()
    # end _on_center_changed

    # Radius changed
    def _on_radius_changed(
            self,
            value
    ):
        """
        Event handler for the radius changing.
        """
        self.update_data()
    # end _on_radius_changed

    # Start angle changed
    def _on_start_angle_changed(
            self,
            value
    ):
        """
        Event handler for the start angle changing.
        """
        self.update_data()
    # end _on_start_angle_changed

    # End angle changed
    def _on_end_angle_changed(
            self,
            value
    ):
        """
        Event handler for the end angle changing.
        """
        self.update_data()
    # end _on_end_angle_changed

    # endregion EVENTS

    # region PRIVATE

    # Scale object
    def _scale_object(
            self,
            m: float
    ):
        """
        Scale the object.

        Args:
            m (float): Scale of the object
        """
        self._scale.value *= m
    # end _scale_object

    # Translate object
    def _translate_object(
            self,
            dp: Point2D
    ):
        """
        Translate the object.

        Args:
            dp (Point2D): Displacement to move the object by
        """
        self._center.x += dp.x
        self._center.y += dp.y
    # end _translate_object

    # Rotate object
    def _rotate_object(
            self,
            angle: float
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate the object by
        """
        self._rotation.value += angle
    # end _rotate_object

    # Create bounding box
    def _create_bbox(
            self,
            border_width: float = 0.0,
            border_color: Color = utils.WHITE.copy()
    ):
        """
        Create the bounding box.
        """
        # Normalize angles to be in the range [0, 2 * PI]
        start_angle = self.start_angle.value % (2 * math.pi)
        end_angle = self.end_angle.value % (2 * math.pi)

        # Calculate the center and radius
        while end_angle < start_angle:
            end_angle += 2 * math.pi
        # end if

        # Calculate the points at the start and end angles
        start_x = self.center.x + self.radius.value * self.scale.value * math.cos(start_angle)
        start_y = self.center.y + self.radius.value * self.scale.value * math.sin(start_angle)
        end_x = self.center.x + self.radius.value * self.scale.value * math.cos(end_angle)
        end_y = self.center.y + self.radius.value * self.scale.value * math.sin(end_angle)

        # Initialize the bounding box with the start and end points
        xmin = min(start_x, end_x)
        xmax = max(start_x, end_x)
        ymin = min(start_y, end_y)
        ymax = max(start_y, end_y)

        # Check if the arc passes through the extrema on the x and y axes
        def check_extrema(
                c_x,
                c_y,
                c_radius,
                c_angle,
                c_xmin,
                c_xmax,
                c_ymin,
                c_ymax
        ):
            """
            Check if the arc passes through the extrema on the x and y axes.
            """
            if start_angle <= c_angle <= end_angle:
                x = c_x + c_radius * math.cos(-c_angle)
                y = c_y + c_radius * math.sin(c_angle)
                return (
                    min(c_xmin, x),
                    max(c_xmax, x),
                    min(c_ymin, y),
                    max(c_ymax, y)
                )
            else:
                return c_xmin, c_xmax, c_ymin, c_ymax
            # end if
        # end check_extrema

        # Critical angles to check
        critical_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]

        # Check the critical angles
        for angle in critical_angles:
            xmin, xmax, ymin, ymax = check_extrema(
                self.center.x,
                self.center.y,
                self.radius.value * self.scale.value,
                angle,
                xmin,
                xmax,
                ymin,
                ymax
            )
        # end for

        return Rectangle(
            upper_left=Point2D(xmin, ymin),
            width=xmax - xmin,
            height=ymax - ymin,
            border_width=border_width,
            border_color=border_color,
            fill=False
        )
    # end _create_bbox

    # endregion PRIVATE

    # region OVERRIDE

    # To string
    def __str__(self):
        """
        Get the string representation of the arc.
        """
        return (
            f"Arc("
            f"center={self.center}, "
            f"radius={self.radius}, "
            f"start_angle={self.start_angle}, "
            f"end_angle={self.end_angle}"
            f"line_width={self.line_width}, "
            f"line_color={self.line_color}, "
            f"fill_color={self.fill_color}, "
            f"scale={self.scale}, "
            f"rotation={self.rotation}, "
            f")"
        )
    # end __str__

    # Return a string representation of the arc.
    def __repr__(self):
        """
        Return a string representation of the arc.
        """
        return self.__str__()
    # end __

    # endregion OVERRIDE

    # region CLASS_METHODS

    @classmethod
    def from_scalar(
            cls,
            center_x: float,
            center_y: float,
            radius: float,
            start_angle: float,
            end_angle: float,
            line_width: float = 0.0,
            line_color: Color = utils.WHITE,
            fill_color: Color = None,
            bbox_border_width: float = 0.5,
            bbox_border_color: Color = utils.RED.copy()
    ):
        """
        Create an arc from scalar values.

        Args:
            center_x (float): X-coordinate of the center
            center_y (float): Y-coordinate of the center
            radius (float): Radius of the arc
            start_angle (float): Start angle of the arc
            end_angle (float): End angle of the arc
            line_width (float): Width of the line
            line_color (Color): Color of the line
            fill_color (Color): Color to fill the arc
            bbox_border_width (float): Width of the bounding box border
            bbox_border_color (Color): Color of the bounding box border

        Returns:
            ArcData: Arc created from scalar values
        """
        return cls(
            cx=center_x,
            cy=center_y,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color,
            bbox_border_width=bbox_border_width,
            bbox_border_color=bbox_border_color
        )
    # end from_scalar

    # endregion CLASS_METHODS

# end Arc


