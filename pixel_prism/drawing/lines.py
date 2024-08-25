from typing import Union

# Imports
import numpy as np
from pixel_prism.animate.able import MovableMixin
from pixel_prism.data import Point2D, Color, Scalar, EventMixin
from pixel_prism.utils import random_color
from . import BoundingBoxMixin, BoundingBox

from .rectangles import Rectangle
from .drawablemixin import DrawableMixin
from .. import utils
from ..base import Context


# A line
class Line(
    DrawableMixin,
    BoundingBoxMixin,
    EventMixin,
    MovableMixin
):
    """
    A class to represent a line in 2D space.
    """

    def __init__(
            self,
            start: Point2D,
            end: Point2D,
            line_width: Scalar = Scalar(1.0),
            line_color: Color = utils.WHITE,
            on_change=None
    ):
        """
        Initialize the line with its start and end points.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
            line_width (float): Width of the line
            line_color (Color): Color of the line
            on_change (callable): On change event
        """
        # Init
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)

        # Start and end points
        self._start = start
        self._end = end
        self._line_width = line_width
        self._line_color = line_color

        # Middle point
        self._middle_point = Point2D(0, 0)

        # Bounding box
        BoundingBoxMixin.__init__(self)

        # Update points
        self.update_points()

        # Set events
        self._start.add_event_listener("on_change", self._start_changed)
        self._end.add_event_listener("on_change", self._end_changed)

        # List of event listeners (per events)
        self.add_event("on_change")
        if on_change: self.add_event_listener("on_change", on_change)
    # end __init__

    # region PROPERTIES

    @property
    def start(self):
        """
        Get the start point of the line.

        Returns:
            Point2D: Start point of the line
        """
        return self._start
    # end start

    @property
    def end(self):
        """
        Get the end point of the line.

        Returns:
            Point2D: End point of the line
        """
        return self._end
    # end end

    @property
    def sx(self):
        """
        Get the X-coordinate of the start point.
        """
        return self._start.x
    # end sx

    @sx.setter
    def sx(self, value):
        """
        Set the X-coordinate of the start point.
        """
        self._start.x = value
    # end sx

    @property
    def sy(self):
        """
        Get the Y-coordinate of the start point.
        """
        return self._start.y
    # end sy

    @sy.setter
    def sy(self, value):
        """
        Set the Y-coordinate of the start point.
        """
        self._start.y = value
    # end sy

    @property
    def ex(self):
        """
        Get the X-coordinate of the end point.
        """
        return self._end.x
    # end ex

    @ex.setter
    def ex(self, value):
        """
        Set the X-coordinate of the end point.
        """
        self._end.x = value
    # end ex

    @property
    def middle_point(self):
        """
        Get the middle point of the line.

        Returns:
            Point2D: Middle point of the line
        """
        return self._middle_point
    # end middle_point

    @property
    def length(self):
        """
        Get the length of the line.

        Returns:
            float: Length of the line
        """
        return np.linalg.norm(self._end.pos - self._start.pos)
    # end length

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

    # endregion PROPERTIES

    # region PUBLIC

    # Update data
    def update_data(
            self
    ):
        """
        Update the data of the line.
        """
        self.update_points()
        self.update_bbox()
    # end update_data

    # Update points
    def update_points(
            self
    ):
        """
        Update the points of the line.
        """
        # Update the middle point
        self._middle_point.set(
            (self._start.x + self._end.x) / 2,
            (self._start.y + self._end.y) / 2
        )
    # end update_points

    # Update bounding box
    def update_bbox(
            self
    ):
        """
        Update the bounding box of the line.
        """
        # Get bounding box
        bbox = self._create_bbox()

        # Update bounding box
        self._bounding_box.upper_left.x = bbox.upper_left.x
        self._bounding_box.upper_left.y = bbox.upper_left.y
        self._bounding_box.width = bbox.width
        self._bounding_box.height = bbox.height
    # end update_bbox

    # Realize
    def realize(
            self,
            context,
            move_to: bool = False,
            build_ratio: float = 1.0
    ):
        """
        Realize the line to the context.

        Args:
            context (cairo.Context): Context to realize the line to
            move_to (bool): Move to the start point
            build_ratio (float): Build ratio
        """
        if move_to:
            context.move_to(self.start.x, self.start.y)
        # end if

        # Realize the line
        if build_ratio == 1.0:
            context.line_to(self.end.x, self.end.y)
        else:
            context.line_to(
                self.start.x + (self.end.x - self.start.x) * build_ratio,
                self.start.y + (self.end.y - self.start.y) * build_ratio
            )
        # end if
    # end realize

    # Draw path (for debugging)
    def draw_path(
            self,
            context: Context
    ):
        """
        Draw the path to the context.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # Select a random int
        color = random_color()

        # Save the context
        context.save()

        # Set the color
        context.set_source_rgb(color)

        # Draw the path
        context.move_to(self.start.x, self.start.y)
        context.line_to(self.end.x, self.end.y)

        # Set the line width
        context.set_line_width(0.1)

        # Stroke the path
        context.stroke()

        # Restore the context
        context.restore()
    # end draw_path

    # Draw points
    def draw_points(
            self,
            context: Context,
            point_radius: float = 0.05
    ):
        """
        Draw the points of the line.

        Args:
            context (cairo.Context): Context to draw the points to
            point_radius (float): Radius of the points
        """
        # Save the context
        context.save()

        # Draw the start point
        context.set_source_rgba(utils.GREEN)
        context.arc(
            self.start.x,
            self.start.y,
            point_radius,
            0.0,
            2 * np.pi
        )
        context.stroke()

        # Draw the start point
        context.set_source_rgba(utils.YELLOW)
        context.arc(
            self.end.x,
            self.end.y,
            point_radius,
            0.0,
            2 * np.pi
        )
        context.stroke()

        # Draw the middle point
        context.set_source_rgba(utils.RED)
        context.arc(
            self.middle_point.x,
            self.middle_point.y,
            point_radius,
            0.0,
            2 * np.pi
        )
        context.stroke()

        # Restore the context
        context.restore()
    # end draw_points

    # Draw the element
    def draw(
            self,
            context: Context,
            move_to: bool = False,
            build_ratio: float = 1.0,
            draw_bboxes: bool = False,
            draw_reference_point: bool = False,
            draw_points: bool = False,
            *args,
            **kwargs
    ):
        """
        Draw the line to the context.

        Args:
            context (Context): Context to draw the line to
            move_to (bool): Move to the start point
            build_ratio (float): Build ratio
            draw_bboxes (bool): Draw bounding boxes
            draw_reference_point (bool): Draw reference point
            draw_points (bool): Draw points
        """
        # Save context
        context.save()

        # Realize the line
        self.realize(context, move_to=True, build_ratio=1.0)

        # Set line color
        context.set_source_rgba(
            self.line_color
        )

        # Set line width
        context.set_line_width(self.line_width.value)
        context.stroke()

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

        # Restore
        context.restore()
    # end draw

    # endregion PUBLIC

    # region EVENTS

    # Start changed
    def _start_changed(
            self,
            x: float,
            y: float
    ):
        """
        Start point changed event.
        """
        self.update_data()
    # end _start_changed

    # End changed
    def _end_changed(
            self,
            x: float,
            y: float
    ):
        """
        End point changed event.
        """
        self.update_data()
    # end _end_changed

    # endregion EVENTS

    # region PRIVATE

    # Create bounding box
    def _create_bbox(self):
        """
        Create the bounding box.
        """
        return BoundingBox.from_objects(
            upper_left=Point2D(
                min(self.start.x, self.end.x),
                min(self.start.y, self.end.y)
            ),
            width=Scalar(abs(self.end.x - self.start.x)),
            height=Scalar(abs(self.end.y - self.start.y))
        )
    # end _create_bbox

    # Move
    def _translate_object(
            self,
            dp: Point2D
    ):
        """
        Move the path by a given displacement.

        Args:
            dp (Point2D): Displacement to move the object by
        """
        # Translate the start and end points
        self._start.x += dp.x
        self._start.y += dp.y
        self._end.x += dp.x
        self._end.y += dp.y

    # end translate

    # Scale object
    def _scale_object(
            self,
            m: Scalar
    ):
        """
        Scale the object.

        Args:
            m (Scalar): Scale of the object
        """
        self._end.x = self._start.x + (self._end.x - self._start.x) * m.value
        self._end.y = self._start.y + (self._end.y - self._start.y) * m.value

    # end _scale_object

    # Rotate object
    def _rotate_object(
            self,
            angle: Union[float, Scalar],
            center: Point2D = None
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate the object by
        """
        if center is None:
            center = self.middle_point.copy()
        # end if

        # Temporary points
        start = self.start.copy()
        end = self.end.copy()

        # Angle value
        angle = angle.value if isinstance(angle, Scalar) else angle

        # Center around "center"
        start.x -= center.x
        start.y -= center.y
        end.x -= center.x
        end.y -= center.y

        # Rotate the start and end points
        self._start.x = start.x * np.cos(angle) - start.y * np.sin(angle) + center.x
        self._start.y = start.x * np.sin(angle) + start.y * np.cos(angle) + center.y
        self._end.x = end.x * np.cos(angle) - end.y * np.sin(angle) + center.x
        self._end.y = end.x * np.sin(angle) + end.y * np.cos(angle) + center.y
    # end _rotate_object

    # endregion PRIVATE

    # region OVERRIDE

    # str
    def __str__(self):
        """
        Get the string representation of the line.

        Returns:
            str: String representation of the line
        """
        return f"Line(start={self.start.pos}, end={self.end.pos})"
    # end __str__

    # repr
    def __repr__(self):
        """
        Get the string representation of the line.

        Returns:
            str: String representation of the line
        """
        return Line.__str__(self)
    # end __repr__

     # endregion OVERRIDE

    # region CLASS_METHODS

    # From objects
    @classmethod
    def from_objects(
            cls,
            start: Point2D,
            end: Point2D,
            line_width: Scalar = Scalar(1.0),
            line_color: Color = utils.WHITE
    ):
        """
        Create a line from two points.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
            line_width (float): Width of the line
            line_color (Color): Color of the line
        """
        return cls(
            start,
            end,
            line_width,
            line_color
        )
    # end from_objects

    # endregion CLASS_METHODS

# end Line

