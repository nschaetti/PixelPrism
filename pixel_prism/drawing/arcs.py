#
#
#

# Imports
from pixel_prism.animate.able import MovableMixin
from pixel_prism.data import Point2D, Scalar

from .rectangles import Rectangle
from .drawablemixin import DrawableMixin


# An arc
class Arc(DrawableMixin, MovableMixin):
    """
    A class to represent a cubic Bezier curve in 2D space.
    """

    def __init__(
            self,
            center: Point2D,
            radius: Scalar,
            start_angle: Scalar,
            end_angle: Scalar,
            bbox: Rectangle = None
    ):
        """
        Initialize the arc with its center, radius, start angle, and end angle.

        Args:
            center (Point2D): Center of the arc
            radius (Scalar): Radius of the arc
            start_angle (Scalar): Start angle of the arc
            end_angle (Scalar): End angle of the arc
            bbox (Rectangle): Bounding box of the arc
        """
        super().__init__()
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.bbox = bbox
    # end __init__

    # Get the width of the arc
    @property
    def width(self):
        """
        Get the width of the arc.
        """
        if self.bbox is None:
            return None
        # end if
        return self.bbox.width
    # end width

    # Get the height of the arc
    @property
    def height(self):
        """
        Get the height of the arc.
        """
        if self.bbox is None:
            return None
        # end if
        return self.bbox.height
    # end height

    # Set center
    def set_center(self, x: float, y: float):
        """
        Set the center of the arc.

        Args:
            x (float): X-coordinate of the center
            y (float): Y-coordinate of the center
        """
        self.center.x = x
        self.center.y = y
    # end set_center

    # Set radius
    def set_radius(self, radius: float):
        """
        Set the radius of the arc.

        Args:
            radius (float): Radius of the arc
        """
        self.radius.set(radius)
    # end set_radius

    # Set start angle
    def set_start_angle(self, angle: float):
        """
        Set the start angle of the arc.

        Args:
            angle (float): Start angle of the arc
        """
        self.start_angle.set(angle)
    # end set_start_angle

    # Set end angle
    def set_end_angle(self, angle: float):
        """
        Set the end angle of the arc.

        Args:
            angle (float): End angle of the arc
        """
        self.end_angle.set(angle)
    # end set_end_angle

    # Move
    def translate(self, dx: float, dy: float):
        """
        Move the rectangle by a delta.

        Args:
            dx (float): Delta X-coordinate
            dy (float): Delta Y-coordinate
        """
        self.center.x += dx
        self.center.y += dy
        if self.bbox is not None:
            self.bbox.translate(dx, dy)
        # end if
    # end translate

    # Draw the element
    def draw(self, context):
        """
        Draw the arc to the context.
        """
        context.arc(
            self.center.x,
            self.center.y,
            self.radius.value,
            self.start_angle.value,
            self.end_angle.value
        )
    # end draw

    @classmethod
    def from_scalar(
            cls,
            center_x: float,
            center_y: float,
            radius: float,
            start_angle: float,
            end_angle: float
    ):
        """
        Create an arc from scalar values.

        Args:
            center_x (float): X-coordinate of the center
            center_y (float): Y-coordinate of the center
            radius (float): Radius of the arc
            start_angle (float): Start angle of the arc
            end_angle (float): End angle of the arc

        Returns:
            ArcData: Arc created from scalar values
        """
        return cls(
            Point2D(center_x, center_y),
            Scalar(radius),
            Scalar(start_angle),
            Scalar(end_angle)
        )
    # end from_scalar

# end Arc


