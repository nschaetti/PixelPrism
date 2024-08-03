

# Imports
from pixel_prism.data import Point2D, Scalar
from pixel_prism.animate.able import MovAble
from .drawablemixin import DrawableMixin


# A 2D rectangle
class Rectangle(DrawableMixin, MovAble):
    """
    A class to represent a rectangle in 2D space.
    """

    def __init__(
            self,
            upper_left: Point2D,
            width: Scalar,
            height: Scalar
    ):
        """
        Initialize the rectangle with its upper left corner, width, and height.

        Args:
            upper_left (Point2D): Upper left corner of the rectangle
            width (Scalar): Width of the rectangle
            height (Scalar): Height of the rectangle
        """
        super().__init__()
        self.upper_left = upper_left
        self.width = width
        self.height = height
    # end __init__

    def set_upper_left(self, x: float, y: float):
        """
        Set the upper left corner of the rectangle.

        Args:
            x (float): X-coordinate of the upper left corner
            y (float): Y-coordinate of the upper left corner
        """
        self.upper_left.set(x, y)
    # end set_upper_left

    def set_width(self, width: float):
        """
        Set the width of the rectangle.

        Args:
            width (float): Width of the rectangle
        """
        self.width.set(width)
    # end set_width

    def set_height(self, height: float):
        """
        Set the height of the rectangle.

        Args:
            height (float): Height of the rectangle
        """
        self.height.set(height)
    # end set_height

    def __str__(self):
        """
        Return a string representation of the rectangle.
        """
        return (
            f"Rectangle(\n"
            f"\tupper_left={self.upper_left},\n"
            f"\twidth={self.width},\n"
            f"\theight={self.height}\n"
            f")"
        )
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the rectangle.
        """
        return Rectangle.__str__()
    # end __repr__

# end Rectangle

