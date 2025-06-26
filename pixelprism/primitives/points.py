

class Point:
    """
    A class to represent a point in 2D space.
    """

    def __init__(self, x, y, size):
        """
        Initialize the point with its coordinates and size.

        Args:
            x (int): X-coordinate of the point
            y (int): Y-coordinate of the point
            size (int): Size of the point
        """
        self.x = x
        self.y = y
        self.size = size
    # end __init__

    def __repr__(self):
        """
        Return a string representation of the point.
        """
        return f"Point(x={self.x}, y={self.y}, size={self.size})"
    # end __repr__

# end Point
