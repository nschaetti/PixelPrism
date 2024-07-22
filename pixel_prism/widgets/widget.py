#
# This file contains the Widget class, which is the base class for all widgets in the Pixel Prism GUI library.
#


class Widget:

    def __init__(self):
        """
        Initialize the widget.
        """
        self.height = None
        self.width = None
        self.y = None
        self.x = None
    # end __init__

    def set_area(
            self,
            x,
            y,
            width,
            height
    ):
        """
        Set the area of the widget.

        Args:
            x (int): X-coordinate of the widget
            y (int): Y-coordinate of the widget
            width (int): Width of the widget
            height (int): Height of the widget
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    # end set_area

    def draw(self, context):
        raise NotImplementedError("Draw method must be implemented by subclasses")
    # end draw

# end Widget
