

# Imports
from .container import Container


class PositionedContainer(Container):
    """
    Positioned container class.
    """

    def set_widget_area(
            self,
            widget,
            x,
            y,
            width,
            height):
        """
        Set the area for a widget in a PositionedContainer.

        Args:
            widget (Widget): The widget to set the area for.
            x (int): The x position.
            y (int): The y position.
            width (int): The width of the widget.
            height (int): The height of the widget.
        """
        widget.set_area(x, y, width, height)
    # end set_widget_area

# end PositionedContainer

