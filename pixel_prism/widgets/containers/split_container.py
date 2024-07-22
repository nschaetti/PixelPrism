

# Import necessary libraries.
from .container import Container


class SplitContainer(Container):
    """
    Split container class.
    """

    def __init__(
            self,
            orientation='horizontal'
    ):
        super().__init__()
        self.orientation = orientation
    # end __init__

    def set_widget_area(self, widget, **kwargs):
        """
        Set the area for widgets in a SplitContainer. Expects two widgets.

        Args:
            widget (Widget): The widget to set the area for.
            kwargs: Additional arguments (not used in this implementation).
        """
        if len(self.widgets) == 0:
            if self.orientation == 'horizontal':
                widget.set_area(0, 0, self.width // 2, self.height)
            else:  # vertical
                widget.set_area(0, 0, self.width, self.height // 2)
        elif len(self.widgets) == 1:
            if self.orientation == 'horizontal':
                widget.set_area(self.width // 2, 0, self.width // 2, self.height)
            else:  # vertical
                widget.set_area(0, self.height // 2, self.width, self.height // 2)
        else:
            raise ValueError("SplitContainer can only contain two widgets.")
        # end if
    # end set_widget_area

# end SplitContainer

