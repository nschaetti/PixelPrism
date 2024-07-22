

# Imports
from pixel_prism.widgets.widget import Widget


class Container(Widget):

    def __init__(self):
        """
        Initialize the container.
        """
        super().__init__()
        self.widgets = []
    # end __init__

    def add_widget(
            self,
            widget,
            **kwargs
    ):
        """
        Add a widget to the container.

        Args:
            widget (Widget): Widget to add to the container
            x (int): X-coordinate of the widget
            y (int): Y-coordinate of the widget
            width (int): Width of the widget
            height (int): Height of the widget
        """
        # Set the widget area based on the container logic
        self.set_widget_area(widget, **kwargs)
        self.widgets.append(widget)
    # end add_widget

    # This method should be overridden by subclasses
    def set_widget_area(self, widget, **kwargs):
        """
        Set the area for a widget. This method should be overridden by subclasses.
        """
        raise NotImplementedError("set_widget_area must be implemented by subclasses")
    # end set_widget_area

    def draw(
            self,
            context
    ):
        """
        Draw the container and its widgets to the context.

        Args:
            context (cairo.Context): Context to draw the container to
        """
        for widget in self.widgets:
            # Create a new sub-context for each widget
            widget_context = context.create()
            widget_context.rectangle(widget.x, widget.y, widget.width, widget.height)
            widget_context.clip()
            widget_context.translate(widget.x, widget.y)
            widget.draw(widget_context)
        # end for
    # end draw

# end Container


