

# Imports
import cairo
from .widget import Widget


class DrawableWidget(Widget):
    """
    Drawable widget
    """

    def __init__(self):
        """
        Initialize the widget.
        """
        super().__init__()
        self.primitives = []
    # end __init__

    # Draw the widget
    def draw(
            self,
            context
    ):
        """
        Draw the widget to the context.

        Args:
            context (cairo.Context): Context to draw the widget to
        """
        # Save context
        context.save()

        # Antialiasing
        context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

        # For each primitive, draw it
        for primitive in self.primitives:
            primitive.draw(context)
        # end for

        # Restore context
        context.restore()
    # end draw

    def add(
            self,
            drawable
    ):
        """
        Add a drawable.

        Args:
            drawable (Drawable): Drawable
        """
        self.primitives.append(drawable)
    # end add

# end DrawableWidget

