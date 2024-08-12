
# Imports
from typing import Tuple
import cairo
from pixel_prism.widgets.widget import Widget
from pixel_prism.base import Context


# Viewport class, container for a scalable and scrollable area
class Viewport(Widget):
    """
    Viewport class, container for a scalable and scrollable area
    """

    # Init
    def __init__(
            self,
            translate: Tuple[int, int] = (0, 0),
            scale: Tuple[float, float] = (1, 1),
            rotate: float = 0
    ):
        """
        Initialize the viewport.
        """
        super().__init__()
        self.x = 0
        self.y = 0
        self.translate = translate
        self.scale = scale
        self.rotate = rotate
        self.widgets = []
    # end __init__

    # Add widget
    def add_widget(self, widget):
        """
        Add a widget to the viewport.

        Args:
            widget (Widget): The widget to add to the viewport.
        """
        self.widgets.append(widget)
    # end add_widget

    # Draw
    def draw(
            self,
            context: Context,
            *args,
            **kwargs
    ):
        """
        Draw the viewport to the context.

        Args:
            context (cairo.Context): Context to draw the viewport to
        """
        # Translate the context
        context.translate(self.translate[0], self.translate[1])

        # Scale the context
        context.scale(self.scale[0], self.scale[1])

        # Rotate the context
        context.rotate(self.rotate)

        # Draw the widgets
        for widget in self.widgets:
            widget.draw(context, *args, **kwargs)
        # end for
    # end draw

    # Render
    def render(
            self,
            context: Context,
            *args,
            **kwargs
    ):
        """
        Render the viewport to the surface.

        Args:
            context (Context): Context to render the viewport to
        """
        # Get draw params
        draw_params = kwargs.get("draw_params", {})

        # Draw the widgets
        self.draw(
            context,
            **draw_params
        )
    # end render

    # Create sub-surface for a widget
    def create_surface(
            self,
            widget: Widget,
            **kwargs
    ):
        """
        Create a sub-surface for a widget.

        Args:
            widget (Widget): The widget to create a surface for.
        """
        return self.surface
    # end create_surface

# end Viewport
