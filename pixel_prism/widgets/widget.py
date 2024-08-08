#
# This file contains the Widget class, which is the base class for all widgets in the Pixel Prism library.
#


import cairo


class Widget:
    """
    Base class for all widgets in the Pixel Prism library.
    """

    def __init__(self):
        """
        Initialize the widget.
        """
        self.surface = None
    # end __init__

    # region PROPERTIES

    @property
    def width(self):
        """
        Get the width of the widget.
        """
        return self.surface.get_width()
    # end width

    @property
    def height(self):
        """
        Get the height of the widget.
        """
        return self.surface.get_height()
    # end height

    # endregion PROPERTIES

    # region PUBLIC

    def render(
            self,
            surface: cairo.ImageSurface,
            *args,
            **kwargs
    ):
        """
        Render the widget to the surface.

        Args:
            surface (cairo.ImageSurface): Surface to render the widget to
            draw_params (dict): Parameters for drawing the widget
        """
        # Get surface and context
        self.surface = surface
        context = cairo.Context(surface)

        # Get draw params
        draw_params = kwargs.get("draw_params", {})

        # Draw the widget
        self.draw(context, **draw_params)
    # end render

    def draw(
            self,
            context: cairo.Context,
            *args,
            **kwargs
    ):
        """
        Draw the widget to the context.

        Args:
            context (cairo.Context): Context to draw the widget to
        """
        pass
    # end draw

    # endregion PUBLIC

# end Widget
