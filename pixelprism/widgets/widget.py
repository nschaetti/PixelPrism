#
# This file contains the Widget class, which is the base class for all widgets in the Pixel Prism library.
#


import cairo

from pixelprism.base import Context


class Widget:
    """
    Base class for all widgets in the Pixel Prism library.
    """

    def __init__(self):
        """
        Initialize the widget.
        """
        self.context = None
    # end __init__

    # region PROPERTIES

    @property
    def width(self):
        """
        Get the width of the widget.
        """
        return self.context.width
    # end width

    @property
    def height(self):
        """
        Get the height of the widget.
        """
        return self.context.height
    # end height

    # endregion PROPERTIES

    # region PUBLIC

    def render(
            self,
            context: Context,
            *args,
            **kwargs
    ):
        """
        Render the widget to the surface.

        Args:
            context (Context): Context to render the widget to
        """
        # Get surface and context
        self.context = context

        # Get draw params
        draw_params = kwargs.get("draw_params", {})

        # Draw the widget
        self.draw(context, **draw_params)
    # end render

    def draw(
            self,
            context: Context,
            *args,
            **kwargs
    ):
        """
        Draw the widget to the context.

        Args:
            context (Context): Context to draw the widget to
        """
        pass
    # end draw

    # endregion PUBLIC

# end Widget
