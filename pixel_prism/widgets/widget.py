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
            surface: cairo.ImageSurface
    ):
        """
        Render the widget to the surface.

        Args:
            surface (cairo.ImageSurface): Surface to render the widget to
        """
        self.surface = surface
        context = cairo.Context(surface)
        self.draw(context)
    # end render

    def draw(
            self,
            context: cairo.Context
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
