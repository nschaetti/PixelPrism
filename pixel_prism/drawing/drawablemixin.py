

class DrawableMixin(object):

    # Set source RGBA
    def set_source_rgba(
            self,
            context,
            color
    ):
        """
        Set the source RGBA of the context.

        Args:
            context (cairo.Context): Context to set the source RGBA of
            color (Color): Color to set
        """
        context.set_source_rgba(
            color.red,
            color.green,
            color.blue,
            color.alpha
        )
    # end set_source_rgba

    # Set source RGB
    def set_source_rgb(
            self,
            context,
            color
    ):
        """
        Set the source RGB of the context.

        Args:
            context (cairo.Context): Context to set the source RGB of
            color (Color): Color to set
        """
        context.set_source_rgb(
            color.red,
            color.green,
            color.blue
        )
    # end set_source_rgb

    def draw(
            self,
            context
    ):
        """
        Draw the point to the context.

        Args:
            context (cairo.Context): Context to draw the point to
        """
        raise NotImplementedError("draw method must be implemented in subclass.")
    # end draw

# end Element


