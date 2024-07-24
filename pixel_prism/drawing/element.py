

class Element(object):

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
