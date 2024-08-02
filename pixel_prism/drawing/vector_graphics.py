

from .drawablemixin import DrawableMixin
from .paths import Path
from pixel_prism.data import VectorGraphicsData, PathData, RectangleData


class VectorGraphics(DrawableMixin, VectorGraphicsData):

    def __init__(self):
        """
        Initialize the vector graphics
        """
        # Init of VectorGraphicsData
        super().__init__()
    # end __init__

    def draw(
            self,
            context
    ):
        """
        Draw the vector graphics to the context.

        Args:
            context (cairo.Context): Context to draw the vector graphics to
        """
        # For each element in the vector graphics
        for element in self.elements:
            # Draw the rectangle
            element.to_drawable().draw(context)
        # end for
        exit()
    # end draw

# end VectorGraphics

