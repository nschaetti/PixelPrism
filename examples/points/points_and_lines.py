
# Imports
import numpy as np
import cairo
from pixel_prism import Animation
from pixel_prism.base import Image, ImageCanvas, DrawableImage
from pixel_prism.drawing import Space2D, Point, Line, Plot


class PointsLinesAnimation(Animation):

    # Init. effects
    def init_effects(
            self
    ):
        """
        Initialize the effects.
        """
        pass
    # end init_effects

    # Process frame
    def process_frame(
            self,
            image_canvas: ImageCanvas,
            t: float,
            frame_number: int
    ):
        """
        Process the frame.

        Args:
            image_canvas (ImageCanvas): Image canvas
            t (float): Time
            frame_number (int): Frame number
        """
        # Create a new transparent layer for drawing
        drawing_image = Image.fill(self.width, self.height, (0, 0, 0, 255.0))
        drawing_layer = DrawableImage.transparent(self.width, self.height)

        # Draw point, line and plot
        space = Space2D()
        space.add(Point(100, 100))
        space.add(Line((150, 150), (300, 300)))
        space.add(Plot(lambda x: 0.01 * (x - 320)**2, (0, 640)))

        # Draw the space
        space.draw(context)

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_image)

        return image_canvas
    # end process_frame

# end PointsLinesAnimation

