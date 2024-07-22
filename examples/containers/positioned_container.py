

# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import PositionedContainer
from pixel_prism.widgets import Point, Line
from pixel_prism.base import DrawableImage, ImageCanvas


# CustomAnimation class
class PositionedContainerAnimation(Animation):

    def init_effects(self):
        pass
    # end init_effects

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
        # Create a DrawableImage
        drawing_layer = DrawableImage.transparent(self.width, self.height)

        # Create a PositionedContainer and add Widgets
        positioned_container = PositionedContainer()
        positioned_container.add_widget(Point(radius=5, color=(1, 0, 0)), x=100, y=100)
        positioned_container.add_widget(Line(start_point=(0, 0), end_point=(50, 50), color=(0, 1, 0)), x=200, y=200)

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(positioned_container)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end PositionedContainerAnimation
