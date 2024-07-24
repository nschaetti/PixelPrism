

# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import SplitContainer, PositionedContainer
from pixel_prism.widgets import Point, Line, Dummy
from pixel_prism.base import DrawableImage, ImageCanvas


# CustomAnimation class
class SplitContainerAnimation(Animation):

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
        split_container = SplitContainer(size=3, orientation='horizontal')
        split_container.add_widget(Dummy((1, 0, 0)), position=0)
        split_container.add_widget(Dummy((0, 1, 0)), position=1)
        split_container.add_widget(Dummy((0, 0, 1)), position=2)

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(split_container)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end SplitContainerAnimation

