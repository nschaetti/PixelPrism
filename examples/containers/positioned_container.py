

# PixelPrism
from pixelprism.animation import Animation
from pixelprism.widgets.containers import SplitContainer, PositionedContainer
from pixelprism.widgets import Point, Line, Dummy
from pixelprism.base import DrawableImage, ImageCanvas


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
        split_container = SplitContainer(size=2, orientation='horizontal')
        split_container.add_widget(Dummy((1, 0, 0)), position=0)
        split_container.add_widget(Dummy((0, 1, 0)), position=1)

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(split_container)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end PositionedContainerAnimation
