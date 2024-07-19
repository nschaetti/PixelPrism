#
# Description: This example demonstrates blending two images using the normal blend mode.
#

# Imports
import numpy as np

# Pixel prism
from pixel_prism import Animation
from pixel_prism.base.imagecanvas import ImageCanvas
from pixel_prism.base import Image
import pixel_prism.base.functional as F


# Normal blend animation class
class BlendNormalAnimation(Animation):
    """
    Normal blend animation.
    """

    # Constructor
    def __init__(
            self,
            input_path,
            output_path,
            display=False,
            debug_frames=False,
            **kwargs
    ):
        """
        Initialize the animation with an input and output path.

        Args:
            input_path (str): Path to the input video
            output_path (str): Path to the output video
            display (bool): Whether to display the video while processing
            debug_frames (bool): Whether to display the layers while processing
            kwarg: Additional keyword arguments
        """
        # Call parent constructor
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            keep_frames=0,
            display=display,
            debug_frames=debug_frames,
            **kwargs
        )

        # Image
        self.image = Image.from_file("examples/blend/assets/circle_alpha.png")
    # end __init__

    def init_effects(
            self
    ):
        """
        Initialize the effects.
        """
        pass
    # end init_effects

    def process_frame(
            self,
            image_canvas: ImageCanvas,
            frame_number,
            total_frames
    ):
        """
        Apply the effects to each frame.

        Args:
            image_canvas (ImageCanvas): Image canvas
            frame_number (int): Frame number
            total_frames (int): Total number of frames
        """
        # Get input image data
        base_layer_image = image_canvas.get_layer("base").image

        # Blend images
        blended_image = F.normal(base_layer_image, self.image)

        # Update image canvas
        image_canvas.update_layer_image("base", blended_image)

        return image_canvas
    # end process_frame

# end BlendNormalAnimation

