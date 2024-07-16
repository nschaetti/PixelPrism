#
# Description: This example demonstrates how to create a custom animation that shows the SIFT points of the
# input video.
#

# Imports
import numpy as np

# Pixel prism
from pixel_prism import Animation
from pixel_prism.base.imagecanvas import ImageCanvas
import pixel_prism.effects.functional as F


# Chromatic shift animation class
class ChromaticShiftAnimation(Animation):
    """
    Example animation that shows chromatic spatial shift.
    """

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

        # Random number between 0 and 25
        random_shift = np.random.randint(0, 25)

        # Apply spatial shift effect
        shifted_image = F.chromatic_spatial_shift_effect(
            image=base_layer_image,
            shift_r=(random_shift, 0),
            shift_g=(0, 0),
            shift_b=(0, 0)
        )

        # Update layer image
        image_canvas.set_layer_image("base", shifted_image)

        return image_canvas
    # end process_frame

# end ChromaticShiftAnimation

