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


# Chromatic temporal persistence animation class
class ChromaticTemporalPersistenceAnimation(Animation):
    """
    Example animation that shows chromatic temporal shift.
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
        random_shift = np.random.randint(0, 10)

        # Apply spatial shift effect
        shifted_image = F.chromatic_temporal_persistence_effect(
            image=base_layer_image,
            persistence_r=random_shift,
            persistence_g=0,
            persistence_b=0,
            prev_frames=self.prev_frames
        )

        # Update layer image
        image_canvas.set_layer_image("base", shifted_image)

        return image_canvas
    # end process_frame

# end ChromaticTemporalPersistenceAnimation

