#
# Description: This example demonstrates how to create a custom animation that shows the SIFT points of the
# input video.
#

# Imports
import numpy as np

# Pixel prism
from pixel_prism import Animation
from pixel_prism.base.imagecanvas import ImageCanvas
import pixel_prism.effects as effects


# Chromatic temporal persistence animation class
class ChromaticTemporalPersistenceAnimation(Animation):
    """
    Example animation that shows chromatic temporal shift.
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
            keep_frames=10,
            display=display,
            debug_frames=debug_frames,
            **kwargs
        )
    # end __init__

    def init_effects(
            self
    ):
        """
        Initialize the effects.
        """
        self.add_effect(
            "chromatic_temporal_persistence",
            effects.ChromaticTemporalPersistenceEffect(
                persistence_r=self.extra_args.get("persistence_r", 40),
                persistence_g=self.extra_args.get("persistence_g", 10),
                persistence_b=self.extra_args.get("persistence_b", 0),
                weight_decay=self.extra_args.get("weight_decay", 'linear')
            )
        )
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
        # random_shift = np.random.randint(0, 10)

        # Apply spatial shift effect
        shifted_image = self.get_effect("chromatic_temporal_persistence").apply(base_layer_image)

        # Update layer image
        image_canvas.set_layer_image("base", shifted_image)

        return image_canvas
    # end process_frame

# end ChromaticTemporalPersistenceAnimation

