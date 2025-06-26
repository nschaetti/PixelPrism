
# Imports
from pixelprism import VideoComposer, Animation

from pixelprism.effects import (
    AdvancedTVEffect,
    ChromaticSpatialShiftEffect,
    ChromaticTemporalPersistenceEffect,
    GlowEffect,
    LenticularDistortionEffect,
    BlurEffect,
    LUTEffect
)


class CustomAnimation(Animation):

    def init_effects(self):
        """
        Initialize the effects specific to this animation.
        """
        self.effects = [
            AdvancedTVEffect(
                pixel_width=6,
                pixel_height=12,
                border_strength=2,
                border_color=(0, 0, 0),
                corner_radius=1,
                blur_kernel_size=0,
                vertical_shift=6
            ),
            # LUTEffect(
            #     lut_path='examples/analog/ressources/Retro_3.cube'
            # ),
            ChromaticSpatialShiftEffect(
                shift_r=(3, 0),
                shift_g=(-2, 0),
                shift_b=(0, 3)
            ),
            ChromaticTemporalPersistenceEffect(
                persistence_r=6,
                persistence_g=3,
                persistence_b=1
            ),
            GlowEffect(
                blur_strength=10,
                intensity=1,
                blend_mode='screen'
            ),
            BlurEffect(
                blur_strength=5
            )
        ]
    # end init_effects

    def process_frame(
            self,
            frame,
            frame_number
    ):
        """
        Process each frame of the video. Should be implemented by derived classes.

        Args:
            frame (np.ndarray): The frame to process
            frame_number (int): The frame number
        """
        for effect in self.effects:
            frame = effect.apply(frame)
        # end for

        return frame
    # end process_frame

# end CustomAnimation


# Main
if __name__ == "__main__":
    import argparse

    # Parse the input arguments
    parser = argparse.ArgumentParser(description="Apply visual effects to videos using PixelPrism.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the output video file")
    args = parser.parse_args()

    # Create the video composer
    composer = VideoComposer(args.input, args.output, CustomAnimation)
    composer.create_video()
# end if
