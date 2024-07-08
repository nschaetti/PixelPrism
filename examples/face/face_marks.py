
# Imports
from pixel_prism import VideoComposer, Animation

from pixel_prism.effects import (
    FacePointsEffect,
    DrawPointsEffect
)


class CustomAnimation(Animation):

    def init_effects(self):
        """
        Initialize the effects specific to this animation.
        """
        self.face_effect = FacePointsEffect()
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
        face_points = self.face_effect.apply(frame)

        draw_points_effect = DrawPointsEffect(face_points)
        frame = draw_points_effect.apply(frame)

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
