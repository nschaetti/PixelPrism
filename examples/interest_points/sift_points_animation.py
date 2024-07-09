
import numpy as np

from pixel_prism import Animation, VideoComposer
from pixel_prism.effects import SIFTPointsEffect, DrawPointsEffect


class CustomAnimation(Animation):
    """
    Custom animation that shows the SIFT points of the input video.
    """

    def init_effects(self):
        """Initialisation des effets spécifiques à cette animation."""
        self.sift_effect = SIFTPointsEffect(num_octaves=4, num_scales=3)
        self.draw_points_effect = DrawPointsEffect(points=[], color=(0, 255, 255), thickness=1)
        self.other_effects = []
    # end init_effects

    def process_frame(
            self,
            image_obj,
            frame_number,
            total_frames
    ):
        """
        Apply the effects to each frame.

        Args:
            image_obj (Image): Image object
            frame_number (int): Frame number
            total_frames (int): Total number of frames
        """
        mid_point = total_frames // 2

        input_image = image_obj.get_layer("input_frame").image

        if frame_number < mid_point:
            # Première partie : image originale avec points SIFT
            sift_points = self.sift_effect.apply(input_image)
            self.draw_points_effect.points = sift_points
            points_layer = self.draw_points_effect.apply(image_obj.get_layer("input_frame").image.copy())
            image_obj.add_layer("points_layer", points_layer)
        else:
            # Deuxième partie : fond noir avec points SIFT
            black_background = np.zeros_like(image_obj.get_layer("input_frame").image)
            sift_points = self.sift_effect.apply(input_image)
            self.draw_points_effect.points = sift_points
            points_layer = self.draw_points_effect.apply(black_background)
            image_obj.add_layer("points_layer", points_layer)

        return image_obj
    # end process_frame

# end CustomAnimation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply visual effects to videos using PixelPrism.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the output video file")
    args = parser.parse_args()

    composer = VideoComposer(args.input, args.output, CustomAnimation)
    composer.create_video()
# end if

