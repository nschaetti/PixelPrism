# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Import necessary libraries
import numpy as np

# Import necessary classes
from pixelprism import Animation, VideoComposer
from pixelprism.effects import SIFTPointsEffect, DrawPointsEffect
from pixelprism.base.image import Image


# Custom animation class
class CustomAnimation(Animation):
    """
    Custom animation that shows the SIFT points of the input video.
    """

    def init_effects(
            self
    ):
        """
        Initialize the effects.
        """
        self.sift_effect = SIFTPointsEffect(num_octaves=4, num_scales=3)
        self.draw_points_effect = DrawPointsEffect(color=(0, 255, 255), thickness=1)
        self.other_effects = []
    # end init_effects

    def process_frame(
            self,
            image_canvas,
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
        mid_point = total_frames // 2

        # Get input image math_old
        base_layer_image = image_canvas.get_layer("base").image

        # Add black background
        if frame_number >= mid_point:
            image_canvas.create_color_layer("black_background", [0, 0, 0, 255])
        # end if

        # Get SIFT points
        sift_points = self.sift_effect.apply(base_layer_image)

        # Create an image object
        trans_im = Image.transparent(
            image_canvas.width,
            image_canvas.height,
        )

        # Draw the SIFT points on the image
        points_image = self.draw_points_effect.apply(trans_im, points=sift_points)

        # Add points
        image_canvas.add_layer("points_layer", points_image)

        return image_canvas
    # end process_frame

# end CustomAnimation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply visual effects to videos using PixelPrism.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the output video file")
    parser.add_argument("--display", action="store_true", help="Display the video while processing")
    parser.add_argument("--debug_frames", type=int, nargs='*', help="List of frame numbers to debug")
    args = parser.parse_args()

    composer = VideoComposer(
        args.input,
        args.output,
        CustomAnimation,
        display=args.display,
        debug_frames=args.debug_frames
    )
    composer.create_video()
# end if

