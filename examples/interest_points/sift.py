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

# Imports
from pixelprism import VideoComposer, Animation

from pixelprism.effects import (
    AdvancedTVEffect,
    ChromaticSpatialShiftEffect,
    ChromaticTemporalPersistenceEffect,
    GlowEffect,
    BlurEffect,
    SIFTPointsEffect,
    DrawPointsEffect
)


class CustomAnimation(Animation):

    def init_effects(self):
        """
        Initialize the effects specific to this animation.
        """
        self.sift_effect = SIFTPointsEffect(num_octaves=2, num_scales=2)
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
        sift_points = self.sift_effect.apply(frame)

        draw_points_effect = DrawPointsEffect(sift_points)
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
