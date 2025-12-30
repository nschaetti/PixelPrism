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

#
# Description: This example demonstrates how to create a custom animation that shows the SIFT points of the
# input video.
#

# Imports
import numpy as np

# Pixel prism
from pixelprism import Animation
from pixelprism.base.imagecanvas import ImageCanvas
import pixelprism.effects.functional as F


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
        # Get input image math_old
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

