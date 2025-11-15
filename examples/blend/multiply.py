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
# Description: This example demonstrates blending two images using the multiply blend mode.
#

# Imports
import numpy as np

# Pixel prism
from pixelprism import Animation
from pixelprism.base.imagecanvas import ImageCanvas
from pixelprism.base import Image
import pixelprism.base.functional as F


# Chromatic temporal persistence animation class
class BlendMultiplyAnimation(Animation):
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
        # Get input image math
        base_layer_image = image_canvas.get_layer("base").image

        # Blend images
        blended_image = F.multiply(base_layer_image, self.image)

        # Update image canvas
        image_canvas.update_layer_image("base", blended_image)

        return image_canvas
    # end process_frame

# end BlendMultiplyAnimation

