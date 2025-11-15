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
# Description: This file contains glow effect.
#

# Imports
import cv2
import numpy as np

# Local
from pixelprism.base.image import Image


def simple_glow(
        image: Image,
        intensity: float = 0.5,
        blur_strength: int = 5,
        blend_mode: str = 'screen'
) -> Image:
    """
    Apply the glow effect to the image.

    Args:
        image (Image): Image to apply the effect to
        intensity (float): Intensity of the glow
        blur_strength (int): Strength of the Gaussian blur
        blend_mode (str): Blend mode for the glow effect

    Returns:
        Image: Image with the glow effect applied
    """
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image.data, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # Apply blend mode
    if blend_mode == 'addition':
        glow_image = cv2.addWeighted(image, 1, blurred_image, intensity, 0)
    elif blend_mode == 'multiply':
        glow_image = cv2.multiply(image, blurred_image)
        glow_image = cv2.addWeighted(image, 1, glow_image, intensity - 1, 0)
    elif blend_mode == 'overlay':
        glow_image = np.where(
            blurred_image > 128, 255 - 2 * (255 - blurred_image) * (255 - image) / 255,
            2 * image * blurred_image / 255
        )
        glow_image = cv2.addWeighted(image, 1, glow_image.astype(np.uint8), intensity, 0)
    else:  # screen is the default blend mode
        inverted_image = 255 - image
        inverted_blur = 255 - blurred_image
        screen_image = cv2.multiply(inverted_image, inverted_blur, scale=1 / 255.0)
        screen_image = 255 - screen_image
        glow_image = cv2.addWeighted(image, 1, screen_image, intensity, 0)
    # end if

    return glow_image
# end simple_glow

