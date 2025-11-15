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
import cv2
import numpy as np

from pixelprism.base.image import Image


def create_tv_overlay(
        shape,
        pixel_width=10,
        pixel_height=10,
        vertical_shift=5,
        border_color=(0, 0, 0),
        border_strength=1,
        corner_radius=0,
        blur_kernel_size=1
):
    """
    Create the overlay for the TV effect

    Args:
        shape (tuple): Shape of the image (height, width, channels)
        pixel_width (int): Width of the pixel
        pixel_height (int): Height of the pixel
        vertical_shift (int): Vertical shift of the pixels
        border_color (tuple): Color of the border
        border_strength (int): Strength of the border
        corner_radius (int): Radius of the corners
        blur_kernel_size (int): Kernel size for the Gaussian blur

    Returns:
        np.ndarray: Overlay image for the TV effect
    """
    height, width, _ = shape
    overlay = np.ones((height, width, 3), dtype=np.float32)

    for j_i, j in enumerate(range(0, width, pixel_width)):
        if j_i % 2 == 0:
            shift = 0
        else:
            shift = vertical_shift
        # end if

        for i in range(-shift, height, pixel_height):
            y = i
            if y > height:
                continue

            sub_img = overlay[y:y + pixel_height, j:j + pixel_width]

            # Dessiner les bords du pixel
            cv2.line(sub_img, (0, 0), (pixel_width, 0), border_color, border_strength)
            cv2.line(sub_img, (0, 0), (0, pixel_height), border_color, border_strength)
            cv2.line(sub_img, (pixel_width, 0), (pixel_width, pixel_height), border_color, border_strength)
            cv2.line(sub_img, (0, pixel_height), (pixel_width, pixel_height), border_color, border_strength)

            # Casser les coins
            if corner_radius > 0:
                cv2.line(sub_img, (0, 0), (corner_radius, corner_radius), border_color, border_strength)
                cv2.line(sub_img, (pixel_width, 0), (pixel_width - corner_radius, corner_radius), border_color, border_strength)
                cv2.line(sub_img, (0, pixel_height), (corner_radius, pixel_height - corner_radius), border_color, border_strength)
                cv2.line(sub_img, (pixel_width, pixel_height), (pixel_width - corner_radius, pixel_height - corner_radius), border_color, border_strength)
            # end if
        # end for
    # end for

    # Apply blur
    if blur_kernel_size > 1:
        overlay = cv2.GaussianBlur(overlay, (blur_kernel_size, blur_kernel_size), 0)
    # end if

    return overlay
# end create_tv_overlay


def tv(
        image: Image,
        overlay: np.ndarray
) -> Image:
    """
    Apply the TV effect to the image

    Args:
        image (Image): Image to apply the effect to
        overlay (np.ndarray): Overlay image for the TV effect

    Returns:
        Image: Image with the TV effect applied
    """
    tv_effect = image * overlay
    return tv_effect.astype(np.uint8)
# end tv


