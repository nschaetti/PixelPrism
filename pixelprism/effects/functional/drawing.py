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
# Description: Module to apply drawing effects to images
#

# Imports
import cv2
import numpy as np

from pixelprism.base.image import Image


def draw_points(
        image: Image,
        points,
        color,
        thickness
):
    """
    Draw points on an image

    Args:
        image (np.ndarray): Image to draw the points on
        points (list): List of points to draw
        color (tuple): Color of the points
        thickness (int): Thickness of the points
    """
    # Alpha ?
    has_alpha = image.has_alpha

    # Create a temporary image
    temp_image = np.zeros_like(image.data[:, :, :3])

    # Draw the points on the temporary image
    for point in points:
        cv2.circle(
            temp_image,
            (int(point.x), int(point.y)), int(point.size / 2),
            color[:3],
            thickness,
            lineType=cv2.LINE_AA
        )
    # end for

    # Add the points to the image
    mask = temp_image > 0
    image.data[:, :, :3][mask] = temp_image[mask]

    # if has_alpha:
    if has_alpha:
        # print(f"Before: {np.mean(image.math_old[:, :, 3])}")
        image.data[:, :, 3][mask[:, :, 1]] = 255
        # print(f"After: {np.mean(image.math_old[:, :, 3])}")
    # end if

    return image
# end draw_points
