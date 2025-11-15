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
# Description: Module to apply color effects to images
#


# Imports
import numpy as np


def apply_lut(
        image,
        lut,
        size
):
    """
    Apply a 3D LUT to an image

    Args:
        image (np.ndarray): Image to apply the LUT to
        lut (np.ndarray): 3D LUT to apply
        size (int): Size of the LUT
    """
    # Normaliser l'image
    image = image / 255.0

    # Appliquer la LUT
    index = (image * (size - 1)).astype(int)
    result = lut[index[:, :, 0], index[:, :, 1], index[:, :, 2]]

    # Remettre l'image à l'échelle de 0 à 255
    return (result * 255).astype(np.uint8)
# end apply_lut

