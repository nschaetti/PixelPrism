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
import numpy as np

# Local
import pixelprism.effects.functional as F
from pixelprism.effects.effect_base import EffectBase


def load_cube_lut(
        file_path
):
    """
    Load a 3D LUT from a .cube file

    Args:
        file_path (str): Path to the .cube file
    """
    # Load the LUT file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # end with

    size = 0
    lut = []

    # Parse the LUT file
    for line in lines:
        if line.startswith('#'):
            continue
        # end if

        if 'TITLE' in line:
            title = line.split()[1]
        # end if

        if 'LUT_3D_SIZE' in line:
            size = int(line.split()[1])
        # end if

        if len(line.split()) == 3:
            lut.append([float(x) for x in line.split()])
        # end if
    # end for

    # Convert the LUT to a numpy array
    lut = np.array(lut)
    lut = lut.reshape((size, size, size, 3))

    return lut, size
# end load_cube_lut


class LUTEffect(EffectBase):

    def __init__(
            self,
            lut_path
    ):
        """
        Initialize the LUT effect with the LUT path

        Args:
            lut_path (str): Path to the .cube file
        """
        self.lut, self.size = load_cube_lut(lut_path)
    # end __init__

    def apply(
            self,
            image,
            **kwargs
    ):
        """
        Apply the LUT effect to the image

        Args:
            image (np.ndarray): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        return F.apply_lut(image, self.lut, self.size)
    # end apply

# end LUTEffect

