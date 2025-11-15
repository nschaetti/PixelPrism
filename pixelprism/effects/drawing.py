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

# Import necessary packages
import cv2
import numpy as np

import pixelprism.effects.functional as F
from .effect_base import EffectBase
from pixelprism.base.image import Image


class DrawPointsEffect(EffectBase):
    """
    Effect to draw points on an image
    """

    def __init__(
            self,
            color=(255, 255, 0),
            thickness=1
    ):
        """
        Initialize the draw points effect with the points to draw

        Args:
            color (tuple): Color of the points to draw
            thickness (int): Thickness of the points to draw
        """
        self.color = color
        self.thickness = thickness
    # end __init__

    def apply(
            self,
            image: Image,
            **kwargs
    ):
        """
        Apply the draw points effect to the image

        Args:
            image (Image): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        # Assert points are provided
        if 'points' not in kwargs:
            raise ValueError("Points must be provided to draw on the image")
        # end if

        # Points
        points = kwargs['points']

        return F.draw_points(image, points, self.color, self.thickness)
    # end apply

# end DrawPointsEffect

