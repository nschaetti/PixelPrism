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

# Local imports
from pixelprism.effects import EffectBase
from pixelprism.primitives import Point
from pixelprism.base.image import Image


class SIFTPoint(Point):
    """
    SIFT point class that extends the Point class with additional SIFT attributes.
    """

    def __init__(
            self,
            x,
            y,
            size,
            angle,
            response,
            octave,
            class_id
    ):
        """
        Initialize the SIFT point with its coordinates, size, angle, response, octave, and class ID.

        Args:
            x (int): X-coordinate of the point
            y (int): Y-coordinate of the point
            size (int): Size of the point
            angle (float): Angle of the point
            response (float): Response of the point
            octave (int): Octave of the point
            class_id (int): Class ID of the point
        """
        super().__init__(x, y, size)
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id
    # end __init__

    def __repr__(self):
        return f"SIFTPoint(x={self.x}, y={self.y}, size={self.size}, angle={self.angle}, response={self.response}, octave={self.octave}, class_id={self.class_id})"
    # end __repr__

# end SIFTPoint


class SIFTPointsEffect(EffectBase):
    """
    Effect to extract SIFT points from an image.
    """

    def __init__(
            self,
            num_octaves=4,
            num_scales=3
    ):
        """
        Initialize the SIFT effect with the number of octaves and scales to use

        Args:
            num_octaves (int): Number of octaves to use in SIFT
            num_scales (int): Number of scales to use in SIFT
        """
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sift = cv2.SIFT_create(nOctaveLayers=num_scales)
    # end __init__

    def apply(
            self,
            image: Image,
            **kwargs
    ):
        """
        Apply the SIFT effect to the input image

        Args:
            image (Image): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        gray_image = cv2.cvtColor(image.data[:, :, :3], cv2.COLOR_BGR2GRAY)
        keypoints = self.sift.detect(gray_image, None)
        sift_points = [
            SIFTPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints
        ]
        return sift_points
    # end apply

# end SIFTPointsEffect

