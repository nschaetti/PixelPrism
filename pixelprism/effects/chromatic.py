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
# Contains the chromatic effects that can be applied to an image.
#

# Imports
import numpy as np

# Local
import pixelprism.effects.functional as F
from pixelprism.effects.effect_base import EffectBase
from pixelprism.base.image import Image


class ChromaticSpatialShiftEffect(EffectBase):
    """
    Spatial shift effect that shifts the image in the x and y directions for each channel.
    """

    def __init__(
            self,
            shift_r=(5, 0),
            shift_g=(-5, 0),
            shift_b=(0, 5)
    ):
        """
        Initialize the spatial shift effect with the shifts for each channel

        Args:
            shift_r (tuple): Shift for the red channel
            shift_g (tuple): Shift for the green channel
            shift_b (tuple): Shift for the blue channel
        """
        self.shift_r = shift_r
        self.shift_g = shift_g
        self.shift_b = shift_b
    # end __init__

    def apply(
            self,
            image,
            **kwargs
    ):
        """
        Apply the spatial shift effect to the image

        Args:
            image (np.ndarray): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        return F.chromatic_spatial_shift_effect(
            image,
            shift_r=self.shift_r,
            shift_g=self.shift_g,
            shift_b=self.shift_b
        )
    # end apply
# end ChromaticSpatialShiftEffect


class ChromaticTemporalPersistenceEffect(EffectBase):
    """
    Temporal persistence effect that blends the current frame with previous frames
    """

    def __init__(
            self,
            persistence_r=5,
            persistence_g=5,
            persistence_b=5,
            weight_decay='linear'
    ):
        """
        Initialize the temporal persistence effect with the number of frames to blend

        Args:
            persistence_r (int): Number of frames to blend for the red channel
            persistence_g (int): Number of frames to blend for the green channel
            persistence_b (int): Number of frames to blend for the blue channel
            weight_decay (str): Weight decay function to use
        """
        self.persistence_r = persistence_r
        self.persistence_g = persistence_g
        self.persistence_b = persistence_b
        self.weight_decay = weight_decay
        self.prev_frames = []
    # end __init__

    def apply(
            self,
            image: Image,
            **kwargs
    ):
        """
        Apply the temporal persistence effect to the image

        Args:
            image (Image): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        # Apply the temporal persistence effect
        shifted_image = F.chromatic_temporal_persistence_effect(
            image=image,
            persistence_r=self.persistence_r,
            persistence_g=self.persistence_g,
            persistence_b=self.persistence_b,
            prev_frames=self.prev_frames,
            weight_decay=self.weight_decay
        )

        # Update previous frames, and pop if the length exceeds the persistence
        self.prev_frames.append(image)
        if len(self.prev_frames) > max(self.persistence_r, self.persistence_g, self.persistence_b):
            self.prev_frames.pop(0)
        # end if

        return shifted_image
    # end apply

# end ChromaticTemporalPersistenceEffect

