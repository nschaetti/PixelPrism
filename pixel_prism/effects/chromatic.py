

# Imports
import numpy as np

import pixel_prism.effects.functional as F
from pixel_prism.effects.effect_base import EffectBase
from pixel_prism.base.image import Image


class ChromaticSpatialShiftEffect(EffectBase):
    """
    Spatial shift effect that shifts the image in the x and y directions for each channel
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
            persistence_b=5
    ):
        """
        Initialize the temporal persistence effect with the number of frames to blend

        Args:
            persistence_r (int): Number of frames to blend for the red channel
            persistence_g (int): Number of frames to blend for the green channel
            persistence_b (int): Number of frames to blend for the blue channel
        """
        self.persistence_r = persistence_r
        self.persistence_g = persistence_g
        self.persistence_b = persistence_b
        self.prev_frames_r = []
        self.prev_frames_g = []
        self.prev_frames_b = []
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
        return F.chromatic_temporal_persistence_effect(
            image=image,
            persistence_r=self.persistence_r,
            persistence_g=self.persistence_g,
            persistence_b=self.persistence_b,
            prev_frames_r=self.prev_frames_r,
            prev_frames_g=self.prev_frames_g,
            prev_frames_b=self.prev_frames_b
        )
    # end apply

# end ChromaticTemporalPersistenceEffect

