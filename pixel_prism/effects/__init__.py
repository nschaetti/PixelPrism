
"""
Pixel Prism Effects - Image Processing and Visual Effects
======================================================

This subpackage provides a collection of image processing effects and filters
that can be applied to images and videos in the Pixel Prism library. These effects
range from basic color adjustments to complex visual effects like chromatic aberration,
glow, and distortion.

Main Components
--------------
- Base Classes:
  - :class:`~pixel_prism.effects.effect_base.EffectBase`: Base class for all effects
  - :class:`~pixel_prism.effects.effect_group.EffectGroup`: Group of effects that can be applied together

- Visual Effects:
  - :class:`~pixel_prism.effects.effects.AdvancedTVEffect`: TV screen effect with scanlines and distortion
  - :class:`~pixel_prism.effects.effects.ChromaticAberrationEffect`: Color channel separation effect
  - :class:`~pixel_prism.effects.effects.LenticularDistortionEffect`: Lens distortion effect
  - :class:`~pixel_prism.effects.effects.GlowEffect`: Glow/bloom effect
  - :class:`~pixel_prism.effects.effects.BlurEffect`: Blur effect

- Chromatic Effects:
  - :class:`~pixel_prism.effects.chromatic.ChromaticSpatialShiftEffect`: Spatial shift of color channels
  - :class:`~pixel_prism.effects.chromatic.ChromaticTemporalPersistenceEffect`: Temporal persistence of color channels

- Color Effects:
  - :class:`~pixel_prism.effects.colors.LUTEffect`: Look-up table color grading effect

- Drawing Effects:
  - :class:`~pixel_prism.effects.drawing.DrawPointsEffect`: Effect for drawing points on an image

- Feature Detection:
  - :class:`~pixel_prism.effects.face.FacePointsEffect`: Effect for detecting and drawing face points
  - :class:`~pixel_prism.effects.interest_points.SIFTPointsEffect`: Effect for detecting and drawing SIFT interest points

These effects can be combined and chained together to create complex visual styles
and image processing pipelines.
"""

# Imports
from .chromatic import ChromaticSpatialShiftEffect, ChromaticTemporalPersistenceEffect
from .colors import LUTEffect
from .drawing import DrawPointsEffect
from .effect_base import EffectBase
from .effect_group import EffectGroup

from .effects import (
    AdvancedTVEffect,
    ChromaticAberrationEffect,
    LenticularDistortionEffect,
    GlowEffect,
    BlurEffect,
)

from .face import FacePoint, FacePointsEffect

from .interest_points import SIFTPointsEffect


__all__ = [
    'AdvancedTVEffect',
    'ChromaticAberrationEffect',
    'LenticularDistortionEffect',
    'GlowEffect',
    'BlurEffect',
    'ChromaticSpatialShiftEffect',
    'ChromaticTemporalPersistenceEffect',
    'LUTEffect',
    'DrawPointsEffect',
    'EffectBase',
    'EffectGroup',
    'FacePoint',
    'FacePointsEffect',
    'SIFTPointsEffect'
]
