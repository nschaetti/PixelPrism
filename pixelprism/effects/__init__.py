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
  - :class:`~pixelprism.effects.effect_base.EffectBase`: Base class for all effects
  - :class:`~pixelprism.effects.effect_group.EffectGroup`: Group of effects that can be applied together

- Visual Effects:
  - :class:`~pixelprism.effects.effects.AdvancedTVEffect`: TV screen effect with scanlines and distortion
  - :class:`~pixelprism.effects.effects.ChromaticAberrationEffect`: Color channel separation effect
  - :class:`~pixelprism.effects.effects.LenticularDistortionEffect`: Lens distortion effect
  - :class:`~pixelprism.effects.effects.GlowEffect`: Glow/bloom effect
  - :class:`~pixelprism.effects.effects.BlurEffect`: Blur effect

- Chromatic Effects:
  - :class:`~pixelprism.effects.chromatic.ChromaticSpatialShiftEffect`: Spatial shift of color channels
  - :class:`~pixelprism.effects.chromatic.ChromaticTemporalPersistenceEffect`: Temporal persistence of color channels

- Color Effects:
  - :class:`~pixelprism.effects.colors.LUTEffect`: Look-up table color grading effect

- Drawing Effects:
  - :class:`~pixelprism.effects.drawing.DrawPointsEffect`: Effect for drawing points on an image

- Feature Detection:
  - :class:`~pixelprism.effects.face.FacePointsEffect`: Effect for detecting and drawing face points
  - :class:`~pixelprism.effects.interest_points.SIFTPointsEffect`: Effect for detecting and drawing SIFT interest points

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
