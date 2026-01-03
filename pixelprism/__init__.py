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
# Copyright (C) 2025 Pixel Prism
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

"""
Pixel Prism - Image Manipulation, Procedural Generation & Visual Effects
========================================================================

Pixel Prism is a creative toolkit for procedural image and video generation. 
It includes support for advanced compositing, GLSL shaders, depth maps,
image segmentation, and AI-powered effects. Automate your workflow with a rich
set of nodes for blending, masking, filtering, and compositional adjustments.
Perfect for artists, designers, and researchers exploring image aesthetics.

This package provides tools for:
- Animation creation and manipulation
- Layer-based image compositing
- Effect pipelines for image processing
- Video composition

Main Components
--------------
- :class:`~pixelprism.animation.Animation`: Base class for creating animations
- :class:`~pixelprism.animation.AnimationViewer`: Simple viewer for animations
- :class:`~pixelprism.effect_pipeline.EffectPipeline`: Pipeline for applying effects to images
- :class:`~pixelprism.layer_manager.LayerManager`: Manager for image layers
- :class:`~pixelprism.render_engine.RenderEngine`: Engine for rendering images
- :class:`~pixelprism.video_composer.VideoComposer`: Composer for creating videos

Utility Functions
----------------
- :func:`~pixelprism.animation.find_animable_mixins`: Find objects that can be animated
- :func:`~pixelprism.basic.p2`: Create a 2D point
- :func:`~pixelprism.basic.t_p2`: Create a 2D point with transform
- :func:`~pixelprism.basic.s`: Create a scalar
- :func:`~pixelprism.basic.t_s`: Create a scalar with transform
- :func:`~pixelprism.basic.c`: Create a color

For more information, visit: https://github.com/nschaetti/PixelPrism
"""

# Imports
from .animation import Animation, AnimationViewer, find_animable_mixins
from .basic import (
    p2,
    t_p2,
    s,
    t_s,
    c
)
from .effect_pipeline import EffectPipeline
from .layer_manager import LayerManager
from .render_engine import RenderEngine
from .video_composer import VideoComposer


# ALL
__all__ = [
    # Shortcuts
    'p2',
    's',
    'c',
    # Animation
    'Animation',
    'AnimationViewer',
    'find_animable_mixins',
    # Effect pipeline and others
    'EffectPipeline',
    'RenderEngine',
    'LayerManager',
    'VideoComposer'
]
