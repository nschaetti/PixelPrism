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
- :class:`~pixel_prism.animation.Animation`: Base class for creating animations
- :class:`~pixel_prism.animation.AnimationViewer`: Simple viewer for animations
- :class:`~pixel_prism.effect_pipeline.EffectPipeline`: Pipeline for applying effects to images
- :class:`~pixel_prism.layer_manager.LayerManager`: Manager for image layers
- :class:`~pixel_prism.render_engine.RenderEngine`: Engine for rendering images
- :class:`~pixel_prism.video_composer.VideoComposer`: Composer for creating videos

Utility Functions
----------------
- :func:`~pixel_prism.animation.find_animable_mixins`: Find objects that can be animated
- :func:`~pixel_prism.basic.p2`: Create a 2D point
- :func:`~pixel_prism.basic.t_p2`: Create a 2D point with transform
- :func:`~pixel_prism.basic.s`: Create a scalar
- :func:`~pixel_prism.basic.t_s`: Create a scalar with transform
- :func:`~pixel_prism.basic.c`: Create a color

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
