
"""
Pixel Prism Primitives - Basic Building Blocks
============================================

This subpackage provides primitive building blocks used throughout the Pixel Prism library.
These primitives are the fundamental components upon which more complex structures are built.

Main Components
--------------
- :class:`~pixel_prism.primitives.image_layer.ImageLayer`: Basic image layer representation
- :class:`~pixel_prism.primitives.points.Point`: Basic point representation

These primitives provide the foundation for more complex structures in the Pixel Prism library.
"""

from .image_layer import ImageLayer
from .points import Point

__all__ = [
    'ImageLayer',
    'Point'
]
