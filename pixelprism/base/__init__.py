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
Pixel Prism Base - Core Image and Coordinate System Classes
=========================================================

This subpackage provides the fundamental classes for image representation, 
coordinate systems, and drawing contexts in the Pixel Prism library.
These classes form the foundation upon which the rest of the library is built.

Main Components
--------------
- Image Representation:
  - :class:`~pixelprism.base.image.Image`: Core class for storing and manipulating image math
  - :class:`~pixelprism.base.image.ImageMode`: Enumeration of image modes (RGB, RGBA, etc.)
  - :class:`~pixelprism.base.imagecanvas.ImageCanvas`: Class for working with multi-layered images
  - :class:`~pixelprism.base.layer.Layer`: Class representing a single layer in an image

- Drawing and Coordinate Systems:
  - :class:`~pixelprism.base.context.Context`: Drawing context for rendering operations
  - :class:`~pixelprism.base.coordsystem.CoordSystem`: Coordinate system for transformations
  - :class:`~pixelprism.base.drawableimage.DrawableImage`: Image that can be drawn on

These classes provide the core functionality for creating, manipulating, and
rendering images in the Pixel Prism library.
"""

# Imports
from .context import Context
from .coordsystem import CoordSystem
from .drawableimage import DrawableImage
from .image import Image, ImageMode
from .imagecanvas import ImageCanvas
from .layer import Layer


__all__ = [
    'Context',
    'CoordSystem',
    'DrawableImage',
    'Image',
    'ImageMode',
    'ImageCanvas',
    'Layer'
]
