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
Pixel Prism Primitives - Basic Building Blocks
============================================

This subpackage provides primitive building blocks used throughout the Pixel Prism library.
These primitives are the fundamental components upon which more complex structures are built.

Main Components
--------------
- :class:`~pixelprism.primitives.image_layer.ImageLayer`: Basic image layer representation
- :class:`~pixelprism.primitives.points.Point`: Basic point representation

These primitives provide the foundation for more complex structures in the Pixel Prism library.
"""

from .image_layer import ImageLayer
from .points import Point

__all__ = [
    'ImageLayer',
    'Point'
]
