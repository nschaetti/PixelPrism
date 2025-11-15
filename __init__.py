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
▗▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▖       ▗▄▄▖ ▗▄▄▖ ▗▄▄▄▖ ▗▄▄▖▗▖  ▗▖
▐▌ ▐▌ █   ▝▚▞▘ ▐▌   ▐▌       ▐▌ ▐▌▐▌ ▐▌  █  ▐▌   ▐▛▚▞▜▌
▐▛▀▘  █    ▐▌  ▐▛▀▀▘▐▌       ▐▛▀▘ ▐▛▀▚▖  █   ▝▀▚▖▐▌  ▐▌
▐▌  ▗▄█▄▖▗▞▘▝▚▖▐▙▄▄▖▐▙▄▄▖    ▐▌   ▐▌ ▐▌▗▄█▄▖▗▄▄▞▘▐▌  ▐▌

             Image Manipulation, Procedural Generation & Visual Effects
                     https://github.com/nschaetti/PixelPrism

@title: Pixel Prism
@author: Nils Schaetti
@category: Image Processing
@reference: https://github.com/nils-schaetti/pixel-prism
@tags: image, pixel, animation, compositing, effects, shader, procedural, generation,
mask, layer, video, transformation, depth, AI, automation, creative, rendering
@description: Pixel Prism is a creative toolkit for procedural image and video
generation. Includes support for advanced compositing, GLSL shaders, depth maps,
image segmentation, and AI-powered effects. Automate your workflow with a rich
set of nodes for blending, masking, filtering, and compositional adjustments.
Perfect for artists, designers, and researchers exploring image aesthetics.
@node list:
    ContourFindingNode

@version: 0.0.1
"""

# Imports
from pixelprism.nodes import (
    ContourFinding,
    VectorsToString,
    SelectChannel,
    GrayScale,
    DrawPolygon
)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ContourFinding": ContourFinding,
    "SelectChannel": SelectChannel,
    "VectorsToString": VectorsToString,
    "GrayScale": GrayScale,
    "DrawPolygon": DrawPolygon
}

