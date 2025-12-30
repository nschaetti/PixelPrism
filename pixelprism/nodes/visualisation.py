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
Pixel Prism Nodes - Visualization Nodes
=====================================

This module provides visualization nodes for displaying and rendering math_old in the Pixel Prism framework.
These nodes implement operations for converting math_old to visual representations,
such as drawing contours, polygons, and displaying vector math_old.
"""

# Imports
import numpy as np
import torch
from skimage.draw import polygon, line
import matplotlib.pyplot as plt


class VectorsToString:
    """
    Node for converting a list of vectors to a string representation.

    This node takes a list of vector objects and converts them to a string representation,
    with each vector on a new line. This is useful for debugging and visualization purposes.

    Attributes:
        INPUT_TYPES (dict): Defines the input types for the node
        RETURN_TYPES (tuple): Defines the return types for the node
        FUNCTION (str): The name of the function to call
        CATEGORY (str): The category of the node
    """

    # Define the input types
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vectors": ("VECTORS",),
            }
        }
    # end INPUT_TYPES

    RETURN_TYPES = ("STRING",)
    FUNCTION = "show_vectors"
    CATEGORY = "PixelPrism"
    # OUTPUT_NODE = True

    def show_vectors(self, vectors):
        """
        Convert a list of vectors to a string representation.

        This method takes a list of vector objects and converts them to a string representation,
        with each vector on a new line.

        Args:
            vectors (list): A list of vector objects to convert to a string.

        Returns:
            tuple: A tuple containing a single string with the string representation of the vectors.
        """
        # Output
        output_str = ""
        for vector in vectors:
            output_str += f"{vector}\n"
        # end for

        return (output_str,)
    # end show_vectors

# end VectorsToString


class DrawPolygon:
    """
    Node for drawing polygons or contours on an image.

    This node takes a list of contours (each contour being a list of points) and draws them
    on a blank image with the specified dimensions. The contours are drawn as connected lines
    with the specified color.

    Attributes:
        INPUT_TYPES (dict): Defines the input types for the node
        RETURN_TYPES (tuple): Defines the return types for the node
        FUNCTION (str): The name of the function to call
        CATEGORY (str): The category of the node
    """

    # Define the input types
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "contours": ("CONTOURS",),
                "height": ("INT",),
                "width": ("INT",),
            },
            "optional": {
                "color": ("STRING", {"default": "#000000"}),
            }
        }
    # end INPUT_TYPES

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "PixelPrism"
    # OUTPUT_NODE = True

    def draw(self, contours, height, width, color="#000000"):
        """
        Draw contours on a blank image.

        This method takes a list of contours and draws them on a blank image with the specified
        dimensions. Each contour is drawn as a series of connected line segments with the specified color.

        Args:
            contours (list): A list of contours, where each contour is a tensor of shape (n, 2)
                containing the (y, x) coordinates of the contour points.
            height (int): The height of the output image.
            width (int): The width of the output image.
            color (str, optional): The color to draw the contours with, specified as a hex string.
                Defaults to "#000000" (black).

        Returns:
            tuple: A tuple containing a single image tensor of shape (1, height, width, 3)
                with the drawn contours.
        """
        # Transform color into RGB
        color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # Create an image
        contour_image = np.zeros((height, width, 3), dtype=np.uint8)
        #
        # # Plot
        # for contour in contours:
        #     rr, cc = polygon(
        #         contour[:, 0].numpy().astype(np.uint32),
        #         contour[:, 1].numpy().astype(np.uint32),
        #         contour_image.shape
        #     )
        #     contour_image[rr, cc] = color
        # # end for
        #
        # # Image
        # image = torch.from_numpy(contour_image)
        #
        # # Add batch dim
        # image = image.unsqueeze(0)

        # Plot
        for contour in contours:
            for i in range(len(contour) - 1):
                y1, x1 = contour[i]
                y2, x2 = contour[i + 1]
                rr, cc = line(
                    int(y1),
                    int(x1),
                    int(y2),
                    int(x2)
                )
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                contour_image[rr, cc] = color
            # end for
        # end for

        # Convert image to tensor
        image = torch.from_numpy(contour_image)

        # Add batch dim
        image = image.unsqueeze(0)

        return (image,)
    # end draw

# end DrawPolygon
