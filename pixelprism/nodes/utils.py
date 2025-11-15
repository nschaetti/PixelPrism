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
Pixel Prism Nodes - Utility Nodes
===============================

This module provides utility nodes for common image processing operations in the Pixel Prism framework.
These nodes implement basic operations like channel selection and grayscale conversion
that are frequently used in image processing workflows.
"""

# Imports
import torch
from skimage.color import rgb2gray


class SelectChannel:
    """
    Node for selecting and separating color channels from an image.

    This node takes an input image and separates it into its individual color channels.
    Each channel is then duplicated across all three output channels to create
    a grayscale representation of that channel.

    Attributes:
        INPUT_TYPES (dict): Defines the input types for the node
        RETURN_TYPES (tuple): Defines the return types for the node (three IMAGE outputs)
        FUNCTION (str): The name of the function to call
        CATEGORY (str): The category of the node
    """

    # Define the input types
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }
    # end INPUT_TYPES

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE")
    FUNCTION = "select_channel"
    CATEGORY = "PixelPrism"
    # OUTPUT_NODE = True

    def select_channel(self, image):
        """
        Separate an image into its individual color channels.

        This method takes an input image and separates it into its individual color channels.
        Each channel is then duplicated across all three output channels to create
        a grayscale representation of that channel.

        Args:
            image (torch.Tensor): The input image tensor. Can be 3D (H,W,C) or 4D (B,H,W,C).

        Returns:
            tuple: A tuple containing three image tensors, one for each color channel
                (R, G, B). Each tensor has the same shape as the input but with the
                selected channel duplicated across all three output channels.

        Raises:
            ValueError: If the input image has an invalid shape.
        """
        ims = list()
        for channel_i in range(image.shape[-1]):
            if image.ndim == 4:
                im = image[:, :, :, channel_i:channel_i+1]
            elif image.ndim == 3:
                im = image[:, :, channel_i:channel_i+1]
            else:
                raise ValueError("Invalid image shape")
            # end if
            im = torch.cat([im, im, im], dim=-1)
            ims.append(im)
        # end for

        return ims
    # end select_channel

# end SelectChannel


class GrayScale:
    """
    Node for converting an image to grayscale.

    This node takes an RGB image and converts it to grayscale using scikit-image's
    rgb2gray function, which applies a weighted sum of the RGB channels
    (0.2125 R + 0.7154 G + 0.0721 B) to create a perceptually accurate grayscale image.

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
                "image": ("IMAGE",)
            }
        }
    # end INPUT_TYPES

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gray_scale"
    CATEGORY = "PixelPrism"
    # OUTPUT_NODE = True

    def gray_scale(self, image):
        """
        Convert an RGB image to grayscale.

        This method converts an RGB image to grayscale using scikit-image's rgb2gray function,
        which applies a weighted sum of the RGB channels (0.2125 R + 0.7154 G + 0.0721 B)
        to create a perceptually accurate grayscale image.

        Args:
            image (torch.Tensor): The input RGB image tensor.

        Returns:
            tuple: A tuple containing a single grayscale image tensor.
        """
        return (torch.from_numpy(rgb2gray(image.numpy())),)
    # end gray_scale

# end GrayScale
