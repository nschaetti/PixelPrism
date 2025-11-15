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
Pixel Prism Nodes - Core Nodes
=============================

This module provides core nodes for image processing in the Pixel Prism framework.
These nodes implement fundamental image processing operations that can be used
as building blocks for more complex workflows.
"""

# Imports
import torch
import skimage as sk


class ContourFinding:
    """
    Node to find contours in an image.

    This node uses scikit-image's find_contours function to detect contours in an image.
    Contours are lines of equal intensity that can be used for shape detection,
    object recognition, and other image analysis tasks.

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
                "image": ("IMAGE",),
            },
            "optional": {
                "channel": ("INT", {"default": 0}),
                "level": ("FLOAT",),
                "fully_connected": ("STRING", {"default": "low"}),
                "positive_orientation": ("STRING", {"default": "low"}),
                "mask": ("MASK",),
            }
        }
    # end INPUT_TYPES

    RETURN_TYPES = ("CONTOURS",)
    FUNCTION = "find_contours"
    CATEGORY = "PixelPrism"
    # OUTPUT_NODE = True

    def find_contours(self, image, channel=0, level=None, fully_connected='low', positive_orientation='low', mask=None):
        """
        Find contours in an image.

        This method uses scikit-image's find_contours function to detect contours in the input image.
        It can process both PyTorch tensors and numpy arrays.

        Args:
            image (torch.Tensor or numpy.ndarray): The input image to find contours in.
                Can be a 2D, 3D, or 4D tensor/array.
            channel (int, optional): The channel index to use if the image has multiple channels.
                Defaults to 0.
            level (float, optional): The value along which to find contours in the array.
                By default, the level is set to 0.5 * (max(image) + min(image)).
            fully_connected (str, optional): Either 'low' or 'high'. Defines the connectivity
                of the neighborhood. Defaults to 'low'.
            positive_orientation (str, optional): Either 'low' or 'high'. Defines whether
                the contour is walked clockwise or counterclockwise. Defaults to 'low'.
            mask (numpy.ndarray, optional): A boolean mask, True where we want to find contours.
                Defaults to None.

        Returns:
            tuple: A tuple containing a list of contours, where each contour is a PyTorch tensor
                of shape (n, 2) containing the (row, column) coordinates of the contour points.
        """
        # print(f"contour_finding: image.shape={image.shape}")
        # Transform to numpy array
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0, :, :, channel].numpy()
            elif image.ndim == 3:
                image = image[:, :, channel].numpy()
            elif image.ndim == 2:
                image = image.numpy()
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            # end if
        # end if

        # Find contours at a constant value of 0.8
        contours = sk.measure.find_contours(
            image=image,
            level=level,
            fully_connected=fully_connected,
            positive_orientation=positive_orientation,
            mask=mask
        )

        # Numpy array to torch tensor
        contours = [torch.from_numpy(contour) for contour in contours]

        return (contours,)
    # end find_contours

# end ContourFinding
