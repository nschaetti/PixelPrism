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

# Imports
import numpy as np

from pixelprism.base import Image


def normal(
        image1: Image,
        image2: Image
) -> Image:
    """
    Blend two images using the normal mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # If the top image has an alpha channel, blend the images
    if image2.has_alpha:
        alpha = image2.data[:, :, 3] / 255.0
        blended_data = np.empty_like(image1.data)

        # Blend the RGB channels using the alpha channel
        for c in range(3):
            blended_data[:, :, c] = alpha * image2.data[:, :, c] + (1 - alpha) * image1.data[:, :, c]
        # end for

        blended_data[:, :, 3] = 255  # Ensure the alpha channel is fully opaque
    else:
        blended_data = image2.data.copy()

    # Return a new Image object
    return Image(blended_data)
# end normal


def multiply(
        image1: Image,
        image2: Image
) -> Image:
    """
    Blend two images using the multiply mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # Convert images to float if they are not already
    data1 = image1.data.astype(np.float32) / 255.0
    data2 = image2.data.astype(np.float32) / 255.0

    # Initialize the blended math
    blended_data = np.empty_like(data1)

    # If the top image has an alpha channel, blend the images
    if image2.has_alpha:
        # Extract the alpha channel
        alpha = data2[:, :, 3]

        # Blend the RGB channels using the alpha channel
        for c in range(3):
            blended_data[:, :, c] = alpha * (data1[:, :, c] * data2[:, :, c]) + (1 - alpha) * data1[:, :, c]
        # end for

        # Ensure the alpha channel is fully opaque
        blended_data[:, :, 3] = data1[:, :, 3]
    else:
        blended_data = data1 * data2
    # end if

    # Convert back to original dtype
    blended_data = (blended_data * 255).astype(image1.data.dtype)

    # Return a new Image object
    return Image(blended_data)
# end multiply


def overlay(
        image1: Image,
        image2: Image
):
    """
    Blend two images using the overlay mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # Convert images to float if they are not already
    data1 = image1.data.astype(np.float32) / 255.0
    data2 = image2.data.astype(np.float32) / 255.0

    # Perform the overlay blend
    blended_data = 1 - (1 - data1) * (1 - data2)

    # Convert back to original dtype
    blended_data = (blended_data * 255).astype(image1.data.dtype)

    # Return a new Image object
    return Image(blended_data)
# end overlay


def hard_light(
        image1: Image,
        image2: Image
):
    """
    Blend two images using the hard light mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # Convert images to float if they are not already
    data1 = image1.data.astype(np.float32) / 255.0
    data2 = image2.data.astype(np.float32) / 255.0

    # Perform the hard light blend
    blended_data = np.where(data2 <= 0.5, 2 * data1 * data2, 1 - 2 * (1 - data1) * (1 - data2))

    # Convert back to original dtype
    blended_data = (blended_data * 255).astype(image1.data.dtype)

    # Return a new Image object
    return Image(blended_data)
# end hard_light


def divide(
        image1: Image,
        image2: Image
):
    """
    Blend two images using the division mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # Convert images to float if they are not already
    data1 = image1.data.astype(np.float32) / 255.0
    data2 = image2.data.astype(np.float32) / 255.0

    # Prevent division by zero by adding a small epsilon
    epsilon = 1e-5
    data2 = np.clip(data2, epsilon, 1.0)

    # Perform the division blend
    blended_data = data1 / data2

    # Clip the values to the range [0, 1]
    blended_data = np.clip(blended_data, 0.0, 1.0)

    # Convert back to original dtype
    blended_data = (blended_data * 255).astype(image1.data.dtype)

    # Return a new Image object
    return Image(blended_data)
# end divide


def add(
        image1: Image,
        image2: Image
):
    """
    Blend two images using the addition mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # Convert images to float if they are not already
    data1 = image1.data.astype(np.float32) / 255.0
    data2 = image2.data.astype(np.float32) / 255.0

    # Perform the addition blend
    blended_data = data1 + data2

    # Clip the values to the range [0, 1]
    blended_data = np.clip(blended_data, 0.0, 1.0)

    # Convert back to original dtype
    blended_data = (blended_data * 255).astype(image1.data.dtype)

    # Return a new Image object
    return Image(blended_data)
# end add


def subtract(
        image1: Image,
        image2: Image
):
    """
    Blend two images using the subtraction mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # Convert images to float if they are not already
    data1 = image1.data.astype(np.float32) / 255.0
    data2 = image2.data.astype(np.float32) / 255.0

    # Perform the subtraction blend
    blended_data = data1 - data2

    # Clip the values to the range [0, 1]
    blended_data = np.clip(blended_data, 0.0, 1.0)

    # Convert back to original dtype
    blended_data = (blended_data * 255).astype(image1.data.dtype)

    # Return a new Image object
    return Image(blended_data)
# end subtract


def difference(
        image1: Image,
        image2: Image
):
    """
    Blend two images using the difference mode.

    Args:
        image1 (Image): The first image (bottom layer).
        image2 (Image): The second image (top layer).

    Returns:
        Image: The blended image.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.data.dtype == image2.data.dtype, "Images must have the same dtype"

    # Convert images to float if they are not already
    data1 = image1.data.astype(np.float32) / 255.0
    data2 = image2.data.astype(np.float32) / 255.0

    # Perform the difference blend
    blended_data = np.abs(data1 - data2)

    # Convert back to original dtype
    blended_data = (blended_data * 255).astype(image1.data.dtype)

    # Return a new Image object
    return Image(blended_data)
# end difference
