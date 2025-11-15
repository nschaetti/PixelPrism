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

#
# functions for applying chromatic effects to images
#

# Imports
from typing import List, Tuple
import cv2
import numpy as np

from pixelprism.base import Image, ImageMode


# Spatial shift effect that shifts the image in the x and y directions for each channel
def chromatic_spatial_shift_effect(
        image,
        shift_r=(5, 0),
        shift_g=(-5, 0),
        shift_b=(0, 5)
) -> Image:
    """
    Apply the spatial shift effect to the image

    Args:
        image (np.ndarray): Image to apply the effect to
        shift_r (tuple): Shift for the red channel
        shift_g (tuple): Shift for the green channel
        shift_b (tuple): Shift for the blue channel

    Returns:
        np.ndarray: Image with the spatial shift effect applied
    """
    channels = cv2.split(image.data)
    b, g, r = channels[:3]
    rows, cols = b.shape

    M_r = np.float32([[1, 0, shift_r[0]], [0, 1, shift_r[1]]])
    M_g = np.float32([[1, 0, shift_g[0]], [0, 1, shift_g[1]]])
    M_b = np.float32([[1, 0, shift_b[0]], [0, 1, shift_b[1]]])

    r_shifted = cv2.warpAffine(r, M_r, (cols, rows))
    g_shifted = cv2.warpAffine(g, M_g, (cols, rows))
    b_shifted = cv2.warpAffine(b, M_b, (cols, rows))

    # New image
    new_image = Image.from_numpy(cv2.merge((b_shifted, g_shifted, r_shifted)))

    # Add alpha if needed
    if image.has_alpha:
        new_image.add_alpha_channel()
    # end if

    # Return an image
    return new_image
# end chromatic_spatial_shift_effect


# Temporal shift effect that shifts the image in the time dimension for each channel
def chromatic_temporal_persistence_effect(
        image: Image,
        persistence_r,
        persistence_g,
        persistence_b,
        prev_frames: List[Image],
        weight_decay='linear'
) -> Image:
    """
    Apply the temporal shift effect to the image

    Args:
        image (Image): Image to apply the effect to
        persistence_r (int): Persistence for the red channel
        persistence_g (int): Persistence for the green channel
        persistence_b (int): Persistence for the blue channel
        prev_frames (List[Image]): List of previous frames
        weight_decay (str): Weight decay function ('linear' or 'exponential')

    Returns:
        Image: Image with the temporal shift effect applied
    """
    # Only with RGB or RBGA images
    assert image.mode in [ImageMode.RGB, ImageMode.RGBA]

    # Split the image into channels
    if image.mode == ImageMode.RGBA:
        b, g, r, _ = image.split()
    else:
        b, g, r = image.split()
    # end if

    # Method compute weights for a channel
    def compute_weights(persistence):
        if weight_decay == 'constant':
            return [1.0 / persistence for _ in range(persistence)]
        if weight_decay == 'linear':
            return list(reversed([1.0 / i if i > 0 else 0 for i in range(2, persistence + 2)]))
        elif weight_decay == 'exponential':
            return list(reversed([1.0 / (2 ** i) if i > 0 else 0 for i in range(2, persistence + 2)]))
        # end if
    # end compute_weights

    # Calculate the weights
    weights_r = compute_weights(persistence_r)
    weights_g = compute_weights(persistence_g)
    weights_b = compute_weights(persistence_b)

    # Calcul weighted average of previous frames
    blended_r = np.zeros_like(r.data, dtype=np.float32)
    blended_g = np.zeros_like(g.data, dtype=np.float32)
    blended_b = np.zeros_like(b.data, dtype=np.float32)

    # We start with image
    blended_r += r.data
    blended_g += g.data
    blended_b += b.data

    # How many previous frames
    n_prev_frames = len(prev_frames)
    n_prev_frames_r = min(n_prev_frames, persistence_r)
    n_prev_frames_g = min(n_prev_frames, persistence_g)
    n_prev_frames_b = min(n_prev_frames, persistence_b)

    # Blend the previous frames
    for frame_i in range(n_prev_frames_r, 0, -1):
        blended_r += prev_frames[-frame_i].get_channel(0) * weights_r[-frame_i]
    # end for
    blended_r = np.clip(blended_r, 0, 255)

    # total_weights_g = 0
    for frame_i in range(n_prev_frames_g, 0, -1):
        blended_g += prev_frames[-frame_i].get_channel(1) * weights_g[-frame_i]
    # end for
    blended_g = np.clip(blended_g, 0, 255)

    # total_weights_b = 0
    for frame_i in range(n_prev_frames_b, 0, -1):
        blended_b += prev_frames[-frame_i].get_channel(2) * weights_b[-frame_i]
    # end for
    blended_b = np.clip(blended_b, 0, 255)

    # Merge channels and return the blended image
    blended = cv2.merge((
        blended_b.astype(np.uint8),
        blended_g.astype(np.uint8),
        blended_r.astype(np.uint8)
    ))

    # To image
    new_image = Image.from_numpy(blended)

    return new_image
# end chromatic_temporal_persistence_effect

