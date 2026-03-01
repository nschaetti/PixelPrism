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
# Copyright (C) 2026 Pixel Prism
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

"""Layer compositing engine for :class:`~pixelprism.base.imagecanvas.ImageCanvas`.

This module provides :class:`RenderEngine`, a stateless utility class that
merges active layers into a final RGBA frame.
"""

# Imports
from __future__ import annotations

import numpy as np

from pixelprism.base import Image, ImageMode
from pixelprism.base.imagecanvas import ImageCanvas


class RenderEngine:
    """Compose all active layers from an image canvas.

    Notes
    -----
    Current implementation supports only the ``normal`` blend mode and alpha
    compositing over an opaque black background.
    """

    @staticmethod
    def render(
            image_canvas: ImageCanvas,
            output_buffer: Image | None = None,
            base_image: Image | None = None
    ) -> Image:
        """Merge all active layers into a single output image.

        Parameters
        ----------
        image_canvas : ImageCanvas
            Canvas containing layers to composite.

        output_buffer : Image | None, default=None
            Optional reusable output buffer to avoid repeated allocations.
        base_image : Image | None, default=None
            Optional base image copied before compositing active layers.

        Returns
        -------
        Image
            Final composited image.
        """
        active_layers = [layer for layer in image_canvas.layers if layer.active]

        # Resolve output buffer
        if output_buffer is None:
            final_image = Image.fill(
                image_canvas.width,
                image_canvas.height,
                (0, 0, 0, 255),
            )
        else:
            if (
                output_buffer.width != image_canvas.width
                or output_buffer.height != image_canvas.height
            ):
                raise ValueError("output_buffer size must match image_canvas size.")
            # end if
            final_image = output_buffer
        # end if

        final_data = final_image.data

        if base_image is not None:
            if (
                base_image.width != image_canvas.width
                or base_image.height != image_canvas.height
            ):
                raise ValueError("base_image size must match image_canvas size.")
            # end if
            final_data[:, :, :] = base_image.data[:, :, :]
        # end if

        # Empty scene
        if not active_layers:
            if base_image is None:
                final_data[:, :, 0:3] = 0
                final_data[:, :, 3] = 255
            # end if
            return final_image
        # end if

        # Fast path: one normal layer can be returned directly
        if (
            base_image is None
            and len(active_layers) == 1
            and active_layers[0].blend_mode == "normal"
        ):
            return active_layers[0].image
        # end if

        # Reset reusable buffer when no base image is provided
        if base_image is None:
            final_data[:, :, 0:3] = 0
            final_data[:, :, 3] = 255
        # end if

        for layer in active_layers:
            if layer.blend_mode == "normal":
                layer_image = layer.image
                layer_data = layer_image.data

                # Opaque layer: direct copy
                if layer_image.mode != ImageMode.RGBA:
                    final_data[:, :, 0:3] = layer_data[:, :, 0:3]
                    continue
                # end if

                alpha_channel = layer_data[:, :, 3]
                if np.all(alpha_channel == 255):
                    final_data[:, :, 0:3] = layer_data[:, :, 0:3]
                    continue
                # end if

                alpha = alpha_channel / 255.0
                for channel in range(0, 3):
                    final_data[:, :, channel] = (
                        alpha * layer_data[:, :, channel]
                        + (1 - alpha) * final_data[:, :, channel]
                    )
                # end for
            # end if
        # end for

        return final_image
    # end def render

# end class RenderEngine
