
"""
Render Engine Module
===================

This module provides the RenderEngine class for rendering image canvases with multiple layers.
"""

import numpy as np

from pixel_prism.base.imagecanvas import ImageCanvas
from pixel_prism.base import ImageMode, Image


class RenderEngine:
    """
    Engine for rendering image canvases with multiple layers.

    The RenderEngine provides functionality to merge multiple image layers into a single
    composite image, handling transparency and blending modes.
    """

    @staticmethod
    def render(image_canvas: ImageCanvas) -> 'Image':
        """
        Merge all layers in the image canvas into a single image.

        This method takes an ImageCanvas containing multiple layers and blends them together
        according to their blend modes and opacity settings to create a final composite image.

        Args:
            image_canvas (ImageCanvas): The image canvas containing layers to render

        Returns:
            Image: The final rendered image, or None if the canvas has no layers
        """
        if not image_canvas.layers:
            return None
        # end if

        # Initialize the final image as a black image
        final_image = Image.fill(image_canvas.width, image_canvas.height, (0, 0, 0, 255))

        # Iterate over all layers and blend them together
        for layer in image_canvas.layers:
            if layer.active:
                if layer.blend_mode == 'normal':
                    # Get the alpha channel if it exists
                    if layer.image.mode == ImageMode.RGBA:
                        alpha = layer.image.data[:, :, 3] / 255.0
                    else:
                        alpha = 1.0
                    # end if

                    # Blend the layers
                    for c in range(0, 3):
                        final_image.data[:, :, c] = alpha * layer.image.data[:, :, c] + (1 - alpha) * final_image.data[:, :, c]
                    # end for
                # end if
            # end if
        # end for

        return final_image
    # end render

# end RenderEngine
