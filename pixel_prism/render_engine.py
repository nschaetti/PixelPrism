
import numpy as np

from pixel_prism.base.imagecanvas import ImageCanvas
from pixel_prism.base import ImageMode, Image


class RenderEngine:

    @staticmethod
    def render(image_canvas: ImageCanvas):
        """
        Merge all layers in the image canvas into a single image.
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
