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
import numpy as np
import torch
from skimage.draw import polygon, line
import matplotlib.pyplot as plt


# Show list of vectors
class VectorsToString:
    """
    Show list of vectors
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
        Show list of vectors
        """
        # Output
        output_str = ""
        for vector in vectors:
            output_str += f"{vector}\n"
        # end for

        return (output_str,)
    # end show_vectors

# end VectorsToString


# Plot
class DrawPolygon:

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
        Plot vectors

        Args:
            contours: Contours
            height: Height
            width: Width
            color: Color
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

