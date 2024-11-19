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
import torch
import skimage as sk


# Node to find contours in an image
class ContourFindingNode:
    """
    Node to find contours in an image
    """

    # Define the input types
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "level": ("FLOAT",),
                "fully_connected": ("STRING", {"default": "low"}),
                "positive_orientation": ("STRING", {"default": "low"}),
                "mask": ("MASK",),
            }
        }
    # end INPUT_TYPES

    RETURN_TYPES = ("VECTORS",)
    FUNCTION = "find_contours"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def find_contours(self, image, level=None, fully_connected='low', positive_orientation='low', mask=None):
        """
        Generate a caption for an image using the Gemini API

        Args:
            image:
            level:
            fully_connected:
            positive_orientation:
            mask:

        Returns:

        """
        print(f"fully_connected: {fully_connected}")
        print(f"positive_orientation: {positive_orientation}")
        # Transform to numpy array
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image[0].numpy()
            elif image.ndim == 2:
                image = image.numpy()
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            # end if
        # end if
        print(f"image: {image.shape}")
        # Find contours at a constant value of 0.8
        contours = sk.measure.find_contours(
            image=image[:, :],
            level=level,
            fully_connected=fully_connected,
            positive_orientation=positive_orientation,
            mask=mask
        )

        print(contours)
        return (contours,)
    # end find_contours

# end ContourFindingNode


# Map the node class to the node name
NODE_CLASS_MAPPINGS = {"ContourFindingNode": ContourFindingNode}
