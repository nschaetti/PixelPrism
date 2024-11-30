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
from skimage.color import rgb2gray


# Select channel
class SelectChannel:
    """
    Select channel node
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

    # Select a channel
    def select_channel(self, image):
        """
        Select channel

        Args:
        - image: Image
        - channel: Channel index

        Returns:
        - Image: Image with selected channel
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


# Gray scale node
class GrayScale:
    """
    Gray scale node
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

    # Node fonction
    def gray_scale(self, image):
        """
        Gray scale

        Args:
            image: Image

        Returns:
            Image: Gray scale image
        """
        return (torch.from_numpy(rgb2gray(image.numpy())),)
    # end gray_scale

# end GrayScale

