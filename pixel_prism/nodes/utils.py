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


# Select channel
class SelectChannel:
    """
    Select channel
    """

    # Define the input types
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": ("INT",),
            }
        }
    # end INPUT_TYPES

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_channel"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def select_channel(self, image, channel):
        """
        Select channel

        Args:
        - image: Image
        - channel: Channel index
        """
        if image.ndim == 4:
            image = image[:, :, :, channel]
        elif image.ndim == 3:
            image = image[:, :, channel]
        else:
            raise ValueError("Invalid image shape")
        # end if
        print(f"SelectChannel: {image.shape}")
        return image
    # end select_channel

# end SelectChannel

