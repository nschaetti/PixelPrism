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


# Show list of vectors
class VectorsToStringNode:
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
    CATEGORY = "image"
    OUTPUT_NODE = True

    def show_vectors(self, vectors):
        """
        Show list of vectors
        """
        # Output
        print(f"Vectors: {vectors}")
        output_str = ""
        for vector in vectors:
            output_str += f"{vector}\n"
        # end for
        print(f"Output: {output_str}")
        return output_str
    # end show_vectors

# end VectorsToStringNode

