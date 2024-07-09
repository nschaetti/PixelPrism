
# Imports
import numpy as np

# Locals
from .layer import Layer


class Image:
    """
    Class to represent an image with multiple layers
    """

    def __init__(self):
        """
        Initialize the image object
        """
        self.layers = []
    # end __init__

    def add_layer(
            self,
            name,
            image,
            blend_mode='normal',
            active=True
    ):
        """
        Add a layer to the image
        """
        self.layers.append(Layer(name, image, blend_mode, active))
    # end add_layer

    def remove_layer(
            self,
            name
    ):
        """
        Remove a layer from the image

        Args:
            name (str): Name of the layer to remove
        """
        self.layers = [layer for layer in self.layers if layer.name != name]
    # end remove_layer

    def merge_layers(self):
        """
        Merge all layers in the image into a single image
        """
        if not self.layers:
            return None
        # end if

        # Initialize the final image as a black image
        final_image = np.zeros_like(self.layers[0].image)

        # Iterate over all layers and blend them together
        for layer in self.layers:
            if layer.active:
                if layer.blend_mode == 'normal':
                    alpha = layer.image[:, :, 3] / 255.0
                    for c in range(0, 3):
                        final_image[:, :, c] = alpha * layer.image[:, :, c] + (1 - alpha) * final_image[:, :, c]
                    # end for
                # end if
            # end if
        # end for

        return final_image
    # end merge_layers

    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
            # end for
        # end for

        return None
    # end get_layer

    def set_layer_active(
            self,
            name,
            active
    ):
        """
        Set the active status of a layer

        Args:
            name (str): Name of the layer to set the active status for
            active (bool): Whether the layer is active
        """
        layer = self.get_layer(name)
        if layer:
            layer.active = active
        # end if
    # end set_layer_active

# end Image

