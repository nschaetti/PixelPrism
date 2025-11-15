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

# Imports
import numpy as np

# Locals
from .image import Image
from .layer import Layer


class ImageCanvas:
    """
    Class to represent an image with multiple layers
    """

    def __init__(
            self,
            width: int = 0,
            height: int = 0
    ):
        """
        Initialize the image object.

        Args:
            width (int): Width of the image
            height (int): Height of the image
        """
        # Empty layers list
        self.layers = []
        self.height = height
        self.width = width
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
        # Check image size
        assert len(self.layers) == 0 or (image.height == self.height and image.width == self.width), \
            f"Image size {image.shape} does not match canvas size ({self.width}, {self.height})."

        # Add the layer
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
        Merge all layers in the image into a single image.
        """
        if not self.layers:
            return None
        # end if

        # Initialize the final image as a black image
        final_image = np.zeros_like(self.layers[0].image)

        # Iterate over all layers and blend them together
        for layer_i, layer in enumerate(self.layers):
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

    # Create a color layer
    def create_color_layer(
            self,
            name,
            color,
            blend_mode='normal',
            active=True
    ):
        """
        Create a color layer with the specified color

        Args:
            name (str): Name of the layer
            color (tuple): Color of the layer
            blend_mode (str): Blend mode for the layer
            active (bool): Whether the layer is active
        """
        layer_image = Image.color(self.width, self.height, color)
        self.add_layer(name, layer_image, blend_mode, active)
    # end create_color_layer

    # Create a transparent layer
    def create_transparent_layer(
            self,
            name,
            blend_mode='normal',
            active=True
    ):
        """
        Create a transparent layer.

        Args:
            name (str): Name of the layer
            blend_mode (str): Blend mode for the layer
            active (bool): Whether the layer is active
        """
        self.add_layer(name, Image.transparent(self.width, self.height), blend_mode, active)
    # end create_transparent_layer

    # Update the image of a layer
    def update_layer_image(
            self,
            name,
            image: Image
    ):
        """
        Update the image of a layer

        Args:
            name (str): Name of the layer
            image (Image): New image math
        """
        layer = self.get_layer(name)
        assert layer is not None, f"Layer {name} does not exist."
        layer.image = image
    # end update_layer_image

    # Set layer image
    def set_layer_image(
            self,
            name,
            image: Image
    ):
        """
        Set the image of a layer

        Args:
            name (str): Name of the layer
            image (Image): New image math
        """
        self.update_layer_image(name, image)
    # end set_layer_image

    # Apply an effect to a layer
    def has_alpha(
            self,
            name
    ):
        """
        Check if a layer has an alpha channel

        Args:
            name (str): Name of the layer
        """
        layer = self.get_layer(name)
        assert layer is not None, f"Layer {name} does not exist."
        return layer.image_data.has_alpha
    # end has_alpha

    # Represent the image canvas as a string, with each layer
    def __repr__(self):
        """
        Represent the image canvas as a string
        """
        return f"ImageCanvas({self.width}, {self.height}, {self.layers})"
    # end __repr__

    # region STATICS

    # Create empty image canvas
    @staticmethod
    def empty(
            width: int,
            height: int
    ):
        """
        Create an empty image canvas

        Args:
            width (int): Width of the image canvas
            height (int): Height of the image canvas
        """
        return ImageCanvas(width, height)
    # end empty

    # Create an image from a NumPy array
    @staticmethod
    def from_numpy(
            image_array,
            add_alpha: bool = True
    ):
        """
        Create an image from a NumPy array

        Args:
            image_array (np.ndarray): Image math as a NumPy array
            add_alpha (bool): Whether to add an alpha channel to the image
        """
        # Check dimension is 3
        assert image_array.ndim == 3, f"Image array must have 3 dimensions, not {image_array.ndim}."

        # Check the number of channels (1, 3, or 4)
        assert image_array.shape[2] in [1, 3, 4], f"Image array must have 1, 3, or 4 channels, not {image_array.shape[2]}."

        # Add an alpha channel if necessary
        if image_array.shape[2] == 3 and add_alpha:
            image_array = np.concatenate(
                [
                    image_array,
                    np.ones((image_array.shape[0], image_array.shape[1], 1), dtype=np.uint8) * 255
                ],
                axis=2
            )
        # end if

        # Create the image canvas
        image_canvas = ImageCanvas(
            width=image_array.shape[1],
            height=image_array.shape[0]
        )

        # Create image
        image = Image(image_array)

        # Add the image as a layer
        image_canvas.add_layer('base', image)

        return image_canvas
    # end from_numpy

    # endregion STATICS

# end Image

