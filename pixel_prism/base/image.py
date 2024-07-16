#
# This file contains the ImageData class, which is used to store image data in a format that is easy to work with.
#

# Imports
from enum import Enum
import numpy as np
import cv2


class ImageMode(Enum):
    GRAY = 'gray'
    RGB = 'rgb'
    RGBA = 'rgba'
    UNKNOWN = 'unknown'
# end ImageChannelType


class Image:
    """
    Class to store image data in a format that is easy to work with
    """

    def __init__(
            self,
            image_array
    ):
        """
        Initialize the image data with an image array

        Args:
            image_array (np.ndarray): Image data as a NumPy array
        """
        # If 2 dim, add channel dim
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=2)
        # end if

        # Set image data
        self.data = image_array
        self.height, self.width, self.channels = image_array.shape
        self.has_alpha = self.channels == 4
        self.mode = self.get_mode()
    # end __init__

    # region PROPERTIES

    @property
    def shape(self):
        """
        Get the number of channels in the image
        """
        return self.data.shape
    # end shape

    # endregion PROPERTIES

    # region PUBLIC

    def get_mode(self):
        """
        Get image node
        """
        if self.channels == 1:
            return ImageMode.GRAY
        elif self.channels == 3:
            return ImageMode.RGB
        elif self.channels == 4:
            return ImageMode.RGBA
        else:
            raise ValueError(f"Unknown image mode: {self.channels} channels.")
        # end if
    # end get_mode

    # get channel
    def get_channel(self, channel):
        """
        Get the specified channel of the image
        """
        return self.data[:, :, channel]
    # end get_channel

    def get_image(self):
        """
        Get the image data as a NumPy array
        """
        return self.data
    # end get_image

    def get_height(self):
        """
        Get the height of the image
        """
        return self.height
    # end get_height

    def get_width(self):
        """
        Get the width of the image
        """
        return self.width
    # end get_width

    def get_shape(self):
        """
        Get the shape of the image
        """
        return self.data.shape
    # end get_shape

    def get_size(self):
        """
        Get the size of the image
        """
        return self.data.size
    # end get_size

    def get_dtype(self):
        """
        Get the data type of the image
        """
        return self.data.dtype
    # end get_dtype

    def get_alpha(self):
        """
        Get the alpha channel of the image
        """
        if self.has_alpha:
            return self.data[:, :, 3]
        # end if
        return None
    # end get_alpha

    def get_rgb(self):
        """
        Get the RGB channels of the image
        """
        return self.data[:, :, :3]
    # end get_rgb

    # Add alpha channel
    def add_alpha_channel(self):
        """
        Add an alpha channel to the image
        """
        if not self.has_alpha:
            alpha_channel = np.full(
                (self.height, self.width, 1),
                255,
                dtype=self.data.dtype
            )
            self.data = np.concatenate([self.data, alpha_channel], axis=2)
            self.channels = self.data.shape[2]
            self.has_alpha = True
        # end if
    # end add_alpha_channel

    # Split
    def split(self):
        """
        Split the image into channels
        """
        # Alpha
        if self.mode == ImageMode.RGBA:
            r, g, b, a = cv2.split(self.data)
            return (
                Image.from_numpy(r),
                Image.from_numpy(g),
                Image.from_numpy(b),
                Image.from_numpy(a)
            )
        elif self.mode == ImageMode.RGB:
            r, g, b = cv2.split(self.data)
            return (
                Image.from_numpy(r),
                Image.from_numpy(g),
                Image.from_numpy(b)
            )
        elif self.mode == ImageMode.GRAY:
            return self
        # end if
    # end split

    # endregion PUBLIC

    # region OVERRIDE

    # Represent as string
    def __repr__(self):
        """
        Represent the image as a string
        """
        return f"Image({self.data.shape}, {self.data.dtype}, {self.mode})"
    # end __repr__

    # endregion OVERRIDE

    # region STATIC

    @staticmethod
    def from_numpy(
            image_array
    ):
        """
        Create an image from a NumPy array

        Args:
            image_array (np.ndarray): Image data as a NumPy array
        """
        return Image(image_array)
    # end from_numpy

    # Create transparent image
    @staticmethod
    def transparent(
            width: int,
            height: int
    ):
        """
        Create a transparent image

        Args:
            width (int): Width of the image
            height (int): Height of the image
        """
        return Image(np.zeros((height, width, 4), dtype=np.uint8))
    # end transparent

    # Create color image
    @staticmethod
    def color(
            width: int,
            height: int,
            color: tuple
    ):
        """
        Create a color image

        Args:
            width (int): Width of the image
            height (int): Height of the image
            color (tuple): Color of the image
        """
        return Image(np.full((height, width, 4), color, dtype=np.uint8))
    # end color

    # endregion Image

# end Image
