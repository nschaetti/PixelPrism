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

#
# This file contains the ImageData class, which is used to store image math_old in a format that is easy to work with.
#

# Imports
from enum import Enum
import numpy as np
import cairo
import requests
import cv2


class ImageMode(Enum):
    GRAY = 'gray'
    RGB = 'rgb'
    RGBA = 'rgba'
    UNKNOWN = 'unknown'
# end ImageChannelType


class Image:
    """
    Class to store image math_old in a format that is easy to work with
    """

    def __init__(
            self,
            image_array
    ):
        """
        Initialize the image math_old with an image array

        Args:
            image_array (np.ndarray): Image math_old as a NumPy array
        """
        # If 2 dim, add channel dim
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=2)
        # end if

        # Set image math_old
        self._data = image_array
        self._height, self._width, self._channels = image_array.shape
        self._has_alpha = self._channels == 4
        self._mode = self.get_mode()
    # end __init__

    # region PROPERTIES

    @property
    def data(self):
        """
        Get the image math_old
        """
        return self._data
    # end math_old

    @property
    def height(self):
        """
        Get the height of the image
        """
        return self._height
    # end height

    @property
    def width(self):
        """
        Get the width of the image
        """
        return self._width
    # end width

    @property
    def channels(self):
        """
        Get the number of channels in the image
        """
        return self._channels
    # end channels

    @property
    def has_alpha(self):
        """
        Get whether the image has an alpha channel
        """
        return self._has_alpha
    # end has_alpha

    @property
    def mode(self):
        """
        Get the mode of the image
        """
        return self._mode
    # end mode

    @property
    def shape(self):
        """
        Get the number of channels in the image
        """
        return self._data.shape
    # end shape

    # endregion PROPERTIES

    # region PUBLIC

    def get_mode(self):
        """
        Get image node
        """
        if self._channels == 1:
            return ImageMode.GRAY
        elif self._channels == 3:
            return ImageMode.RGB
        elif self._channels == 4:
            return ImageMode.RGBA
        else:
            raise ValueError(f"Unknown image mode: {self._channels} channels.")
        # end if
    # end get_mode

    # get channel
    def get_channel(self, channel):
        """
        Get the specified channel of the image
        """
        return self._data[:, :, channel:channel + 1]
    # end get_channel

    def get_image(self):
        """
        Get the image math_old as a NumPy array
        """
        return self._data
    # end get_image

    def get_height(self):
        """
        Get the height of the image
        """
        return self._height
    # end get_height

    def get_width(self):
        """
        Get the width of the image
        """
        return self._width
    # end get_width

    def get_shape(self):
        """
        Get the shape of the image
        """
        return self._data.shape
    # end get_shape

    def get_size(self):
        """
        Get the size of the image
        """
        return self._data.size
    # end get_size

    def get_dtype(self):
        """
        Get the math_old type of the image
        """
        return self._data.dtype
    # end get_dtype

    def get_alpha(self):
        """
        Get the alpha channel of the image
        """
        if self.has_alpha:
            return self._data[:, :, 3]
        # end if
        return None
    # end get_alpha

    def get_rgb(self):
        """
        Get the RGB channels of the image
        """
        return self._data[:, :, :3]
    # end get_rgb

    # Add alpha channel
    def add_alpha_channel(self):
        """
        Add an alpha channel to the image
        """
        if not self._has_alpha:
            alpha_channel = np.full(
                (self._height, self._width, 1),
                255,
                dtype=self._data.dtype
            )
            self._data = np.concatenate([self._data, alpha_channel], axis=2)
            self._channels = self._data.shape[2]
            self._has_alpha = True
        # end if
    # end add_alpha_channel

    # Split
    def split(self):
        """
        Split the image into channels
        """
        # Alpha
        if self._mode == ImageMode.RGBA:
            r, g, b, a = cv2.split(self._data)
            return (
                Image.from_numpy(r),
                Image.from_numpy(g),
                Image.from_numpy(b),
                Image.from_numpy(a)
            )
        elif self._mode == ImageMode.RGB:
            r, g, b = cv2.split(self._data)
            return (
                Image.from_numpy(r),
                Image.from_numpy(g),
                Image.from_numpy(b)
            )
        elif self._mode == ImageMode.GRAY:
            return self
        # end if
    # end split

    # Create a cairo surface
    def create_drawing_context(self):
        """
        Create a Cairo surface from the image
        """
        surface = cairo.ImageSurface.create_for_data(
            self._data,
            cairo.FORMAT_ARGB32,
            self._width,
            self._height
        )

        # Create context
        context = cairo.Context(surface)

        return surface, context
    # end create_drawing_context

    # Save image to file
    def save(
            self,
            file_path
    ):
        """
        Save the image to a file

        Args:
            file_path (str): Path to the file
        """
        cv2.imwrite(file_path, self._data)
    # end save

    # endregion PUBLIC

    # region OVERRIDE

    # Represent as string
    def __repr__(self):
        """
        Represent the image as a string
        """
        return f"Image({self._data.shape}, {self._data.dtype}, {self._mode})"
    # end __repr__

    # endregion OVERRIDE

    # region STATIC

    @classmethod
    def from_numpy(
            cls,
            image_array
    ):
        """
        Create an image from a NumPy array

        Args:
            image_array (np.ndarray): Image math_old as a NumPy array
        """
        return cls(image_array)
    # end from_numpy

    @classmethod
    def from_file(
            cls,
            file_path
    ):
        """
        Create an image from a file

        Args:
            file_path (str): Path to the image file
        """
        image_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        # Test if image is loaded
        if image_array is None:
            raise ValueError(f"Unable to load image from file: {file_path}")
        # end if

        return cls(image_array)
    # end from_file

    @classmethod
    def from_url(
            cls,
            url
    ):
        """
        Create an image from a URL

        Args:
            url (str): URL of the image
        """
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        image_array = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        return cls(image_array)
    # end from_url

    # Create transparent image
    @classmethod
    def transparent(
            cls,
            width: int,
            height: int,
            *args,
            **kwargs
    ):
        """
        Create a transparent image

        Args:
            width (int): Width of the image
            height (int): Height of the image
        """
        return cls(np.zeros((height, width, 4), dtype=np.uint8), *args, **kwargs)
    # end transparent

    # Create color image
    @classmethod
    def color(
            cls,
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
        return cls(np.full((height, width, 4), color, dtype=np.uint8))
    # end color

    # Create color image
    @classmethod
    def fill(
            cls,
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
        return cls.color(width, height, color)
    # end fill

    # Create similar image with color
    @classmethod
    def fill_like(
            cls,
            image: 'Image',
            color: tuple
    ):
        """
        Create a color image

        Args:
            image (Image): Image to copy
            color (tuple): Color of the image
        """
        return cls.color(image._width, image._height, color)
    # end fill_like

    # endregion Image

# end Image
