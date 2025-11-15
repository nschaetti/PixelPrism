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
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

#
# Drawable image
#

# Imports
import cairo
from .context import Context
from .image import Image
from .coordsystem import CoordSystem


# DrawableImage class
class DrawableImage(Image):

    # Constructor
    def __init__(
            self,
            image_array,
            coord_system: CoordSystem
    ):
        """
        Initialize the image math with an image array

        Args:
            image_array (np.ndarray): Image math as a NumPy array
            coord_system (CoordSystem): Coordinate system for the image
        """
        super().__init__(image_array)
        self._context = self.create_context(coord_system)
        self._root_container = None
        self._coord_system = coord_system
    # end __init__

    @property
    def root_container(self):
        """
        Get the root container for the image.
        """
        return self._root_container
    # end root_container

    @property
    def context(self):
        """
        Get the Cairo context for drawing.
        """
        return self._context
    # end context

    def create_context(
            self,
            coord_system: CoordSystem
    ):
        """
        Create a Cairo surface and context from the image math.

        Args:
            coord_system (CoordSystem): Coordinate system
        """
        return Context.from_image(
            image=self,
            coord_system=coord_system
        )
    # end create_context

    def get_context(self):
        """
        Get the Cairo context for drawing.
        """
        return self._context
    # end get

    # Set root container
    def set_root_container(
            self,
            container
    ):
        """
        Set the root container for the image.

        Args:
            container (Container): Root container
        """
        self._root_container = container
    # end set_root_container

    # Render the image
    def render(
            self,
            *args,
            **kwargs
    ):
        """
        Render the image to the context.
        """
        if self.root_container:
            # Setup coordinate system
            self.context.setup_context()

            # Render the root container
            self.root_container.render(
                self.context,
                *args,
                **kwargs
            )
        else:
            raise ValueError("Root container not set.")
        # end if
        return self
    # end render

    def save(
            self,
            file_path
    ):
        """
        Save the image, ensuring Cairo surface is written to file.

        Args:
            file_path (str): Path to the file
        """
        self._context.surface.write_to_png(file_path)
    # end save

# end DrawableImage
