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

class Layer:
    """
    Class to represent a layer in an image
    """

    def __init__(
            self,
            name,
            image,
            blend_mode='normal',
            active=True
    ):
        """
        Initialize the layer with a name, image, blend mode, and active status

        Args:
            name (str): Name of the layer
            image (np.ndarray): Image data for the layer
            blend_mode (str): Blend mode for the layer
            active (bool): Whether the layer is active
        """
        self.name = name
        self.image = image
        self.blend_mode = blend_mode
        self.active = active
    # end __init__

    def __repr__(self):
        """
        Return a string representation of the layer
        """
        return f"Layer(name={self.name}, blend_mode={self.blend_mode}, active={self.active}, image={self.image})"
    # end __repr__

# end Layer

