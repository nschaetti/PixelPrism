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

# PixelPrism
from pixelprism.animation import Animation
from pixelprism.widgets.containers import SplitContainer, PositionedContainer
from pixelprism.widgets import Point, Line, Dummy
from pixelprism.base import DrawableImage, ImageCanvas


# CustomAnimation class
class SplitContainerAnimation(Animation):

    def init_effects(self):
        pass
    # end init_effects

    def process_frame(
            self,
            image_canvas: ImageCanvas,
            t: float,
            frame_number: int
    ):
        """
        Process the frame.

        Args:
            image_canvas (ImageCanvas): Image canvas
            t (float): Time
            frame_number (int): Frame number
        """
        # Create a DrawableImage
        drawing_layer = DrawableImage.transparent(self.width, self.height)

        # Create a PositionedContainer and add Widgets
        split_container = SplitContainer(size=3, orientation='horizontal')
        split_container.add_widget(Dummy((1, 0, 0)), position=0)
        split_container.add_widget(Dummy((0, 1, 0)), position=1)
        split_container.add_widget(Dummy((0, 0, 1)), position=2)

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(split_container)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end SplitContainerAnimation

