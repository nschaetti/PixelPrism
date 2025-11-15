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
import cairo
from pixelprism import Animation
from pixelprism.base import Image, ImageCanvas, DrawableImage
from pixelprism.drawing import Space2D, Point, Line, Plot


class PointsLinesAnimation(Animation):

    # Init. effects
    def init_effects(
            self
    ):
        """
        Initialize the effects.
        """
        pass
    # end init_effects

    # Process frame
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
        # Create a new transparent layer for drawing
        drawing_image = Image.fill(self.width, self.height, (0, 0, 0, 255.0))
        drawing_layer = DrawableImage.transparent(self.width, self.height)

        # Draw point, line and plot
        space = Space2D()
        space.add(Point(100, 100))
        space.add(Line((150, 150), (300, 300)))
        space.add(Plot(lambda x: 0.01 * (x - 320)**2, (0, 640)))

        # Draw the space
        space.draw(context)

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_image)

        return image_canvas
    # end process_frame

# end PointsLinesAnimation

