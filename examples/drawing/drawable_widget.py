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

# Imports
from pixelprism import s, p2, c
from pixelprism.animation import Animation
from pixelprism.widgets.containers import Viewport
from pixelprism.widgets import DrawableWidget
from pixelprism.base import DrawableImage, ImageCanvas, CoordSystem
from pixelprism.drawing import Line, Circle


# DrawableWidgetAnimation class
class DrawableWidgetAnimation(Animation):

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    def build(self):
        """
        Build the animation.
        """
        # Coordinate system
        coord_system = CoordSystem(
            image_width=self.width,
            image_height=self.height,
            size=10
        )

        # Create a Viewport
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Position of the circle
        circle_position = coord_system.uls.copy()

        # Add points and lines
        circle1 = Circle(position=circle_position, radius=s(0.04), line_width=s(0.02), fill_color=c('RED').copy(), line_color=c('YELLOW').copy())
        line = Line.from_objects(start=circle_position, end=coord_system.center, line_width=s(0.02))

        # Add the circle and line to the drawable widget
        drawable_widget.add(line)
        drawable_widget.add(circle1)

        # Animate point 1
        anim = circle1.fadein(1).move(2, coord_system.urs).move(2, coord_system.lrs).move(2, coord_system.lls)
        anim.move(2, coord_system.uls).fadeout(1)

        # Animate line
        line.fadein(1).fadeout(1, 9)

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            circle1=circle1,
            line=line
        )
    # end build

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
        # Create a DrawableImage
        drawing_layer = DrawableImage.transparent(
            self.width,
            self.height,
            coord_system=self.obj("coord_system")
        )

        # Get the viewport and drawable widget
        viewport = self.obj("viewport")

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(viewport)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end DrawableWidgetAnimation
