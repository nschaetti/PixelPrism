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
import math

# Imports
from pixelprism import p2, s, c
from pixelprism.animation import Animation
from pixelprism.widgets.containers import Viewport
from pixelprism.widgets import DrawableWidget
from pixelprism.base import DrawableImage, ImageCanvas, CoordSystem
from pixelprism.drawing import Line, DebugGrid, Rectangle
from pixelprism.animate import Move, EaseInOutInterpolator, FadeIn, FadeOut, Range
from pixelprism.math import Point2D, Scalar, Color, Style, Transform


# Animation class
class DrawRectanglesAnimation(Animation):

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    def build_rectangle1(self):
        """
        Build the first line.
        """
        return Rectangle(
            upper_left=Point2D(-10, -5),
            width=Scalar(20),
            height=Scalar(10),
            style=Style(line_color=c('MAGENTA'), fill_color=c('CYAN'), line_width=s(0.4))
        )
    # end build_rectangle1

    # Build second rectangle
    def build_rectangle2(self):
        """
        Build the second line.
        """
        # Rotation
        rotation = Scalar(0)

        # Transform
        transform = Transform(
            position=Point2D(0, 0),
            rotation=rotation,
            scale=Point2D(1, 1)
        )

        # Animate
        rotation.range(8, math.pi * 2.0)

        return Rectangle(
            upper_left=Point2D(-5, -2.5),
            width=Scalar(10),
            height=Scalar(5),
            style=Style(line_color=c('TEAL'), fill_color=c('ORANGE'), line_width=s(0.3)),
            transform=transform
        )
    # end build_rectangle2

    # Build third rectangle
    def build_rectangle3(self):
        """
                Build the third line.
                """
        # Rotation
        position = p2(0, 0)

        # Transform
        transform = Transform(
            position=position,
            rotation=Scalar(0.0),
            scale=p2(1, 1)
        )

        # Animate
        position.move(4, p2(25, 0))
        position.move(4, p2(-25, 0), 4)

        # Line
        rectangle = Rectangle(
            upper_left=Point2D(-1.5, -10),
            width=Scalar(3),
            height=Scalar(20),
            style=Style(line_color=c('GOLD'), fill_color=c('NAVY'), line_width=s(0.3)),
            transform=transform
        )

        return rectangle
    # end build_rectangle3

    # Build fourth rectangle
    def build_rectangle4(self):
        """
        Build the fourth line.
        """
        # Rotation
        rotation = Scalar(0)

        # Transform
        transform = Transform(
            position=Point2D(0, 0),
            rotation=rotation,
            scale=Point2D(1, 1)
        )

        # Animate
        rotation.range(8, -math.pi * 2.0)

        # Line
        rectangle = Rectangle(
            upper_left=Point2D(5, 5),
            width=Scalar(3),
            height=Scalar(10),
            style=Style(line_color=c('GREEN'), fill_color=c('RED'), line_width=s(0.3)),
            transform=transform
        )

        return rectangle
    # end build_rectangle4

    # Build fifth rectangle
    def build_rectangle5(self, coord_system: CoordSystem):
        """
        Build the fifth line.
        """
        # Rotation
        position = p2(0, 0)
        scale = p2(1, 1)
        rotation = Scalar(0)

        # Transform
        transform = Transform(
            position=position,
            rotation=rotation,
            scale=scale
        )

        # Forth
        position.move(4, coord_system.urs)
        scale.move(4, p2(2, 0.5))
        rotation.range(4, math.pi)

        # Back
        position.move(4, p2(0, 0), 4)
        scale.move(4, p2(1, 1), 4)
        rotation.range(4, 0, 4)

        # Line
        rectangle = Rectangle(
            upper_left=Point2D(-5, -5),
            width=Scalar(10),
            height=Scalar(10),
            style=Style(line_color=c('BLUE'), fill_color=c('YELLOW'), line_width=s(0.3)),
            transform=transform
        )

        return rectangle
    # end build_rectangle5

    # Build sixth rectangle
    def build_rectangle6(self, coord_system: CoordSystem):
        """
        Build the sixth line.
        """
        # First transform
        position = p2(0, 0)
        rotation = Scalar(0)

        # Transform 1
        transform1 = Transform(
            position=position
        )

        # Transform 2
        transform2 = Transform(
            rotation=rotation,
            parent=transform1
        )

        # Aimation
        position.move(4, coord_system.us)
        rotation.range(4, math.pi * 2.0, 4)

        # Line
        rectangle = Rectangle(
            upper_left=Point2D(-2, -2),
            width=Scalar(4),
            height=Scalar(4),
            style=Style(line_color=c('SALMON'), fill_color=c('VIOLET'), line_width=s(0.2)),
            transform=transform2
        )

        return rectangle
    # end build_rectangle6

    def build(self):
        """
        Build the animation.
        """
        # Coordinate system
        coord_system = CoordSystem(
            image_width=self.width,
            image_height=self.height,
            size=100
        )

        # Create a Viewport
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Create a debug grid
        debug_grid = DebugGrid(
            coord_system=coord_system,
            grid_style=Style(line_color=c('DARKER_GRAY'), line_width=s(0.1)),
            axis_style=Style(line_color=c('TEAL'), line_width=s(0.2)),
            major_width=s(10),
            minor_width=s(5)
        )

        # Create rectangles
        rectangle1 = self.build_rectangle1()
        rectangle2 = self.build_rectangle2()
        rectangle3 = self.build_rectangle3()
        rectangle4 = self.build_rectangle4()
        rectangle5 = self.build_rectangle5(coord_system)
        rectangle6 = self.build_rectangle6(coord_system)

        # Add debug grid
        drawable_widget.add(debug_grid)
        drawable_widget.add(rectangle1)
        drawable_widget.add(rectangle2)
        drawable_widget.add(rectangle3)
        drawable_widget.add(rectangle4)
        drawable_widget.add(rectangle5)
        drawable_widget.add(rectangle6)

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            debug_grid=debug_grid,
            rectangle1=rectangle1,
            rectangle2=rectangle2,
            rectangle3=rectangle3,
            rectangle4=rectangle4,
            rectangle5=rectangle5,
            rectangle6=rectangle6
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
        Process a frame of the animation

        Args:
            image_canvas (ImageCanvas): Image canvas
            t (float): Time
            frame_number (int): Frame number
        """
        # Create a drawable image
        drawing_layer = DrawableImage.transparent(
            width=self.width,
            height=self.height,
            coord_system=self.obj("coord_system")
        )

        # Get the viewport
        viewport = self.obj("viewport")

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(viewport)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end DrawLinesAnimation

