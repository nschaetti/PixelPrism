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
from pixelprism.drawing import Line, DebugGrid
from pixelprism.animate import Move, EaseInOutInterpolator, FadeIn, FadeOut, Range
from pixelprism.math import Point2D, Scalar, Color, Style, Transform


# Animation class
class DrawLinesAnimation(Animation):

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    def build_line1(self):
        """
        Build the first line.
        """
        return Line(
            start=Point2D(-10, -10),
            end=Point2D(10, 10),
            style=Style(line_color=c('MAGENTA'), line_width=s(0.4))
        )
    # end build_line1

    def build_line2(self):
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

         # Line
        line = Line(
            start=Point2D(-10, 10),
            end=Point2D(10, -10),
            style=Style(line_color=c('ORANGE'), line_width=s(0.4)),
            transform=transform
        )

        return line
    # end build_line2

    # Build line 3
    def build_line3(self):
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
        line = Line(
            start=p2(0, 10),
            end=p2(0, -10),
            style=Style(line_color=c('RED'), line_width=s(0.4)),
            transform=transform
        )

        return line
    # end build_line3

    # Build line 4
    def build_line4(self):
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
        rotation.range(8, math.pi * 2.0)

        # Line
        line = Line(
            start=p2(0, 10),
            end=p2(10, 0),
            style=Style(line_color=c('BLUE'), line_width=s(0.4)),
            transform=transform
        )

        return line
    # end build_line4

    # Build line 5
    def build_line5(self, coord_system: CoordSystem):
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

        # Back
        position.move(4, coord_system.urs)
        scale.move(4, p2(0.5, 0.5))
        rotation.range(4, math.pi)

        # Forward
        position.move(4, p2(0.0, 0.0), 4)
        scale.move(4, p2(1.0, 1.0), 4)
        rotation.range(4, 0.0, 4)

        # Line
        line = Line(
            start=p2(10, 10),
            end=p2(-10, -10),
            style=Style(line_color=c('GREEN'), line_width=s(0.4)),
            transform=transform
        )

        return line
    # end build_line5

    # Build line 6
    def build_line6(self, coord_system: CoordSystem):
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
        line = Line(
            start=p2(-10, 0),
            end=p2(10, 0),
            style=Style(line_color=c('YELLOW'), line_width=s(0.4)),
            transform=transform2
        )

        return line
    # end build_line6

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

        # Create lines
        line1 = self.build_line1()
        line2 = self.build_line2()
        line3 = self.build_line3()
        line4 = self.build_line4()
        line5 = self.build_line5(coord_system)
        line6 = self.build_line6(coord_system)

        # Add debug grid
        drawable_widget.add(debug_grid)
        drawable_widget.add(line1)
        drawable_widget.add(line2)
        drawable_widget.add(line3)
        drawable_widget.add(line4)
        drawable_widget.add(line5)
        drawable_widget.add(line6)

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            debug_grid=debug_grid,
            line1=line1,
            line2=line2,
            line3=line3,
            line4=line4,
            line5=line5,
            line6=line6
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

