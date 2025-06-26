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

# PixelPrism
import numpy as np
from pixelprism import p2, s, c
from pixelprism.animation import Animation
from pixelprism.widgets.containers import Viewport
from pixelprism.widgets import DrawableWidget
from pixelprism.base import DrawableImage, ImageCanvas, CoordSystem
from pixelprism.drawing import Line
from pixelprism.animate import Move, EaseInOutInterpolator, FadeIn, FadeOut, Range
from pixelprism.data import Point2D, Scalar, Color


# JoinedLinesAnimation class
class JoinedLinesAnimation(Animation):

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

        # Create a point and line
        shared_point = coord_system.uls.copy()

        # Opacity and thickness
        opacity = s(0.0)
        thickness = s(0.02)
        cyan = c("CYAN")
        line_color = Color.from_objects(s(cyan.red), s(cyan.green), s(cyan.blue), opacity)

        # Two lines
        line1 = Line.from_objects(shared_point, coord_system.urs, line_width=thickness, line_color=line_color)
        line2 = Line.from_objects(shared_point, coord_system.lrs, line_width=thickness, line_color=line_color)

        # Add the lines to the drawable widget
        drawable_widget.add(line1)
        drawable_widget.add(line2)

        # Animate shared point
        anim = shared_point.move(2, coord_system.lls, 1).move(2, coord_system.center)
        anim.move(2, coord_system.uls)

        # Animate opacity and thickness
        opacity.range(1, s(1.0)).range(1, s(0.0), 7)
        thickness.range(4, s(0.08)).range(4, s(0.02), 4)

        # Add
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            line1=line1,
            line2=line2
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

# end JoinedLinesAnimation
