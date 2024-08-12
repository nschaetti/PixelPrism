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
# Animation of an equation.
# Build and highlight
#
import math

from pixel_prism import utils
# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.animate import Move, EaseInOutInterpolator, Range, Call
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas
from pixel_prism.drawing import MathTex, Line
from pixel_prism.data import Point2D, Scalar


# DrawableWidgetAnimation class
class LineAnimation(Animation):

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    # Build first arc
    def build_first_arc(
            self
    ):
        """
        Build the first arc.
        """
        # Create an ARC on upper left
        line1 = Line(
            sx=1920 / 4.0 - 300,
            sy=1080 / 4.0 - 200,
            ex=1920 / 4.0 + 300,
            ey=1080 / 4.0 + 200,
            line_color=utils.RED.copy(),
            line_width=4.0,
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy(),
        )

        # Animate end angle
        self.animate(
            Move(
                line1.start,
                start_time=0,
                end_time=3,
                target_value=Point2D(1920 / 4.0 - 300, 1080 / 4.0 + 200),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Animate end angle
        self.animate(
            Move(
                line1.start,
                start_time=3,
                end_time=6,
                target_value=Point2D(1920 / 4.0 - 300, 1080 / 4.0 - 200),
                interpolator=EaseInOutInterpolator()
            )
        )

        return line1
    # end build_first_arc

    # Build second line
    def build_second_line(
            self
    ):
        """
        Build the second line.
        """
        # Create an ARC on upper left
        line2 = Line(
            sx=1920 / 4.0 * 3 - 300,
            sy=1080 / 4.0 - 200,
            ex=1920 / 4.0 * 3,
            ey=1080 / 4.0,
            line_color=utils.RED.copy(),
            line_width=4.0,
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy(),
        )

        # Change value of scale
        self.animate(
            Call(
                line2.scale,
                times=[2, 4, 6],
                values=[[Scalar(0.5)], [Scalar(2.0)], [Scalar(1.5)]],
            )
        )

        return line2
    # end build_second_line

    # Build three lines
    def build_three_lines(
            self
    ):
        """
        Build the three lines.
        """
        # Create an ARC on upper left
        line3 = Line(
            sx=1920 / 4.0 - 150,
            sy=1080 / 4.0 * 3 - 150,
            ex=1920 / 4.0 + 150,
            ey=1080 / 4.0 * 3 + 150,
            line_color=utils.RED.copy(),
            line_width=4.0,
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy(),
        )

        # Change value of scale
        self.animate(
            Call(
                line3.rotate,
                times=[2, 4, 6],
                values=[[Scalar(math.pi / 2.0)], [Scalar(math.pi / 2.0)], [Scalar(math.pi / 2.0)]],
            )
        )

        return line3
    # end build_three_lines

    # Build fourth line
    def build_fourth_line(
            self
    ):
        """
        Build the fourth line.
        """
        # Create an ARC on upper left
        line4 = Line(
            sx=1920 / 4.0 * 3 - 150,
            sy=1080 / 4.0 * 3 - 150,
            ex=1920 / 4.0 * 3 + 150,
            ey=1080 / 4.0 * 3 + 150,
            line_color=utils.RED.copy(),
            line_width=4.0,
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy(),
        )

        # Change value of scale
        self.animate(
            Call(
                line4.translate,
                times=[2, 4, 6],
                values=[[Point2D(10, 10)], [Point2D(10, 10)], [Point2D(10, 10)]],
            )
        )

        return line4
    # end build_fourth_line

    def build(self):
        """
        Build the animation.
        """
        # Add the LaTeX widget to the viewport or a container
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Create lines
        line1 = self.build_first_arc()
        line2 = self.build_second_line()
        line3 = self.build_three_lines()
        line4 = self.build_fourth_line()

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(line1)
        drawable_widget.add(line2)
        drawable_widget.add(line3)
        drawable_widget.add(line4)

        # Add objects
        self.add(
            viewport=viewport,
            drawable_widget=drawable_widget,
            line1=line1,
            line2=line2,
            line3=line3,
            line4=line4
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
        drawing_layer = DrawableImage.transparent(self.width, self.height)

        # Get the viewport and drawable widget
        viewport = self.obj("viewport")

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(viewport)
        drawing_layer.render(
            draw_params={
                'draw_bboxes': True,
                'draw_reference_point': True,
                'draw_points': True
            }
        )

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end LineAnimation


