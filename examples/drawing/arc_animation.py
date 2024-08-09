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
from pixel_prism.drawing import MathTex, Arc
from pixel_prism.data import Point2D, Scalar


# DrawableWidgetAnimation class
class ShapesAnimation(Animation):

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
        arc1 = Arc(
            cx=1920 / 4.0,
            cy=1080 / 4.0,
            radius=200,
            start_angle=0.0,
            end_angle=0.0,
            line_color=utils.RED.copy(),
            line_width=4.0,
            fill_color=utils.GREEN.copy(),
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy(),

        )

        # Animate end angle
        self.animate(
            Range(
                arc1.start_angle,
                start_time=0,
                end_time=7,
                target_value=math.pi * 2,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Animate end angle
        self.animate(
            Range(
                arc1.end_angle,
                start_time=0,
                end_time=4,
                target_value=math.pi * 2,
                interpolator=EaseInOutInterpolator()
            )
        )

        return arc1
    # end build_first_arc

    # Create second arc
    def build_second_arc(
            self
    ):
        """
        Build the second arc.
        """
        # Create an ARC on upper left
        arc2 = Arc(
            cx=1920 / 2.0 + 250,
            cy=1080 / 4.0,
            radius=200,
            start_angle=0.0,
            end_angle=math.pi,
            line_color=utils.RED.copy(),
            line_width=4.0,
            fill_color=utils.GREEN.copy(),
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy()
        )

        # Move the Arc
        self.animate(
            Move(
                arc2.center,
                start_time=0,
                end_time=3,
                target_value=Point2D(1920 - 250, 1080 / 4.0),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Move the Arc
        self.animate(
            Move(
                arc2.center,
                start_time=3,
                end_time=6,
                target_value=Point2D(1920 / 2.0 + 250, 1080 / 4.0),
                interpolator=EaseInOutInterpolator()
            )
        )

        return arc2
    # end build_second_arc

    # Create third arc
    def build_third_arc(
            self
    ):
        """
        Build the third arc.
        """
        # Create an ARC on upper left
        arc3 = Arc(
            cx=1920 / 4.0,
            cy=1080 / 4.0 * 3.0,
            radius=100,
            start_angle=0.0,
            end_angle=math.pi * 1.5,
            line_color=utils.RED.copy(),
            line_width=4.0,
            fill_color=utils.GREEN.copy(),
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy()
        )

        # Change value of scale
        self.animate(
            Call(
                arc3.scale,
                times=[2, 4, 6],
                values=[[Scalar(2.0)], [Scalar(0.5)], [Scalar(2.0)]],
            )
        )

        return arc3
    # end build_third_arc

    # Build fourth arc
    def build_fourth_arc(
            self
    ):
        """
        Build the fourth arc.
        """
        # Create an ARC on upper left
        arc4 = Arc(
            cx=1920 / 4.0 * 3.0,
            cy=1080 / 4.0 * 3.0,
            radius=200,
            start_angle=0.0,
            end_angle=math.pi,
            line_color=utils.RED.copy(),
            line_width=4.0,
            fill_color=utils.GREEN.copy(),
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy()
        )

        # Change value of scale
        self.animate(
            Call(
                arc4.rotate,
                times=[2, 4, 6],
                values=[[Scalar(math.pi / 2.0)], [Scalar(math.pi / 2.0)], [Scalar(math.pi / 2.0)]],
            )
        )

        return arc4
    # end build_fourth_arc

    def build(self):
        """
        Build the animation.
        """
        # Add the LaTeX widget to the viewport or a container
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Create arcs
        arc1 = self.build_first_arc()
        arc2 = self.build_second_arc()
        arc3 = self.build_third_arc()
        arc4 = self.build_fourth_arc()

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(arc1)
        drawable_widget.add(arc2)
        drawable_widget.add(arc3)
        drawable_widget.add(arc4)

        # Add objects
        self.add(
            viewport=viewport,
            drawable_widget=drawable_widget,
            arc1=arc1,
            arc2=arc2
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

# end MathTexAnimation


