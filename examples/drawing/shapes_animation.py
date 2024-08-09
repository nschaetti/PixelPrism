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
from pixel_prism.animate import Move, EaseInOutInterpolator, Range
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

    def build(self):
        """
        Build the animation.
        """
        # Add the LaTeX widget to the viewport or a container
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Create an ARC on upper left
        arc1 = Arc(
            cx=250,
            cy=1080 / 4.0,
            radius=200,
            start_angle=0.0,
            end_angle=0.0,
            line_color=utils.RED.copy(),
            line_width=4.0,
            fill_color=utils.GREEN.copy(),
            bbox_border_width=2,
            bbox_border_color=utils.BLUE.copy()
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

        # Move the Arc
        self.animate(
            Move(
                arc1.center,
                start_time=0,
                end_time=7,
                target_value=Point2D(1920 / 2.0 - 250, 1080 / 4.0),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(arc1)

        # Add objects
        self.add(
            viewport=viewport,
            drawable_widget=drawable_widget,
            arc1=arc1
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
            draw_params={'draw_bboxes': True}
        )

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end MathTexAnimation


