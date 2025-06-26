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

from pixelprism import utils
from pixelprism.animation import Animation
from pixelprism.animate import Move, EaseInOutInterpolator, Range, Call
from pixelprism.widgets.containers import Viewport
from pixelprism.widgets import DrawableWidget
from pixelprism.base import DrawableImage, ImageCanvas, CoordSystem
from pixelprism.drawing import MathTex, Line, Arc
from pixelprism.data import Point2D, Scalar


# DrawableWidgetAnimation class
class RelativeCoorsAnimation(Animation):

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

        # Add the LaTeX widget to the viewport or a container
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Radius, line width, line color, fill color
        radius = Scalar(0.2)
        line_width = Scalar(0.02)
        line_color = utils.YELLOW.copy()
        fill_color = utils.RED.copy()

        # Center
        center = Arc.from_objects(
            center=coord_system.center,
            radius=radius,
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi * 2),
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color
        )

        # Upper right
        upper_right = Arc.from_objects(
            center=coord_system.upper_right,
            radius=radius,
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi * 2),
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color
        )

        # Lower right
        lower_right = Arc.from_objects(
            center=coord_system.lower_right,
            radius=radius,
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi * 2),
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color
        )

        # Upper left
        upper_left = Arc.from_objects(
            center=coord_system.upper_left,
            radius=radius,
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi * 2),
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color
        )

        # Lower left
        lower_left = Arc.from_objects(
            center=coord_system.lower_left,
            radius=radius,
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi * 2),
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color
        )

        # Moving point
        moving_point = Arc.from_objects(
            center=coord_system.upper_left_square.copy(),
            radius=radius,
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi * 2),
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color
        )

        # Add points
        drawable_widget.add(center)
        drawable_widget.add(upper_right)
        drawable_widget.add(lower_right)
        drawable_widget.add(upper_left)
        drawable_widget.add(lower_left)
        drawable_widget.add(moving_point)

        # Animate end angle
        self.animate(
            Move(
                moving_point.center,
                start_time=0,
                end_time=2,
                target_value=coord_system.upper_right_square,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Animate end angle
        self.animate(
            Move(
                moving_point.center,
                start_time=2,
                end_time=4,
                target_value=coord_system.lower_right_square,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Animate end angle
        self.animate(
            Move(
                moving_point.center,
                start_time=4,
                end_time=6,
                target_value=coord_system.lower_left_square,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Animate end angle
        self.animate(
            Move(
                moving_point.center,
                start_time=6,
                end_time=8,
                target_value=coord_system.upper_left_square,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            center=center,
            upper_right=upper_right,
            lower_right=lower_right,
            upper_left=upper_left,
            lower_left=lower_left,
            moving_point=moving_point
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

# end LineAnimation


