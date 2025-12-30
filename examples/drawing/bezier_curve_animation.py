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

#
# Animation of an equation.
# Build and highlight
#

# Imports
import math
from pixelprism import p2, s, c
from pixelprism.animation import Animation
from pixelprism.widgets.containers import Viewport
from pixelprism.widgets import DrawableWidget
from pixelprism.base import DrawableImage, ImageCanvas, CoordSystem
from pixelprism.drawing import CubicBezierCurve
from pixelprism.math_old import Scalar


# DrawableWidgetAnimation class
class CurveAnimation(Animation):

    CURVE_LINE_WIDTH = 0.02

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    # Build first cruve
    def build_first_curve(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the first curve.
        """
        # Create an ARC on upper left
        curve1 = CubicBezierCurve.from_objects(
            start=coord_system.uls - p2(1.5, 1),
            control1=p2(0.5, 0.5),
            control2=p2(-0.5, -0.5),
            end=coord_system.uls + p2(1.5, 0),
            line_color=c('WHITE').copy(),
            line_width=s(self.CURVE_LINE_WIDTH)
        )

        # Animate start
        curve1.start.move(4, coord_system.uls - p2(1.5, -1)).move(4, coord_system.uls - p2(1.5, 1))
        curve1.control2.move(4, p2(0.5, 0.5)).move(8, p2(-0.5, -0.5))
        curve1.control1.move(4, p2(-0.5, -0.5)).move(4, p2(0.5, 0.5))

        return curve1
    # end build_first_curve

    # Build second cruve
    def build_second_curve(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the first curve.
        """
        # Create an ARC on upper left
        curve2 = CubicBezierCurve.from_objects(
            start=coord_system.urs - p2(2, 1),
            control1=p2(0.0, 1.0),
            control2=p2(0.0, 1.0),
            end=coord_system.urs + p2(0, -1),
            line_color=c('WHITE').copy(),
            line_width=s(self.CURVE_LINE_WIDTH)
        )

        # Animate start
        curve2.move(4, coord_system.urs - p2(2, 1) + p2(2, 0)).move(4, coord_system.urs - p2(2, 1))

        return curve2
    # end build_second_curve

    # Build third curve
    def build_third_curve(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the first curve.
        """
        # Position and length
        position = Scalar(0.0)
        path_length = Scalar(0.25)

        # Create an ARC on upper left
        curve3 = CubicBezierCurve.from_objects(
            start=coord_system.lls + p2(-1, -1),
            control1=p2(0.0, 1.0),
            control2=p2(0.0, -1.0),
            end=coord_system.lls + p2(1, 1),
            position=position,
            path_length=path_length,
            line_color=c('WHITE').copy(),
            line_width=s(self.CURVE_LINE_WIDTH)
        )

        # Animate position
        position.range(4, s(0.5)).range(4, s(0.0))
        path_length.range(4, s(1.0), 4)

        return curve3
    # end build_third_curve

    # Build fourth curve
    def build_fourth_curve(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the first curve.
        """
        # Position and length
        position = s(0.0)
        path_length = s(0.25)

        curve4 = CubicBezierCurve.from_objects(
            start=coord_system.lrs - p2(1.5, 0.5),
            control1=p2(0.0, 1.0),
            control2=p2(0.0, 1.0),
            end=coord_system.lrs + p2(0, -0.5),
            position=position,
            path_length=path_length,
            line_color=c('WHITE').copy(),
            line_width=s(self.CURVE_LINE_WIDTH)
        )

        # Translate, rotate and scale
        curve4.call([2, 7], 'translate', [[p2(1.0, 0.0)], [p2(-1.0, 0.0)]])
        curve4.call([3, 6], 'rotate', [[s(math.pi / 4.0), p2(1.89, -1.67)], [s(-math.pi / 4.0), p2(1.89, -1.67)]])
        curve4.call([4, 5], 'scale', [[s(1.5), p2(1.89, -1.67)], [s(0.6666666), p2(1.89, -1.67)]])

        # Animate position
        path_length.range(8, s(1.0))

        return curve4
    # end build_fourth_curve

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

        # Create arcs
        curve1 = self.build_first_curve(coord_system)
        curve2 = self.build_second_curve(coord_system)
        curve3 = self.build_third_curve(coord_system)
        curve4 = self.build_fourth_curve(coord_system)

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(curve1)
        drawable_widget.add(curve2)
        drawable_widget.add(curve3)
        drawable_widget.add(curve4)

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            curve1=curve1,
            curve2=curve2,
            curve3=curve3,
            curve4=curve4
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
        drawing_layer.render(
            draw_params={
                'draw_bboxes': True,
                'draw_points': True,
                'draw_reference_point': True,
                'draw_control_points': True
            }
        )

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end MathTexAnimation


