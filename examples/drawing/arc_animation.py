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
import math
from pixelprism import utils, p2, s
from pixelprism.animation import Animation
from pixelprism.animate import Move, EaseInOutInterpolator, Range, Call
from pixelprism.widgets.containers import Viewport
from pixelprism.widgets import DrawableWidget
from pixelprism.base import DrawableImage, ImageCanvas, CoordSystem
from pixelprism.drawing import Arc
from pixelprism.math_old import Point2D, Scalar


# DrawableWidgetAnimation class
class ArcAnimation(Animation):

    ARC_LINE_WIDTH = 0.02
    ARC_RADIUS = 1.0

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    # Build first arc
    def build_first_arc(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the first arc.
        """
        # Create an ARC on upper left
        arc1 = Arc.from_objects(
            center=coord_system.upper_left_square,
            radius=Scalar(self.ARC_RADIUS),
            start_angle=Scalar(0.0),
            end_angle=Scalar(0.0),
            line_color=utils.RED.copy(),
            line_width=Scalar(self.ARC_LINE_WIDTH),
            fill_color=utils.GREEN.copy()
        )

        # Animate start and end angles
        arc1.start_angle.range(8, math.pi * 2)
        arc1.end_angle.range(4, math.pi * 2)

        return arc1
    # end build_first_arc

    # Create second arc
    def build_second_arc(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the second arc.
        """
        # Create an ARC on upper left
        arc2 = Arc.from_objects(
            center=coord_system.upper_right_square - p2(1.0, 0.0),
            radius=s(self.ARC_RADIUS),
            start_angle=s(0.0),
            end_angle=s(math.pi),
            line_color=utils.RED.copy(),
            line_width=s(self.ARC_LINE_WIDTH),
            fill_color=utils.GREEN.copy()
        )

        # Move the arc (by the center)
        anim = arc2.center.move(4, coord_system.upper_right_square + p2(1.0, 0.0))
        anim.move(4, coord_system.upper_right_square - p2(1.0, 0.0))

        return arc2
    # end build_second_arc

    # Create third arc
    def build_third_arc(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the third arc.
        """
        # Create an ARC on upper left
        arc3 = Arc.from_objects(
            center=coord_system.lower_left_square,
            radius=Scalar(self.ARC_RADIUS / 2.0),
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi * 1.5),
            line_color=utils.RED.copy(),
            line_width=Scalar(self.ARC_LINE_WIDTH),
            fill_color=utils.GREEN.copy()
        )

        # Change value of scale
        arc3.scale.call([2, 4, 6], [[Scalar(2.0)], [Scalar(0.5)], [Scalar(2.0)]])

        return arc3
    # end build_third_arc

    # Build fourth arc
    def build_fourth_arc(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the fourth arc.
        """
        # Create an ARC on upper left
        arc4 = Arc.from_objects(
            center=coord_system.lower_right_square,
            radius=Scalar(self.ARC_RADIUS),
            start_angle=Scalar(0.0),
            end_angle=Scalar(math.pi),
            line_color=utils.RED.copy(),
            line_width=Scalar(self.ARC_LINE_WIDTH),
            fill_color=utils.GREEN.copy()
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
        arc1 = self.build_first_arc(coord_system)
        arc2 = self.build_second_arc(coord_system)
        arc3 = self.build_third_arc(coord_system)
        arc4 = self.build_fourth_arc(coord_system)

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(arc1)
        drawable_widget.add(arc2)
        drawable_widget.add(arc3)
        drawable_widget.add(arc4)

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            arc1=arc1,
            arc2=arc2,
            arc3=arc3,
            arc4=arc4
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
                'draw_reference_point': True,
                'draw_points': True
            }
        )

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end ArcAnimation


