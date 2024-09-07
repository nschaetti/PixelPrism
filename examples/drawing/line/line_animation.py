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
from pixel_prism import utils, p2, s
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas, CoordSystem
from pixel_prism.drawing import Line


# DrawableWidgetAnimation class
class LineAnimation(Animation):

    LINE_WIDTH = 0.02

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    # Build first arc
    def build_first_arc(
            self,
            coord_system
    ):
        """
        Build the first arc.
        """
        # Create an ARC on upper left
        line1 = Line.from_objects(
            start=coord_system.uls - p2(1.5, 1),
            end=coord_system.uls + p2(1.5, 1),
            line_color=utils.RED.copy(),
            line_width=s(self.LINE_WIDTH)
        )

        # Animate start
        line1.start.move(3, coord_system.uls + p2(-1.5, 1)).move(3, coord_system.uls - p2(1.5, 1))

        return line1
    # end build_first_arc

    # Build second line
    def build_second_line(
            self,
            coord_system
    ):
        """
        Build the second line.
        """
        # Create an ARC on upper left
        line2 = Line.from_objects(
            start=coord_system.urs - p2(1.5, 1),
            end=coord_system.urs.copy(),
            line_color=utils.RED.copy(),
            line_width=s(self.LINE_WIDTH)
        )

        # Change value of scale
        line2.call([2, 4, 6], 'scale', [[s(0.5)], [s(2.0)], [s(1.5)]])

        return line2
    # end build_second_line

    # Build three lines
    def build_three_lines(
            self,
            coord_system
    ):
        """
        Build the three lines.
        """
        # Create an ARC on upper left
        line3 = Line.from_objects(
            start=coord_system.lls - p2(0.75, 0.75),
            end=coord_system.lls + p2(0.75, 0.75),
            line_color=utils.RED.copy(),
            line_width=s(self.LINE_WIDTH)
        )

        # Change value of scale
        line3.call([2, 4, 6], 'rotate', [[s(math.pi / 2.0)], [s(math.pi / 2.0)], [s(math.pi / 2.0)]])

        return line3
    # end build_three_lines

    # Build fourth line
    def build_fourth_line(
            self,
            coord_system
    ):
        """
        Build the fourth line.
        """
        # Create an ARC on upper left
        line4 = Line.from_objects(
            start=coord_system.lrs - p2(0.75, 0.75),
            end=coord_system.lrs + p2(0.75, 0.75),
            line_color=utils.RED.copy(),
            line_width=s(self.LINE_WIDTH)
        )

        # Change value of scale
        line4.call([2, 4, 6], 'translate', [[p2(0.1, 0.1)], [p2(0.1, 0.1)], [p2(0.1, 0.1)]])

        return line4
    # end build_fourth_line

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

        # Create lines
        line1 = self.build_first_arc(coord_system)
        line2 = self.build_second_line(coord_system)
        line3 = self.build_three_lines(coord_system)
        line4 = self.build_fourth_line(coord_system)

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(line1)
        drawable_widget.add(line2)
        drawable_widget.add(line3)
        drawable_widget.add(line4)

        # Add objects
        self.add(
            coord_system=coord_system,
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

# end LineAnimation


