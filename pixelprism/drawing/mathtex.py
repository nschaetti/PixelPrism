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
# This file contains the MathTex class, which is a widget that can be drawn
#

# Imports
from typing import List, Optional, Any
import os
import cairo
import tempfile
from pixelprism.utils import render_latex_to_svg, draw_svg
from pixelprism.animate.able import (
    BuildableMixin,
    DestroyableMixin
)
from pixelprism.animate import FadeableMixin, MovableMixin
from pixelprism.math import Point2D, Color
import pixelprism.utils as utils
from .drawablemixin import DrawableMixin
from .vector_graphics import VectorGraphics


def generate_temp_svg_filename():
    """
    Generate a random filename for a temporary SVG file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    return temp_filename
# end generate_temp_svg_filename


class MathTex(
    DrawableMixin,
    MovableMixin,
    FadeableMixin,
    BuildableMixin,
    DestroyableMixin
):

    def __init__(
            self,
            latex,
            position,
            scale: Point2D = Point2D(1, 1),
            color: Color = utils.WHITE.copy(),
            refs: Optional[List] = None,
            keep_svg: bool = False,
            debug: bool = False
    ):
        """
        Initialize the MathTex object.

        Args:
            latex (str): The latex string to render
            position (Point2D): The position of the MathTex object
            color (tuple): The color of the latex string
            refs (list): List of references
            keep_svg (bool): Whether to keep the SVG file
            debug (bool): Whether to print debug messages
        """
        super().__init__()
        self.latex = latex
        self.position = position
        self.scale = scale
        self.color = color
        self.keep_svg = keep_svg
        self.debug = debug
        self.math_graphics = self.generate_vector_graphics(refs)
    # end __init__

    # Generate the vector graphics
    def generate_vector_graphics(
            self,
            refs: Optional[List] = None
    ) -> VectorGraphics:
        """
        Generate the vector graphics for the MathTex object.
        """
        # Generate a random filename for the SVG file
        random_svg_path = generate_temp_svg_filename()

        # Generate the SVG file from the math string
        self.update_svg(random_svg_path)

        # Create the vector graphics object
        vector_graphics = VectorGraphics.from_svg(
            random_svg_path,
            scale=self.scale,
            color=self.color,
            refs=refs,
        )

        # Delete the temporary SVG file
        if not self.keep_svg:
            os.remove(random_svg_path)
        # end if

        return vector_graphics
    # end generate_vector_graphics

    def update_svg(self, svg_path):
        """
        Update the svg file with the latex string.

        Args:
            svg_path (str): The path to the SVG file
        """
        render_latex_to_svg(self.latex, svg_path)
    # end update_svg

    # Set alpha
    def set_alpha(self, alpha):
        """
        Set the alpha of the MathTex object.
        """
        self.math_graphics.set_alpha(alpha)
    # end set_alpha

    def draw(self, context):
        """
        Draw the MathTex object to the context.

        Args:
            context (cairo.Context): Context to draw the MathTex object to
        """
        # Get the position
        x, y = self.position.get()

        # Translate the context
        context.save()
        context.translate(x, y)

        # draw_svg(context, self.svg_path, x, y, color=self.color)
        # Draw the vector graphics
        self.math_graphics.draw(
            context,
            draw_bboxes=False,
            draw_reference_point=False,
            draw_paths=False
        )

        # Restore the context
        context.restore()
    # end draw

    # region OVERRIDE

    def _create_bbox(
            self,
            border_width: float = 0.0,
            border_color: Color = utils.WHITE.copy()
    ):
        """
        Create the bounding box.
        """
        return None
    # end _create_bbox

    def __getitem__(self, index):
        """
        Get the element at the specified index.
        """
        return self.math_graphics[index]

    # end __getitem__

    def __setitem__(self, index, value):
        """
        Set the element at the specified index.
        """
        self.math_graphics[index] = value
    # end __setitem__

    # endregion OVERRIDE

    # region MOVABLE

    # Start moving
    def start_move(
            self,
            start_value: Any
    ):
        """
        Start moving the MathTex object.

        Args:
            start_value (any): The start position of the object
        """
        self.start_position = self.position.copy()
    # end start_moving

    # Animate move
    def animate_move(self, t, duration, interpolated_t, env_value):
        """
        Animate moving the MathTex object.
        """
        # New x, y
        self.position.x = self.start_position.x * (1 - interpolated_t) + env_value.x * interpolated_t
        self.position.y = self.start_position.y * (1 - interpolated_t) + env_value.y * interpolated_t
    # end animate_move

    # endregion MOVABLE

    # region FADE_IN

    def start_fadein(self, start_value: Any):
        """
        Start fading in the MathTex object.
        """
        self.math_graphics.start_fadein(start_value)
    # end start_fadein

    def animate_fadein(self, t, duration, interpolated_t, end_value):
        """
        Animate fading in the MathTex object.
        """
        self.math_graphics.animate_fadein(t, duration, interpolated_t, end_value)
    # end animate_fadein

    # endregion FADE_IN

    # region FADE_OUT

    def start_fadeout(self, start_value: Any):
        """
        Start fading out the MathTex object.
        """
        self.math_graphics.start_fadeout(start_value)
    # end start_fadeout

    def animate_fadeout(self, t, duration, interpolated_t, end_value):
        """
        Animate fading out the MathTex object.
        """
        self.math_graphics.animate_fadeout(t, duration, interpolated_t, end_value)
    # end animate_fadeout

    # endregion FADE_OUT

    # region BUILD

    # Initialize building
    def init_build(self):
        """
        Initialize building the MathTex object.
        """
        self.math_graphics.init_build()
    # end init_build

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the MathTex object.
        """
        self.math_graphics.start_build(start_value)
    # end start_build

    # Animate building
    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the MathTex object.
        """
        self.math_graphics.animate_build(t, duration, interpolated_t, env_value)
    # end animate_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the MathTex object.
        """
        self.math_graphics.end_build(end_value)
    # end end_build

    # Finish building
    def finish_build(self):
        """
        Finish building the MathTex object.
        """
        self.math_graphics.finish_build()
    # end finish_build

    # endregion BUILD

    # region DESTROY

    # Initialize destroying
    def init_destroy(self):
        """
        Initialize destroying the MathTex object.
        """
        self.math_graphics.init_destroy()
    # end init_destroy

    # Start building
    def start_destroy(self, start_value: Any):
        """
        Start destroying the MathTex object.
        """
        self.math_graphics.start_destroy(start_value)
    # end start_destroy

    # Animate building
    def animate_destroy(self, t, duration, interpolated_t, env_value):
        """
        Animate destroying the MathTex object.
        """
        self.math_graphics.animate_destroy(t, duration, interpolated_t, env_value)
    # end animate_destroy

    # End building
    def end_destroy(self, end_value: Any):
        """
        End destroying the MathTex object.
        """
        self.math_graphics.end_destroy(end_value)
    # end end_destroy

    # Finish building
    def finish_destroy(self):
        """
        Finish destroying the MathTex object.
        """
        self.math_graphics.finish_destroy()
    # end finish_destroy

    # endregion DESTROY

# end MathTex
