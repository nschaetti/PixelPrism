#
# This file contains the MathTex class, which is a widget that can be drawn
#

# Imports
from typing import List, Optional, Any
import os
import cairo
import tempfile

from pixel_prism.utils import render_latex_to_svg, draw_svg
from pixel_prism.animate.able import FadeInableMixin, FadeOutableMixin, MovableMixin
from pixel_prism.data import Point2D, Color
import pixel_prism.utils as utils
from .drawablemixin import DrawableMixin
from .vector_graphics import VectorGraphics


def generate_temp_svg_filename():
    """
    Generate a random filename for a temporary SVG file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
    temp_filename = temp_file.name
    print(temp_filename)
    temp_file.close()
    return temp_filename
# end generate_temp_svg_filename


class MathTex(DrawableMixin, MovableMixin, FadeInableMixin, FadeOutableMixin):

    def __init__(
            self,
            latex,
            position,
            scale: Point2D = Point2D(1, 1),
            color: Color = utils.WHITE,
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

# end MathTex
