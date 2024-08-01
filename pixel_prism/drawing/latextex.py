#
# This file contains the MathTex class, which is a widget that can be drawn
#

# Imports
import os
from typing import Tuple, Any
import cairo
import tempfile

from pixel_prism.utils import render_latex_to_svg, draw_svg
from pixel_prism.animate.able import FadeInAble, FadeOutAble
from pixel_prism.data import Point2D, VectorGraphics
from .drawable import Drawable


def generate_temp_svg_filename():
    """
    Generate a random filename for a temporary SVG file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    return temp_filename
# end generate_temp_svg_filename


class MathTex(Drawable, FadeInAble, FadeOutAble):

    def __init__(self, latex, position, color=(0, 0, 0), font_size=20):
        """
        Initialize the MathTex object.

        Args:
            latex (str): The latex string to render
            position (Point2D): The position of the MathTex object
            color (tuple): The color of the latex string
            font_size (int): The font size of the latex string
        """
        super().__init__()
        self.latex = latex
        self.position = position
        self.color = color
        self.font_size = font_size
        self.math_graphics = self.generate_vector_graphics()
    # end __init__

    # end generate_temp_svg_filename

    # Generate the vector graphics
    def generate_vector_graphics(self):
        """
        Generate the vector graphics for the MathTex object.
        """
        # Generate a random filename for the SVG file
        random_svg_path = generate_temp_svg_filename()

        # Generate the SVG file from the math string
        self.update_svg(random_svg_path)

        # Create the vector graphics object
        vector_graphics = VectorGraphics.from_svg(random_svg_path)

        # Delete the temporary SVG file
        os.remove(random_svg_path)

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
        x, y = self.position.get()
        # draw_svg(context, self.svg_path, x, y, color=self.color)
    # end draw

# end MathTex
