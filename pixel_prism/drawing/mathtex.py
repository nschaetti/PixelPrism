#
# This file contains the MathTex class, which is a widget that can be drawn
#

# Imports
import os
import cairo
import tempfile

from pixel_prism.utils import render_latex_to_svg, draw_svg
from pixel_prism.animate.able import FadeInAble, FadeOutAble
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
    temp_file.close()
    return temp_filename
# end generate_temp_svg_filename


class MathTex(DrawableMixin, FadeInAble, FadeOutAble):

    def __init__(
            self,
            latex,
            position,
            scale: Point2D = Point2D(1, 1),
            color: Color = utils.WHITE,
            font_size=20
    ):
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
        self.scale = scale
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
        vector_graphics = VectorGraphics.from_svg(
            random_svg_path,
            scale=self.scale,
            color=self.color
        )

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
        # Get the position
        x, y = self.position.get()

        # Translate the context
        context.save()
        context.translate(x, y)

        # draw_svg(context, self.svg_path, x, y, color=self.color)
        # Draw the vector graphics
        self.math_graphics.draw(context)

        # Restore the context
        context.restore()
    # end draw

# end MathTex
