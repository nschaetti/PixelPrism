#
# This file contains the MathTex class, which is a widget that can be drawn
#

# Imports
from typing import Tuple, Any
import cairo

from pixel_prism.utils import render_latex_to_svg, draw_svg
from pixel_prism.animate.able import FadeInAble, FadeOutAble
from pixel_prism.data import Point2D, Scalar
from .element import Element


class MathTex(Element, FadeInAble, FadeOutAble):

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
        self.svg_path = 'latex.svg'
        self.update_svg()
    # end __init__

    def update_svg(self):
        """
        Update the svg file with the latex string.
        """
        render_latex_to_svg(self.latex, self.svg_path)
    # end update_svg

    def draw(self, context):
        """
        Draw the MathTex object to the context.

        Args:
            context (cairo.Context): Context to draw the MathTex object to
        """
        x, y = self.position.get()
        draw_svg(context, self.svg_path, x, y, color=self.color)
    # end draw

# end MathTex
