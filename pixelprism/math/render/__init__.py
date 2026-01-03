"""
Rendering utilities for symbolic math expressions.

This subpackage provides a clean split between converting expressions into
LaTeX strings and compiling LaTeX into vector graphics suitable for final
renders.
"""

from __future__ import annotations

import io
import os
import tempfile

import cairosvg
from PIL import Image

from pixelprism.utils.latex import render_latex_to_svg
from ..math_expr import MathExpr
from .cache import LatexRenderCache
from .latex import to_latex


__all__ = ["to_latex", "render_latex", "latex_to_svg", "LatexRenderCache"]


def latex_to_svg(latex: str, output_path: str) -> None:
    """
    Render a LaTeX math string to an SVG file.

    Parameters
    ----------
    latex :
        LaTeX string to compile. It is assumed to represent math mode content.
    output_path :
        Destination path for the generated SVG file.
    """
    render_latex_to_svg(latex, output_path)
# end def latex_to_svg


def render_latex(expr: MathExpr, output_path: str | None = None) -> Image.Image:
    """
    Convert ``expr`` to LaTeX, render it as SVG, and return a PIL image.

    Parameters
    ----------
    expr :
        Expression tree to render.
    output_path : str, optional
        Destination path for the SVG file. When ``None`` (default), a temporary
        file is created and removed after rendering.

    Returns
    -------
    PIL.Image.Image
        Image object containing the rendered SVG content.
    """
    temp_path: str | None = None
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        temp_file.close()
        temp_path = temp_file.name
        target_path = temp_path
    else:
        target_path = output_path
    # end if

    latex = to_latex(expr)
    latex_to_svg(latex, target_path)
    with open(target_path, "rb") as svg_file:
        svg_bytes = svg_file.read()
    # end with
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        scale=3,
        background_color="white"
    )
    with Image.open(io.BytesIO(png_bytes)) as img:
        rendered = img.copy()
    # end with

    if temp_path is not None:
        os.remove(temp_path)
    # end if
    return rendered
# end def render_latex
