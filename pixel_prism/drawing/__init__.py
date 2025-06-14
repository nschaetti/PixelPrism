"""
Pixel Prism Drawing - Vector Graphics and Shape Drawing
====================================================

This subpackage provides classes for creating and manipulating vector graphics,
geometric shapes, and paths in the Pixel Prism library. It includes functionality
for drawing basic shapes, complex paths, and applying transformations.

Main Components
--------------
- Basic Shapes:
  - :class:`~pixel_prism.drawing.rectangles.Rectangle`: Rectangle shape
  - :class:`~pixel_prism.drawing.circle.Circle`: Circle shape
  - :class:`~pixel_prism.drawing.lines.Line`: Line shape
  - :class:`~pixel_prism.drawing.arcs.Arc`: Arc shape

- Curves and Paths:
  - :class:`~pixel_prism.drawing.curves.CubicBezierCurve`: Cubic Bezier curve
  - :class:`~pixel_prism.drawing.curves.QuadraticBezierCurve`: Quadratic Bezier curve
  - :class:`~pixel_prism.drawing.paths.Path`: Complex path composed of segments
  - :class:`~pixel_prism.drawing.paths.PathSegment`: Base class for path segments
  - :class:`~pixel_prism.drawing.paths.PathLine`: Line segment in a path
  - :class:`~pixel_prism.drawing.paths.PathBezierCubic`: Cubic Bezier segment in a path
  - :class:`~pixel_prism.drawing.paths.PathBezierQuadratic`: Quadratic Bezier segment in a path
  - :class:`~pixel_prism.drawing.paths.PathArc`: Arc segment in a path

- Transformations:
  - :class:`~pixel_prism.drawing.transforms.Translate2D`: 2D translation
  - :class:`~pixel_prism.drawing.transforms.Scale2D`: 2D scaling
  - :class:`~pixel_prism.drawing.transforms.Rotate2D`: 2D rotation
  - :class:`~pixel_prism.drawing.transforms.SkewX2D`: X-axis skew
  - :class:`~pixel_prism.drawing.transforms.SkewY2D`: Y-axis skew
  - :class:`~pixel_prism.drawing.transforms.Matrix2D`: 2D transformation matrix

- Utilities:
  - :class:`~pixel_prism.drawing.bounding_box.BoundingBox`: Bounding box for shapes
  - :class:`~pixel_prism.drawing.boundingboxmixin.BoundingBoxMixin`: Mixin for objects with bounding boxes
  - :class:`~pixel_prism.drawing.drawablemixin.DrawableMixin`: Mixin for drawable objects
  - :class:`~pixel_prism.drawing.debug_grid.DebugGrid`: Grid for debugging
  - :class:`~pixel_prism.drawing.mathtex.MathTex`: Mathematical text rendering
  - :class:`~pixel_prism.drawing.vector_graphics.VectorGraphics`: Vector graphics container

These classes provide the foundation for creating and manipulating vector graphics
and geometric shapes in the Pixel Prism library.
"""

# Imports
from .arcs import Arc
from .bounding_box import BoundingBox
from .boundingboxmixin import BoundingBoxMixin
from .circle import Circle
from .curves import CubicBezierCurve, QuadraticBezierCurve
from .debug_grid import DebugGrid
from .drawablemixin import DrawableMixin
from .lines import Line
from .mathtex import MathTex
from .paths import (
    Path,
    PathSegment,
    PathLine,
    PathBezierCubic,
    PathBezierQuadratic,
    PathArc
)
from .rectangles import Rectangle
from .transforms import (
    Translate2D,
    Scale2D,
    Rotate2D,
    SkewX2D,
    SkewY2D,
    Matrix2D
)
from .vector_graphics import VectorGraphics

# ALL
__all__ = [
    "Arc",
    "BoundingBox",
    "Circle",
    "CubicBezierCurve",
    "QuadraticBezierCurve",
    "Line",
    "MathTex",
    "DebugGrid",
    # Minxin
    "BoundingBoxMixin",
    "DrawableMixin",
    # Path
    "PathLine",
    "Path",
    "PathSegment",
    "PathBezierCubic",
    "PathBezierQuadratic",
    "PathArc",
    # Rectangle
    "Rectangle",
    "Translate2D",
    "Scale2D",
    "Rotate2D",
    "SkewX2D",
    "SkewY2D",
    "Matrix2D",
    "VectorGraphics"
]
