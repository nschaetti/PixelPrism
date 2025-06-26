"""
Pixel Prism Drawing - Vector Graphics and Shape Drawing
====================================================

This subpackage provides classes for creating and manipulating vector graphics,
geometric shapes, and paths in the Pixel Prism library. It includes functionality
for drawing basic shapes, complex paths, and applying transformations.

Main Components
--------------
- Basic Shapes:
  - :class:`~pixelprism.drawing.rectangles.Rectangle`: Rectangle shape
  - :class:`~pixelprism.drawing.circle.Circle`: Circle shape
  - :class:`~pixelprism.drawing.lines.Line`: Line shape
  - :class:`~pixelprism.drawing.arcs.Arc`: Arc shape

- Curves and Paths:
  - :class:`~pixelprism.drawing.curves.CubicBezierCurve`: Cubic Bezier curve
  - :class:`~pixelprism.drawing.curves.QuadraticBezierCurve`: Quadratic Bezier curve
  - :class:`~pixelprism.drawing.paths.Path`: Complex path composed of segments
  - :class:`~pixelprism.drawing.paths.PathSegment`: Base class for path segments
  - :class:`~pixelprism.drawing.paths.PathLine`: Line segment in a path
  - :class:`~pixelprism.drawing.paths.PathBezierCubic`: Cubic Bezier segment in a path
  - :class:`~pixelprism.drawing.paths.PathBezierQuadratic`: Quadratic Bezier segment in a path
  - :class:`~pixelprism.drawing.paths.PathArc`: Arc segment in a path

- Transformations:
  - :class:`~pixelprism.drawing.transforms.Translate2D`: 2D translation
  - :class:`~pixelprism.drawing.transforms.Scale2D`: 2D scaling
  - :class:`~pixelprism.drawing.transforms.Rotate2D`: 2D rotation
  - :class:`~pixelprism.drawing.transforms.SkewX2D`: X-axis skew
  - :class:`~pixelprism.drawing.transforms.SkewY2D`: Y-axis skew
  - :class:`~pixelprism.drawing.transforms.Matrix2D`: 2D transformation matrix

- Utilities:
  - :class:`~pixelprism.drawing.bounding_box.BoundingBox`: Bounding box for shapes
  - :class:`~pixelprism.drawing.boundingboxmixin.BoundingBoxMixin`: Mixin for objects with bounding boxes
  - :class:`~pixelprism.drawing.drawablemixin.DrawableMixin`: Mixin for drawable objects
  - :class:`~pixelprism.drawing.debug_grid.DebugGrid`: Grid for debugging
  - :class:`~pixelprism.drawing.mathtex.MathTex`: Mathematical text rendering
  - :class:`~pixelprism.drawing.vector_graphics.VectorGraphics`: Vector graphics container

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
