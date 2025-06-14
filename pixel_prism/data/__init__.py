"""
Pixel Prism Data - Core Data Structures and Types
===============================================

This subpackage provides fundamental data structures and types used throughout the Pixel Prism library.
These include geometric primitives, color representations, transformation matrices, and utility classes
for data handling and event management.

Main Components
--------------
- Geometric Primitives:
  - :class:`~pixel_prism.data.points.Point2D`: 2D point representation
  - :class:`~pixel_prism.data.points.Point3D`: 3D point representation
  - :class:`~pixel_prism.data.points.TPoint2D`: 2D point with transformation
  - :class:`~pixel_prism.data.scalar.Scalar`: Scalar value representation
  - :class:`~pixel_prism.data.scalar.TScalar`: Scalar with transformation

- Matrices and Transformations:
  - :class:`~pixel_prism.data.matrices.Matrix2D`: 2D matrix representation
  - :class:`~pixel_prism.data.matrices.TMatrix2D`: Transformable 2D matrix
  - :class:`~pixel_prism.data.transform.Transform`: Transformation class

- Visual Styling:
  - :class:`~pixel_prism.data.colors.Color`: Color representation
  - :class:`~pixel_prism.data.style.Style`: Visual styling properties

- Data Management:
  - :class:`~pixel_prism.data.data.Data`: Base class for data objects
  - :class:`~pixel_prism.data.array_data.ArrayData`: Array-based data

- Event Handling:
  - :class:`~pixel_prism.data.events.Event`: Event class
  - :class:`~pixel_prism.data.events.EventType`: Event type enumeration

- Decorators:
  - :func:`~pixel_prism.data.decorators.call_before`: Call a function before a method
  - :func:`~pixel_prism.data.decorators.call_after`: Call a function after a method

These classes provide the core data structures and types used throughout the Pixel Prism library
for representing and manipulating geometric objects, colors, and other data.
"""

# Array data
from .array_data import ArrayData

# Data
from .data import Data

# Decorators
from .decorators import (
    call_before,
    call_after
)

# Events
from .events import (
    Event,
    EventType
)

# Colors
from .colors import (
    Color
)

# Matrices
from .matrices import (
    Matrix2D,
    TMatrix2D
)

# Points
from .points import (
    Point,
    Point2D,
    Point3D,
    TPoint2D,
    TPoint3D
)

# Scalar
from .scalar import (
    Scalar,
    TScalar
)

# Style
from .style import (
    Style
)

from .transform import (
    Transform
)

# ALL
__all__ = [
    # Array data
    "ArrayData",
    # Data
    "Data",
    # Decorators
    "call_after",
    "call_before",
    # Events
    "Event",
    "EventType",
    # Colors
    "Color",
    # Matrices
    "Matrix2D",
    "TMatrix2D",
    # Points
    "Point",
    "Point2D",
    "Point3D",
    "TPoint2D",
    "TPoint3D",
    # Scalar
    "Scalar",
    "TScalar",
    # Style
    "Style",
    # Transform
    "Transform"
]
