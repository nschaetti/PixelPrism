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

"""
Pixel Prism Data - Core Data Structures and Types
===============================================

This subpackage provides fundamental math structures and types used throughout the Pixel Prism library.
These include geometric primitives, color representations, transformation matrices, and utility classes
for math handling and event management.

Main Components
--------------
- Geometric Primitives:
  - :class:`~pixelprism.math.points.Point2D`: 2D point representation
  - :class:`~pixelprism.math.points.Point3D`: 3D point representation
  - :class:`~pixelprism.math.points.TPoint2D`: 2D point with transformation
  - :class:`~pixelprism.math.scalar.Scalar`: Scalar value representation
  - :class:`~pixelprism.math.scalar.TScalar`: Scalar with transformation

- Matrices and Transformations:
  - :class:`~pixelprism.math.matrices.Matrix2D`: 2D matrix representation
  - :class:`~pixelprism.math.matrices.TMatrix2D`: Transformable 2D matrix
  - :class:`~pixelprism.math.transform.Transform`: Transformation class

- Visual Styling:
  - :class:`~pixelprism.math.colors.Color`: Color representation
  - :class:`~pixelprism.math.style.Style`: Visual styling properties

- Data Management:
  - :class:`~pixelprism.math.math.Data`: Base class for math objects
  - :class:`~pixelprism.math.array_data.ArrayData`: Array-based math

- Event Handling:
  - :class:`~pixelprism.math.events.Event`: Event class
  - :class:`~pixelprism.math.events.EventType`: Event type enumeration

- Decorators:
  - :func:`~pixelprism.math.decorators.call_before`: Call a function before a method
  - :func:`~pixelprism.math.decorators.call_after`: Call a function after a method

These classes provide the core math structures and types used throughout the Pixel Prism library
for representing and manipulating geometric objects, colors, and other math.
"""

# Array math
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
    # Array math
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
