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
# Copyright (C) 2025 Pixel Prism
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


"""Symbolic 2D geometry helpers built on top of :mod:`pixelprism.math`.

The classes in this module are lightweight wrappers around scalar
``MathExpr`` instances so they compose naturally with variables, constants,
and operator trees already used by the animation engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union, cast

from . import as_expr
from .dtype import DType
from .math_leaves import var
from .typing import MathExpr
from .functional import builders as FB
from .functional import elementwise as EL
from .functional import trigo as TR


__all__ = [
    "ScalarIn",
    "Point1",
    "Point2",
    "Point3",
    "Line2",
    "Line3",
    "Affine2",
    "Affine3",
    "Circle2",
    "Ellipse2",
]


ScalarIn = Union[MathExpr, int, float]


def _as_scalar(expr: ScalarIn) -> MathExpr:
    """Convert ``expr`` to a scalar symbolic expression."""
    out = as_expr(expr)
    if out.shape.rank != 0:
        raise TypeError(f"Expected scalar expression, got shape {out.shape}.")
    # end if
    return out
# end def _as_scalar


def _add(a: ScalarIn, b: ScalarIn) -> MathExpr:
    return EL.add(cast(Any, a), cast(Any, b))
# end def _add


def _sub(a: ScalarIn, b: ScalarIn) -> MathExpr:
    return EL.sub(cast(Any, a), cast(Any, b))
# end def _sub


def _mul(a: ScalarIn, b: ScalarIn) -> MathExpr:
    return EL.mul(cast(Any, a), cast(Any, b))
# end def _mul


def _div(a: ScalarIn, b: ScalarIn) -> MathExpr:
    return EL.div(cast(Any, a), cast(Any, b))
# end def _div


def _neg(a: ScalarIn) -> MathExpr:
    return EL.neg(cast(Any, a))
# end def _neg


def _sqrt(a: ScalarIn) -> MathExpr:
    return EL.sqrt(cast(Any, a))
# end def _sqrt


def _sin(a: ScalarIn) -> MathExpr:
    return TR.sin(cast(Any, a))
# end def _sin


def _cos(a: ScalarIn) -> MathExpr:
    return TR.cos(cast(Any, a))
# end def _cos


def _atan2(y: ScalarIn, x: ScalarIn) -> MathExpr:
    return TR.atan2(cast(Any, y), cast(Any, x))
# end def _atan2


def _vector2(x: ScalarIn, y: ScalarIn) -> MathExpr:
    return FB.vector([cast(Any, x), cast(Any, y)])
# end def _vector2


def _vector1(x: ScalarIn) -> MathExpr:
    return FB.vector([cast(Any, x)])
# end def _vector1


def _vector3(x: ScalarIn, y: ScalarIn, z: ScalarIn) -> MathExpr:
    return FB.vector([cast(Any, x), cast(Any, y), cast(Any, z)])
# end def _vector3


def _matrix(rows) -> MathExpr:
    return FB.matrix(cast(Any, rows))
# end def _matrix


@dataclass(frozen=True)
class Point1:
    """Symbolic 1D point.

    Parameters
    ----------
    x : MathExpr or int or float
        Scalar coordinate.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> from pixelprism.math import Point1
    >>> p = Point1(2.0)
    >>> p.eval().value
    array([2.])
    """

    x: MathExpr

    def __init__(self, x: ScalarIn):
        """Create a 1D point from a scalar symbolic value.

        Parameters
        ----------
        x : MathExpr or int or float
            Coordinate value.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> Point1(1.5)
        Point1(x=...)
        """
        object.__setattr__(self, "x", _as_scalar(x))
    # end def __init__

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Point1":
        """Create a variable point with deterministic naming.

        Parameters
        ----------
        prefix : str
            Variable name prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Point1
            Variable point using ``{prefix}.x``.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> p = Point1.vars("cursor")
        >>> p.x.name
        'cursor.x'
        """
        return Point1(var(f"{prefix}.x", dtype=dtype, shape=()))
    # end def vars

    def to_vector(self) -> MathExpr:
        """Return the point as a rank-1 symbolic vector.

        Returns
        -------
        MathExpr
            Expression with shape ``(1,)``.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> Point1(3.0).to_vector().shape.dims
        (1,)
        """
        return _vector1(self.x)
    # end def to_vector

    def eval(self):
        """Evaluate point coordinates.

        Returns
        -------
        Tensor
            Tensor with shape ``(1,)``.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> Point1(3.0).eval().value
        array([3.])
        """
        return self.to_vector().eval()
    # end def eval

    def translated(self, dx: ScalarIn) -> "Point1":
        """Return a translated point.

        Parameters
        ----------
        dx : MathExpr or int or float
            Translation offset.

        Returns
        -------
        Point1
            Translated point.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> Point1(1.0).translated(2.0).eval().value
        array([3.])
        """
        return Point1(_add(self.x, _as_scalar(dx)))
    # end def translated

    def scaled(self, sx: ScalarIn, center: Optional["Point1"] = None) -> "Point1":
        """Scale the point, optionally around a center.

        Parameters
        ----------
        sx : MathExpr or int or float
            Scale factor.
        center : Point1, optional
            Center of scaling. Defaults to origin when omitted.

        Returns
        -------
        Point1
            Scaled point.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> Point1(3.0).scaled(2.0).eval().value
        array([6.])
        """
        factor = _as_scalar(sx)
        if center is None:
            return Point1(_mul(factor, self.x))
        # end if
        return Point1(_add(center.x, _mul(factor, _sub(self.x, center.x))))
    # end def scaled

    def distance_sq(self, other: "Point1") -> MathExpr:
        """Return squared distance to another 1D point.

        Parameters
        ----------
        other : Point1
            Target point.

        Returns
        -------
        MathExpr
            Squared distance expression.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> Point1(1.0).distance_sq(Point1(4.0)).eval().item()
        9.0
        """
        dx = _sub(self.x, other.x)
        return _mul(dx, dx)
    # end def distance_sq

    def distance(self, other: "Point1") -> MathExpr:
        """Return distance to another 1D point.

        Parameters
        ----------
        other : Point1
            Target point.

        Returns
        -------
        MathExpr
            Distance expression.

        Examples
        --------
        >>> from pixelprism.math import Point1
        >>> Point1(1.0).distance(Point1(4.0)).eval().item()
        3.0
        """
        return _sqrt(self.distance_sq(other))
    # end def distance

# end class Point1


@dataclass(frozen=True)
class Point2:
    """Symbolic 2D point.

    Parameters
    ----------
    x : MathExpr or int or float
        X coordinate.
    y : MathExpr or int or float
        Y coordinate.

    Examples
    --------
    >>> from pixelprism.math import Point2
    >>> Point2(1.0, 2.0).eval().value
    array([1., 2.])
    """

    x: MathExpr
    y: MathExpr

    def __init__(self, x: ScalarIn, y: ScalarIn):
        """Create a 2D point from scalar symbolic values.

        Parameters
        ----------
        x : MathExpr or int or float
            X coordinate.
        y : MathExpr or int or float
            Y coordinate.
        """
        object.__setattr__(self, "x", _as_scalar(x))
        object.__setattr__(self, "y", _as_scalar(y))
    # end def __init__

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Point2":
        """Create a variable 2D point.

        Parameters
        ----------
        prefix : str
            Variable name prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Point2
            Variable point using ``{prefix}.x`` and ``{prefix}.y``.

        Examples
        --------
        >>> from pixelprism.math import Point2
        >>> p = Point2.vars("p")
        >>> (p.x.name, p.y.name)
        ('p.x', 'p.y')
        """
        return Point2(
            var(f"{prefix}.x", dtype=dtype, shape=()),
            var(f"{prefix}.y", dtype=dtype, shape=()),
        )
    # end def vars

    def to_vector(self) -> MathExpr:
        """Return the point as a rank-1 symbolic vector.

        Returns
        -------
        MathExpr
            Expression with shape ``(2,)``.

        Examples
        --------
        >>> from pixelprism.math import Point2
        >>> Point2(0.0, 1.0).to_vector().shape.dims
        (2,)
        """
        return _vector2(self.x, self.y)
    # end def to_vector

    def eval(self):
        """Evaluate point coordinates.

        Returns
        -------
        Tensor
            Tensor with shape ``(2,)``.

        Examples
        --------
        >>> from pixelprism.math import Point2
        >>> Point2(1.0, 2.0).eval().shape.dims
        (2,)
        """
        return self.to_vector().eval()
    # end def eval

    def translated(self, dx: ScalarIn, dy: ScalarIn) -> "Point2":
        """Return a translated 2D point.

        Parameters
        ----------
        dx : MathExpr or int or float
            X offset.
        dy : MathExpr or int or float
            Y offset.

        Returns
        -------
        Point2
            Translated point.

        Examples
        --------
        >>> from pixelprism.math import Point2
        >>> Point2(1.0, 2.0).translated(3.0, -1.0).eval().value
        array([4., 1.])
        """
        return Point2(
            _add(self.x, _as_scalar(dx)),
            _add(self.y, _as_scalar(dy)),
        )
    # end def translated

    def scaled(
            self,
            sx: ScalarIn,
            sy: Optional[ScalarIn] = None,
            center: Optional["Point2"] = None
    ) -> "Point2":
        """Scale the point, optionally around a center.

        Parameters
        ----------
        sx : MathExpr or int or float
            Scale on X.
        sy : MathExpr or int or float, optional
            Scale on Y. Uses ``sx`` when omitted.
        center : Point2, optional
            Scaling center.

        Returns
        -------
        Point2
            Scaled point.

        Examples
        --------
        >>> from pixelprism.math import Point2
        >>> Point2(2.0, 3.0).scaled(2.0).eval().value
        array([4., 6.])
        """
        transform = Affine2.scale(sx=sx, sy=sy, center=center)
        return transform.apply_point(self)
    # end def scaled

    def rotated(self, angle: ScalarIn, center: Optional["Point2"] = None) -> "Point2":
        """Rotate the point by an angle in radians.

        Parameters
        ----------
        angle : MathExpr or int or float
            Rotation angle in radians.
        center : Point2, optional
            Rotation center. Uses origin when omitted.

        Returns
        -------
        Point2
            Rotated point.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Point2
        >>> Point2(1.0, 0.0).rotated(np.pi / 2).eval().shape.dims
        (2,)
        """
        transform = Affine2.rotation(theta=angle, center=center)
        return transform.apply_point(self)
    # end def rotated

    def distance_sq(self, other: "Point2") -> MathExpr:
        """Return squared Euclidean distance.

        Parameters
        ----------
        other : Point2
            Target point.

        Returns
        -------
        MathExpr
            Squared distance expression.

        Examples
        --------
        >>> from pixelprism.math import Point2
        >>> Point2(0.0, 0.0).distance_sq(Point2(3.0, 4.0)).eval().item()
        25.0
        """
        dx = _sub(self.x, other.x)
        dy = _sub(self.y, other.y)
        return _add(_mul(dx, dx), _mul(dy, dy))
    # end def distance_sq

    def distance(self, other: "Point2") -> MathExpr:
        """Return Euclidean distance.

        Parameters
        ----------
        other : Point2
            Target point.

        Returns
        -------
        MathExpr
            Distance expression.

        Examples
        --------
        >>> from pixelprism.math import Point2
        >>> Point2(0.0, 0.0).distance(Point2(3.0, 4.0)).eval().item()
        5.0
        """
        return _sqrt(self.distance_sq(other))
    # end def distance

# end class Point2


@dataclass(frozen=True)
class Point3:
    """Symbolic 3D point.

    Parameters
    ----------
    x : MathExpr or int or float
        X coordinate.
    y : MathExpr or int or float
        Y coordinate.
    z : MathExpr or int or float
        Z coordinate.

    Examples
    --------
    >>> from pixelprism.math import Point3
    >>> Point3(1.0, 2.0, 3.0).eval().shape.dims
    (3,)
    """

    x: MathExpr
    y: MathExpr
    z: MathExpr

    def __init__(self, x: ScalarIn, y: ScalarIn, z: ScalarIn):
        """Create a 3D point from scalar symbolic values.

        Parameters
        ----------
        x : MathExpr or int or float
            X coordinate.
        y : MathExpr or int or float
            Y coordinate.
        z : MathExpr or int or float
            Z coordinate.
        """
        object.__setattr__(self, "x", _as_scalar(x))
        object.__setattr__(self, "y", _as_scalar(y))
        object.__setattr__(self, "z", _as_scalar(z))
    # end def __init__

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Point3":
        """Create a variable 3D point.

        Parameters
        ----------
        prefix : str
            Variable name prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Point3
            Variable point using ``{prefix}.x/.y/.z``.

        Examples
        --------
        >>> from pixelprism.math import Point3
        >>> p = Point3.vars("cam")
        >>> p.z.name
        'cam.z'
        """
        return Point3(
            var(f"{prefix}.x", dtype=dtype, shape=()),
            var(f"{prefix}.y", dtype=dtype, shape=()),
            var(f"{prefix}.z", dtype=dtype, shape=()),
        )
    # end def vars

    def to_vector(self) -> MathExpr:
        """Return the point as a rank-1 symbolic vector.

        Returns
        -------
        MathExpr
            Expression with shape ``(3,)``.

        Examples
        --------
        >>> from pixelprism.math import Point3
        >>> Point3(1.0, 2.0, 3.0).to_vector().shape.dims
        (3,)
        """
        return _vector3(self.x, self.y, self.z)
    # end def to_vector

    def eval(self):
        """Evaluate point coordinates.

        Returns
        -------
        Tensor
            Tensor with shape ``(3,)``.

        Examples
        --------
        >>> from pixelprism.math import Point3
        >>> Point3(1.0, 2.0, 3.0).eval().shape.dims
        (3,)
        """
        return self.to_vector().eval()
    # end def eval

    def translated(self, dx: ScalarIn, dy: ScalarIn, dz: ScalarIn) -> "Point3":
        """Return a translated 3D point.

        Parameters
        ----------
        dx : MathExpr or int or float
            X offset.
        dy : MathExpr or int or float
            Y offset.
        dz : MathExpr or int or float
            Z offset.

        Returns
        -------
        Point3
            Translated point.

        Examples
        --------
        >>> from pixelprism.math import Point3
        >>> Point3(1.0, 2.0, 3.0).translated(1.0, -1.0, 2.0).eval().value
        array([2., 1., 5.])
        """
        return Point3(
            _add(self.x, _as_scalar(dx)),
            _add(self.y, _as_scalar(dy)),
            _add(self.z, _as_scalar(dz)),
        )
    # end def translated

    def scaled(
            self,
            sx: ScalarIn,
            sy: Optional[ScalarIn] = None,
            sz: Optional[ScalarIn] = None,
            center: Optional["Point3"] = None
    ) -> "Point3":
        """Scale the point, optionally around a center.

        Parameters
        ----------
        sx : MathExpr or int or float
            Scale on X.
        sy : MathExpr or int or float, optional
            Scale on Y. Uses ``sx`` when omitted.
        sz : MathExpr or int or float, optional
            Scale on Z. Uses ``sx`` when omitted.
        center : Point3, optional
            Scaling center.

        Returns
        -------
        Point3
            Scaled point.

        Examples
        --------
        >>> from pixelprism.math import Point3
        >>> Point3(1.0, 2.0, 3.0).scaled(2.0).eval().shape.dims
        (3,)
        """
        transform = Affine3.scale(sx=sx, sy=sy, sz=sz, center=center)
        return transform.apply_point(self)
    # end def scaled

    def rotated_x(self, angle: ScalarIn, center: Optional["Point3"] = None) -> "Point3":
        """Rotate the point around the X axis.

        Parameters
        ----------
        angle : MathExpr or int or float
            Rotation angle in radians.
        center : Point3, optional
            Rotation center.

        Returns
        -------
        Point3
            Rotated point.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Point3
        >>> Point3(0.0, 1.0, 0.0).rotated_x(np.pi / 2).eval().shape.dims
        (3,)
        """
        return Affine3.rotation_x(theta=angle, center=center).apply_point(self)
    # end def rotated_x

    def rotated_y(self, angle: ScalarIn, center: Optional["Point3"] = None) -> "Point3":
        """Rotate the point around the Y axis.

        Parameters
        ----------
        angle : MathExpr or int or float
            Rotation angle in radians.
        center : Point3, optional
            Rotation center.

        Returns
        -------
        Point3
            Rotated point.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Point3
        >>> Point3(1.0, 0.0, 0.0).rotated_y(np.pi / 2).eval().shape.dims
        (3,)
        """
        return Affine3.rotation_y(theta=angle, center=center).apply_point(self)
    # end def rotated_y

    def rotated_z(self, angle: ScalarIn, center: Optional["Point3"] = None) -> "Point3":
        """Rotate the point around the Z axis.

        Parameters
        ----------
        angle : MathExpr or int or float
            Rotation angle in radians.
        center : Point3, optional
            Rotation center.

        Returns
        -------
        Point3
            Rotated point.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Point3
        >>> Point3(1.0, 0.0, 0.0).rotated_z(np.pi / 2).eval().shape.dims
        (3,)
        """
        return Affine3.rotation_z(theta=angle, center=center).apply_point(self)
    # end def rotated_z

    def distance_sq(self, other: "Point3") -> MathExpr:
        """Return squared Euclidean distance.

        Parameters
        ----------
        other : Point3
            Target point.

        Returns
        -------
        MathExpr
            Squared distance expression.

        Examples
        --------
        >>> from pixelprism.math import Point3
        >>> Point3(0.0, 0.0, 0.0).distance_sq(Point3(1.0, 2.0, 2.0)).eval().item()
        9.0
        """
        dx = _sub(self.x, other.x)
        dy = _sub(self.y, other.y)
        dz = _sub(self.z, other.z)
        return _add(_add(_mul(dx, dx), _mul(dy, dy)), _mul(dz, dz))
    # end def distance_sq

    def distance(self, other: "Point3") -> MathExpr:
        """Return Euclidean distance.

        Parameters
        ----------
        other : Point3
            Target point.

        Returns
        -------
        MathExpr
            Distance expression.

        Examples
        --------
        >>> from pixelprism.math import Point3
        >>> Point3(0.0, 0.0, 0.0).distance(Point3(1.0, 2.0, 2.0)).eval().item()
        3.0
        """
        return _sqrt(self.distance_sq(other))
    # end def distance

# end class Point3


@dataclass(frozen=True)
class Line2:
    """Symbolic 2D line segment.

    Parameters
    ----------
    start : Point2
        Segment start point.
    end : Point2
        Segment end point.

    Examples
    --------
    >>> from pixelprism.math import Point2, Line2
    >>> line = Line2(Point2(0.0, 0.0), Point2(2.0, 0.0))
    >>> line.point_at(0.5).eval().value
    array([1., 0.])
    """

    start: Point2
    end: Point2

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Line2":
        """Create a variable line with named endpoints.

        Parameters
        ----------
        prefix : str
            Variable name prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Line2
            Variable line using ``{prefix}.start.*`` and ``{prefix}.end.*``.

        Examples
        --------
        >>> from pixelprism.math import Line2
        >>> line = Line2.vars("edge")
        >>> line.start.x.name
        'edge.start.x'
        """
        return Line2(
            start=Point2.vars(f"{prefix}.start", dtype=dtype),
            end=Point2.vars(f"{prefix}.end", dtype=dtype),
        )
    # end def vars

    def direction(self) -> Point2:
        """Return segment direction ``end - start``.

        Returns
        -------
        Point2
            Direction vector encoded as point-like pair.

        Examples
        --------
        >>> from pixelprism.math import Point2, Line2
        >>> Line2(Point2(0.0, 0.0), Point2(2.0, 1.0)).direction().eval().value
        array([2., 1.])
        """
        return Point2(
            _sub(self.end.x, self.start.x),
            _sub(self.end.y, self.start.y),
        )
    # end def direction

    def point_at(self, t: ScalarIn) -> Point2:
        """Return point on the segment parametric line.

        Parameters
        ----------
        t : MathExpr or int or float
            Parameter value. ``0`` gives ``start``, ``1`` gives ``end``.

        Returns
        -------
        Point2
            Point at ``start + t * (end - start)``.

        Examples
        --------
        >>> from pixelprism.math import Point2, Line2
        >>> Line2(Point2(0.0, 0.0), Point2(2.0, 0.0)).point_at(0.5).eval().value
        array([1., 0.])
        """
        tau = _as_scalar(t)
        return Point2(
            _add(self.start.x, _mul(tau, _sub(self.end.x, self.start.x))),
            _add(self.start.y, _mul(tau, _sub(self.end.y, self.start.y))),
        )
    # end def point_at

    def length_sq(self) -> MathExpr:
        """Return squared segment length.

        Returns
        -------
        MathExpr
            Squared length expression.

        Examples
        --------
        >>> from pixelprism.math import Point2, Line2
        >>> Line2(Point2(0.0, 0.0), Point2(3.0, 4.0)).length_sq().eval().item()
        25.0
        """
        return self.start.distance_sq(self.end)
    # end def length_sq

    def length(self) -> MathExpr:
        """Return segment length.

        Returns
        -------
        MathExpr
            Length expression.

        Examples
        --------
        >>> from pixelprism.math import Point2, Line2
        >>> Line2(Point2(0.0, 0.0), Point2(3.0, 4.0)).length().eval().item()
        5.0
        """
        return self.start.distance(self.end)
    # end def length

    def transformed(self, affine: "Affine2") -> "Line2":
        """Transform both endpoints with an affine map.

        Parameters
        ----------
        affine : Affine2
            Transform to apply.

        Returns
        -------
        Line2
            Transformed line segment.

        Examples
        --------
        >>> from pixelprism.math import Point2, Line2, Affine2
        >>> line = Line2(Point2(0.0, 0.0), Point2(1.0, 0.0))
        >>> line.transformed(Affine2.translation(1.0, 2.0)).start.eval().value
        array([1., 2.])
        """
        return affine.apply_line(self)
    # end def transformed

# end class Line2


@dataclass(frozen=True)
class Line3:
    """Symbolic 3D line segment.

    Parameters
    ----------
    start : Point3
        Segment start point.
    end : Point3
        Segment end point.

    Examples
    --------
    >>> from pixelprism.math import Point3, Line3
    >>> line = Line3(Point3(0.0, 0.0, 0.0), Point3(1.0, 0.0, 0.0))
    >>> line.point_at(0.5).eval().value
    array([0.5, 0. , 0. ])
    """

    start: Point3
    end: Point3

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Line3":
        """Create a variable line with named endpoints.

        Parameters
        ----------
        prefix : str
            Variable name prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Line3
            Variable line.

        Examples
        --------
        >>> from pixelprism.math import Line3
        >>> line = Line3.vars("seg")
        >>> line.end.z.name
        'seg.end.z'
        """
        return Line3(
            start=Point3.vars(f"{prefix}.start", dtype=dtype),
            end=Point3.vars(f"{prefix}.end", dtype=dtype),
        )
    # end def vars

    def direction(self) -> Point3:
        """Return segment direction ``end - start``.

        Returns
        -------
        Point3
            Direction vector encoded as point-like triple.

        Examples
        --------
        >>> from pixelprism.math import Point3, Line3
        >>> Line3(Point3(0.0, 0.0, 0.0), Point3(0.0, 1.0, 2.0)).direction().eval().value
        array([0., 1., 2.])
        """
        return Point3(
            _sub(self.end.x, self.start.x),
            _sub(self.end.y, self.start.y),
            _sub(self.end.z, self.start.z),
        )
    # end def direction

    def point_at(self, t: ScalarIn) -> Point3:
        """Return point on the parametric line.

        Parameters
        ----------
        t : MathExpr or int or float
            Parameter value.

        Returns
        -------
        Point3
            Point at ``start + t * (end - start)``.

        Examples
        --------
        >>> from pixelprism.math import Point3, Line3
        >>> Line3(Point3(0.0, 0.0, 0.0), Point3(2.0, 0.0, 0.0)).point_at(0.5).eval().value
        array([1., 0., 0.])
        """
        tau = _as_scalar(t)
        return Point3(
            _add(self.start.x, _mul(tau, _sub(self.end.x, self.start.x))),
            _add(self.start.y, _mul(tau, _sub(self.end.y, self.start.y))),
            _add(self.start.z, _mul(tau, _sub(self.end.z, self.start.z))),
        )
    # end def point_at

    def length_sq(self) -> MathExpr:
        """Return squared segment length.

        Returns
        -------
        MathExpr
            Squared length expression.

        Examples
        --------
        >>> from pixelprism.math import Point3, Line3
        >>> Line3(Point3(0.0, 0.0, 0.0), Point3(1.0, 2.0, 2.0)).length_sq().eval().item()
        9.0
        """
        return self.start.distance_sq(self.end)
    # end def length_sq

    def length(self) -> MathExpr:
        """Return segment length.

        Returns
        -------
        MathExpr
            Length expression.

        Examples
        --------
        >>> from pixelprism.math import Point3, Line3
        >>> Line3(Point3(0.0, 0.0, 0.0), Point3(1.0, 2.0, 2.0)).length().eval().item()
        3.0
        """
        return self.start.distance(self.end)
    # end def length

    def transformed(self, affine: "Affine3") -> "Line3":
        """Transform both endpoints with an affine map.

        Parameters
        ----------
        affine : Affine3
            Transform to apply.

        Returns
        -------
        Line3
            Transformed line segment.

        Examples
        --------
        >>> from pixelprism.math import Point3, Line3, Affine3
        >>> line = Line3(Point3(0.0, 0.0, 0.0), Point3(1.0, 0.0, 0.0))
        >>> line.transformed(Affine3.translation(1.0, 2.0, 3.0)).start.eval().value
        array([1., 2., 3.])
        """
        return affine.apply_line(self)
    # end def transformed

# end class Line3


@dataclass(frozen=True)
class Affine2:
    """Symbolic 2D affine transform represented by 2x3 coefficients.

    The mapping is:

    ``x' = a*x + b*y + tx``
    ``y' = c*x + d*y + ty``

    Examples
    --------
    >>> from pixelprism.math import Point2, Affine2
    >>> t = Affine2.translation(2.0, -1.0)
    >>> t.apply_point(Point2(1.0, 2.0)).eval().value
    array([3., 1.])
    """

    a: MathExpr
    b: MathExpr
    tx: MathExpr
    c: MathExpr
    d: MathExpr
    ty: MathExpr

    def __init__(
            self,
            a: ScalarIn,
            b: ScalarIn,
            tx: ScalarIn,
            c: ScalarIn,
            d: ScalarIn,
            ty: ScalarIn
    ):
        """Create an affine transform from 2x3 scalar coefficients.

        Parameters
        ----------
        a, b, tx, c, d, ty : MathExpr or int or float
            Coefficients of the affine transform.
        """
        object.__setattr__(self, "a", _as_scalar(a))
        object.__setattr__(self, "b", _as_scalar(b))
        object.__setattr__(self, "tx", _as_scalar(tx))
        object.__setattr__(self, "c", _as_scalar(c))
        object.__setattr__(self, "d", _as_scalar(d))
        object.__setattr__(self, "ty", _as_scalar(ty))
    # end def __init__

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Affine2":
        """Create variable affine coefficients under a prefix.

        Parameters
        ----------
        prefix : str
            Variable name prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Affine2
            Variable transform.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.vars("m").tx.name
        'm.tx'
        """
        return Affine2(
            a=var(f"{prefix}.a", dtype=dtype, shape=()),
            b=var(f"{prefix}.b", dtype=dtype, shape=()),
            tx=var(f"{prefix}.tx", dtype=dtype, shape=()),
            c=var(f"{prefix}.c", dtype=dtype, shape=()),
            d=var(f"{prefix}.d", dtype=dtype, shape=()),
            ty=var(f"{prefix}.ty", dtype=dtype, shape=()),
        )
    # end def vars

    @staticmethod
    def identity() -> "Affine2":
        """Return identity transform.

        Returns
        -------
        Affine2
            Identity map.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.identity().matrix3().shape.dims
        (3, 3)
        """
        return Affine2(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # end def identity

    @staticmethod
    def translation(tx: ScalarIn, ty: ScalarIn) -> "Affine2":
        """Return translation transform.

        Parameters
        ----------
        tx : MathExpr or int or float
            Translation on X.
        ty : MathExpr or int or float
            Translation on Y.

        Returns
        -------
        Affine2
            Translation transform.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.translation(1.0, 2.0).tx.eval().item()
        1.0
        """
        return Affine2(1.0, 0.0, tx, 0.0, 1.0, ty)
    # end def translation

    @staticmethod
    def scale(
            sx: ScalarIn,
            sy: Optional[ScalarIn] = None,
            center: Optional[Point2] = None
    ) -> "Affine2":
        """Return scaling transform, optionally around a center.

        Parameters
        ----------
        sx : MathExpr or int or float
            Scale on X.
        sy : MathExpr or int or float, optional
            Scale on Y. Uses ``sx`` when omitted.
        center : Point2, optional
            Scaling center.

        Returns
        -------
        Affine2
            Scaling transform.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.scale(2.0).a.eval().item()
        2.0
        """
        sy_expr = _as_scalar(sx) if sy is None else _as_scalar(sy)
        base = Affine2(_as_scalar(sx), 0.0, 0.0, 0.0, sy_expr, 0.0)
        if center is None:
            return base
        # end if
        return (
            Affine2.translation(center.x, center.y)
            .compose(base)
            .compose(Affine2.translation(_neg(center.x), _neg(center.y)))
        )
    # end def scale

    @staticmethod
    def shear(shx: ScalarIn = 0.0, shy: ScalarIn = 0.0) -> "Affine2":
        """Return 2D shear transform.

        Parameters
        ----------
        shx : MathExpr or int or float, default 0.0
            Shear of X as function of Y.
        shy : MathExpr or int or float, default 0.0
            Shear of Y as function of X.

        Returns
        -------
        Affine2
            Shear transform.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.shear(1.0, 0.0).b.eval().item()
        1.0
        """
        return Affine2(1.0, shx, 0.0, shy, 1.0, 0.0)
    # end def shear

    @staticmethod
    def rotation(theta: ScalarIn, center: Optional[Point2] = None) -> "Affine2":
        """Return rotation transform in radians.

        Parameters
        ----------
        theta : MathExpr or int or float
            Rotation angle in radians.
        center : Point2, optional
            Rotation center.

        Returns
        -------
        Affine2
            Rotation transform.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Affine2
        >>> Affine2.rotation(np.pi / 2).matrix3().shape.dims
        (3, 3)
        """
        angle = _as_scalar(theta)
        cos_t = _cos(angle)
        sin_t = _sin(angle)
        base = Affine2(cos_t, _neg(sin_t), 0.0, sin_t, cos_t, 0.0)
        if center is None:
            return base
        # end if
        return (
            Affine2.translation(center.x, center.y)
            .compose(base)
            .compose(Affine2.translation(_neg(center.x), _neg(center.y)))
        )
    # end def rotation

    def matrix2x3(self) -> MathExpr:
        """Return transform as symbolic matrix with shape ``(2, 3)``.

        Returns
        -------
        MathExpr
            Matrix expression.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.identity().matrix2x3().shape.dims
        (2, 3)
        """
        return _matrix([
            [self.a, self.b, self.tx],
            [self.c, self.d, self.ty],
        ])
    # end def matrix2x3

    def matrix3(self) -> MathExpr:
        """Return homogeneous symbolic matrix with shape ``(3, 3)``.

        Returns
        -------
        MathExpr
            Homogeneous matrix expression.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.identity().matrix3().shape.dims
        (3, 3)
        """
        return _matrix([
            [self.a, self.b, self.tx],
            [self.c, self.d, self.ty],
            [0.0, 0.0, 1.0],
        ])
    # end def matrix3

    def apply_point(self, point: Point2) -> Point2:
        """Apply transform to a 2D point.

        Parameters
        ----------
        point : Point2
            Point to transform.

        Returns
        -------
        Point2
            Transformed point.

        Examples
        --------
        >>> from pixelprism.math import Affine2, Point2
        >>> Affine2.translation(1.0, 0.0).apply_point(Point2(1.0, 2.0)).eval().value
        array([2., 2.])
        """
        x_new = _add(
            _add(_mul(self.a, point.x), _mul(self.b, point.y)),
            self.tx,
        )
        y_new = _add(
            _add(_mul(self.c, point.x), _mul(self.d, point.y)),
            self.ty,
        )
        return Point2(x_new, y_new)
    # end def apply_point

    def apply_line(self, line: Line2) -> Line2:
        """Apply transform to both line endpoints.

        Parameters
        ----------
        line : Line2
            Line segment to transform.

        Returns
        -------
        Line2
            Transformed line segment.

        Examples
        --------
        >>> from pixelprism.math import Affine2, Line2, Point2
        >>> line = Line2(Point2(0.0, 0.0), Point2(1.0, 0.0))
        >>> Affine2.translation(1.0, 0.0).apply_line(line).start.eval().value
        array([1., 0.])
        """
        return Line2(
            start=self.apply_point(line.start),
            end=self.apply_point(line.end),
        )
    # end def apply_line

    def compose(self, other: "Affine2") -> "Affine2":
        """Compose two affine transforms.

        Parameters
        ----------
        other : Affine2
            Transform applied first.

        Returns
        -------
        Affine2
            Composition ``self @ other``.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.translation(1.0, 0.0).compose(Affine2.identity()).tx.eval().item()
        1.0
        """
        return Affine2(
            a=_add(_mul(self.a, other.a), _mul(self.b, other.c)),
            b=_add(_mul(self.a, other.b), _mul(self.b, other.d)),
            tx=_add(
                _add(_mul(self.a, other.tx), _mul(self.b, other.ty)),
                self.tx,
            ),
            c=_add(_mul(self.c, other.a), _mul(self.d, other.c)),
            d=_add(_mul(self.c, other.b), _mul(self.d, other.d)),
            ty=_add(
                _add(_mul(self.c, other.tx), _mul(self.d, other.ty)),
                self.ty,
            ),
        )
    # end def compose

    def inverse(self) -> "Affine2":
        """Return symbolic inverse transform.

        Returns
        -------
        Affine2
            Inverse transform.

        Examples
        --------
        >>> from pixelprism.math import Affine2
        >>> Affine2.translation(2.0, -3.0).inverse().tx.eval().item()
        -2.0
        """
        det = _sub(_mul(self.a, self.d), _mul(self.b, self.c))
        inv_a = _div(self.d, det)
        inv_b = _div(_neg(self.b), det)
        inv_c = _div(_neg(self.c), det)
        inv_d = _div(self.a, det)
        inv_tx = _div(
            _sub(_mul(self.b, self.ty), _mul(self.d, self.tx)),
            det,
        )
        inv_ty = _div(
            _sub(_mul(self.c, self.tx), _mul(self.a, self.ty)),
            det,
        )
        return Affine2(inv_a, inv_b, inv_tx, inv_c, inv_d, inv_ty)
    # end def inverse

# end class Affine2


@dataclass(frozen=True)
class Affine3:
    """Symbolic 3D affine transform represented by 3x4 coefficients.

    Examples
    --------
    >>> from pixelprism.math import Affine3, Point3
    >>> t = Affine3.translation(1.0, 2.0, 3.0)
    >>> t.apply_point(Point3(0.0, 0.0, 0.0)).eval().value
    array([1., 2., 3.])
    """

    a: MathExpr
    b: MathExpr
    c: MathExpr
    tx: MathExpr
    d: MathExpr
    e: MathExpr
    f: MathExpr
    ty: MathExpr
    g: MathExpr
    h: MathExpr
    i: MathExpr
    tz: MathExpr

    def __init__(
            self,
            a: ScalarIn,
            b: ScalarIn,
            c: ScalarIn,
            tx: ScalarIn,
            d: ScalarIn,
            e: ScalarIn,
            f: ScalarIn,
            ty: ScalarIn,
            g: ScalarIn,
            h: ScalarIn,
            i: ScalarIn,
            tz: ScalarIn,
    ):
        """Create a 3D affine transform from 3x4 scalar coefficients.

        Parameters
        ----------
        a, b, c, tx, d, e, f, ty, g, h, i, tz : MathExpr or int or float
            Coefficients of the affine transform.
        """
        object.__setattr__(self, "a", _as_scalar(a))
        object.__setattr__(self, "b", _as_scalar(b))
        object.__setattr__(self, "c", _as_scalar(c))
        object.__setattr__(self, "tx", _as_scalar(tx))
        object.__setattr__(self, "d", _as_scalar(d))
        object.__setattr__(self, "e", _as_scalar(e))
        object.__setattr__(self, "f", _as_scalar(f))
        object.__setattr__(self, "ty", _as_scalar(ty))
        object.__setattr__(self, "g", _as_scalar(g))
        object.__setattr__(self, "h", _as_scalar(h))
        object.__setattr__(self, "i", _as_scalar(i))
        object.__setattr__(self, "tz", _as_scalar(tz))
    # end def __init__

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Affine3":
        """Create variable affine coefficients.

        Parameters
        ----------
        prefix : str
            Variable name prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Affine3
            Variable transform.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.vars("a3").tz.name
        'a3.tz'
        """
        return Affine3(
            a=var(f"{prefix}.a", dtype=dtype, shape=()),
            b=var(f"{prefix}.b", dtype=dtype, shape=()),
            c=var(f"{prefix}.c", dtype=dtype, shape=()),
            tx=var(f"{prefix}.tx", dtype=dtype, shape=()),
            d=var(f"{prefix}.d", dtype=dtype, shape=()),
            e=var(f"{prefix}.e", dtype=dtype, shape=()),
            f=var(f"{prefix}.f", dtype=dtype, shape=()),
            ty=var(f"{prefix}.ty", dtype=dtype, shape=()),
            g=var(f"{prefix}.g", dtype=dtype, shape=()),
            h=var(f"{prefix}.h", dtype=dtype, shape=()),
            i=var(f"{prefix}.i", dtype=dtype, shape=()),
            tz=var(f"{prefix}.tz", dtype=dtype, shape=()),
        )
    # end def vars

    @staticmethod
    def identity() -> "Affine3":
        """Return identity transform.

        Returns
        -------
        Affine3
            Identity map.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.identity().matrix4().shape.dims
        (4, 4)
        """
        return Affine3(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        )
    # end def identity

    @staticmethod
    def translation(tx: ScalarIn, ty: ScalarIn, tz: ScalarIn) -> "Affine3":
        """Return translation transform.

        Parameters
        ----------
        tx : MathExpr or int or float
            Translation on X.
        ty : MathExpr or int or float
            Translation on Y.
        tz : MathExpr or int or float
            Translation on Z.

        Returns
        -------
        Affine3
            Translation transform.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.translation(1.0, 2.0, 3.0).tz.eval().item()
        3.0
        """
        return Affine3(
            1.0, 0.0, 0.0, tx,
            0.0, 1.0, 0.0, ty,
            0.0, 0.0, 1.0, tz,
        )
    # end def translation

    @staticmethod
    def scale(
            sx: ScalarIn,
            sy: Optional[ScalarIn] = None,
            sz: Optional[ScalarIn] = None,
            center: Optional[Point3] = None
    ) -> "Affine3":
        """Return scaling transform, optionally around a center.

        Parameters
        ----------
        sx : MathExpr or int or float
            Scale on X.
        sy : MathExpr or int or float, optional
            Scale on Y.
        sz : MathExpr or int or float, optional
            Scale on Z.
        center : Point3, optional
            Scaling center.

        Returns
        -------
        Affine3
            Scaling transform.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.scale(2.0).a.eval().item()
        2.0
        """
        sx_expr = _as_scalar(sx)
        sy_expr = sx_expr if sy is None else _as_scalar(sy)
        sz_expr = sx_expr if sz is None else _as_scalar(sz)
        base = Affine3(
            sx_expr, 0.0, 0.0, 0.0,
            0.0, sy_expr, 0.0, 0.0,
            0.0, 0.0, sz_expr, 0.0,
        )
        if center is None:
            return base
        # end if
        return (
            Affine3.translation(center.x, center.y, center.z)
            .compose(base)
            .compose(Affine3.translation(_neg(center.x), _neg(center.y), _neg(center.z)))
        )
    # end def scale

    @staticmethod
    def rotation_x(theta: ScalarIn, center: Optional[Point3] = None) -> "Affine3":
        """Return rotation around X axis.

        Parameters
        ----------
        theta : MathExpr or int or float
            Rotation angle in radians.
        center : Point3, optional
            Rotation center.

        Returns
        -------
        Affine3
            Rotation transform.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Affine3
        >>> Affine3.rotation_x(np.pi / 2).matrix4().shape.dims
        (4, 4)
        """
        angle = _as_scalar(theta)
        c = _cos(angle)
        s = _sin(angle)
        base = Affine3(
            1.0, 0.0, 0.0, 0.0,
            0.0, c, _neg(s), 0.0,
            0.0, s, c, 0.0,
        )
        if center is None:
            return base
        # end if
        return (
            Affine3.translation(center.x, center.y, center.z)
            .compose(base)
            .compose(Affine3.translation(_neg(center.x), _neg(center.y), _neg(center.z)))
        )
    # end def rotation_x

    @staticmethod
    def rotation_y(theta: ScalarIn, center: Optional[Point3] = None) -> "Affine3":
        """Return rotation around Y axis.

        Parameters
        ----------
        theta : MathExpr or int or float
            Rotation angle in radians.
        center : Point3, optional
            Rotation center.

        Returns
        -------
        Affine3
            Rotation transform.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Affine3
        >>> Affine3.rotation_y(np.pi / 2).matrix4().shape.dims
        (4, 4)
        """
        angle = _as_scalar(theta)
        c = _cos(angle)
        s = _sin(angle)
        base = Affine3(
            c, 0.0, s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            _neg(s), 0.0, c, 0.0,
        )
        if center is None:
            return base
        # end if
        return (
            Affine3.translation(center.x, center.y, center.z)
            .compose(base)
            .compose(Affine3.translation(_neg(center.x), _neg(center.y), _neg(center.z)))
        )
    # end def rotation_y

    @staticmethod
    def rotation_z(theta: ScalarIn, center: Optional[Point3] = None) -> "Affine3":
        """Return rotation around Z axis.

        Parameters
        ----------
        theta : MathExpr or int or float
            Rotation angle in radians.
        center : Point3, optional
            Rotation center.

        Returns
        -------
        Affine3
            Rotation transform.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Affine3
        >>> Affine3.rotation_z(np.pi / 2).matrix4().shape.dims
        (4, 4)
        """
        angle = _as_scalar(theta)
        c = _cos(angle)
        s = _sin(angle)
        base = Affine3(
            c, _neg(s), 0.0, 0.0,
            s, c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        )
        if center is None:
            return base
        # end if
        return (
            Affine3.translation(center.x, center.y, center.z)
            .compose(base)
            .compose(Affine3.translation(_neg(center.x), _neg(center.y), _neg(center.z)))
        )
    # end def rotation_z

    @staticmethod
    def shear(
            xy: ScalarIn = 0.0,
            xz: ScalarIn = 0.0,
            yx: ScalarIn = 0.0,
            yz: ScalarIn = 0.0,
            zx: ScalarIn = 0.0,
            zy: ScalarIn = 0.0,
    ) -> "Affine3":
        """Return a general 3D shear transform.

        Parameters
        ----------
        xy, xz, yx, yz, zx, zy : MathExpr or int or float
            Shear coefficients.

        Returns
        -------
        Affine3
            Shear transform.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.shear(xy=1.0).b.eval().item()
        1.0
        """
        return Affine3(
            1.0, xy, xz, 0.0,
            yx, 1.0, yz, 0.0,
            zx, zy, 1.0, 0.0,
        )
    # end def shear

    def matrix3x4(self) -> MathExpr:
        """Return transform as symbolic matrix with shape ``(3, 4)``.

        Returns
        -------
        MathExpr
            Matrix expression.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.identity().matrix3x4().shape.dims
        (3, 4)
        """
        return _matrix([
            [self.a, self.b, self.c, self.tx],
            [self.d, self.e, self.f, self.ty],
            [self.g, self.h, self.i, self.tz],
        ])
    # end def matrix3x4

    def matrix4(self) -> MathExpr:
        """Return homogeneous symbolic matrix with shape ``(4, 4)``.

        Returns
        -------
        MathExpr
            Homogeneous matrix expression.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.identity().matrix4().shape.dims
        (4, 4)
        """
        return _matrix([
            [self.a, self.b, self.c, self.tx],
            [self.d, self.e, self.f, self.ty],
            [self.g, self.h, self.i, self.tz],
            [0.0, 0.0, 0.0, 1.0],
        ])
    # end def matrix4

    def apply_point(self, point: Point3) -> Point3:
        """Apply transform to a 3D point.

        Parameters
        ----------
        point : Point3
            Point to transform.

        Returns
        -------
        Point3
            Transformed point.

        Examples
        --------
        >>> from pixelprism.math import Affine3, Point3
        >>> Affine3.translation(1.0, 2.0, 3.0).apply_point(Point3(0.0, 0.0, 0.0)).eval().value
        array([1., 2., 3.])
        """
        x_new = _add(_add(_add(_mul(self.a, point.x), _mul(self.b, point.y)), _mul(self.c, point.z)), self.tx)
        y_new = _add(_add(_add(_mul(self.d, point.x), _mul(self.e, point.y)), _mul(self.f, point.z)), self.ty)
        z_new = _add(_add(_add(_mul(self.g, point.x), _mul(self.h, point.y)), _mul(self.i, point.z)), self.tz)
        return Point3(x_new, y_new, z_new)
    # end def apply_point

    def apply_line(self, line: Line3) -> Line3:
        """Apply transform to both line endpoints.

        Parameters
        ----------
        line : Line3
            Line segment to transform.

        Returns
        -------
        Line3
            Transformed line segment.

        Examples
        --------
        >>> from pixelprism.math import Affine3, Line3, Point3
        >>> line = Line3(Point3(0.0, 0.0, 0.0), Point3(1.0, 0.0, 0.0))
        >>> Affine3.translation(1.0, 0.0, 0.0).apply_line(line).start.eval().value
        array([1., 0., 0.])
        """
        return Line3(
            start=self.apply_point(line.start),
            end=self.apply_point(line.end),
        )
    # end def apply_line

    def compose(self, other: "Affine3") -> "Affine3":
        """Compose two affine transforms.

        Parameters
        ----------
        other : Affine3
            Transform applied first.

        Returns
        -------
        Affine3
            Composition ``self @ other``.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.translation(1.0, 0.0, 0.0).compose(Affine3.identity()).tx.eval().item()
        1.0
        """
        return Affine3(
            a=_add(_add(_mul(self.a, other.a), _mul(self.b, other.d)), _mul(self.c, other.g)),
            b=_add(_add(_mul(self.a, other.b), _mul(self.b, other.e)), _mul(self.c, other.h)),
            c=_add(_add(_mul(self.a, other.c), _mul(self.b, other.f)), _mul(self.c, other.i)),
            tx=_add(_add(_add(_mul(self.a, other.tx), _mul(self.b, other.ty)), _mul(self.c, other.tz)), self.tx),
            d=_add(_add(_mul(self.d, other.a), _mul(self.e, other.d)), _mul(self.f, other.g)),
            e=_add(_add(_mul(self.d, other.b), _mul(self.e, other.e)), _mul(self.f, other.h)),
            f=_add(_add(_mul(self.d, other.c), _mul(self.e, other.f)), _mul(self.f, other.i)),
            ty=_add(_add(_add(_mul(self.d, other.tx), _mul(self.e, other.ty)), _mul(self.f, other.tz)), self.ty),
            g=_add(_add(_mul(self.g, other.a), _mul(self.h, other.d)), _mul(self.i, other.g)),
            h=_add(_add(_mul(self.g, other.b), _mul(self.h, other.e)), _mul(self.i, other.h)),
            i=_add(_add(_mul(self.g, other.c), _mul(self.h, other.f)), _mul(self.i, other.i)),
            tz=_add(_add(_add(_mul(self.g, other.tx), _mul(self.h, other.ty)), _mul(self.i, other.tz)), self.tz),
        )
    # end def compose

    def inverse(self) -> "Affine3":
        """Return symbolic inverse transform.

        Returns
        -------
        Affine3
            Inverse transform.

        Examples
        --------
        >>> from pixelprism.math import Affine3
        >>> Affine3.translation(1.0, 2.0, 3.0).inverse().tx.eval().item()
        -1.0
        """
        det = _add(
            _sub(_mul(self.a, _sub(_mul(self.e, self.i), _mul(self.f, self.h))), _mul(self.b, _sub(_mul(self.d, self.i), _mul(self.f, self.g)))),
            _mul(self.c, _sub(_mul(self.d, self.h), _mul(self.e, self.g))),
        )

        inv_a = _div(_sub(_mul(self.e, self.i), _mul(self.f, self.h)), det)
        inv_b = _div(_sub(_mul(self.c, self.h), _mul(self.b, self.i)), det)
        inv_c = _div(_sub(_mul(self.b, self.f), _mul(self.c, self.e)), det)

        inv_d = _div(_sub(_mul(self.f, self.g), _mul(self.d, self.i)), det)
        inv_e = _div(_sub(_mul(self.a, self.i), _mul(self.c, self.g)), det)
        inv_f = _div(_sub(_mul(self.c, self.d), _mul(self.a, self.f)), det)

        inv_g = _div(_sub(_mul(self.d, self.h), _mul(self.e, self.g)), det)
        inv_h = _div(_sub(_mul(self.b, self.g), _mul(self.a, self.h)), det)
        inv_i = _div(_sub(_mul(self.a, self.e), _mul(self.b, self.d)), det)

        inv_tx = _neg(_add(_add(_mul(inv_a, self.tx), _mul(inv_b, self.ty)), _mul(inv_c, self.tz)))
        inv_ty = _neg(_add(_add(_mul(inv_d, self.tx), _mul(inv_e, self.ty)), _mul(inv_f, self.tz)))
        inv_tz = _neg(_add(_add(_mul(inv_g, self.tx), _mul(inv_h, self.ty)), _mul(inv_i, self.tz)))

        return Affine3(
            inv_a, inv_b, inv_c, inv_tx,
            inv_d, inv_e, inv_f, inv_ty,
            inv_g, inv_h, inv_i, inv_tz,
        )
    # end def inverse

# end class Affine3


@dataclass(frozen=True)
class Circle2:
    """Symbolic 2D circle represented by center and radius.

    Parameters
    ----------
    center : Point2
        Circle center.
    radius : MathExpr or int or float
        Circle radius.

    Examples
    --------
    >>> from pixelprism.math import Circle2, Point2
    >>> c = Circle2(Point2(0.0, 0.0), 2.0)
    >>> c.point_at(0.0).eval().value
    array([2., 0.])
    """

    center: Point2
    radius: MathExpr

    def __init__(self, center: Point2, radius: ScalarIn):
        """Create a circle from center and radius.

        Parameters
        ----------
        center : Point2
            Circle center.
        radius : MathExpr or int or float
            Circle radius.
        """
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", _as_scalar(radius))
    # end def __init__

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Circle2":
        """Create a variable circle.

        Parameters
        ----------
        prefix : str
            Variable prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Circle2
            Variable circle with names ``{prefix}.cx/.cy/.r``.

        Examples
        --------
        >>> from pixelprism.math import Circle2
        >>> Circle2.vars("c").radius.name
        'c.r'
        """
        return Circle2(
            center=Point2(
                var(f"{prefix}.cx", dtype=dtype, shape=()),
                var(f"{prefix}.cy", dtype=dtype, shape=()),
            ),
            radius=var(f"{prefix}.r", dtype=dtype, shape=()),
        )
    # end def vars

    def eval_center(self):
        """Evaluate center.

        Returns
        -------
        Tensor
            Tensor with shape ``(2,)``.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(1.0, 2.0), 3.0).eval_center().value
        array([1., 2.])
        """
        return self.center.eval()
    # end def eval_center

    def eval_radius(self):
        """Evaluate radius.

        Returns
        -------
        Tensor
            Scalar tensor.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(0.0, 0.0), 3.0).eval_radius().item()
        3.0
        """
        return self.radius.eval()
    # end def eval_radius

    def point_at(self, angle: ScalarIn) -> Point2:
        """Return a point on the circumference.

        Parameters
        ----------
        angle : MathExpr or int or float
            Polar angle in radians.

        Returns
        -------
        Point2
            Point on the circle.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(0.0, 0.0), 2.0).point_at(0.0).eval().value
        array([2., 0.])
        """
        theta = _as_scalar(angle)
        return Point2(
            _add(self.center.x, _mul(self.radius, _cos(theta))),
            _add(self.center.y, _mul(self.radius, _sin(theta))),
        )
    # end def point_at

    def distance_to_point_sq(self, point: Point2) -> MathExpr:
        """Return squared distance from center to point.

        Parameters
        ----------
        point : Point2
            Query point.

        Returns
        -------
        MathExpr
            Squared distance expression.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(0.0, 0.0), 1.0).distance_to_point_sq(Point2(3.0, 4.0)).eval().item()
        25.0
        """
        return self.center.distance_sq(point)
    # end def distance_to_point_sq

    def contains_metric(self, point: Point2) -> MathExpr:
        """Return implicit containment metric.

        Parameters
        ----------
        point : Point2
            Query point.

        Returns
        -------
        MathExpr
            Value ``distance_sq(point) - radius**2``.
            Inside/on-boundary corresponds to ``<= 0``.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(0.0, 0.0), 2.0).contains_metric(Point2(1.0, 0.0)).eval().item()
        -3.0
        """
        return _sub(self.distance_to_point_sq(point), _mul(self.radius, self.radius))
    # end def contains_metric

    def translated(self, dx: ScalarIn, dy: ScalarIn) -> "Circle2":
        """Translate the circle.

        Parameters
        ----------
        dx : MathExpr or int or float
            X offset.
        dy : MathExpr or int or float
            Y offset.

        Returns
        -------
        Circle2
            Translated circle.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(0.0, 0.0), 1.0).translated(1.0, 2.0).center.eval().value
        array([1., 2.])
        """
        return Circle2(self.center.translated(dx, dy), self.radius)
    # end def translated

    def rotated(self, angle: ScalarIn, center: Optional[Point2] = None) -> "Circle2":
        """Rotate circle center. Radius is unchanged.

        Parameters
        ----------
        angle : MathExpr or int or float
            Rotation angle in radians.
        center : Point2, optional
            Rotation center.

        Returns
        -------
        Circle2
            Rotated circle.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(1.0, 0.0), 1.0).rotated(np.pi / 2).center.eval().shape.dims
        (2,)
        """
        return Circle2(self.center.rotated(angle, center=center), self.radius)
    # end def rotated

    def scaled(self, s: ScalarIn, center: Optional[Point2] = None) -> "Circle2":
        """Uniformly scale center and radius.

        Parameters
        ----------
        s : MathExpr or int or float
            Uniform scale factor.
        center : Point2, optional
            Scaling center for the circle center.

        Returns
        -------
        Circle2
            Scaled circle.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2
        >>> Circle2(Point2(0.0, 0.0), 2.0).scaled(0.5).radius.eval().item()
        1.0
        """
        factor = _as_scalar(s)
        new_center = self.center.scaled(factor, center=center)
        return Circle2(new_center, _mul(self.radius, factor))
    # end def scaled

    def transformed(self, affine: Affine2) -> "Ellipse2":
        """Transform the circle into an ellipse.

        Parameters
        ----------
        affine : Affine2
            Affine transform.

        Returns
        -------
        Ellipse2
            Equivalent transformed ellipse.

        Examples
        --------
        >>> from pixelprism.math import Circle2, Point2, Affine2
        >>> e = Circle2(Point2(0.0, 0.0), 1.0).transformed(Affine2.scale(2.0, 1.0))
        >>> e.rx.eval().item()
        2.0
        """
        center = affine.apply_point(self.center)

        sxx = _add(_mul(affine.a, affine.a), _mul(affine.b, affine.b))
        syy = _add(_mul(affine.c, affine.c), _mul(affine.d, affine.d))
        sxy = _add(_mul(affine.a, affine.c), _mul(affine.b, affine.d))

        trace = _add(sxx, syy)
        delta = _sqrt(_add(_mul(_sub(sxx, syy), _sub(sxx, syy)), _mul(4.0, _mul(sxy, sxy))))

        lambda1 = _div(_add(trace, delta), 2.0)
        lambda2 = _div(_sub(trace, delta), 2.0)

        rx = _mul(self.radius, _sqrt(lambda1))
        ry = _mul(self.radius, _sqrt(lambda2))
        angle = _mul(0.5, _atan2(_mul(2.0, sxy), _sub(sxx, syy)))

        return Ellipse2(center=center, rx=rx, ry=ry, angle=angle)
    # end def transformed

# end class Circle2


@dataclass(frozen=True)
class Ellipse2:
    """Symbolic 2D ellipse with center, radii and orientation.

    Parameters
    ----------
    center : Point2
        Ellipse center.
    rx : MathExpr or int or float
        Radius along local X axis.
    ry : MathExpr or int or float
        Radius along local Y axis.
    angle : MathExpr or int or float, default 0.0
        Ellipse orientation in radians.

    Examples
    --------
    >>> from pixelprism.math import Ellipse2, Point2
    >>> e = Ellipse2(Point2(0.0, 0.0), 3.0, 2.0)
    >>> e.point_at(0.0).eval().value
    array([3., 0.])
    """

    center: Point2
    rx: MathExpr
    ry: MathExpr
    angle: MathExpr

    def __init__(self, center: Point2, rx: ScalarIn, ry: ScalarIn, angle: ScalarIn = 0.0):
        """Create an ellipse from center, radii and angle.

        Parameters
        ----------
        center : Point2
            Ellipse center.
        rx : MathExpr or int or float
            Radius on local X axis.
        ry : MathExpr or int or float
            Radius on local Y axis.
        angle : MathExpr or int or float, default 0.0
            Orientation in radians.
        """
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "rx", _as_scalar(rx))
        object.__setattr__(self, "ry", _as_scalar(ry))
        object.__setattr__(self, "angle", _as_scalar(angle))
    # end def __init__

    @staticmethod
    def vars(prefix: str, dtype: DType = DType.R) -> "Ellipse2":
        """Create a variable ellipse.

        Parameters
        ----------
        prefix : str
            Variable prefix.
        dtype : DType, default DType.R
            Variable dtype.

        Returns
        -------
        Ellipse2
            Variable ellipse with names ``{prefix}.cx/.cy/.rx/.ry/.angle``.

        Examples
        --------
        >>> from pixelprism.math import Ellipse2
        >>> Ellipse2.vars("e").angle.name
        'e.angle'
        """
        return Ellipse2(
            center=Point2(
                var(f"{prefix}.cx", dtype=dtype, shape=()),
                var(f"{prefix}.cy", dtype=dtype, shape=()),
            ),
            rx=var(f"{prefix}.rx", dtype=dtype, shape=()),
            ry=var(f"{prefix}.ry", dtype=dtype, shape=()),
            angle=var(f"{prefix}.angle", dtype=dtype, shape=()),
        )
    # end def vars

    def point_at(self, t: ScalarIn) -> Point2:
        """Return a point on the ellipse boundary.

        Parameters
        ----------
        t : MathExpr or int or float
            Parametric angle in radians.

        Returns
        -------
        Point2
            Boundary point.

        Examples
        --------
        >>> from pixelprism.math import Ellipse2, Point2
        >>> Ellipse2(Point2(0.0, 0.0), 3.0, 2.0).point_at(0.0).eval().value
        array([3., 0.])
        """
        tau = _as_scalar(t)
        ct = _cos(tau)
        st = _sin(tau)
        ca = _cos(self.angle)
        sa = _sin(self.angle)

        x_local = _mul(self.rx, ct)
        y_local = _mul(self.ry, st)

        x = _add(self.center.x, _sub(_mul(ca, x_local), _mul(sa, y_local)))
        y = _add(self.center.y, _add(_mul(sa, x_local), _mul(ca, y_local)))
        return Point2(x, y)
    # end def point_at

    def translated(self, dx: ScalarIn, dy: ScalarIn) -> "Ellipse2":
        """Translate the ellipse.

        Parameters
        ----------
        dx : MathExpr or int or float
            X offset.
        dy : MathExpr or int or float
            Y offset.

        Returns
        -------
        Ellipse2
            Translated ellipse.

        Examples
        --------
        >>> from pixelprism.math import Ellipse2, Point2
        >>> Ellipse2(Point2(0.0, 0.0), 2.0, 1.0).translated(1.0, 2.0).center.eval().value
        array([1., 2.])
        """
        return Ellipse2(self.center.translated(dx, dy), self.rx, self.ry, self.angle)
    # end def translated

    def rotated(self, delta: ScalarIn, center: Optional[Point2] = None) -> "Ellipse2":
        """Rotate ellipse center and orientation.

        Parameters
        ----------
        delta : MathExpr or int or float
            Added orientation angle in radians.
        center : Point2, optional
            Rotation center for ellipse center.

        Returns
        -------
        Ellipse2
            Rotated ellipse.

        Examples
        --------
        >>> import numpy as np
        >>> from pixelprism.math import Ellipse2, Point2
        >>> Ellipse2(Point2(1.0, 0.0), 2.0, 1.0).rotated(np.pi / 2).center.eval().shape.dims
        (2,)
        """
        return Ellipse2(
            center=self.center.rotated(delta, center=center),
            rx=self.rx,
            ry=self.ry,
            angle=_add(self.angle, _as_scalar(delta)),
        )
    # end def rotated

    def scaled(
            self,
            sx: ScalarIn,
            sy: Optional[ScalarIn] = None,
            center: Optional[Point2] = None
    ) -> "Ellipse2":
        """Scale ellipse center and radii.

        Parameters
        ----------
        sx : MathExpr or int or float
            Scale factor on X.
        sy : MathExpr or int or float, optional
            Scale factor on Y. Uses ``sx`` when omitted.
        center : Point2, optional
            Scaling center for the ellipse center.

        Returns
        -------
        Ellipse2
            Scaled ellipse.

        Examples
        --------
        >>> from pixelprism.math import Ellipse2, Point2
        >>> Ellipse2(Point2(0.0, 0.0), 2.0, 1.0).scaled(2.0).rx.eval().item()
        4.0
        """
        sx_expr = _as_scalar(sx)
        sy_expr = sx_expr if sy is None else _as_scalar(sy)
        return Ellipse2(
            center=self.center.scaled(sx_expr, sy_expr, center=center),
            rx=_mul(self.rx, sx_expr),
            ry=_mul(self.ry, sy_expr),
            angle=self.angle,
        )
    # end def scaled

    def contains_metric(self, point: Point2) -> MathExpr:
        """Return implicit containment metric.

        Parameters
        ----------
        point : Point2
            Query point.

        Returns
        -------
        MathExpr
            Value ``(x'/rx)^2 + (y'/ry)^2 - 1`` in local ellipse coordinates.
            Inside/on-boundary corresponds to ``<= 0``.

        Examples
        --------
        >>> from pixelprism.math import Ellipse2, Point2
        >>> Ellipse2(Point2(0.0, 0.0), 2.0, 1.0).contains_metric(Point2(1.0, 0.0)).eval().item()
        -0.75
        """
        dx = _sub(point.x, self.center.x)
        dy = _sub(point.y, self.center.y)
        ca = _cos(self.angle)
        sa = _sin(self.angle)

        x_local = _add(_mul(ca, dx), _mul(sa, dy))
        y_local = _sub(_mul(_neg(sa), dx), _mul(ca, dy))

        term_x = _div(_mul(x_local, x_local), _mul(self.rx, self.rx))
        term_y = _div(_mul(y_local, y_local), _mul(self.ry, self.ry))
        return _sub(_add(term_x, term_y), 1.0)
    # end def contains_metric

    def transformed(self, affine: Affine2) -> "Ellipse2":
        """Transform the ellipse by an affine map.

        Parameters
        ----------
        affine : Affine2
            Affine transform.

        Returns
        -------
        Ellipse2
            Transformed ellipse.

        Examples
        --------
        >>> from pixelprism.math import Ellipse2, Point2, Affine2
        >>> e = Ellipse2(Point2(0.0, 0.0), 2.0, 1.0).transformed(Affine2.translation(1.0, 0.0))
        >>> e.center.eval().value
        array([1., 0.])
        """
        ca = _cos(self.angle)
        sa = _sin(self.angle)

        ux_x = _mul(self.rx, ca)
        ux_y = _mul(self.rx, sa)
        uy_x = _mul(_neg(self.ry), sa)
        uy_y = _mul(self.ry, ca)

        vx_x = _add(_mul(affine.a, ux_x), _mul(affine.b, ux_y))
        vx_y = _add(_mul(affine.c, ux_x), _mul(affine.d, ux_y))
        vy_x = _add(_mul(affine.a, uy_x), _mul(affine.b, uy_y))
        vy_y = _add(_mul(affine.c, uy_x), _mul(affine.d, uy_y))

        sxx = _add(_mul(vx_x, vx_x), _mul(vy_x, vy_x))
        syy = _add(_mul(vx_y, vx_y), _mul(vy_y, vy_y))
        sxy = _add(_mul(vx_x, vx_y), _mul(vy_x, vy_y))

        trace = _add(sxx, syy)
        delta = _sqrt(_add(_mul(_sub(sxx, syy), _sub(sxx, syy)), _mul(4.0, _mul(sxy, sxy))))
        lambda1 = _div(_add(trace, delta), 2.0)
        lambda2 = _div(_sub(trace, delta), 2.0)

        center = affine.apply_point(self.center)
        rx = _sqrt(lambda1)
        ry = _sqrt(lambda2)
        angle = _mul(0.5, _atan2(_mul(2.0, sxy), _sub(sxx, syy)))
        return Ellipse2(center=center, rx=rx, ry=ry, angle=angle)
    # end def transformed

# end class Ellipse2
