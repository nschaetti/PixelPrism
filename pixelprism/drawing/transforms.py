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

# Imports
import numpy as np
from pixelprism.math import Point2D, Scalar


class Transform(object):
    """
    A class to represent a transform.
    """
    pass
# end Transform


# Translate transform
class Translate2D(Transform):
    """
    A class to represent a translate transform.
    """

    def __init__(self, tx=0, ty=0):
        """
        Initialize the translate transform with its X and Y offsets.

        Args:
            x (float): X offset
            y (float): Y offset
        """
        super().__init__()
        self.translate = Point2D(tx, ty)
    # end __init__

    # Get
    def get(self):
        """
        Get the X and Y offsets of the translate transform.

        Returns:
            tuple: X and Y offsets of the translate transform
        """
        return self.translate
    # end get

    # Set
    def set(self, tx=0, ty=0):
        """
        Set the X and Y offsets of the translate transform.

        Args:
            tx (float): X offset
            ty (float): Y offset
        """
        self.translate.x = tx
        self.translate.y = ty
    # end set

# end Translate2D


# Scale transform
class Scale2D(Transform):
    """
    A class to represent a scale transform.
    """

    def __init__(self, sx=1, sy=1):
        """
        Initialize the scale transform with its X and Y scale factors.

        Args:
            sx (float): X scale factor
            sy (float): Y scale factor
        """
        super().__init__()
        self.scale = Point2D(sx, sy)
    # end __init__

    # Get
    def get(self):
        """
        Get the X and Y scale factors of the scale transform.

        Returns:
            tuple: X and Y scale factors of the scale transform
        """
        return self.scale
    # end get

    # Set
    def set(self, sx=1, sy=1):
        """
        Set the X and Y scale factors of the scale transform.

        Args:
            sx (float): X scale factor
            sy (float): Y scale factor
        """
        self.scale.x = sx
        self.scale.y = sy
    # end

# end Scale2D


# Rotate transform
class Rotate2D(Transform):
    """
    A class to represent a rotate transform.
    """

    def __init__(self, cx: float = 0, cy: float = 0, angle=0):
        """
        Initialize the rotate transform with its rotation angle.

        Args:
            angle (float): Rotation angle in degrees
        """
        super().__init__()
        self.center = Point2D(cx, cy)
        self.angle = Scalar(angle)
    # end __init__

    # Get
    def get(self):
        """
        Get the rotation angle of the rotate transform.

        Returns:
            float: Rotation angle of the rotate transform
        """
        return self.center.x, self.center.y, self.angle.value
    # end get

    # Set
    def set(self, cx: float = 0, cy: float = 0, angle=0):
        """
        Set the center and rotation angle of the rotate transform.

        Args:
            cx (float): X-coordinate of the center
            cy (float): Y-coordinate of the center
            angle (float): Rotation angle in degrees
        """
        self.center.x = cx
        self.center.y = cy
        self.angle.value = angle
    # end set

# end Rotate2D


# Skew transform
class SkewX2D(Transform):
    """
    A class to represent a skew transform along the X-axis.
    """

    def __init__(self, angle=0):
        """
        Initialize the skew transform with its skew angle along the X-axis.

        Args:
            angle (float): Skew angle in degrees
        """
        super().__init__()
        self.angle = Scalar(angle)
    # end __init__

    # Get
    def get(self):
        """
        Get the skew angle of the skew transform along the X-axis.

        Returns:
            float: Skew angle of the skew transform along the X-axis
        """
        return self.angle.value
    # end get

    # Set
    def set(self, angle=0):
        """
        Set the skew angle of the skew transform along the X-axis.

        Args:
            angle (float): Skew angle in degrees
        """
        self.angle.value = angle
    # end set

# end SkewX2D


class SkewY2D(SkewX2D):
    pass
# end SkewY2D


# Matrix transform
class Matrix2D(Transform):
    """
    A class to represent a matrix transform.
    """

    def __init__(self, xx=1, yx=0, xy=0, yy=1, x0=0, y0=0):
        """
        Initialize the matrix transform with its matrix coefficients.

        Args:
            xx (float): Coefficient xx
            yx (float): Coefficient yx
            xy (float): Coefficient xy
            yy (float): Coefficient yy
            x0 (float): Coefficient x0
            y0 (float): Coefficient y0
        """
        super().__init__()
        self.xx = Scalar(xx)
        self.yx = Scalar(yx)
        self.xy = Scalar(xy)
        self.yy = Scalar(yy)
        self.x0 = Scalar(x0)
        self.y0 = Scalar(y0)
    # end __init__

    # Get
    def get(self):
        """
        Get the matrix coefficients of the matrix transform.

        Returns:
            np.ndarray: Matrix coefficients of the matrix transform
        """
        return (
            self.xx.value,
            self.yx.value,
            self.xy.value,
            self.yy.value,
            self.x0.value,
            self.y0.value
        )
    # end get

    # Set
    def set(self, xx=1, yx=0, xy=0, yy=1, x0=0, y0=0):
        """
        Set the matrix coefficients of the matrix transform.

        Args:
            xx (float): Coefficient xx
            yx (float): Coefficient yx
            xy (float): Coefficient xy
            yy (float): Coefficient yy
            x0 (float): Coefficient x0
            y0 (float): Coefficient y0
        """
        self.xx.value = xx
        self.yx.value = yx
        self.xy.value = xy
        self.yy.value = yy
        self.x0.value = x0
        self.y0.value = y0
    # end set

# end Matrix2D
