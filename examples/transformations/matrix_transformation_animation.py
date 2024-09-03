#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np

# Imports
from pixel_prism import utils, p2
from pixel_prism.animate import Range, EaseInOutInterpolator
from pixel_prism.data import Scalar, tpoint2d, meshgrid, tmatrix2d, TMatrix2D
from pixel_prism.data.matrices import mv_t, mm_t
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas, CoordSystem
from pixel_prism.drawing import Circle, Line


# MatrixTransformationAnimation class
class MatrixTransformationAnimation(Animation):

    # Range
    RANGE = 10
    NUM_POINTS = 97

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    # Build the animation
    def build(self):
        """
        Build the animation.
        """
        # Coordinate system
        coord_system = CoordSystem(
            image_width=self.width,
            image_height=self.height,
            size=10
        )

        # Add the LaTeX widget to the viewport or a container
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Create grid of points
        points = list()
        points += [(p2(x, self.RANGE), p2(x, -self.RANGE)) for x in np.linspace(-self.RANGE, self.RANGE, self.NUM_POINTS)]
        points += [(p2(self.RANGE, y), p2(-self.RANGE, y)) for y in np.linspace(-self.RANGE, self.RANGE, self.NUM_POINTS)]
        axe_points = [(p2(self.RANGE*1.5, 0), p2(-self.RANGE*1.5, 0)), (p2(0, self.RANGE*1.5), p2(0, -self.RANGE*1.5))]
        animated_point = p2(1, 1)

        # Create transformation matrix
        scale_x = Scalar(1.0)
        scale_y = Scalar(1.0)
        shear_x = Scalar(0.0)
        shear_y = Scalar(0.0)
        angle = Scalar(0.0)
        trans_matrix = TMatrix2D.stretching(scale_x, scale_y)
        trans_matrix = mm_t(trans_matrix, TMatrix2D.shearing(shear_x, shear_y))
        trans_matrix = mm_t(trans_matrix, TMatrix2D.rotation(angle))

        # Create tpoint2d from points
        tpoints = [(mv_t(trans_matrix, ps), mv_t(trans_matrix, pe)) for (ps, pe) in points]
        axe_tpoints = [(mv_t(trans_matrix, ps), mv_t(trans_matrix, pe)) for (ps, pe) in axe_points]
        animated_tpoint = mv_t(trans_matrix, animated_point)

        # Main grid from tpoints
        lines = [
            Line.from_objects(
                start=ps,
                end=pe,
                line_color=utils.from_hex("#54D2EF", alpha=0.9),
                line_width=Scalar(0.008)
            ) for pi, (ps, pe) in enumerate(tpoints)
            if pi % 2 == 0
        ]

        # Second grid from tpoints
        lines += [
            Line.from_objects(
                start=ps,
                end=pe,
                line_color=utils.from_hex("#54D2EF", alpha=0.6),
                line_width=Scalar(0.005)
            ) for pi, (ps, pe) in enumerate(tpoints)
            if pi % 2 == 1
        ]

        # Axes lines
        lines += [
            Line.from_objects(
                start=ps,
                end=pe,
                line_color=utils.WHITE.copy(),
                line_width=Scalar(0.015)
            ) for pi, (ps, pe) in enumerate(axe_tpoints)
        ]

        # Points
        circle = Circle(
            position=animated_tpoint,
            radius=Scalar(0.08),
            fill_color=utils.RED.copy(),
            line_color=utils.YELLOW.copy(),
            line_width=Scalar(0.01)
        )

        # Animate stretch, shear
        self.animate(scale_x.range(2, 4).range(2, 1))
        self.animate(scale_y.range(2, 4, 4).range(2, 1))
        self.animate(shear_x.range(2, 1, 8).range(2, 0))
        self.animate(shear_y.range(2, 1, 12).range(2, 0))
        self.animate(angle.range(2, np.pi, 16).range(2, 0))
        self.animate(
            animated_point
            .move(4, p2(1, -1))
            .move(4, p2(-1, -1), 4)
            .move(4, p2(-1, 1), 8)
            .move(4, p2(1, 1), 12)
        )

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(lines)
        drawable_widget.add(circle)

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            lines=lines,
            circle=circle
        )
    # end build

    # Process frame
    def process_frame(
            self,
            image_canvas: ImageCanvas,
            t: float,
            frame_number: int
    ):
        """
        Process the frame.

        Args:
            image_canvas (ImageCanvas): Image canvas
            t (float): Time
            frame_number (int): Frame number
        """
        # Create a DrawableImage
        drawing_layer = DrawableImage.transparent(
            self.width,
            self.height,
            coord_system=self.obj("coord_system")
        )

        # Get the viewport and drawable widget
        viewport = self.obj("viewport")

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(viewport)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end MatrixTransformationAnimation

