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

#
# Animation of a vector graphics.
# Build and highlight
#

# Imports
import math
from pixelprism import p2, s, c
from pixelprism import utils
from pixelprism.animate.animate import Rotate, Scale
from pixelprism.animation import Animation
from pixelprism.animate import Move, EaseInOutInterpolator, Call
from pixelprism.widgets.containers import Viewport
from pixelprism.widgets import DrawableWidget
from pixelprism.base import DrawableImage, ImageCanvas, CoordSystem
from pixelprism.drawing import (
    Path,
    PathSegment,
    PathLine,
    PathBezierCubic,
    PathArc,
    VectorGraphics
)
from pixelprism.data import Scalar


# A simple vector graphics animation
class VectorGraphicsAnimation(Animation):
    """
    Vector graphics animation.
    """

    PATH_LINE_WIDTH = 0.02
    PATH_SIZE = 0.7

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    # Build first path
    def build_first_path(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the first curve.
        """
        # Decalage point
        d_point = p2(1.0, 0.0)

        # Curve control points
        control_points = [
            coord_system.center + d_point + p2(self.PATH_SIZE, 0),
            coord_system.center + d_point + p2(0, self.PATH_SIZE),
            p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 2.0),
            p2(0, self.PATH_SIZE / 2.0),
            coord_system.center + d_point + p2(0, -self.PATH_SIZE)
        ]

        # Subpath control points
        subpath_control_points = [
            coord_system.center + d_point + p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 4.0)
        ]

        # Create path segment
        path_segment = PathSegment.from_objects(
            start=control_points[0],
            elements=[
                PathLine(control_points[0], control_points[1]),
                PathArc(center=coord_system.center + d_point, radius=s(self.PATH_SIZE), start_angle=s(math.pi / 2),
                        end_angle=s(math.pi)),
                PathBezierCubic(start=coord_system.center + d_point + p2(-self.PATH_SIZE, 0),
                                control1=control_points[2], control2=control_points[3], end=control_points[4])
            ]
        )

        # Create path
        path1 = Path.from_objects(
            path=path_segment,
            subpaths=[
                PathSegment.rectangle(
                    lower_left=subpath_control_points[0],
                    width=Scalar(self.PATH_SIZE),
                    height=Scalar(self.PATH_SIZE / 2.0)
                )
            ],
            line_color=utils.RED.copy(),
            line_width=Scalar(self.PATH_LINE_WIDTH),
            fill_color=utils.GREEN.copy(),
            closed_path=True
        )

        return path1
    # end build_first_path

    # Build second path
    def build_second_path(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the second curve.

        Args:
            coord_system (CoordSystem): Coordinate system
        """
        # Decalage point
        d_point = p2(-1.0, 0.0)

        # Create path segment
        path_segment = PathSegment.from_objects(
            start=coord_system.center + d_point + p2(self.PATH_SIZE, 0),
            elements=[
                PathLine(
                    coord_system.center + d_point + p2(self.PATH_SIZE, 0),
                    coord_system.center + d_point + p2(0, self.PATH_SIZE)
                ),
                PathArc(
                    center=coord_system.center + d_point,
                    radius=s(self.PATH_SIZE),
                    start_angle=s(math.pi / 2),
                    end_angle=s(math.pi)
                ),
                PathBezierCubic(
                    start=coord_system.center + d_point + p2(-self.PATH_SIZE, 0),
                    control1=p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 2.0),
                    control2=p2(0, self.PATH_SIZE / 2.0),
                    end=coord_system.center + d_point + p2(0, -self.PATH_SIZE)
                )
            ]
        )

        # Rectangle
        rectangle = PathSegment.rectangle(
            lower_left=coord_system.center + d_point + p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 4.0),
            width=Scalar(self.PATH_SIZE),
            height=Scalar(self.PATH_SIZE / 2.0)
        )

        # Create path
        path2 = Path.from_objects(
            path=path_segment,
            subpaths=[rectangle],
            line_color=utils.RED.copy(),
            line_width=Scalar(self.PATH_LINE_WIDTH),
            fill_color=utils.GREEN.copy(),
            closed_path=True
        )

        return path2
    # end build_second_path

    def build_vector_graphics(self, coord_system: CoordSystem):
        """
        Build the vector graphics.
        """
        # Get paths
        path1 = self.build_first_path(coord_system)
        path2 = self.build_second_path(coord_system)

        # Create a vector graphics
        vector_graphics = VectorGraphics(
            paths=[path1, path2]
        )


    # end build_vector_graphics

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

        # Create a vector graphics
        vector_graphics = VectorGraphics(

        )

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
        drawing_layer.render(
            draw_params={
                'draw_bboxes': True,
                'draw_points': True,
                'draw_reference_point': True,
                'draw_control_points': True
            }
        )

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end VectorGraphicsAnimation
