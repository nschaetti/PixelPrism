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
# Animation of an path.
# Build and highlight
#

# Imports
import math
from pixel_prism import p2, s, c
from pixel_prism import utils
from pixel_prism.animate.animate import Rotate, Scale
from pixel_prism.animation import Animation
from pixel_prism.animate import Move, EaseInOutInterpolator, Call
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas, CoordSystem
from pixel_prism.drawing import (
    Path,
    PathSegment,
    PathLine,
    PathBezierCubic,
    PathArc
)
from pixel_prism.data import Scalar


# A path animation class
class PathAnimation(Animation):

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
        # Curve control points
        control_points = [
            coord_system.upper_left_square + p2(self.PATH_SIZE, 0),
            coord_system.upper_left_square + p2(0, self.PATH_SIZE),
            p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 2.0),
            p2(0, self.PATH_SIZE / 2.0),
            coord_system.upper_left_square + p2(0, -self.PATH_SIZE)
        ]

        # Subpath control points
        subpath_control_points = [
            coord_system.upper_left_square +  p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 4.0)
        ]

        # Create path segment
        path_segment = PathSegment.from_objects(
            start=control_points[0],
            elements=[
                PathLine(control_points[0], control_points[1]),
                PathArc(center=coord_system.upper_left_square, radius=s(self.PATH_SIZE), start_angle=s(math.pi / 2), end_angle=s(math.pi)),
                PathBezierCubic(start=coord_system.upper_left_square + p2(-self.PATH_SIZE, 0), control1=control_points[2], control2=control_points[3], end=control_points[4])
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

        # Moving rectangle
        self.animate(
            Move(
                subpath_control_points[0],
                start_time=0,
                end_time=8,
                target_value=coord_system.upper_left_square + p2(-self.PATH_SIZE * 1.5   , -self.PATH_SIZE / 4.0),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Moving starting point
        self.animate(
            Move(
                control_points[0],
                start_time=0,
                end_time=4,
                target_value=coord_system.upper_left_square + p2(self.PATH_SIZE * 1.5, 0),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Moving point 2
        self.animate(
            Move(
                control_points[1],
                start_time=0,
                end_time=4,
                target_value=coord_system.upper_left_square + p2(0, self.PATH_SIZE * 1.5),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Moving control 1
        self.animate(
            Move(
                control_points[2],
                start_time=0,
                end_time=4,
                target_value=p2(0, -self.PATH_SIZE),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Moving control 1
        self.animate(
            Move(
                control_points[3],
                start_time=4,
                end_time=8,
                target_value=p2(0, -self.PATH_SIZE),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Moving end
        self.animate(
            Move(
                control_points[4],
                start_time=0,
                end_time=8,
                target_value=coord_system.upper_left_square + p2(self.PATH_SIZE * 0.5, -self.PATH_SIZE),
                interpolator=EaseInOutInterpolator()
            )
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
        # Create path segment
        path_segment = PathSegment.from_objects(
            start=coord_system.upper_right_square + p2(self.PATH_SIZE, 0),
            elements=[
                PathLine(
                    coord_system.upper_right_square + p2(self.PATH_SIZE, 0),
                    coord_system.upper_right_square + p2(0, self.PATH_SIZE)
                ),
                PathArc(
                    center=coord_system.upper_right_square,
                    radius=s(self.PATH_SIZE),
                    start_angle=s(math.pi / 2),
                    end_angle=s(math.pi)
                ),
                PathBezierCubic(
                    start=coord_system.upper_right_square + p2(-self.PATH_SIZE, 0),
                    control1=p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 2.0),
                    control2=p2(0, self.PATH_SIZE / 2.0),
                    end=coord_system.upper_right_square + p2(0, -self.PATH_SIZE)
                )
            ]
        )

        # Rectangle
        rectangle = PathSegment.rectangle(
            lower_left=coord_system.upper_right_square + p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 4.0),
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

        # Moving rectangle
        self.animate(
            Move(
                path_segment,
                start_time=0,
                end_time=8,
                target_value=coord_system.upper_right_square + p2(self.PATH_SIZE - 0.5, 0),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Moving rectangle
        self.animate(
            Move(
                rectangle,
                start_time=0,
                end_time=8,
                target_value=coord_system.upper_right_square + p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 4.0 + 0.4),
                interpolator=EaseInOutInterpolator()
            )
        )

        return path2
    # end build_second_path

    # Build the third path
    def build_third_path(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the third curve.

        Args:
            coord_system (CoordSystem): Coordinate system
        """
        # Create path segment
        path_segment = PathSegment.from_objects(
            start=coord_system.lower_left_square + p2(self.PATH_SIZE, 0),
            elements=[
                PathLine(
                    coord_system.lower_left_square + p2(self.PATH_SIZE, 0),
                    coord_system.lower_left_square + p2(0, self.PATH_SIZE)
                ),
                PathArc(
                    center=coord_system.lower_left_square,
                    radius=s(self.PATH_SIZE),
                    start_angle=s(math.pi / 2),
                    end_angle=s(math.pi)
                ),
                PathBezierCubic(
                    start=coord_system.lower_left_square + p2(-self.PATH_SIZE, 0),
                    control1=p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 2.0),
                    control2=p2(0, self.PATH_SIZE / 2.0),
                    end=coord_system.lower_left_square + p2(0, -self.PATH_SIZE)
                )
            ]
        )

        # Rectangle
        rectangle = PathSegment.rectangle(
            lower_left=coord_system.lower_left_square + p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 4.0),
            width=Scalar(self.PATH_SIZE),
            height=Scalar(self.PATH_SIZE / 2.0)
        )

        # Create path
        path3 = Path.from_objects(
            path=path_segment,
            subpaths=[rectangle],
            line_color=utils.RED.copy(),
            line_width=Scalar(self.PATH_LINE_WIDTH),
            fill_color=utils.GREEN.copy(),
            closed_path=True
        )

        # Rotating internal path
        self.animate(
            Rotate(
                rectangle,
                start_time=0,
                end_time=8,
                target_value=math.pi * 2.0,
                interpolator=EaseInOutInterpolator(),
                center=coord_system.lower_left_square
            )
        )

        # Rotating internal path
        self.animate(
            Rotate(
                path_segment,
                start_time=0,
                end_time=8,
                target_value=-math.pi * 2.0,
                interpolator=EaseInOutInterpolator(),
                center=coord_system.lower_left_square
            )
        )

        return path3
    # end build_third_path

    # Build fourth path
    def build_fourth_path(
            self,
            coord_system: CoordSystem
    ):
        """
        Build the fourth curve.

        Args:
            coord_system (CoordSystem): Coordinate system
        """
        # Create path segment
        path_segment = PathSegment.from_objects(
            start=coord_system.lower_right_square + p2(self.PATH_SIZE, 0),
            elements=[
                PathLine(
                    coord_system.lower_right_square + p2(self.PATH_SIZE, 0),
                    coord_system.lower_right_square + p2(0, self.PATH_SIZE)
                ),
                PathArc(
                    center=coord_system.lower_right_square.copy(),
                    radius=s(self.PATH_SIZE),
                    start_angle=s(math.pi / 2),
                    end_angle=s(math.pi)
                ),
                PathBezierCubic(
                    start=coord_system.lower_right_square + p2(-self.PATH_SIZE, 0),
                    control1=p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 2.0),
                    control2=p2(0, self.PATH_SIZE / 2.0),
                    end=coord_system.lower_right_square + p2(0, -self.PATH_SIZE)
                )
            ]
        )

        # Rectangle
        rectangle = PathSegment.rectangle(
            lower_left=coord_system.lower_right_square + p2(-self.PATH_SIZE / 2.0, -self.PATH_SIZE / 4.0),
            width=Scalar(self.PATH_SIZE),
            height=Scalar(self.PATH_SIZE / 2.0)
        )

        # Create path
        path4 = Path.from_objects(
            path=path_segment,
            subpaths=[rectangle],
            line_color=utils.RED.copy(),
            line_width=Scalar(self.PATH_LINE_WIDTH),
            fill_color=utils.GREEN.copy(),
            closed_path=True
        )

        # Moving path
        self.animate(
            Move(
                path4,
                start_time=0,
                end_time=2,
                target_value=p2(self.PATH_SIZE, 0.0),
                interpolator=EaseInOutInterpolator(),
                relative=True
            )
        )

        # Rotate path
        self.animate(
            Rotate(
                path4,
                start_time=2,
                end_time=4,
                target_value=math.pi * 2.0,
                interpolator=EaseInOutInterpolator(),
                center=path4.path.start
            )
        )

        # Moving path
        self.animate(
            Move(
                path4,
                start_time=4,
                end_time=6,
                target_value=p2(-self.PATH_SIZE, 0.0),
                interpolator=EaseInOutInterpolator(),
                relative=True
            )
        )

        # Scale path
        self.animate(
            Scale(
                path4,
                start_time=6,
                end_time=8,
                target_value=Scalar(2.0),
                interpolator=EaseInOutInterpolator(),
                center=path4.path.start
            )
        )

        # Scale path
        self.animate(
            Scale(
                path4,
                start_time=8,
                end_time=10,
                target_value=Scalar(0.5),
                interpolator=EaseInOutInterpolator(),
                center=coord_system.lower_right_square + p2(self.PATH_SIZE, 0)
            )
        )

        return path4
    # end build_fourth_path

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

        # Create paths
        path1 = self.build_first_path(coord_system)
        path2 = self.build_second_path(coord_system)
        path3 = self.build_third_path(coord_system)
        path4 = self.build_fourth_path(coord_system)

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(path1)
        drawable_widget.add(path2)
        drawable_widget.add(path3)
        drawable_widget.add(path4)

        # Add objects
        self.add(
            coord_system=coord_system,
            viewport=viewport,
            drawable_widget=drawable_widget,
            path1=path1,
            path2=path2,
            path3=path3,
            path4=path4
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

# end PathAnimation
