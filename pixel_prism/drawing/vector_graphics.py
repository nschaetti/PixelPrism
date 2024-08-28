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

# Imports
from typing import Iterator, List, Optional, Any, Union
from pixel_prism.animate.able import (
    MovableMixin,
    FadeInableMixin,
    FadeOutableMixin,
    BuildableMixin,
    DestroyableMixin
)
from pixel_prism.data import Point2D, Color, EventMixin, ObjectChangedEvent
from pixel_prism import utils
from pixel_prism.utils.svg import parse_svg, parse_path
from pixel_prism.utils import Anchor
import svgpathtools as svg

from . import BoundingBoxMixin
from .rectangles import Rectangle
from .paths import (
    Path,
    PathSegment,
    PathLine,
    PathBezierQuadratic,
    PathBezierCubic,
    PathArc
)
from .transforms import *
from .drawablemixin import DrawableMixin


def create_transform(
        trans
):
    """
    Apply an SVG transform to a Cairo context.

    Args:
        trans (str): SVG transform
    """
    # Translate
    if trans is None or trans == '':
        return None
    elif trans.startswith('translate'):
        values = trans[10:-1].split(',')
        tx = float(values[0])
        ty = float(values[1]) if len(values) > 1 else 0.0
        transform = Translate2D(tx, ty)
    # Scale
    elif trans.startswith('scale'):
        values = trans[6:-1].split(',')
        sx = float(values[0])
        sy = float(values[1]) if len(values) > 1 else sx
        transform = Scale2D(sx, sy)
    # Rotate
    elif trans.startswith('rotate'):
        values = trans[7:-1].split(',')
        angle = float(values[0])
        cx = float(values[1]) if len(values) > 1 else 0.0
        cy = float(values[2]) if len(values) > 2 else 0.0
        transform = Rotate2D(cx, cy, angle)
    elif trans.startswith('skewX'):
        angle = np.radians(float(trans[6:-1]))
        transform = SkewX2D(angle)
    elif trans.startswith('skewY'):
        angle = np.radians(float(trans[6:-1]))
        transform = SkewY2D(angle)
    elif trans.startswith('matrix'):
        values = trans[7:-1].split(',')
        transform = Matrix2D(
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5])
        )
    else:
        raise ValueError(f"Unknown transform: {trans}")
    # end if

    return transform
# end create_transform


# Load an SVG
def load_svg(
        svg_path,
        vector_graphics: 'VectorGraphics',
        color: Color = utils.WHITE,
        anchor_point: Anchor = utils.Anchor.UPPER_LEFT,
        refs: Optional[List] = None
):
    """
    Load an SVG file and return the paths and transformations.

    Args:
        svg_path (str): Path to the SVG file
        vector_graphics (VectorGraphicsData): Vector graphics object to load the SVG into
        color (Color): Color of the SVG
        centered (bool): Whether to center the SVG
        anchor_point (int): Anchor point for the SVG
        refs (list): List of references
    """
    # Parse the SVG file
    paths = parse_svg(svg_path)

    assert refs is None or len(paths) == len(refs), "Number of paths and references must match"

    # Draw the paths
    for el_i, element in enumerate(paths):
        # Get transformations
        transform = create_transform(element['transform'])

        # Element position
        x = element['x']
        y = element['y']

        # We have a path
        if element['type'] == 'path':
            # Main & subpaths
            main_path = None
            sub_paths = list()

            # Get subpaths
            subpaths_parsed = element['data'].d().split('M')

            # For each subpaths
            subpath_i = 0
            for subpath in subpaths_parsed:
                # Skip empty subpaths
                if not subpath.strip():
                    continue
                # end if

                # Move to the first point
                subpath = 'M' + subpath.strip()

                # Parse the subpath
                sub_path = parse_path(subpath)

                # Segment elements
                segment_elements = list()

                # Draw the segments
                for segment in sub_path:
                    # Bounding box
                    segment_bbox = segment.bbox()

                    # Add the segment
                    if isinstance(segment, svg.Line):
                        line = PathLine.from_objects(
                            start=Point2D(segment.start.real + x, segment.start.imag + y),
                            end=Point2D(segment.end.real + x, segment.end.imag + y)
                        )
                        segment_elements.append(line)
                    elif isinstance(segment, svg.CubicBezier):
                        segment_elements.append(
                            PathBezierCubic.from_objects(
                                start=Point2D(segment.start.real + x, segment.start.imag + y),
                                control1=Point2D(segment.control1.real + x, segment.control1.imag + y),
                                control2=Point2D(segment.control2.real + x, segment.control2.imag + y),
                                end=Point2D(segment.end.real + x, segment.end.imag + y),
                                bounding_box=Rectangle.from_bbox(
                                    bbox=(
                                        segment_bbox[0] + x,
                                        segment_bbox[1] + x,
                                        segment_bbox[2] + y,
                                        segment_bbox[3] + y
                                    )
                                )
                            )
                        )
                    elif isinstance(segment, svg.QuadraticBezier):
                        segment_elements.append(
                            PathBezierQuadratic.from_objects(
                                start=Point2D(segment.start.real + x, segment.start.imag + y),
                                control=Point2D(segment.control.real + x, segment.control.imag + y),
                                end=Point2D(segment.end.real + x, segment.end.imag + y),
                                bounding_box=Rectangle.from_bbox(
                                    bbox=(
                                        segment_bbox[0] + x,
                                        segment_bbox[1] + x,
                                        segment_bbox[2] + y,
                                        segment_bbox[3] + y
                                    )
                                )
                            )
                        )
                    elif isinstance(segment, svg.Arc):
                        segment_elements.append(
                            PathArc.from_objects(
                                center=Point2D(segment.center.real + x, segment.center.imag + y),
                                radius=Scalar(segment.radius),
                                start_angle=Scalar(segment.start_angle),
                                end_angle=Scalar(segment.end_angle),
                                bounding_box=Rectangle.from_bbox(
                                    bbox=(
                                        segment_bbox[0] + x,
                                        segment_bbox[1] + x,
                                        segment_bbox[2] + y,
                                        segment_bbox[3] + y
                                    )
                                )
                            )
                        )
                    else:
                        raise ValueError(f"Unknown segment type: {segment}")
                    # end if
                # end for

                # New path
                path_segment = PathSegment(segment_elements)

                # Add the subpath to the path
                if subpath_i == 0:
                    main_path = path_segment
                else:
                    sub_paths.append(path_segment)
                # end if

                # Increment the subpath index
                subpath_i += 1
            # end for

            # New path
            path_data = Path(
                origin=Point2D(0, 0),
                path=main_path,
                subpaths=sub_paths,
                line_width=Scalar(0.0),
                transform=transform,
                fill_color=color.copy()
            )

            # Add the path to the vector graphics
            vector_graphics.add(path_data, refs[el_i] if refs is not None else None)
        elif element['type'] == 'rect':
            # Add a rectangle
            rec = Rectangle(
                upper_left=Point2D(
                    element['x'],
                    element['y']
                ),
                width=element['width'],
                height=element['height'],
                border_width=Scalar(0.0),
                fill_color=color.copy()
            )
            vector_graphics.add(rec, refs[el_i] if refs is not None else None)
        else:
            raise ValueError(f"Unknown element type: {element['type']}")
        # end if
    # end for

    # Compute the bounding box
    bbox = vector_graphics.bounding_box
    print(f"Bounding box: {bbox}")
    print(type(bbox.width))
    # Put to anchor point
    for element in vector_graphics.elements:
        if anchor_point == utils.Anchor.UPPER_LEFT:
            element.translate(-bbox.x1, -bbox.y1)
        elif anchor_point == utils.Anchor.UPPER_CENTER:
            element.translate(-bbox.x1 - bbox.width.value / 2.0, -bbox.y1)
        elif anchor_point == utils.Anchor.UPPER_RIGHT:
            element.translate(-bbox.x2, -bbox.y1)
        elif anchor_point == utils.Anchor.MIDDLE_LEFT:
            element.translate(-bbox.x1, -bbox.y1 - bbox.height.value / 2.0)
        elif anchor_point == utils.Anchor.MIDDLE_CENTER:
            element.translate(-bbox.x1 - bbox.width.value / 2.0, -bbox.y1 - bbox.height.value / 2.0)
        elif anchor_point == utils.Anchor.MIDDLE_RIGHT:
            element.translate(-bbox.x2, -bbox.y1 - bbox.height.value / 2)
        elif anchor_point == utils.Anchor.LOWER_LEFT:
            element.translate(-bbox.x1, -bbox.y2)
        elif anchor_point == utils.Anchor.LOWER_CENTER:
            element.translate(-bbox.x1 - bbox.width.value / 2, -bbox.y2)
        elif anchor_point == utils.Anchor.LOWER_RIGHT:
            element.translate(-bbox.x2, -bbox.y2)
        # end if
    # end for
# end load_svg


# A class to group paths
class PathGroup(
    BoundingBoxMixin,
    EventMixin
):
    """
    A class to group paths.
    """

    def __init__(
            self,
            paths: Optional[List[Path]] = None
    ):
        """
        Initialize the path group.

        Args:
            paths (List[Path]): Paths to group
        """
        # Properties
        self._paths = paths if paths is not None else list()

        # Bounding box
        BoundingBoxMixin.__init__(self)
        EventMixin.__init__(self)

        # Events
        self.add_event('on_change')
        for path in self._paths:
            path.add_event_listener("on_change", self._on_path_changed)
        # end for
    # end __init__

    # region PROPERTIES

    @property
    def paths(self) -> List[Path]:
        """
        Get the paths of the path group.

        Returns:
            List[Path]: Paths of the path group
        """
        return self._paths
    # end paths

    # endregion PROPERTIES

    # region PUBLIC

    # Update data
    def update_data(self):
        """
        Update the data of the vector graphics.
        """
        self.update_bbox()
    # end update_data

    # Update bounding box
    def update_bbox(self):
        """
        Update the bounding box of the vector graphics.
        """
        if len(self._paths) > 0:
            # Get the min and max values
            min_x = min([el.bounding_box.x1 for el in self._paths])
            min_y = min([el.bounding_box.y1 for el in self._paths])
            max_x = max([el.bounding_box.x2 for el in self._paths])
            max_y = max([el.bounding_box.y2 for el in self._paths])

            # Update the bounding box
            self._bounding_box.upper_left.x = min_x
            self._bounding_box.upper_left.y = min_y
            self._bounding_box.width = max_x - min_x
            self._bounding_box.height = max_y - min_y
        # end if
    # end update_bbox

    # endregion PUBLIC

    # region EVENT

    # On path changed
    def _on_path_changed(self, event):
        """
        On path changed event.
        """
        self.update_data()
        self.dispatch_event('on_change', event=event)
    # end _on_path_changed

    # endregion EVENT

    # region FADE_IN

    def init_fadein(self):
        """
        Initialize the fadein animation.
        """
        pass
    # end init_fadein

    def start_fadein(
            self,
            start_value: Any,
            *args,
            **kwargs
    ):
        """
        Start fading in the vector graphic.
        """
        # Reset the last animated element
        self.last_animated_element = -1
    # end start_fadein

    def animate_fadein(
            self,
            t,
            duration,
            interpolated_t,
            end_value
    ):
        """
        Animate fading in the vector graphic.
        """
        # Time of animation divided by elements
        t_per_element = duration / len(self.elements)

        # Check on which element we are
        i = min(int(interpolated_t * len(self.elements)), len(self.elements) - 1)

        # Element time
        element_t = (interpolated_t - i / len(self.elements)) * len(self.elements)

        # Check if it is the first time
        if i != self.last_animated_element:
            self.last_animated_element = i
            if i > 0:  # Start the fadein animation
                self.elements[i - 1].end_fadein(1)
            # end if
            self.elements[i].start_fadein(0)
        else:
            # Animate the element
            self.elements[i].animate_fadein(
                element_t,
                t_per_element,
                element_t,
                end_value
            )
        # end if
    # end animate_fadein

    def end_fadein(self, end_value: Any):
        """
        End the fade-in animation.
        """
        pass
    # end end_fadein

    def finish_fadein(self):
        """
        Finish the fade-in animation.
        """
        pass
    # end finish_fadein

    # endregion FADE_IN

    # region OVERRIDE

    # Get path by index
    def __getitem__(self, item):
        """
        Get the path by index.
        """
        return self._paths[item]
    # end __getitem__

    # Set path by index
    def __setitem__(self, key, value):
        """
        Set the path by index.
        """
        self._paths[key] = value
    # end __setitem__

    # Get the length
    def __len__(self):
        """
        Get the length of the path group.
        """
        return len(self._paths)
    # end __len__

    # Get the iterator
    def __iter__(self):
        """
        Get the iterator of the path group.
        """
        return iter(self._paths)
    # end __iter__

    # Get the string representation
    def __str__(self):
        """
        Get the string representation of the path group.
        """
        return f"PathGroup(paths={self._paths})"
    # end __str__

    # endregion OVERRIDE

# end PathGroup



class VectorGraphics(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    FadeInableMixin,
    FadeOutableMixin,
    BuildableMixin,
    DestroyableMixin
):
    """
    Vector graphics class.
    """

    def __init__(
            self,
            paths=None
    ):
        """
        Initialize the vector graphics

        Args:
            paths (List[Path]): Paths of the vector graphics
        """
        # Initialize the elements
        self._path_group = PathGroup(elements)
        self._references = {}
        self._index = 0

        # Init of VectorGraphicsData
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)
        FadeInableMixin.__init__(self)
        FadeOutableMixin.__init__(self)
        BuildableMixin.__init__(self, True, 1.0)

        # Fadein, fadeout
        self.last_animated_element = None
        self.build_animated_element = None

        # Debugging circle
        """self.reference_point = Circle(
            0,
            0,
            fill_color=utils.RED.copy().change_alpha(0.75),
            border_color=utils.WHITE.copy(),
            radius=Scalar(0.5),
            border_width=Scalar(0.05)
        )"""
    # end __init__

    # region PROPERTIES

    @property
    def path_group(self) -> PathGroup:
        """
        Get the path group of the vector graphics.

        Returns:
            PathGroup: Path group of the vector graphics
        """
        return self._path_group
    # end path_group

    @property
    def references(self) -> dict:
        """
        Get the references of the vector graphics.

        Returns:
            dict: References of the vector graphics
        """
        return self._references
    # end references

    # endregion PROPERTIES

    # region PUBLIC

    # Update bounding box
    def update_bbox(self):
        """
        Update the bounding box of the vector graphics.
        """
        # Update the bounding box
        self._bounding_box.upper_left.x = self._path_group.bounding_box.x
        self._bounding_box.upper_left.y = self._path_group.bounding_box.y
        self._bounding_box.width = self._path_group.bounding_box.width
        self._bounding_box.height = self._path_group.bounding_box.height
    # end update_bbox

    # Set alpha
    def set_alpha(self, alpha: float):
        """
        Set the alpha of the vector graphics.

        Args:
            alpha (float): Alpha value
        """
        for element in self.elements:
            element.set_alpha(alpha)
        # end for
    # end set_alpha

    # Add
    def add(
            self,
            element: Union[List[Any], Any],
            ref: str = None
    ):
        """
        Add an element to the vector graphic.

        Args:
            element: Element to add to the vector graphic
            ref (str): Reference of the element
        """
        if isinstance(element, list):
            for e_i, el in enumerate(element):
                self.elements.append(el)
                if ref is not None:
                    self.references[ref[e_i]] = el
                # end if
            # end for
        else:
            self.elements.append(element)
            if ref is not None:
                self.references[ref] = element
            # end if
        # end if

        # Update the bounding box
        self.update_bbox()
    # end add

    # endregion PUBLIC

    # region DRAW

    # Draw bounding box anchors
    def draw_bounding_box_anchors(self, context):
        """
        Draw the bounding box anchors of the vector graphics.
        """
        # Bounding box
        path_bbox = self.bounding_box

        # Draw upper left position
        upper_left = path_bbox.upper_left
        context.rectangle(
            upper_left.x - 0.25,
            upper_left.y - 0.25,
            0.5,
            0.5
        )
        context.set_source_rgba(255, 255, 255, 1)
        context.fill()

        # Draw upper left position
        context.rectangle(
            path_bbox.x2 - 0.25,
            path_bbox.y2 - 0.25,
            0.5,
            0.5
        )
        context.set_source_rgba(255, 255, 255, 1)
        context.fill()

        # Draw text upper left
        context.set_font_size(0.6)
        point_position = f"({path_bbox.x1:0.02f}, {path_bbox.y1:0.02f})"
        extents = context.text_extents(point_position)
        context.move_to(path_bbox.x1 - extents.width / 2, path_bbox.y1 - extents.height)
        context.show_text(point_position)
        context.fill()

        # Draw text bottom right
        context.set_font_size(0.6)
        point_position = f"({path_bbox.x2:0.02f}, {path_bbox.y2:0.02f})"
        extents = context.text_extents(point_position)
        context.move_to(path_bbox.x2 - extents.width / 2, path_bbox.y2 + extents.height * 2)
        context.show_text(point_position)
        context.fill()
    # end draw_bbox_anchors

    # Draw bounding box
    def draw_bounding_box(self, context):
        """
        Draw the bounding box of the vector graphics.
        """
        # Set the color and draw the rectangle
        self._bounding_box.draw(context)
    # end draw_bbox

    def draw(
            self,
            context,
            draw_bboxes: bool = False,
            draw_reference_point: bool = False,
            draw_paths: bool = False
    ):
        """
        Draw the vector graphics to the context.

        Args:
            context (cairo.Context): Context to draw the vector graphics to
            draw_bboxes (bool): Whether to draw the debug information
            draw_reference_point (bool): Whether to draw the reference point
            draw_paths (bool): Whether to draw the paths
        """
        # Move the context
        context.save()
        context.translate(self.position.x, self.position.y)
        context.scale(self.scale.x, self.scale.y)

        # Draw a circle
        """if draw_reference_point:
            self.reference_point.draw(context)
        # end if"""

        # For each element in the vector graphics
        for element in self.elements:
            element.draw(
                context,
                draw_bboxes=draw_bboxes,
                draw_reference_point=draw_reference_point
            )
        # end for

        # Draw rectangle bounding box
        if draw_bboxes:
            for element in self.elements:
                if type(element) is Rectangle:
                    # Draw the bounding box
                    element.draw_bounding_box(context)
                    element.draw_bounding_box_anchors(context)
                # end if
            # end for

            # Draw VG bounding box
            self.draw_bounding_box(context)
            self.draw_bounding_box_anchors(context)
        # end if

        # Draw paths
        if draw_paths:
            for element in self.elements:
                if type(element) is Path:
                    element.draw_paths(context)
                # end if
            # end for
        # end

        # Restore the context
        context.restore()
    # end draw

    # endregion DRAW

    # region PRIVATE

    # Get bounding box
    def _create_bbox(
            self,
            border_width: float = 1.0,
            border_color: Color = utils.WHITE
    ):
        """
        Get the bounding box of the vector graphics.

        Args:
            border_width (float): Width of the border
            border_color (Color): Color of the border
        """
        if len(self.elements) > 0:
            # Get the min and max values
            min_x = min([el.bounding_box.x1 for el in self.elements])
            min_y = min([el.bounding_box.y1 for el in self.elements])
            max_x = max([el.bounding_box.x2 for el in self.elements])
            max_y = max([el.bounding_box.y2 for el in self.elements])

            return Rectangle.from_objects(
                upper_left=Point2D(min_x, min_y),
                width=Scalar(max_x - min_x),
                height=Scalar(max_y - min_y),
                border_color=border_color,
                border_width=Scalar(border_width),
                fill=False
            )
        # end if

        # Empty bounding box
        return Rectangle.from_objects(
            upper_left=Point2D(0, 0),
            width=Scalar(0),
            height=Scalar(0),
            border_color=border_color,
            border_width=Scalar(border_width),
            fill=False
        )
    # end _create_bbox

    # endregion PRIVATE

    # region MOVABLE

    # Start moving
    def start_move(
            self,
            start_value: Any,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Start moving the vector graphic.
        """
        if relative:
            self.movablemixin_state.start_position = Point2D.null()
            self.movablemixin_state.last_position = Point2D.null()
        else:
            self.movablemixin_state.start_position = self.movable_position.copy()
        # end if
    # end start_moving

    # Animate move
    def animate_move(
            self,
            t,
            duration,
            interpolated_t,
            env_value,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Animate moving the vector graphic.
        """
        # New x, y
        self.position.x = self.start_position.x * (1 - interpolated_t) + env_value.x * interpolated_t
        self.position.y = self.start_position.y * (1 - interpolated_t) + env_value.y * interpolated_t
    # end animate_move

    # endregion MOVABLE

    # region FADE_IN

    def start_fadein(
            self,
            start_value: Any,
            *args,
            **kwargs
    ):
        """
        Start fading in the vector graphic.
        """
        # Reset the last animated element
        self.last_animated_element = -1
    # end start_fadein

    def animate_fadein(
            self,
            t,
            duration,
            interpolated_t,
            end_value
    ):
        """
        Animate fading in the vector graphic.
        """
        # Time of animation divided by elements
        t_per_element = duration / len(self.elements)

        # Check on which element we are
        i = min(int(interpolated_t * len(self.elements)), len(self.elements) - 1)

        # Element time
        element_t = (interpolated_t - i / len(self.elements)) * len(self.elements)

        # Check if it is the first time
        if i != self.last_animated_element:
            self.last_animated_element = i
            if i > 0: # Start the fadein animation
                self.elements[i - 1].end_fadein(1)
            # end if
            self.elements[i].start_fadein(0)
        else:
            # Animate the element
            self.elements[i].animate_fadein(
                element_t,
                t_per_element,
                element_t,
                end_value
            )
        # end if
    # end animate_fadein

    # endregion FADE_IN

    # region FADE_OUT

    def start_fadeout(self, start_value: Any):
        """
        Start fading out the vector graphic.
        """
        self.last_animated_element = -1
    # end start_fadeout

    def animate_fadeout(self, t, duration, interpolated_t, end_value):
        """
        Animate fading out the vector graphic.
        """
        # Time of animation divided by elements
        t_per_element = duration / len(self.elements)

        # Check on which element we are
        i = min(int((1 - interpolated_t) * len(self.elements)), len(self.elements) - 1)

        # Element time
        element_t = (interpolated_t - i / len(self.elements)) * len(self.elements)

        # Check if it is the first time
        if i != self.last_animated_element:
            self.last_animated_element = i
            if i < len(self.elements) - 1:
                self.elements[i + 1].end_fadeout(1)
            # end if
            self.elements[i].start_fadeout(0)
        else:
            # Animate the element
            self.elements[i].animate_fadeout(
                element_t,
                t_per_element,
                element_t,
                end_value
            )
        # end if
    # end animate_fadeout

    # endregion FADE_OUT

    # region BUILD

    # Initialize building
    def init_build(self):
        """
        Initialize building the vector graphic.
        """
        super().init_build()
        for element in self.elements:
            element.init_build()
        # end for
    # end init_build

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the vector graphic.
        """
        super().start_build(start_value)
        self.build_animated_element = None
    # end start_build

    # Animate building
    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the vector graphic.
        """
        # Time of animation divided by elements
        t_per_element = duration / len(self.elements)

        # Get which element we are
        i = min(int(interpolated_t * len(self.elements)), len(self.elements) - 1)

        # New element ?
        if i != self.build_animated_element:
            self.build_animated_element = i
            if i > 0:
                self.elements[i - 1].end_build(1)
            # end if
            self.elements[i].start_build(0)
        # end if

        # Check on which element we are
        for e_i, element in enumerate(self.elements):
            if e_i == i:
                element.animate_build(
                    t,
                    t_per_element,
                    (interpolated_t - i / len(self.elements)) * len(self.elements),
                    env_value
                )
            # end if
            # element.animate_build(t, duration, interpolated_t, env_value)
        # end for
    # end animate_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the vector graphic.
        """
        super().end_build(end_value)
        self.build_animated_element = None
    # end end_build

    # Finish
    def finish_build(self):
        """
        Finish building the vector graphic.
        """
        super().finish_build()
        for element in self.elements:
            element.finish_build()
        # end for
    # end finish_build

    # endregion BUILD

    # region DESTROY

    # Initialize destroying
    def init_destroy(self):
        """
        Initialize destroying the vector graphic.
        """
        super().init_destroy()
        for element in self.elements:
            element.init_destroy()
        # end for
    # end init_destroy

    # Start destroying
    def start_destroy(self, start_value: Any):
        """
        Start destroying the vector graphic.
        """
        super().start_destroy(start_value)
        self.build_animated_element = None
    # end start_destroy

    # Animate destroying
    def animate_destroy(self, t, duration, interpolated_t, env_value):
        """
        Animate destroying the vector graphic.
        """
        # Time of animation divided by elements
        t_per_element = duration / len(self.elements)

        # Get which element we are
        i = min(int((1 - interpolated_t) * len(self.elements)), len(self.elements) - 1)

        # New element ?
        if i != self.build_animated_element:
            self.build_animated_element = i
            if i < len(self.elements) - 1:
                self.elements[i + 1].end_destroy(0)
            # end if
            self.elements[i].start_destroy(0)
        # end if

        # Check on which element we are
        for e_i, element in enumerate(self.elements):
            if e_i == i:
                new_interpolated_t = (interpolated_t - (len(self.elements) - i - 1) / len(self.elements)) * len(self.elements)
                element.animate_destroy(
                    t,
                    t_per_element,
                    # (interpolated_t - (len(self.elements) - i - 1) / len(self.elements)) * len(self.elements),
                    new_interpolated_t,
                    env_value
                )
            # end if
        # end for
    # end animate_destroy

    # End destroying
    def end_destroy(self, end_value: Any):
        """
        End destroying the vector graphic.
        """
        super().end_build(end_value)
        self.build_animated_element = None
    # end end_destroy

    # Finish destroying
    def finish_destroy(self):
        """
        Finish destroying the vector graphic.
        """
        super().finish_destroy()
        for element in self.elements:
            element.finish_destroy()
        # end for
    # end finish_destroy

    # endregion DESTROY

    # region OVERRIDE

    def __str__(self):
        """
        Get the string representation of the vector graphic.
        """
        return f"VectorGraphics(elements={self.elements})"
    # end __str__

    def __repr__(self):
        """
        Get the string representation of the vector graphic.
        """
        return self.__str__()
    # end __repr__

    def __len__(self):
        """
        Get the number of elements in the vector graphic.
        """
        return len(self.elements)
    # end __len__

    def __getitem__(self, index):
        """
        Get the element at the specified index.
        """
        if isinstance(index, str):
            return self.references[index]
        else:
            return self.elements[index]
        # end if
    # end __getitem__

    def __setitem__(self, index, value):
        """
        Set the element at the specified index.
        """
        self.elements[index] = value
    # end __setitem__

    def __delitem__(self, index):
        """
        Delete the element at the specified index.
        """
        del self.elements[index]
    # end __delitem__

    def __iter__(self) -> Iterator:
        """
        Return an iterator object for the elements.
        """
        self._index = 0
        return self
    # end __iter__

    def __next__(self):
        """
        Return the next element in the iteration.
        """
        if self._index < len(self.elements):
            result = self.elements[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
        # end if
    # end __next__

    # endregion OVERRIDE

    # region CLASS_METHODS

    @classmethod
    def from_svg(
            cls,
            svg_path,
            position: Point2D = Point2D(0, 0),
            scale: Point2D = Point2D(1, 1),
            color: Color = utils.WHITE,
            refs: Optional[List] = None,
    ):
        """
        Create a vector graphic from an SVG string.

        Args:
            cls (type): Class
            svg_path (str): SVG string
            position (Point2D): Position of the vector graphic
            scale (Point2D): Scale of the vector graphic
            color (Color): Color of the vector graphic
            refs (list): List of references

        Returns:
            VectorGraphicsData: Vector graphic
        """
        # Create a new vector graphic
        vector_graphics = cls(
            position=position,
            scale=scale
        )

        # Parse the SVG string
        load_svg(
            svg_path,
            vector_graphics,
            color=color,
            anchor_point=utils.Anchor.MIDDLE_CENTER,
            refs=refs
        )

        return vector_graphics
    # end from_svg

    # endregion CLASS_METHODS

# end VectorGraphics

