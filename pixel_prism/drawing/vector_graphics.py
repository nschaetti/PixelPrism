

from typing import Iterator, List, Optional, Any
from pixel_prism.animate.able import MovableMixin, FadeInableMixin, FadeOutableMixin, BuildableMixin
from pixel_prism.data import Point2D, Color
from pixel_prism import utils
from pixel_prism.utils.svg import parse_svg, parse_path
from pixel_prism.utils import Anchor
from pixel_prism.drawing import Circle
import svgpathtools as svg

from .lines import Line
from .curves import CubicBezierCurve, QuadraticBezierCurve
from .arcs import Arc
from .rectangles import Rectangle
from .paths import Path, PathSegment
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
            # New path
            path_data = Path(
                # origin=Point2D(x, y),
                origin=Point2D(0, 0),
                line_width=Scalar(0.0),
                transform=transform,
                fill_color=color.copy()
            )

            # Get subpaths
            subpaths = element['data'].d().split('M')

            # For each subpaths
            subpath_i = 0
            for subpath in subpaths:
                # Skip empty subpaths
                if not subpath.strip():
                    continue
                # end if

                # Move to the first point
                subpath = 'M' + subpath.strip()

                # Parse the subpath
                sub_path = parse_path(subpath)

                # New path
                subpath_data = PathSegment()

                # Draw the segments
                for segment in sub_path:
                    if isinstance(segment, svg.Line):
                        line = Line(
                            start=Point2D(segment.start.real + x, segment.start.imag + y),
                            end=Point2D(segment.end.real + x, segment.end.imag + y),
                            bbox=Rectangle.from_bbox(segment.bbox(), translate=(x, y))
                        )
                        subpath_data.add(line)
                    elif isinstance(segment, svg.CubicBezier):
                        subpath_data.add(
                            CubicBezierCurve(
                                start=Point2D(segment.start.real + x, segment.start.imag + y),
                                control1=Point2D(segment.control1.real + x, segment.control1.imag + y),
                                control2=Point2D(segment.control2.real + x, segment.control2.imag + y),
                                end=Point2D(segment.end.real + x, segment.end.imag + y),
                                bbox=Rectangle.from_bbox(segment.bbox(), translate=(x, y))
                            )
                        )
                    elif isinstance(segment, svg.QuadraticBezier):
                        subpath_data.add(
                            QuadraticBezierCurve(
                                start=Point2D(segment.start.real + x, segment.start.imag + y),
                                control=Point2D(segment.control.real + x, segment.control.imag + y),
                                end=Point2D(segment.end.real + x, segment.end.imag + y),
                                bbox=Rectangle.from_bbox(segment.bbox(), translate=(x, y))
                            )
                        )
                    elif isinstance(segment, svg.Arc):
                        subpath_data.add(
                            Arc(
                                center=Point2D(segment.center.real + x, segment.center.imag + y),
                                radius=Scalar(segment.radius),
                                start_angle=Scalar(segment.start_angle),
                                end_angle=Scalar(segment.end_angle),
                                bbox=Rectangle.from_bbox(segment.bbox(), translate=(x, y))
                            )
                        )
                    else:
                        raise ValueError(f"Unknown segment type: {segment}")
                    # end if
                # end for

                # Add the subpath to the path
                if subpath_i == 0:
                    path_data.path = subpath_data
                else:
                    path_data.add_subpath(subpath_data)
                # end if

                # Increment the subpath index
                subpath_i += 1
            # end for

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
    bbox = vector_graphics.get_bbox()

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


class VectorGraphics(DrawableMixin, MovableMixin, FadeInableMixin, FadeOutableMixin, BuildableMixin):

    def __init__(
            self,
            position: Point2D,
            scale: Point2D,
            elements=None
    ):
        """
        Initialize the vector graphics

        Args:
            elements (list): Elements of the vector graphics
        """
        # Init of VectorGraphicsData
        super().__init__()

        # Initialize the elements
        self.position = position
        self.scale = scale
        self.elements = elements if elements is not None else []
        self.references = {}
        self._index = 0 # Index of the current

        # Fadein, fadeout
        self.last_animated_element = None
        self.build_animated_element = None

        # Debugging circle
        self.reference_point = Circle(
            0,
            0,
            fill_color=utils.RED.copy().change_alpha(0.75),
            border_color=utils.WHITE.copy(),
            radius=Scalar(0.5),
            border_width=Scalar(0.05)
        )
    # end __init__

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

    # Get bounding box
    def get_bbox(
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
        # Get the min and max values
        min_x = min([el.get_bbox().x1 for el in self.elements])
        min_y = min([el.get_bbox().y1 for el in self.elements])
        max_x = max([el.get_bbox().x2 for el in self.elements])
        max_y = max([el.get_bbox().y2 for el in self.elements])

        return Rectangle(
            upper_left=Point2D(min_x, min_y),
            width=max_x - min_x,
            height=max_y - min_y,
            border_color=border_color,
            border_width=border_width,
            fill=False
        )
    # end get_bbox

    # Add
    def add(self, element, ref: str = None):
        """
        Add an element to the vector graphic.

        Args:
            element: Element to add to the vector graphic
            ref (str): Reference of the element
        """
        self.elements.append(element)

        if ref is not None:
            self.references[ref] = element
        # end if
    # end add

    # Draw bounding box anchors
    def draw_bounding_box_anchors(self, context):
        """
        Draw the bounding box anchors of the vector graphics.
        """
        # Bounding box
        path_bbox = self.get_bbox()

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
        self.get_bbox(
            border_width=0.12,
            border_color=utils.RED.copy()
        ).draw(context)
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
        if draw_reference_point:
            self.reference_point.draw(context)
        # end if

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

    # region MOVABLE

    # Start moving
    def start_move(
            self,
            start_value: Any
    ):
        """
        Start moving the vector graphic.
        """
        self.start_position = self.position.copy()
    # end start_moving

    # Animate move
    def animate_move(self, t, duration, interpolated_t, env_value):
        """
        Animate moving the vector graphic.
        """
        # New x, y
        self.position.x = self.start_position.x * (1 - interpolated_t) + env_value.x * interpolated_t
        self.position.y = self.start_position.y * (1 - interpolated_t) + env_value.y * interpolated_t
    # end animate_move

    # endregion MOVABLE

    # region FADE_IN

    def start_fadein(self, start_value: Any):
        """
        Start fading in the vector graphic.
        """
        # Reset the last animated element
        self.last_animated_element = -1
    # end start_fadein

    def animate_fadein(self, t, duration, interpolated_t, end_value):
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

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the vector graphic.
        """
        super().start_build(start_value)
        self.build_animated_element = None
    # end start_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the vector graphic.
        """
        super().end_build(end_value)
        self.build_animated_element = None
    # end end_build

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

    # endregion BUILD

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

# end VectorGraphics

