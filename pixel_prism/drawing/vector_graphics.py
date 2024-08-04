

from typing import Iterator
from pixel_prism.animate.able import MovAble, FadeInAble, FadeOutAble
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
        anchor_point: Anchor = utils.Anchor.UPPER_LEFT
):
    """
    Load an SVG file and return the paths and transformations.

    Args:
        svg_path (str): Path to the SVG file
        vector_graphics (VectorGraphicsData): Vector graphics object to load the SVG into
        color (Color): Color of the SVG
        centered (bool): Whether to center the SVG
        anchor_point (int): Anchor point for the SVG
    """
    # Parse the SVG file
    paths = parse_svg(svg_path)

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
                fill_color=color
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
                                enter=Point2D(segment.center.real + x, segment.center.imag + y),
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
            vector_graphics.add(path_data)
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
                fill_color=color
            )
            vector_graphics.add(rec)
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


class VectorGraphics(DrawableMixin, MovAble, FadeInAble, FadeOutAble):

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
        self._index = 0 # Index of the current

        # Debugging circle
        self.debug_circle = Circle(
            0,
            0,
            fill_color=utils.RED.copy().change_alpha(0.75),
            radius=Scalar(0.4),
            border_width=Scalar(0)
        )
    # end __init__

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
    def add(self, element):
        """
        Add an element to the vector graphic.

        Args:
            element: Element to add to the vector graphic
        """
        self.elements.append(element)
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
            context
    ):
        """
        Draw the vector graphics to the context.

        Args:
            context (cairo.Context): Context to draw the vector graphics to
        """
        # Move the context
        context.save()
        context.translate(self.position.x, self.position.y)
        context.scale(self.scale.x, self.scale.y)

        # Draw a circle
        self.debug_circle.draw(context)

        # For each element in the vector graphics
        for element in self.elements:
            element.draw(context)
        # end for

        # Draw rectangle bounding box
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

        # Restore the context
        context.restore()
    # end draw

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
        return self.elements[index]
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
            color: Color = utils.WHITE
    ):
        """
        Create a vector graphic from an SVG string.

        Args:
            cls (type): Class
            svg_path (str): SVG string
            position (Point2D): Position of the vector graphic
            scale (Point2D): Scale of the vector graphic
            color (Color): Color of the vector graphic

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
            anchor_point=utils.Anchor.MIDDLE_CENTER
        )

        return vector_graphics
    # end from_svg

# end VectorGraphics

