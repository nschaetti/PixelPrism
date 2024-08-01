

# Imports
import svgpathtools as svg
import pprint
from pixel_prism.utils import parse_svg, parse_path

from .data import Data
from .points import Point2D
from .lines import Line
from .curves import CubicBezierCurve, QuadraticBezierCurve
from .arcs import Arc
from .rectangles import Rectangle
from .scalar import Scalar
from .paths import Path, PathSegment
from .transforms import *


def create_transform(trans):
    """
    Apply an SVG transform to a Cairo context.

    Args:
        trans (str): SVG transform
    """
    # Translate
    if trans is None:
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
        vector_graphics: 'VectorGraphics'
):
    """
    Load an SVG file and return the paths and transformations.

    Args:
        svg_path (str): Path to the SVG file
        vector_graphics (VectorGraphics): Vector graphics object to load the SVG into
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
                origin=Point2D(x, y),
                transform=transform
            )

            # Get subpaths
            subpaths = element['data'].d().split('M')

            # For each subpaths
            for subpath_i, subpath in enumerate(subpaths):
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
                        subpath_data.add(
                            Line(
                                start=Point2D(segment.start.real, segment.start.imag),
                                end=Point2D(segment.end.real, segment.end.imag)
                            )
                        )
                    elif isinstance(segment, svg.CubicBezier):
                        subpath_data.add(
                            CubicBezierCurve(
                                start=Point2D(segment.start.real, segment.start.imag),
                                control1=Point2D(segment.control1.real, segment.control1.imag),
                                control2=Point2D(segment.control2.real, segment.control2.imag),
                                end=Point2D(segment.end.real, segment.end.imag)
                            )
                        )
                    elif isinstance(segment, svg.QuadraticBezier):
                        subpath_data.add(
                            QuadraticBezierCurve(
                                start=Point2D(segment.start.real, segment.start.imag),
                                control=Point2D(segment.control.real, segment.control.imag),
                                end=Point2D(segment.end.real, segment.end.imag)
                            )
                        )
                    elif isinstance(segment, svg.Arc):
                        subpath_data.add(
                            Arc(
                                center=Point2D(segment.center.real, segment.center.imag),
                                radius=Scalar(segment.radius),
                                start_angle=Scalar(segment.start_angle),
                                end_angle=Scalar(segment.end_angle)
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
            # end for

            # Add the path to the vector graphics
            vector_graphics.add(path_data)
        elif element['type'] == 'rect':
            # Add a rectangle
            rec = Rectangle(
                upper_left=Point2D(element['x'], element['y']),
                width=element['width'],
                height=element['height']
            )
            vector_graphics.add(rec)
        # end if
    # end for
# end load_svg


# A class to represent a vector graphic in 2D space.
class VectorGraphics(Data):
    """
    A class to represent a vector graphic in 2D space.
    """

    def __init__(self, elements=None):
        """
        Initialize the vector graphic with its elements.

        Args:
            elements (list): Elements of the vector graphic
        """
        super().__init__()

        # Initialize the elements
        self.elements = elements if elements is not None else []
    # end __init__

    # Add
    def add(self, element):
        """
        Add an element to the vector graphic.

        Args:
            element: Element to add to the vector graphic
        """
        self.elements.append(element)
    # end add

    # Get
    def get(self):
        """
        Get the elements of the vector graphic.

        Returns:
            list: Elements of the vector graphic
        """
        return self.elements
    # end get

    # Set
    def set(self, elements):
        """
        Set the elements of the vector graphic.

        Args:
            elements (list): Elements of the vector graphic
        """
        self.elements = elements
    # end set

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

    @staticmethod
    def from_svg(svg_path):
        """
        Create a vector graphic from an SVG string.

        Args:
            svg_path (str): SVG string

        Returns:
            VectorGraphics: Vector graphic
        """
        # Create a new vector graphic
        vector_graphics = VectorGraphics()

        # Parse the SVG string
        load_svg(svg_path, vector_graphics)

        return vector_graphics
    # end from_svg

# end VectorGraphics
