

from typing import List

import cairo

from pixel_prism.data import Point2D, Color, Scalar
import pixel_prism.utils as utils
from pixel_prism.drawing import Circle
from pixel_prism.animate.able import MovAble
from .drawablemixin import DrawableMixin


# Path segment
class PathSegment(DrawableMixin, MovAble):
    """
    A class to represent a path segment.
    """

    def __init__(self, elements=None):
        """
        Initialize the path segment with no elements.
        """
        # Constructors
        DrawableMixin.__init__(self)
        MovAble.__init__(self)

        # Default elements
        if elements is None:
            elements = []
        # end if

        self.elements = elements
    # end __init__

    # Add
    def add(self, element):
        """
        Add an element to the path.

        Args:
            element: Element to add to the path
        """
        self.elements.append(element)
    # end add

    def draw(self, context):
        """
        Draw the path segment.
        """
        # For each element in the segment
        for element in self.elements:
            element.draw(context)
        # end for
    # end draw

    # Move
    def move(self, dx: float, dy: float):
        """
        Move the path by a given displacement.

        Args:
            dx (float): Displacement in the X-direction
            dy (float): Displacement in the Y-direction
        """
        for element in self.elements:
            element.move(dx, dy)
        # end for
    # end move

    def __len__(self):
        """
        Get the number of elements in the path.
        """
        return len(self.elements)
    # end __len__

    def __getitem__(self, index):
        """
        Get the element at the given index in the path.

        Args:
            index (int): Index of the element to get
        """
        return self.elements[index]
    # end __getitem__

    def __setitem__(self, index, value):
        """
        Set the element at the given index in the path."

        Args:
            index (int): Index of the element to set
            value: Value to set the element
        """
        self.elements[index] = value
    # end __setitem__

    def __delitem__(self, index):
        """
        Delete the element at the given index in the path.

        Args:
            index (int): Index of the element to delete
        """
        del self.elements[index]
    # end __delitem__

    def __str__(self):
        """
        Get the string representation of the path.
        """
        return f"PathSegment(elements={self.elements})"
    # end __str__

    def __repr__(self):
        """
        Get the string representation of the path.
        """
        return self.__str__()
    # end __repr__

    @classmethod
    def from_data(
            cls,
            path_segment_data: 'PathSegment'
    ):
        """
        Create a path segment from data.

        Args:
            path_segment_data (PathSegmentData): Data to create the path segment from

        Returns:
        """
        return PathSegment(
            elements=path_segment_data.elements
        )
    # end

# end PathSegment


class Path(DrawableMixin, MovAble):
    """
    A simple path class that can be drawn to a cairo context.
    """

    def __init__(
            self,
            origin: Point2D,
            line_width: Scalar,
            line_color: Color = utils.WHITE,
            fill_color: Color = None,
            path: PathSegment = None,
            subpaths: List[PathSegment] = None,
            transform=None,
    ):
        """
        Initialize the path.

        Args:
            origin (Point2D): Origin of the path
            path (PathSegment): Path segment of the path
            subpaths (List[PathSegment]): Subpaths of the path
            transform: Transformation to apply to the path
        """
        # Constructors
        DrawableMixin.__init__(self)
        MovAble.__init__(self)

        # Add the subpaths
        if subpaths is None:
            subpaths = []
        # end if

        # Initialize the elements
        self.origin = origin
        self.line_width = line_width
        self.line_color = line_color
        self.fill_color = fill_color
        self.path = path
        self.subpaths = subpaths
        self.transform = transform

        # Debug circle
        self.debug_circle = Circle(
            0,
            0,
            fill_color=utils.BLUE.copy(),
            radius=Scalar(1),
            border_width=Scalar(0)
        )
    # end __init__

    # Add
    def add(self, element):
        """
        Add an element to the path.

        Args:
            element: Element to add to the path
        """
        self.path.add(element)
    # end add

    # Add subpath
    def add_subpath(self, subpath: PathSegment):
        """
        Add a subpath to the path.

        Args:
            subpath (PathSegmentList): Subpath to add to the path
        """
        self.subpaths.append(subpath)
    # end add_subpath

    # Get subpaths
    def get_subpaths(self):
        """
        Get the subpaths of the path.

        Returns:
            list: Subpaths of the path
        """
        return self.subpaths
    # end get_subpaths

    # Set subpaths
    def set_subpaths(self, subpaths):
        """
        Set the subpaths of the path.

        Args:
            subpaths (list): Subpaths of the path
        """
        self.subpaths = subpaths
    # end set_subpaths

    def draw(self, context):
        """
        Draw the path to the context.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # Save context
        context.save()
        context.translate(self.origin.x, self.origin.y)
        context.set_fill_rule(cairo.FillRule.WINDING)
        # self.debug_circle.draw(context)

        # Apply transform
        # ...

        # Draw path
        context.new_path()
        self.path.draw(context)

        # For each path segments
        for segment in self.subpaths:
            # New subpath
            context.new_sub_path()

            # Draw the subpath
            segment.draw(context)
        # end for

        # Close the path
        # context.close_path()

        # Fill if color is set
        if self.fill_color is not None:
            # Set color
            context.set_source_rgba(
                self.fill_color.red,
                self.fill_color.green,
                self.fill_color.blue,
                self.fill_color.opacity
            )

            # If stroke or not
            if self.line_width.value == 0:
                context.fill()
            else:
                context.fill_preserve()
            # end if
        # end if

        # Stroke
        if self.line_width.value > 0:
            # Set line color and width
            context.set_line_width(self.line_width.value)

            # Set color
            context.set_source_rgba(
                self.line_color.red,
                self.line_color.green,
                self.line_color.blue,
                self.line_color.opacity
            )

            # Stroke
            context.stroke()
        # end if

        # Restore the context
        context.restore()
    # end draw

    def __len__(self):
        return len(self.path)
    # end __len__

    def __getitem__(self, index):
        return self.path[index]
    # end __getitem__

    def __setitem__(self, index, value):
        self.path[index] = value
    # end __setitem__

    def __delitem__(self, index):
        del self.path[index]
    # end __delitem__

    # str
    def __str__(self):
        """
        Get the string representation of the path.
        """
        return (
            f"Path("
            f"path={self.path},"
            f"subpaths={self.subpaths},"
            f"transform={self.transform.__str__() if self.transform is not None else 'None'}"
            f")"
        )
    # end __str__

    # repr
    def __repr__(self):
        """
        Get the string representation of the path.
        """
        return self.__str__()
    # end __repr__

# end Path

