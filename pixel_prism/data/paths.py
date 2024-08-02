

# Imports
from typing import List

from pixel_prism.mixins import DrawableDataMixin
from pixel_prism.animate.able import MovAble

from .data import Data
from .points import Point2D


# A segment
class PathSegmentData(Data, DrawableDataMixin, MovAble):
    """
    A class to represent a path.
    """

    # Constructor
    def __init__(self, elements=None):
        """
        Initialize the path with no elements.

        Args:
            elements (list): Elements of the path
        """
        super().__init__()

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

    # Get
    def get(self):
        """
        Get the elements of the path.

        Returns:
            list: Elements of the path
        """
        return self.elements
    # end get

    # Set
    def set(self, elements):
        """
        Set the elements of the path.

        Args:
            elements (list): Elements of the path
        """
        self.elements = elements
    # end set

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

    # Transform into a drawable element
    def to_drawable(self):
        """
        Transform the path into a drawable element.
        """
        from pixel_prism.drawing import PathSegment
        return PathSegment(
            elements=self.elements
        )
    # end to_drawable

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
    # end move

# end PathSegmentData


# A path
class PathData(Data, DrawableDataMixin, MovAble):
    """
    A class to represent a path in 2D space.
    """

    def __init__(
            self,
            origin: Point2D = None,
            path: PathSegmentData = None,
            subpaths: List[PathSegmentData] = None,
            transform=None
    ):
        """
        Initialize the path with its segments.

        Args:
            path (PathSegmentList): Path segment list
            subpaths (list): Subpaths
        """
        super().__init__()

        # Add the subpaths
        if subpaths is None:
            subpaths = []
        # end if

        # Initialize the elements
        self.origin = origin
        self.path = path
        self.subpaths = subpaths
        self.transform = transform
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
    def add_subpath(self, subpath: PathSegmentData):
        """
        Add a subpath to the path.

        Args:
            subpath (PathSegmentList): Subpath to add to the path
        """
        self.subpaths.append(subpath)
    # end add_subpath

    # Get
    def get(self):
        """
        Get the elements of the path.

        Returns:
            list: Elements of the path
        """
        return self.path
    # end get

    # Get subpaths
    def get_subpaths(self):
        """
        Get the subpaths of the path.

        Returns:
            list: Subpaths of the path
        """
        return self.subpaths
    # end get_subpaths

    # Set
    def set(self, path_element_list):
        """
        Set the elements of the path.

        Args:
            path_element_list (list): Elements of the path
        """
        self.path = path_element_list
    # end set

    # Set subpaths
    def set_subpaths(self, subpaths):
        """
        Set the subpaths of the path.

        Args:
            subpaths (list): Subpaths of the path
        """
        self.subpaths = subpaths
    # end set_subpaths

    # Transform into a drawable element
    def to_drawable(self):
        """
        Transform the path into a drawable element.
        """
        from pixel_prism.drawing import Path
        return Path(
            origin=self.origin,
            path=self.path,
            subpaths=self.subpaths,
            transform=self.transform
        )
    # end to_drawable

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

# end PathData

