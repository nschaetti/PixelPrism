

# Imports
from typing import List
from pixel_prism.animate.able import MovAble

from .data import Data


# A segment
class PathSegment(Data, MovAble):
    """
    A class to represent a path.
    """

    # Constructor
    def __init__(self):
        """
        Initialize the path with no elements.
        """
        super().__init__()
        self.elements = []
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
        return len(self.elements)
    # end __len__

    def __getitem__(self, index):
        return self.elements[index]
    # end __getitem__

    def __setitem__(self, index, value):
        self.elements[index] = value
    # end __setitem__

    def __delitem__(self, index):
        del self.elements[index]
    # end __delitem__

    def __str__(self):
        """
        Get the string representation of the path.
        """
        return f"PathSegment(\n\t{self.elements}\n)"
    # end __str__

    def __repr__(self):
        """
        Get the string representation of the path.
        """
        return f"PathSegment(\n\t{self.elements}\n)"
    # end __repr__

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

# end PathSegment


# A path
class Path(Data, MovAble):
    """
    A class to represent a path in 2D space.
    """

    def __init__(
            self,
            path: PathSegment = None,
            subpaths: List[PathSegment] = None
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
        self.path = path
        self.subpaths = subpaths
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
        return f"Path(\n\t{self.path},\n\t{self.subpaths}\n)"
    # end __str__

    # repr
    def __repr__(self):
        """
        Get the string representation of the path.
        """
        return f"Path(\n\t{self.path},\n\t{self.subpaths}\n)"
    # end __repr__

# end Path

