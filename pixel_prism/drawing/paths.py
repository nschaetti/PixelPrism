

from .drawablemixin import DrawableMixin
from pixel_prism.data import PathData, PathSegmentData


# Path segment
class PathSegment(DrawableMixin, PathSegmentData):
    """
    A class to represent a path segment.
    """

    def __init__(self, elements=None):
        """
        Initialize the path segment with no elements.
        """
        # Constructors
        DrawableMixin.__init__(self)
        PathSegmentData.__init__(self, elements=elements)
    # end __init__

    def draw(self, context):
        """
        Draw the path segment.
        """
        # For each element in the segment
        for element in self.elements:
            element.to_drawable().draw(context)
        # end for
    # end draw

    @classmethod
    def from_data(
            cls,
            path_segment_data: PathSegmentData
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


class Path(DrawableMixin, PathData):
    """
    A simple path class that can be drawn to a cairo context.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the path.

        Args:
            *args: Arguments to pass to the PathData constructor
            **kwargs: Keyword arguments to pass to the PathData constructor
        """
        # Constructors
        DrawableMixin.__init__(self)
        PathData.__init__(self, *args, **kwargs)
    # end __init__

    def draw(self, context):
        """
        Draw the path to the context.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # Save and translate the context
        context.save()
        context.translate(self.origin.x, self.origin.y)

        # Apply transform
        # ...

        # Draw path
        context.new_path()
        self.path.to_drawable().draw(context)

        # For each path segments
        for segment in self.subpaths:
            # New subpath
            context.new_sub_path()

            # Draw the subpath
            segment.to_drawable().draw(context)
        # end for

        # Close the path
        context.close_path()

        # Restore the context
        context.restore()
    # end draw

    @classmethod
    def from_data(
            cls,
            path_data: PathData
    ):
        """
        Create a path from data.

        Args:
            path_data (PathData): Data to create the path from

        Returns:
        """
        return Path(
            origin=path_data.origin,
            path=path_data.path,
            subpaths=path_data.subpaths,
            transform=path_data.transform
        )
    # end from_data

# end Path

