
# Imports
from typing import List, Any
import cairo
import numpy as np
from pixel_prism.data import Point2D, Color, Scalar
import pixel_prism.utils as utils
from pixel_prism.drawing import Circle, Line
from pixel_prism.animate.able import (
    MovableMixin,
    FadeInableMixin,
    FadeOutableMixin,
    BuildableMixin,
    DestroyableMixin
)
from .drawablemixin import DrawableMixin
from .transforms import (
    Translate2D,
    Rotate2D,
    Scale2D,
    SkewX2D,
    SkewY2D,
    Matrix2D
)


# Path segment
class PathSegment(
    DrawableMixin,
    MovableMixin,
    BuildableMixin,
    DestroyableMixin
):
    """
    A class to represent a path segment.
    """

    def __init__(
            self,
            elements=None,
            is_built: bool = True,
            build_ratio: float = 1.0
    ):
        """
        Initialize the path segment with no elements.
        """
        # Constructors
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)
        BuildableMixin.__init__(self, is_built, build_ratio)

        # Default elements
        if elements is None:
            elements = []
        # end if

        # Initialize the elements
        self.elements = elements

        # Compute length
        self.length = sum([element.length for element in self.elements]) if len(self.elements) > 0 else 0
    # end __init__

    # Get bounding box
    def get_bbox(
            self,
            border_width: float = 1.0,
            border_color: Color = utils.WHITE.copy()
    ):
        """
        Get the bounding box of the path segment.

        Args:
            border_width (float): Width of the border
            border_color (Color): Color of the border
        """
        # Get the bounding box of the path segment
        bbox = self.elements[0].bbox

        # For each element in the path segment
        for element in self.elements[1:]:
            # Update the bounding box
            bbox = bbox.union(element.bbox)
        # end for

        # Set fill, border color and witdth
        bbox.fill = False
        bbox.border_color = border_color
        bbox.border_width.value = border_width

        # Return the bounding box
        return bbox
    # end get_bbox

    # Add
    def add(self, element):
        """
        Add an element to the path.

        Args:
            element: Element to add to the path
        """
        self.elements.append(element)
        self.length += element.length
    # end add

    # Move segments
    def move(
            self,
            dx: float,
            dy: float
    ):
        """
        Move the path segment by a given displacement.

        Args:
            dx (float): Displacement in the X-direction
            dy (float): Displacement in the Y-direction
        """
        for element in self.elements:
            element.move(dx, dy)
        # end for
    # end move

    # Draw bounding box
    def draw_bounding_box(self, context):
        """
        Draw the bounding box of the path segment.

        Args:
            context (cairo.Context): Context to draw the bounding box to
        """
        # Draw the bounding box of the elements
        for element in self.elements:
            # Draw path bounding box
            path_bbox = element.bbox
            path_bbox.fill = False
            path_bbox.border_color = utils.BLUE.copy()
            path_bbox.border_width.value = 0.03
            path_bbox.draw(context)
        # end for
    # end draw_bounding_box

    # Draw path
    def draw_path(self, context):
        """
        Draw the path of the path segment.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # For each element in the segment
        for element in self.elements:
            element.draw_path(context)
        # end for
    # end draw_path

    def draw(self, context):
        """
        Draw the path segment.
        """
        # Get first element
        first_element = self.elements[0]

        # Move to the first element
        context.move_to(first_element.start.x, first_element.start.y)

        if self.is_built:
            # For each element in the segment
            for element in self.elements:
                element.draw(context)
            # end for
        else:
            # How many elements based on build ratio
            num_elements = int(len(self.elements) * self.build_ratio)

            # Share of time for each element
            element_share = 1 / len(self.elements)

            # For each element in the segment
            for i in range(num_elements):
                # Build ratio for this element
                element_build_ratio = min((self.build_ratio - i * element_share) / element_share, 1.0)
                element = self.elements[i]
                element.draw(
                    context=context,
                    move_to=False,
                    build_ratio=element_build_ratio
                )
            # end for
        # end if
    # end draw

    # Move
    def translate(self, dx: float, dy: float):
        """
        Move the path by a given displacement.

        Args:
            dx (float): Displacement in the X-direction
            dy (float): Displacement in the Y-direction
        """
        for element in self.elements:
            element.translate(dx, dy)
        # end for
    # end translate

    # region BUILD

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the vector graphic.
        """
        super().start_build(start_value)
    # end start_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the vector graphic.
        """
        super().end_build(end_value)
    # end end_build

    # Animate building
    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the vector graphic.
        """
        super().animate_build(t, duration, interpolated_t, env_value)
    # end animate_build

    # endregion BUILD

    # region DESTROY

    # Start building
    def start_destroy(self, start_value: Any):
        """
        Start building the vector graphic.
        """
        super().start_destroy(start_value)
    # end start_destroy

    # End building
    def end_destroy(self, end_value: Any):
        """
        End building the vector graphic.
        """
        super().end_destroy(end_value)
    # end end_destroy

    # Animate building
    def animate_destroy(self, t, duration, interpolated_t, env_value):
        """
        Animate building the vector graphic.
        """
        super().animate_destroy(t, duration, interpolated_t, env_value)
    # end animate_destroy

    # endregion DESTROY

    # region OVERRIDE

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

    # endregion OVERRIDE

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


class Path(
    DrawableMixin,
    MovableMixin,
    FadeInableMixin,
    FadeOutableMixin,
    BuildableMixin,
    DestroyableMixin
):
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
            is_built: bool = True,
            build_ratio: float = 1.0
    ):
        """
        Initialize the path.

        Args:
            origin (Point2D): Origin of the path
            path (PathSegment): Path segment of the path
            subpaths (List[PathSegment]): Subpaths of the path
            transform: Transformation to apply to the path
            is_built (bool): Is the path built
            build_ratio (float): Build ratio of the path
        """
        # Constructors
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)
        FadeInableMixin.__init__(self)
        FadeOutableMixin.__init__(self)
        BuildableMixin.__init__(self, is_built, build_ratio)

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

        # Set length
        self.length = self.compute_length()

        # Debug circle
        self.debug_circle = Circle(
            0,
            0,
            fill_color=utils.BLUE.copy(),
            border_color=utils.WHITE.copy(),
            radius=Scalar(0.2),
            border_width=Scalar(0.01)
        )
    # end __init__

    # Compute length
    def compute_length(self):
        """
        Compute the length of the path.
        """
        # Add path length
        length = self.path.length if self.path is not None else 0

        # Add length of subpaths
        for subpath in self.subpaths:
            length += subpath.length
        # end for

        return length
    # end compute_length

    # Set alpha
    def set_alpha(
            self,
            alpha: float
    ):
        """
        Set the alpha value of the path.

        Args:
            alpha (float): Alpha value to set
        """
        self.line_color.alpha = alpha
        self.fill_color.alpha = alpha
    # end set_alpha

    def get_bbox(
            self,
            border_width: float = 1.0,
            border_color: Color = utils.WHITE.copy()
    ):
        """
        Get the bounding box of the path.

        Args:
            border_width (float): Width of the border
            border_color (Color): Color of the border
        """
        # Get the bounding box of the path
        bbox = self.path.get_bbox()

        # For each subpath
        for subpath in self.subpaths:
            # Update the bounding box
            bbox = bbox.union(subpath.get_bbox())
        # end for

        # Set fill, border color and witdth
        bbox.fill = False
        bbox.border_color = border_color
        bbox.border_width.value = border_width

        # Return the bounding box
        return bbox
    # end get_bbox

    # Add
    def add(self, element):
        """
        Add an element to the path.

        Args:
            element: Element to add to the path
        """
        self.path.add(element)
        self.length = self.compute_length()
    # end add

    # Add subpath
    def add_subpath(self, subpath: PathSegment):
        """
        Add a subpath to the path.

        Args:
            subpath (PathSegmentList): Subpath to add to the path
        """
        self.subpaths.append(subpath)
        self.length = self.compute_length()
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
        self.length = self.compute_length()
    # end set_subpaths

    # Translate path
    def translate(
            self,
            dx: float,
            dy: float
    ):
        """
        Translate the path by a given displacement.

        Args:
            dx (float): Displacement in the X-direction
            dy (float): Displacement in the Y-direction
        """
        # Translate the path
        self.origin.x += dx
        self.origin.y += dy

        # Move path
        self.path.translate(dx, dy)

        # Translate the subpaths
        for subpath in self.subpaths:
            # Translate the subpath
            subpath.translate(dx, dy)
        # end for
    # end translate

    # Draw bounding box anchors
    def draw_bounding_box_anchors(
            self,
            context
    ):
        """
        Draw the bounding box anchors of the path.

        Args:
            context (cairo.Context): Context to draw the bounding box anchors to
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
    # end draw_bounding_box_anchors

    # Draw bounding boxes
    def draw_bounding_box(
            self,
            context
    ):
        """
        Draw the bounding box of the path.

        Args:
            context (cairo.Context): Context to draw the bounding box to
        """
        # Save context
        context.save()

        # Draw bb of segments
        for subpath in self.subpaths:
            # Draw the bounding box of the subpath
            subpath.draw_bounding_box(context)
        # end for

        # Draw segments bb of path
        self.path.draw_bounding_box(context)

        # Draw subpathsbounding box
        for subpath in self.subpaths:
            # Get the bounding box
            path_bbox = subpath.get_bbox(0.07, utils.YELLOW.copy())

            # Draw the bounding box
            path_bbox.draw(context)
        # end for

        # Draw path bounding box
        path_bbox = self.get_bbox(0.12, utils.GREEN.copy())
        path_bbox.draw(context)

        # Restore context
        context.restore()
    # end draw_bounding_box

    def apply_transform(
            self,
            context,
            transform
    ):
        """
        Apply an SVG transform to a Cairo context.

        Args:
            context (cairo.Context): Context to apply the transform to
            transform (str): SVG transform
        """
        # Translate
        if isinstance(transform, Translate2D):
            context.translate(transform.translate.x, transform.translate.y)
        # Scale
        elif isinstance(transform, Scale2D):
            context.scale(transform.scale.x, transform.scale.y)
        # Rotate
        elif isinstance(transform, Rotate2D):
            context.translate(transform.center.x, transform.center.y)
            context.rotate(np.radians(transform.angle.value))
            context.translate(-transform.center.x, -transform.center.y)
        elif isinstance(transform, SkewX2D):
            angle = np.radians(float(transform.angle.value))
            matrix = cairo.Matrix(1, np.tan(angle), 0, 1, 0, 0)
            context.transform(matrix)
        elif isinstance(transform, SkewY2D):
            angle = np.radians(float(transform.angle.value))
            matrix = cairo.Matrix(1, 0, np.tan(angle), 1, 0, 0)
            context.transform(matrix)
        elif isinstance(transform, Matrix2D):
            matrix = cairo.Matrix(
                float(transform.xx.value),
                float(transform.yx.value),
                float(transform.xy.value),
                float(transform.yy.value),
                float(transform.x0.value),
                float(transform.y0.value)
            )
            context.transform(matrix)
        else:
            raise ValueError(f"Unknown transform: {transform}")
        # end if
    # end apply_transform

    # Draw path components
    def draw_paths(
            self,
            context
    ):
        """
        Draw the path components to the context.

        Args:
            context (cairo.Context): Context to draw the path components to
        """
        # Save context
        context.save()

        # Draw path
        self.path.draw_path(context)

        # Draw subpaths
        for subpath in self.subpaths:
            subpath.draw_path(context)
        # end for

        # Restore context
        context.restore()
    # end draw_paths

    def draw(
            self, context,
            draw_bboxes: bool = False,
            draw_reference_point: bool = False
    ):
        """
        Draw the path to the context.

        Args:
            context (cairo.Context): Context to draw the path to
            draw_bboxes (bool): Whether to draw the bounding boxes
            draw_reference_point (bool): Whether to draw the debug information
        """
        # Save context
        context.save()
        # context.translate(self.origin.x, self.origin.y)
        context.set_fill_rule(cairo.FillRule.WINDING)

        # Reference point
        if draw_reference_point:
            self.debug_circle.draw(context)
        # end if

        # Apply transform
        if self.transform is not None:
            self.apply_transform(context, self.transform)
        # end if

        # Draw path
        context.new_path()
        self.path.draw(context)

        # For each path segments
        for segment in self.subpaths:
            # New sub path
            context.new_sub_path()

            # Draw the sub path
            segment.draw(context)
        # end for

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

        # Draw the bounding box and anchors
        if draw_bboxes:
            self.draw_bounding_box(context)
            self.draw_bounding_box_anchors(context)
        # end if

        # Restore the context
        context.restore()
    # end draw

    # region FADE_IN

    def start_fadein(self, start_value: Any):
        """
        Start fading in the path segment.

        Args:
            start_value (any): The start value of the path segment
        """
        self.set_alpha(0)
    # end start_fadein

    def end_fadein(self, end_value: Any):
        """
        End fading in the path segment.
        """
        self.set_alpha(1)
    # end end

    def animate_fadein(self, t, duration, interpolated_t, env_value):
        """
        Animate fading in the path segment.
        """
        self.set_alpha(interpolated_t)
    # end animate_fadein

    # endregion FADE_IN

    # region FADE_OUT

    def start_fadeout(self, start_value: Any):
        """
        Start fading out the path segment.
        """
        self.set_alpha(1)
    # end start_fadeout

    def end_fadeout(self, end_value: Any):
        """
        End fading out the path segment.
        """
        self.set_alpha(0)
    # end end_fadeout

    def animate_fadeout(self, t, duration, interpolated_t, target_value):
        """
        Animate fading out the path segment.
        """
        self.set_alpha(1 - interpolated_t)
    # end animate_fadeout

    # endregion FADE_OUT

    # region BUILD

    # Initialize building
    def init_build(self):
        """
        Initialize building the object.
        """
        print(f"Path init_build")
        super().init_build()
        self.path.init_build()
        for subpath in self.subpaths:
            subpath.init_build()
        # end for
    # end init_build

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the object.
        """
        super().start_build(start_value)
        self.path.start_build(start_value)
        for subpath in self.subpaths:
            subpath.start_build(start_value)
        # end for
    # end start_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the object.
        """
        super().end_build(end_value)

        # End elements
        self.path.end_build(end_value)
        for subpath in self.subpaths:
            subpath.end_build(end_value)
        # end for
    # end end_build

    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the object.
        """
        # Animate building
        super().animate_build(t, duration, interpolated_t, env_value)

        # Animate build for path
        self.path.animate_build(t, duration, interpolated_t, env_value)

        # Animate build for each subpath
        for subpath in self.subpaths:
            subpath.animate_build(t, duration, interpolated_t, env_value)
        # end for
    # end animate_build

    # Finish building
    def finish_build(self):
        """
        Finish building the object.
        """
        super().finish_build()
        self.path.finish_build()
        for subpath in self.subpaths:
            subpath.finish_build()
        # end for
    # end finish_build

    # endregion BUILD

    # region DESTROY

    # Initialize destroying
    def init_destroy(self):
        """
        Initialize destroying the object.
        """
        super().init_destroy()
        self.path.init_destroy()
        for subpath in self.subpaths:
            subpath.init_destroy()
        # end for
    # end init_destroy

    # Start destroying
    def start_destroy(self, start_value: Any):
        """
        Start building the object.
        """
        super().start_destroy(start_value)
        self.path.start_destroy(start_value)
        for subpath in self.subpaths:
            subpath.start_destroy(start_value)
        # end for
    # end start_destroy

    def animate_destroy(self, t, duration, interpolated_t, env_value):
        """
        Animate building the object.
        """
        # Animate building
        super().animate_destroy(t, duration, interpolated_t, env_value)

        # Animate build for path
        self.path.animate_destroy(t, duration, interpolated_t, env_value)

        # Animate build for each subpath
        for subpath in self.subpaths:
            subpath.animate_destroy(t, duration, interpolated_t, env_value)
        # end for
    # end animate_destroy

    # End destroying
    def end_destroy(self, end_value: Any):
        """
        End building the object.
        """
        super().end_destroy(end_value)

        # End elements
        self.path.end_destroy(end_value)
        for subpath in self.subpaths:
            subpath.end_destroy(end_value)
        # end for
    # end end_destroy

    # Finish destroying
    def finish_destroy(self):
        """
        Finish destroying the object.
        """
        super().finish_destroy()
        self.path.finish_destroy()
        for subpath in self.subpaths:
            subpath.finish_destroy()
        # end for
    # end finish_destroy

    # endregion DESTROY

    # region OVERRIDE

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

    # endregion OVERRIDE

# end Path
