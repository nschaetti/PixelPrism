from pixel_prism import utils
from pixel_prism.data import Color


class DrawableMixin(object):

    # Constructor
    def __init__(
            self,
            has_bbox: bool = True,
            bbox_border_width: float = 1.0,
            bbox_border_color: Color = utils.WHITE
    ):
        """
        Initialize the drawable mixin.
        """
        super().__init__()

        # Bounding box
        if has_bbox:
            self._bounding_box = self._create_bbox(
                border_width=bbox_border_width,
                border_color=bbox_border_color
            )
        else:
            self._bounding_box = None
        # end if
        self.has_bbox = has_bbox
    # end __init__

    # region PROPERTIES

    # Bounding box
    @property
    def bounding_box(self):
        return self._bounding_box
    # end bounding_box

    # endregion PROPERTIES

    # region PUBLIC

    # Update bounding box
    def update_bbox(
            self
    ):
        """
        Update the bounding box of the drawable.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.update_bbox method must be implemented in subclass.")
    # end update_bbox

    # Set source RGBA
    def set_source_rgba(
            self,
            context,
            color
    ):
        """
        Set the source RGBA of the context.

        Args:
            context (cairo.Context): Context to set the source RGBA of
            color (Color): Color to set
        """
        context.set_source_rgba(
            color.red,
            color.green,
            color.blue,
            color.alpha
        )
    # end set_source_rgba

    # Set source RGB
    def set_source_rgb(
            self,
            context,
            color
    ):
        """
        Set the source RGB of the context.

        Args:
            context (cairo.Context): Context to set the source RGB of
            color (Color): Color to set
        """
        context.set_source_rgb(
            color.red,
            color.green,
            color.blue
        )
    # end set_source_rgb

    # Translate object (to override)
    def _translate_object(
            self,
            dx: float,
            dy: float
    ):
        """
        Translate the object.

        Args:
            dx (float): Delta x
            dy (float): Delta y
        """
        raise NotImplementedError(f"{self.__class__.__name__}._translate_object method must be implemented in subclass.")
    # _translate_object

    # Translate object
    def translate(
            self,
            dx: float,
            dy: float
    ):
        """
        Translate the object.

        Args:
            dx (float): Delta x
            dy (float): Delta y
        """
        self._translate_object(dx, dy)
        if self.bounding_box is not None:
            self.bounding_box.translate(dx, dy)
        # end if
    # end translate

    # Scale object (to override)
    def _scale_object(
            self,
            sx: float,
            sy: float
    ):
        """
        Scale the object.

        Args:
            sx (float): Scale x
            sy (float): Scale y
        """
        raise NotImplementedError(f"{self.__class__.__name__}._scale_object method must be implemented in subclass.")
    # end _scale_object

    # Scale object
    def scale(
            self,
            sx: float,
            sy: float
    ):
        """
        Scale the object.

        Args:
            sx (float): Scale x
            sy (float): Scale y
        """
        self._scale_object(sx, sy)
        if self.bounding_box is not None:
            self.bounding_box.scale(sx, sy)
        # end if
    # end scale

    # Rotate object (to override)
    def _rotate_object(
            self,
            angle: float
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate
        """
        raise NotImplementedError(f"{self.__class__.__name__}._rotate_object method must be implemented in subclass.")
    # end _rotate_object

    # Rotate object
    def rotate(
            self,
            angle: float
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate
        """
        self._rotate_object(angle)
        if self.bounding_box is not None:
            self.bounding_box.rotate(angle)
        # end if
    # end rotate

    # Draw bounding box anchors
    def draw_bbox_anchors(
            self,
            context
    ):
        """
        Draw the bounding box anchors to the context.

        Args:
            context (cairo.Context): Context to draw the bounding box anchors to
        """
        if self.bounding_box is not None:
            self.bounding_box.draw_anchors(context)
        # end if
    # end draw_bbox_anchors

    # Draw bounding box
    def draw_bbox(
            self,
            context
    ):
        """
        Draw the bounding box to the context.

        Args:
            context (cairo.Context): Context to draw the bounding box to
        """
        if self.bounding_box is not None:
            self.bounding_box.draw(context)
        # end if
    # end draw_bbox

    # Realize
    def realize(
            self,
            *args,
            **kwargs
    ):
        """
        Create path from context

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError(f"{self.__class__.__name__}.realize method must be implemented in subclass.")
    # end realize

    def draw(
            self,
            *args,
            **kwargs
    ):
        """
        Draw the point to the context.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        raise NotImplementedError(f"{self.__class__.__name__}.draw method must be implemented in subclass.")
    # end draw

    # endregion PUBLIC

    # region PRIVATE

    # Create bounding box
    def _create_bbox(
            self,
            border_width: float = 0.0,
            border_color: Color = utils.WHITE.copy()
    ):
        """
        Create the bounding box.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _create_bbox method."
        )
    # end _create_bbox

    # endregion PRIVATE

# end DrawableMixin


