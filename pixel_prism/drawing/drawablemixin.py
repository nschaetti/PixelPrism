#
#
#

# Imports
from pixel_prism import utils
from pixel_prism.base import Context
from pixel_prism.data import Color, Scalar


class DrawableMixin:

    # region PUBLIC

    # Update object data
    def update_data(
            self
    ):
        """
        Update the data of the drawable.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.update_data method must be implemented in subclass.")
    # end update_data

    # Translate object
    def translate(
            self,
            *args,
            **kwargs
    ):
        """
        Translate the object.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        self._translate_object(*args, **kwargs)
    # end translate

    # Scale object
    def scale(
            self,
            *args,
            **kwargs
    ):
        """
        Scale the object.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        self._scale_object(*args, **kwargs)
    # end scale

    # Rotate object
    def rotate(
            self,
            *args,
            **kwargs
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate
        """
        self._rotate_object(*args, **kwargs)
    # end rotate

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

    # region EVENTS

    # On scale changed
    def _on_scale_changed(
            self,
            scale
    ):
        """
        Handle scale changed event.

        Args:
            scale (float): New scale
        """
        self.update_data()
    # end _on_scale_changed

    # On rotation changed
    def _on_rotation_changed(
            self,
            rotation
    ):
        """
        Handle rotation changed event.

        Args:
            rotation (float): New rotation
        """
        self.update_data()
    # end _on_rotation_changed

    # endregion EVENTS

    # region PRIVATE

    # Scale object (to override)
    def _scale_object(
            self,
            *args,
            **kwargs
    ):
        """
        Scale the object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._scale_object method must be implemented in subclass."
        )
    # end _scale_object

    # Translate object (to override)
    def _translate_object(
            self,
            *args,
            **kwargs
    ):
        """
        Translate the object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._translate_object method must be implemented in subclass."
        )
    # _translate_object

    # Rotate object (to override)
    def _rotate_object(
            self,
            *args,
            **kwargs
    ):
        """
        Rotate the object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._rotate_object method must be implemented in subclass.")
    # end _rotate_object

    # endregion PRIVATE

# end DrawableMixin


