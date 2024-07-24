

# Imports
from .image import Image
import cairo


# DrawableImage class
class DrawableImage(Image):

    # Constructor
    def __init__(
            self,
            image_array
    ):
        """
        Initialize the image data with an image array

        Args:
            image_array (np.ndarray): Image data as a NumPy array
        """
        super().__init__(image_array)
        self.surface = self.create_surface()
        self.root_container = None
    # end __init__

    def create_surface(self):
        """
        Create a Cairo surface and context from the image data.
        """
        surface = cairo.ImageSurface.create_for_data(
            self.data,
            cairo.FORMAT_ARGB32,
            self.width,
            self.height
        )
        return surface
    # end create_surface

    def get_context(self):
        """
        Get the Cairo context for drawing.
        """
        return cairo.Context(self.surface)
    # end get

    # Set root container
    def set_root_container(
            self,
            container
    ):
        """
        Set the root container for the image.

        Args:
            container (Container): Root container
        """
        self.root_container = container
    # end set_root_container

    # Render the image
    def render(self):
        """
        Render the image to the context.
        """
        if self.root_container:
            self.root_container.render(self.surface)
        else:
            raise ValueError("Root container not set.")
        # end if
        return self
    # end render

    def save(
            self,
            file_path
    ):
        """
        Save the image, ensuring Cairo surface is written to file.

        Args:
            file_path (str): Path to the file
        """
        self.surface.write_to_png(file_path)
    # end save

# end DrawableImage
