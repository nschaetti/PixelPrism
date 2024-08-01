#
# Animation of an equation.
#

# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas
from pixel_prism.drawing import Point, Line, MathTex
from pixel_prism.animate import Move, EaseInOutInterpolator, FadeIn, FadeOut
from pixel_prism.data import Point2D, Scalar


# DrawableWidgetAnimation class
class MathTexAnimation(Animation):

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    def build(self):
        """
        Build the animation.
        """
        # Create a Point2D for the position of the LaTeX widget
        latex_position = Point2D(-800, 400)

        # Créer un widget LaTeX
        latex_widget = MathTex(
            "g(x)",
            latex_position,
            color=(1.0, 1.0, 1.0)
        )

        # Ajouter le widget au viewport ou à un conteneur
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(latex_widget)

        # Add objects
        self.add(
            viewport=viewport,
            drawable_widget=drawable_widget,
            latex_widget=latex_widget
        )
    # end build

    # Process frame
    def process_frame(
            self,
            image_canvas: ImageCanvas,
            t: float,
            frame_number: int
    ):
        """
        Process the frame.

        Args:
            image_canvas (ImageCanvas): Image canvas
            t (float): Time
            frame_number (int): Frame number
        """
        # Create a DrawableImage
        drawing_layer = DrawableImage.transparent(self.width, self.height)

        # Get the viewport and drawable widget
        viewport = self.obj("viewport")

        # Set the root container and render the drawing layer
        drawing_layer.set_root_container(viewport)
        drawing_layer.render()

        # Add the new drawing layer to the image canvas
        image_canvas.add_layer('drawing', drawing_layer)

        return image_canvas
    # end process_frame

# end MathTexAnimation


