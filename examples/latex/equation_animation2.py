#
# Animation of an equation.
#

# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.animate import Move, EaseInOutInterpolator, FadeIn, FadeOut, Build
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas
from pixel_prism.drawing import MathTex
from pixel_prism.data import Point2D


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
        latex_position = Point2D(1920 / 4.0, 1080 / 4.0)

        # Créer un widget LaTeX
        latex_widget = MathTex(
            "g(x) = \\frac{\partial Q}{\partial t}",
            latex_position,
            scale=Point2D(15, 15),
            refs=["g", "(", "x", ")", "=", "partial1", "Q", "bar", "partial2",  "t"],
            debug=True
        )

        # Ajouter le widget au viewport ou à un conteneur
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Add the LaTeX widget to the drawable widget
        drawable_widget.add(latex_widget)

        # Build the math tex object
        self.animate(
            Build(
                latex_widget,
                start_time=0,
                end_time=1,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Add transitions for point1
        self.animate(
            Move(
                latex_widget,
                start_time=1,
                end_time=3,
                target_value=Point2D(1920 / 4.0 * 3.0, 1080 / 4.0 * 3.0),
                interpolator=EaseInOutInterpolator()
            )
        )

        self.animate(
            Move(
                latex_widget,
                start_time=4,
                end_time=6,
                target_value=Point2D(1920 / 4.0, 1080 / 4.0),
                interpolator=EaseInOutInterpolator()
            )
        )

        # Math FadeOut
        self.animate(
            FadeOut(
                latex_widget,
                start_time=6,
                end_time=7,
                interpolator=EaseInOutInterpolator()
            )
        )

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


