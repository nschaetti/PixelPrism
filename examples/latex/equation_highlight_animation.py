#
# Animation of an equation.
# Build and highlight
#

# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.animate import Move, EaseInOutInterpolator, FadeOut, Build, Destroy
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
        latex_position = Point2D(1920 / 2.0, 1080 / 2.0)

        # Create a LaTeX widget
        latex_widget = MathTex(
            "g(x) = \\frac{\partial Q}{\partial t}",
            latex_position,
            scale=Point2D(20, 20),
            refs=["g", "(", "x", ")", "=", "partial1", "Q", "bar", "partial2",  "t"],
            debug=True
        )

        # Add the LaTeX widget to the viewport or a container
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

        # Destroy the math tex object
        self.animate(
            Destroy(
                latex_widget,
                start_time=7,
                end_time=8,
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


