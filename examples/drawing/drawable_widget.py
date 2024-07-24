
# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas
from pixel_prism.drawing import Point, Line
from pixel_prism.animate import Move


# DrawableWidgetAnimation class
class DrawableWidgetAnimation(Animation):

    # Init effects
    def init_effects(self):
        pass
    # end init_effects

    def build(self):
        """
        Build the animation.
        """
        # Create a Viewport
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Add points and lines
        point1 = Point(50, 50)
        point2 = Point(110, 110)
        line = Line((60, 60), (210, 110))

        drawable_widget.add(point1)
        drawable_widget.add(point2)
        drawable_widget.add(line)

        # Keep the widget
        self.add_object("viewport", viewport)
        self.add_object("drawable_widget", drawable_widget)
        self.add_object("point1", point1)
        self.add_object("point2", point2)

        # Add transition point 1
        self.add_transition(
            Move(
                point1,
                start_time=0,
                end_time=2,
                start_value=(50, 50),
                end_value=(100, 100)
            )
        )

        # Add transition point 2
        self.add_transition(
            Move(
                point2,
                start_time=0,
                end_time=2,
                start_value=(110, 110),
                end_value=(150, 150)
            )
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

# end DrawableWidgetAnimation
