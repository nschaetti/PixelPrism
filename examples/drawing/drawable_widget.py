
# PixelPrism
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas
from pixel_prism.drawing import Point, Line
from pixel_prism.animate import Move, EaseInOutInterpolator, FadeIn, FadeOut


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
        point1 = Point(50, 50, radius=25)
        line = Line((100, 540), (1820, 540))

        drawable_widget.add(point1)
        drawable_widget.add(line)

        # Keep the widget
        self.add_object("viewport", viewport)
        self.add_object("drawable_widget", drawable_widget)
        self.add_object("point1", point1)
        self.add_object("line", line)

        # Fade in point 1
        self.animate(
            FadeIn(
                "FadeInPoint1",
                point1,
                start_time=0,
                end_time=1,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Add transitions for point1
        self.animate(
            Move(
                "Move1",
                point1,
                start_time=1,
                end_time=3,
                target_value=(1870, 50),
                interpolator=EaseInOutInterpolator()
            )
        )
        self.animate(
            Move(
                "Move2",
                point1,
                start_time=3,
                end_time=5,
                target_value=(1870, 1030),
                interpolator=EaseInOutInterpolator()
            )
        )
        self.animate(
            Move(
                "Move3",
                point1,
                start_time=5,
                end_time=7,
                target_value=(50, 1030),
                interpolator=EaseInOutInterpolator()
            )
        )
        self.animate(
            Move(
                "Move4",
                point1,
                start_time=7,
                end_time=9,
                target_value=(50, 50),
                interpolator=EaseInOutInterpolator()
            )
        )
        self.animate(
            FadeOut(
                "FadeOutPoint1",
                point1,
                start_time=9,
                end_time=10,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Add fade-in transition for line
        self.animate(
            FadeIn(
                "FadeIn1",
                line,
                start_time=0,
                end_time=1,
                interpolator=EaseInOutInterpolator()
            )
        )

        # Add fade-out transition for line
        self.animate(
            FadeOut(
                "FadeOut1",
                line,
                start_time=9,
                end_time=10,
                interpolator=EaseInOutInterpolator()
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
