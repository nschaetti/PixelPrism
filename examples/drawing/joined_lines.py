
# PixelPrism
import numpy as np
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas
from pixel_prism.drawing import Line
from pixel_prism.animate import Move, EaseInOutInterpolator, FadeIn, FadeOut, Range
from pixel_prism.data import Point2D, Scalar


# JoinedLinesAnimation class
class JoinedLinesAnimation(Animation):

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

        # Create a point and line
        shared_point = Point2D(50, 50)

        # Shared scalar and thickness
        opacity = Scalar(0.0)
        thickness = Scalar(2)

        # First line
        line1 = Line(shared_point, Point2D(1820, 50), opacity=opacity, thickness=thickness)

        # Second line
        line2 = Line(shared_point, Point2D(1820, 1030), opacity=opacity, thickness=thickness)

        # Add the lines to the drawable widget
        drawable_widget.add(line1)
        drawable_widget.add(line2)

        # Keep the widget
        self.add_object("viewport", viewport)
        self.add_object("drawable_widget", drawable_widget)
        self.add_object("line1", line1)
        self.add_object("line2", line2)

        # Animate shared point
        self.animate(Move(shared_point, 1, 3, (1000, 540), EaseInOutInterpolator()))
        self.animate(Move(shared_point,3, 5, (50, 1030), EaseInOutInterpolator()))
        self.animate(Move(shared_point, 5, 7,(50, 50), EaseInOutInterpolator()))

        # Animate opacity
        self.animate(Range(opacity, 0, 1, 1.0, EaseInOutInterpolator()))
        self.animate(Range(opacity, 7, 8, 0.0, EaseInOutInterpolator()))

        # Animate thickness
        self.animate(Range(thickness, 0, 4, 5, EaseInOutInterpolator()))
        self.animate(Range(thickness, 4, 8, 2, EaseInOutInterpolator()))
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

# end JoinedLinesAnimation
