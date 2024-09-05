
# PixelPrism
import numpy as np
from pixel_prism import p2, s
from pixel_prism.animation import Animation
from pixel_prism.widgets.containers import Viewport
from pixel_prism.widgets import DrawableWidget
from pixel_prism.base import DrawableImage, ImageCanvas, CoordSystem
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
        # Coordinate system
        coord_system = CoordSystem(
            image_width=self.width,
            image_height=self.height,
            size=10
        )

        # Create a Viewport
        viewport = Viewport()

        # Drawable widget
        drawable_widget = DrawableWidget()
        viewport.add_widget(drawable_widget)

        # Create a point and line
        shared_point = coord_system.upper_left_square

        # Two lines
        line1 = Line.from_objects(shared_point, coord_system.upper_right_square)
        line2 = Line.from_objects(shared_point, coord_system.lower_right_square)

        # Add the lines to the drawable widget
        drawable_widget.add(line1)
        drawable_widget.add(line2)

        # Animate shared point
        anim = shared_point.move(2, coord_system.lower_left_square, 1).move(2, coord_system.center)
        anim.move(2, coord_system.upper_left_square)

        # Animate opacity
        self.animate(Range(opacity, 0, 1, 1.0, EaseInOutInterpolator()))
        self.animate(Range(opacity, 7, 8, 0.0, EaseInOutInterpolator()))

        # Animate thickness
        self.animate(Range(thickness, 0, 4, 5, EaseInOutInterpolator()))
        self.animate(Range(thickness, 4, 8, 2, EaseInOutInterpolator()))

        # Add
        self.add(
            viewport=viewport,
            drawable_widget=drawable_widget,
            line1=line1,
            line2=line2
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

# end JoinedLinesAnimation
