#
# Description: Base class for creating animations.
# Subclasses should implement the `process_frame` method.
#

# Imports
from typing import Optional
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Locals
from pixel_prism.base.imagecanvas import ImageCanvas
from pixel_prism.render_engine import RenderEngine
from pixel_prism.utils import setup_logger


# Setup logger
logger = setup_logger(__name__)


class Animation:
    """
    Base class for creating animations.
    Subclasses should implement the `process_frame` method.
    """

    def __init__(
            self,
            input_video: Optional[str] = None,
            output_video: str = 'output.mp4',
            duration: int = None,
            fps: int = 30,
            width: int = 1920,
            height: int = 1080,
            keep_frames: int = 0,
            debug_frames: int = False,
            save_frames: bool = False,
            **kwargs
    ):
        """
        Initialize the animation with an input and output path.

        Args:
            input_video (str): Path to the input video
            output_video (str): Path to the output video
            duration (int): Duration of the video in seconds
            fps (int): Frames per second
            width (int): Width of the video
            height (int): Height of the video
            keep_frames (int): Number of frames to keep in memory
            debug_frames (int): Whether to display the layers while processing
            save_frames (bool): Whether to save the frames to disk
            kwarg: Additional keyword arguments
        """
        # Input video given
        if input_video:
            self.cap = cv2.VideoCapture(input_video)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / self.fps if duration is None else duration
        else:
            self.cap = None
            self.fps = fps
            assert fps is not None, "FPS must be specified if input video is not given"
            assert duration is not None, "Duration must be specified if input video is not given"
            self.frame_count = int(fps * duration) if duration else None
            self.width = width
            self.height = height
            self.duration = duration
        # end if

        # Properties
        self.input_video = input_video
        self.output_video = output_video
        self.time_info = None
        self.keep_frames = keep_frames
        self.debug_frames = debug_frames if debug_frames is not None else []
        self.debug_output_dir = "debug"

        # Extra keyword arguments
        self.extra_args = kwargs

        # Keep frames
        self.prev_frames = []

        # Dictionary to store the effects
        self.effects = {}

        # Objects
        self.objects = {}

        # Transitions
        self.transitions = []

        # Create the debug output directory
        if self.debug_frames:
            os.makedirs(self.debug_output_dir, exist_ok=True)
        # end if

        # Save outout frames
        self.save_frames = save_frames
        if self.save_frames:
            self.save_frames_path = os.path.join(Path(output_video).parent, "frames")
            os.makedirs(
                self.save_frames_path,
                exist_ok=True
            )
        # end if

        # Build the animation
        self.build()
    # end __init__

    # region PROPERTIES

    @property
    def step(self):
        """
        Get the step size for the animation.
        """
        return 1 / self.fps
    # end step

    # endregion PROPERTIES

    def init_effects(
            self
    ):
        """
        Initialize the effects, called before starting the video.
        """
        pass
    # end init_effects

    # Build the animation
    def build(
            self
    ):
        """
        Build the animation.
        """
        pass
    # end build

    # Add an effect
    def add_effect(
            self,
            name,
            effect
    ):
        """
        Add an effect to the animation.

        Args:
            name (str): Name of the effect
            effect (EffectBase): Effect to add to the animation
        """
        self.effects[name] = effect
    # end add_effect

    # Get an effect
    def get_effect(
            self,
            name
    ):
        """
        Get an effect from the animation.

        Args:
            name (str): Name of the effect

        Returns:
            EffectBase: Effect object
        """
        return self.effects.get(name)
    # end get_effect

    # Add object
    def add_object(
            self,
            name,
            obj
    ):
        """
        Add a widget to the animation.

        Args:
            name (str): Name of the widget
            obj (object): Widget object
        """
        self.objects[name] = obj
    # end add_object

    # Get object
    def get_object(
            self,
            name
    ):
        """
        Get a widget from the animation.

        Args:
            name (str): Name of the widget

        Returns:
            object: Object
        """
        return self.objects.get(name)
    # end get_object

    # Get object
    def obj(self, name):
        """
        Get a widget from the animation.

        Args:
            name (str): Name of the widget

        Returns:
            object: Object
        """
        return self.get_object(name)
    # end get_object

    # Remove an effect
    def remove_effect(
            self,
            name
    ):
        """
        Remove an effect from the animation.

        Args:
            name (str): Name of the effect
        """
        self.effects.pop(name)
    # end remove_effect

    def animate(
            self,
            transition
    ):
        """
        Add a transition to the animation.

        Args:
            transition (Transition): Transition object
        """
        self.transitions.append(transition)
    # end animate

    def apply_transitions(
            self,
            t
    ):
        """
        Apply transitions to the animation.

        Args:
            t (float): Time
        """
        for transition in self.transitions:
            transition.update(t)
        # end for
    # end apply_transitions

    def process_frame(
            self,
            image_canvas: ImageCanvas,
            t: float,
            frame_number: int
    ):
        """
        Process each frame of the video. Should be implemented by derived classes.
        """
        raise NotImplementedError("Subclasses should implement this method!")
    # end process_frame

    def display_layers(
            self,
            image_canvas
    ):
        """
        Display the layers of the image object.

        Args:
            image_canvas (ImageCanvas): Image object
        """
        num_layers = len(image_canvas.layers)
        fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
        if len(image_canvas.layers) == 1:
            axes = [axes]
        # end if

        # For each layer
        for ax, layer in zip(axes, image_canvas.layers):
            ax.imshow(cv2.cvtColor(layer.image[:, :, :3], cv2.COLOR_BGR2RGB))
            ax.set_title(layer.name)
            ax.axis('off')
        # end for

        # Pause
        plt.pause(0.001)
        plt.show()
    # end display_layers

    def save_debug_layers(
            self,
            image_canvas,
            frame_number
    ):
        """
        Save the debug layers to a file.

        Args:
            image_canvas (ImageCanvas): Image object
            frame_number (int): Frame number
        """
        fig, axes = plt.subplots(1, len(image_canvas.layers), figsize=(15, 5))
        if len(image_canvas.layers) == 1:
            axes = [axes]
        # end if

        # For each layer
        for ax, layer in zip(axes, image_canvas.layers):
            ax.imshow(cv2.cvtColor(layer.image.data[:, :, :3], cv2.COLOR_BGR2RGB))
            ax.set_title(layer.name)
            ax.axis('off')
        # end for

        # Save the figure
        debug_frame_path = os.path.join(self.debug_output_dir, f'frame_{frame_number}.png')

        # Save the figure
        plt.savefig(debug_frame_path)
        plt.close()
    # end save_debug_layers

    def compose_video(
            self
    ):
        """
        Compose the final video by applying `process_frame` to each frame.
        """
        fps = self.fps
        width = self.width
        height = self.height

        # Open the video capture
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not open the output video file for writing: {self.output_video}")
        # end if
        logger.info(f"Input video: {self.input_video}, FPS: {fps}, width: {width}, height: {height}")
        logger.info(f"Output video: {self.output_video}, fourcc: {fourcc}, out: {out}")

        # Frame count
        frame_count = self.frame_count

        # Process each frame of the video
        with tqdm(total=frame_count, desc="Processing video") as pbar:
            # Frame number
            frame_number = 0

            # Get the first frame, or create a blank frame
            if self.cap:
                ret, frame = self.cap.read()
            else:
                ret = True
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            # end if

            # Initialize the effects
            self.init_effects()

            # Process each frame
            while frame_number < frame_count and ret:
                # Create canvas from frame
                if self.cap:
                    image_canvas = ImageCanvas.from_numpy(frame, add_alpha=True)
                else:
                    image_canvas = ImageCanvas(width, height)
                # end if

                # Apply transitions
                t = frame_number / fps
                self.apply_transitions(t)

                # Process the frame
                image_canvas = self.process_frame(
                    image_canvas=image_canvas,
                    t=frame_number / fps,
                    frame_number=frame_number
                )

                # Keep the frame
                if self.keep_frames:
                    self.prev_frames.append(image_canvas)
                    if len(self.prev_frames) > self.keep_frames:
                        self.prev_frames.pop(0)
                    # end if
                # end if

                # Compose final image
                final_frame = RenderEngine.render(image_canvas)

                # Ensure the final frame is the correct size and type
                assert final_frame.data.shape[:2] == (height, width), "Final frame size mismatch"
                assert final_frame.data.dtype == np.uint8, "Final frame data type mismatch"

                # Save as RGB
                out.write(final_frame.data[:, :, :3])

                # Save frames
                if self.save_frames:
                    frame_path = os.path.join(self.save_frames_path, f'frame_{frame_number}.png')
                    final_frame.save(frame_path)
                # end if

                # Save debug layers
                if frame_number in self.debug_frames:
                    self.save_debug_layers(image_canvas, frame_number)
                # end if

                # Read next frame
                if self.cap:
                    ret, frame = self.cap.read()
                # end if

                pbar.update(1)
                frame_number += 1
            # end while
        # end with

        # Release the video capture and writer
        if self.cap: self.cap.release()
        out.release()
    # end compose_video

# end Animation

