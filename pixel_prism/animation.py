

# Imports
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Locals
from pixel_prism.base.imagecanvas import ImageCanvas
from pixel_prism.render_engine import RenderEngine


class Animation:
    """
    Base class for creating animations. Subclasses should implement the `process_frame` method.
    """

    def __init__(
            self,
            input_path,
            output_path,
            keep_frames=0,
            display=False,
            debug_frames=False,
            **kwargs
    ):
        """
        Initialize the animation with an input and output path.

        Args:
            input_path (str): Path to the input video
            output_path (str): Path to the output video
            keep_frames (int): Number of frames to keep in memory
            display (bool): Whether to display the video while processing
            debug_frames (bool): Whether to display the layers while processing
            kwarg: Additional keyword arguments
        """
        self.input_path = input_path
        self.output_path = output_path
        self.display = display
        self.keep_frames = keep_frames
        self.debug_frames = debug_frames if debug_frames is not None else []
        self.debug_output_dir = os.path.splitext(output_path)[0] + "_debug"

        # Extra keyword arguments
        self.extra_args = kwargs

        # Keep frames
        self.prev_frames = []

        # Dictionary to store the effects
        self.effects = {}

        # Display the video
        if self.display:
            plt.ion()
        # end if

        # Create the debug output directory
        if self.debug_frames:
            os.makedirs(self.debug_output_dir, exist_ok=True)
        # end if
    # end __init__

    def init_effects(
            self
    ):
        """
        Initialize the effects, called before starting the video.
        """
        pass
    # end init_effects

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

    def process_frame(
            self,
            image_canvas,
            frame_number,
            total_frames
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
        cap = cv2.VideoCapture(self.input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        # Frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Display the video
        if self.display:
            fig, ax = plt.subplots()
        # end if

        # Process each frame of the video
        with tqdm(total=frame_count, desc="Processing video") as pbar:
            # Get frame
            ret, frame = cap.read()
            self.init_effects()

            # Process each frame
            frame_number = 0
            while cap.isOpened() and ret:
                # Create canvas from frame
                image_canva = ImageCanvas.from_numpy(frame, add_alpha=True)

                # Process the frame
                image_canva = self.process_frame(
                    image_canva,
                    frame_number,
                    frame_count
                )

                # Keep the frame
                if self.keep_frames:
                    self.prev_frames.append(image_canva)
                    if len(self.prev_frames) > self.keep_frames:
                        self.prev_frames.pop(0)
                    # end if
                # end if

                # Compose final image
                final_frame = RenderEngine.render(image_canva)

                # Save as RGB
                out.write(final_frame[:, :, :3])

                # Display the frame
                if self.display:
                    plt.clf()  # Clear the current figure
                    plt.imshow(cv2.cvtColor(final_frame[:, :, :3], cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.pause(0.001)
                    plt.draw()
                # end if

                # Save debug layers
                if frame_number in self.debug_frames:
                    self.save_debug_layers(image_canva, frame_number)
                # end if

                ret, frame = cap.read()
                pbar.update(1)
                frame_number += 1
            # end while
        # end with

        # Release the video capture and writer
        cap.release()
        out.release()

        # Close the display
        if self.display:
            plt.ioff()
            plt.show()
        # end if
    # end compose_video

# end Animation

