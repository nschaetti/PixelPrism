

# Imports
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Locals
from pixel_prism.base.image import Image


class Animation:
    """
    Base class for creating animations. Subclasses should implement the `process_frame` method.
    """

    def __init__(
            self,
            input_path,
            output_path,
            display=False,
            debug_frames=False
    ):
        """
        Initialize the animation with an input and output path.

        Args:
            input_path (str): Path to the input video
            output_path (str): Path to the output video
            display (bool): Whether to display the video while processing
            debug (bool): Whether to display the layers while processing
        """
        self.input_path = input_path
        self.output_path = output_path
        self.display = display
        self.debug_frames = debug_frames if debug_frames is not None else []
        self.debug_output_dir = os.path.splitext(output_path)[0] + "_debug"

        if self.display:
            plt.ion()  # Interactive mode on

        if self.debug_frames:
            os.makedirs(self.debug_output_dir, exist_ok=True)
    # end __init__

    def init_effects(
            self
    ):
        """
        Initialize the effects, called before starting the video.
        """
        pass
    # end init_effects

    def process_frame(
            self,
            image_obj,
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
            image_obj
    ):
        """
        Display the layers of the image object.

        Args:
            image_obj (Image): Image object
        """
        num_layers = len(image_obj.layers)
        fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
        if len(image_obj.layers) == 1:
            axes = [axes]
        # end if

        # For each layer
        for ax, layer in zip(axes, image_obj.layers):
            ax.imshow(cv2.cvtColor(layer.image[:, :, :3], cv2.COLOR_BGR2RGB))
            ax.set_title(layer.name)
            ax.axis('off')
        # end for

        # Pause
        plt.pause(0.001)
        plt.show()
    # end display_layers

    def save_debug_layers(self, image_obj, frame_number):
        fig, axes = plt.subplots(1, len(image_obj.layers), figsize=(15, 5))
        if len(image_obj.layers) == 1:
            axes = [axes]

        for ax, layer in zip(axes, image_obj.layers):
            ax.imshow(cv2.cvtColor(layer.image[:, :, :3], cv2.COLOR_BGR2RGB))
            ax.set_title(layer.name)
            ax.axis('off')

        debug_frame_path = os.path.join(self.debug_output_dir, f'frame_{frame_number}.png')
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
                image_obj = Image()
                image_obj.add_layer(
                    "input_frame",
                    np.dstack([frame, np.ones_like(frame[:, :, 0]) * 255])
                )
                image_obj = self.process_frame(
                    image_obj,
                    frame_number,
                    frame_count
                )
                final_frame = image_obj.merge_layers()
                out.write(final_frame[:, :, :3])

                if self.display:
                    plt.clf()  # Clear the current figure
                    plt.imshow(cv2.cvtColor(final_frame[:, :, :3], cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.pause(0.001)
                    plt.draw()
                # end if

                if frame_number in self.debug_frames:
                    self.save_debug_layers(image_obj, frame_number)

                ret, frame = cap.read()
                pbar.update(1)
                frame_number += 1
            # end while
        # end with

        # Release the video capture and writer
        cap.release()
        out.release()

        # Close the display
        if self.display or self.debug:
            plt.ioff()
            plt.show()
        # end if
    # end compose_video

# end Animation

