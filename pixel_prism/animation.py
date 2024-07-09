

# Imports
import cv2
from tqdm import tqdm
import numpy as np

# Locals
from pixel_prism.base.image import Image


class Animation:
    """
    Base class for creating animations. Subclasses should implement the `process_frame` method.
    """

    def __init__(
            self,
            input_path,
            output_path
    ):
        """
        Initialize the animation with an input and output path.

        Args:
            input_path (str): Path to the input video
            output_path (str): Path to the output video
        """
        self.input_path = input_path
        self.output_path = output_path
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
                out.write(final_frame[:, :, :3])  # Sauvegarder en RGB

                ret, frame = cap.read()
                pbar.update(1)
                frame_number += 1
            # end while
        # end with

        # Release the video capture and writer
        cap.release()
        out.release()
    # end compose_video

# end Animation

