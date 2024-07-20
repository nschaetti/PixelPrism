
# Imports


class VideoComposer:
    def __init__(
            self,
            input_path,
            output_path,
            duration: int,
            fps: int,
            width: int,
            height: int,
            animation_class,
            debug_frames=False,
            save_frames: bool = False,
            **kwargs
    ):
        """
        Initialize the video composer with an input and output path.

        Args:
            input_path (str): Path to the input video
            output_path (str): Path to the output video
            duration (int): Duration of the video in seconds
            fps (int): Frames per second
            width (int): Width of the video
            height (int): Height of the video
            animation_class: Class of the animation to compose
            debug_frames (bool): Whether to display the layers while processing
            save_frames (bool): Whether to save the frames to disk
            kwarg: Additional keyword
        """
        self.animation = animation_class(
            input_video=input_path,
            output_video=output_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            debug_frames=debug_frames,
            save_frames=save_frames,
            **kwargs
        )
    # end __init__

    def create_video(self):
        """
        Create the video by composing the animation.
        """
        self.animation.compose_video()
    # end create_video

# end VideoComposer
