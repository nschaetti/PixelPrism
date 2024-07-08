

class VideoComposer:
    def __init__(
            self,
            input_path,
            output_path,
            animation_class
    ):
        """
        Initialize the video composer with an input and output path.

        Args:
            input_path (str): Path to the input video
            output_path (str): Path to the output video
            animation_class (Animation): Animation class to use
        """
        self.animation = animation_class(input_path, output_path)
    # end __init__

    def create_video(self):
        """
        Create the video by composing the animation.
        """
        self.animation.compose_video()
    # end create_video

# end VideoComposer
