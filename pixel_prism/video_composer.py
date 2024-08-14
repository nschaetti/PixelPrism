#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
from .animation import AnimationViewer


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
            viewer: bool = True,
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
            viewer (bool): Whether to display the video after creation
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
            viewer=viewer,
            **kwargs
        )

        # AnimationVideo object
        self.viewer = AnimationViewer(self.animation) if viewer else None
    # end __init__

    def create_video(self):
        """
        Create the video by composing the animation.
        """
        if self.viewer is not None:
            self.viewer.run()
        else:
            self.animation.compose_video()
        # end if
    # end create_video

# end VideoComposer
