# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2026 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Compose output videos from animation classes.

This module provides :class:`VideoComposer`, a small orchestration layer that
instantiates an :class:`~pixelprism.animation.Animation`, then runs it either
through an interactive viewer or directly in headless rendering mode.
"""

# Imports
from __future__ import annotations

from typing import Any, Type

from .animation import Animation, AnimationViewer


class VideoComposer:
    """High-level wrapper that composes a video from an animation class.

    Parameters
    ----------
    input_path : str | None
        Optional path to an input video used by the animation.
    output_path : str
        Path where the rendered video is written.
    duration : int
        Target duration in seconds when no input video is provided.
    fps : int
        Output frames per second.
    width : int
        Output video width in pixels.
    height : int
        Output video height in pixels.
    animation_class : Type[Animation]
        Animation class to instantiate.
    debug_frames : bool | tuple[int, ...] | list[int], default=False
        Debug frame configuration forwarded to :class:`Animation`.
    save_frames : bool, default=False
        Whether the animation should save individual rendered frames.
    viewer : bool, default=True
        If ``True``, run the interactive viewer while composing.
    **kwargs : Any
        Additional keyword arguments forwarded to ``animation_class``.

    Attributes
    ----------
    animation : Animation
        Instantiated animation object.
    viewer : AnimationViewer | None
        Optional interactive viewer.
    """
    def __init__(
            self,
            input_path: str | None,
            output_path: str,
            duration: int,
            fps: int,
            width: int,
            height: int,
            animation_class: Type[Animation],
            debug_frames=False,
            save_frames: bool = False,
            viewer: bool = True,
            **kwargs: Any
    ):
        """Initialize the video composer.

        Parameters
        ----------
        input_path : str | None
            Optional path to an input video used by the animation.
        output_path : str
            Path where the rendered video is written.
        duration : int
            Target duration in seconds when no input video is provided.
        fps : int
            Output frames per second.
        width : int
            Output video width in pixels.
        height : int
            Output video height in pixels.
        animation_class : Type[Animation]
            Animation class to instantiate.
        debug_frames : bool | tuple[int, ...] | list[int], default=False
            Debug frame configuration forwarded to :class:`Animation`.
        save_frames : bool, default=False
            Whether the animation should save individual rendered frames.
        viewer : bool, default=True
            If ``True``, run the interactive viewer while composing.
        **kwargs : Any
            Additional keyword arguments forwarded to ``animation_class``.
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

        # Animation viewer object
        self.viewer = AnimationViewer(self.animation) if viewer else None
    # end def __init__

    def create_video(self) -> None:
        """Compose the output video.

        Notes
        -----
        If a viewer is available, composition is controlled by
        :class:`AnimationViewer`. Otherwise, composition runs directly via
        :meth:`Animation.compose_video`.
        """
        if self.viewer is not None:
            self.viewer.run()
        else:
            self.animation.compose_video()
        # end if
    # end def create_video

# end class VideoComposer
