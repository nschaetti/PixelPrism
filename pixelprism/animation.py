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


"""Animation primitives and rendering loop.

This module defines:

- :func:`find_animable_mixins` to recursively discover animable objects,
- :class:`AnimationViewer` to preview rendering with a Tkinter window,
- :class:`Animation` as the base class for timeline-driven video generation.

Subclasses should override :meth:`Animation.build` and
:meth:`Animation.process_frame`.
"""

# Imports
from typing import Optional
import os
import queue
import threading
import cv2
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from .animate import Animate, Animator
from .animate.able import AnimableMixin
from .base.image import Image as PixelImage
from .base.imagecanvas import ImageCanvas
from .render_engine import RenderEngine
from .utils import setup_logger


# Setup logger
logger = setup_logger(__name__)


def find_animable_mixins(obj, visited=None) -> list:
    """Recursively collect all :class:`AnimableMixin` instances.

    Parameters
    ----------
    obj : object
        Root object to inspect.
    visited : set[int] | None, default=None
        Internal set of visited object ids, used to avoid recursive loops.

    Returns
    -------
    list
        Flat list of discovered animable objects.
    """
    if visited is None:
        visited = set()
    # end if

    # Avoid revisiting objects that have already been inspected (to prevent infinite recursion)
    if id(obj) in visited:
        return []
    # end if

    # Mark the object as visited
    visited.add(id(obj))

    # List of found objects
    animable_objects = []

    # Check if the current object is an instance of AnimableMixin
    if isinstance(obj, AnimableMixin):
        animable_objects.append(obj)
    # end if

    # Check if object is for inspection
    def is_animeclass(obj):
        """Check whether ``obj`` class advertises animeclass metadata."""
        return hasattr(obj.__class__, '_attrs_to_inspect') and obj.__class__.is_animeclass()
    # end def is_animeclass

    # Get animeclass attributes
    def animeclass_attributes(obj):
        """Return attribute names marked for animeclass inspection."""
        return obj.__class__.animeclass_attributes()
    # end def animeclass_attributes

    # Check if the class of this object is marked for inspection with @animeclass
    if is_animeclass(obj):
        # Iterate over the attributes marked with @animeattribut in the class
        for attr_name in animeclass_attributes(obj):
            try:
                # Get the attribute value
                attr_value = getattr(obj, attr_name)

                # If the attribute is a list, propagate through the list elements
                if isinstance(attr_value, list):
                    for item in attr_value:
                        animable_objects.extend(find_animable_mixins(item, visited))
                    # end for
                # end if
                # If the attribute is a dictionary, propagate through the values
                elif isinstance(attr_value, dict):
                    for key, value in attr_value.items():
                        animable_objects.extend(find_animable_mixins(value, visited))
                    # end for
                # If the attribute itself is an animeclass, propagate it
                else:
                    animable_objects.extend(find_animable_mixins(attr_value, visited))
                # end if
            except AttributeError:
                # Ignore attributes that cannot be accessed
                continue
            # end try
        # end for
    # end if

    return animable_objects
# end def find_animable_mixins


class AnimationViewer(tk.Tk):
    """Simple Tkinter-based preview window for an :class:`Animation`.

    Parameters
    ----------
    animation : Animation
        Animation instance to compose and preview.
    """

    def __init__(self, animation):
        """Initialize the viewer widgets and rendering thread state.

        Parameters
        ----------
        animation : Animation
            Animation instance associated with this viewer.
        """
        super().__init__()

        # Set the animation
        self.animation = animation
        self.title("Animation Viewer")
        self.geometry(f"{self.animation.width}x{self.animation.height+50}")  # Adjust height for button
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Frame for the control elements (button and progress bar)
        self.control_frame = tk.Frame(self)
        self.control_frame.pack(fill=tk.X)

        # Button to stop the animation (left side)
        self.stop_button = tk.Button(self.control_frame, text="Arrêter", command=self.stop_animation)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Progress bar (right side)
        self.progress = ttk.Progressbar(self.control_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, padx=10, pady=10, expand=True)

        # Canvas for displaying images
        self.canvas = tk.Canvas(self, width=self.animation.width, height=self.animation.height)
        self.canvas.pack()

        # Initialize buffer
        self.buffer = None
        self.img = None

        # Image
        self.stop = False
        self.thread = None  # To hold the animation thread
    # end def __init__

    def stop_animation(self):
        """Request a graceful stop of the rendering process."""
        self.stop = True
    # end def stop_animation

    def on_close(self):
        """Handle window close and synchronize with worker thread."""
        self.stop = True
        self.destroy()

        # Wait for the thread to finish if it exists
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # end if
    # end def on_close

    def update_image(self, image):
        """Refresh the preview canvas with a new frame.

        Parameters
        ----------
        image : np.ndarray
            BGR image produced by the renderer.
        """
        # Convert BGR image to RGB for Tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))

        # Use the buffer to reduce flickering
        if self.buffer is None:
            self.buffer = self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        else:
            self.canvas.itemconfig(self.buffer, image=img)
        # end if

        # Keep a reference to the image to prevent garbage collection
        self.img = img
    # end def update_image

    def update_progress(self, value):
        """Update progress bar value.

        Parameters
        ----------
        value : float
            Progress percentage in ``[0, 100]``.
        """
        self.progress['value'] = value
        self.update_idletasks()  # Refresh the GUI
    # end def update_progress

    def run(self):
        """Start composition in a background thread and run Tk main loop."""
        self.thread = threading.Thread(target=self.animation.compose_video, args=(self,))
        self.thread.start()
        self.mainloop()

        # Ensure the thread is finished before exiting
        if self.thread.is_alive():
            self.thread.join()
        # end if
    # end def run

# end class AnimationViewer


class Animation:
    """Base class for creating timeline-driven animations.

    Parameters
    ----------
    input_video : str | None, default=None
        Optional input video path. If provided, FPS, frame count and dimensions
        are inferred from the source unless overridden by ``duration``.
    output_video : str, default="output.mp4"
        Output video path.
    duration : int | None, default=None
        Output duration in seconds when no input video is provided.
    fps : int, default=30
        Output frame rate when no input video is provided.
    width : int, default=1920
        Output width when no input video is provided.
    height : int, default=1080
        Output height when no input video is provided.
    keep_frames : int, default=0
        Number of previously rendered frames to keep in memory.
    debug_frames : int | list[int] | tuple[int, ...], default=False
        Frame indices for layer debug snapshots.
    save_frames : bool, default=False
        Whether to write each rendered frame as an image file.
    **kwargs
        Extra arguments available to subclasses.
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
        """Initialize runtime state and build the animation graph.

        Parameters
        ----------
        input_video : str | None, default=None
            Optional input video path.
        output_video : str, default="output.mp4"
            Output video path.
        duration : int | None, default=None
            Output duration in seconds.
        fps : int, default=30
            Frame rate when no input video is used.
        width : int, default=1920
            Output width when no input video is used.
        height : int, default=1080
            Output height when no input video is used.
        keep_frames : int, default=0
            Number of previous frames retained in memory.
        debug_frames : int | list[int] | tuple[int, ...], default=False
            Frame indices to dump debug layer previews.
        save_frames : bool, default=False
            Whether individual frame images should be written.
        **kwargs
            Extra keyword arguments consumed by subclasses.
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
    # end def __init__

    # region PROPERTIES

    @property
    def step(self):
        """float: Time step between two consecutive frames."""
        return 1 / self.fps
    # end def step

    # endregion PROPERTIES

    # region PUBLIC

    def init_effects(
            self
    ):
        """Initialize effects before frame processing starts.

        Notes
        -----
        Subclasses can override this hook to reset effect state.
        """
        pass
    # end def init_effects

    # Build the animation
    def build(
            self
    ):
        """Build animation objects and register transitions.

        Notes
        -----
        Subclasses should override this method.
        """
        pass
    # end def build

    # Add an effect
    def add_effect(
            self,
            name,
            effect
    ):
        """Register an effect by name.

        Parameters
        ----------
        name : str
            Effect identifier.
        effect : object
            Effect instance.
        """
        self.effects[name] = effect
    # end def add_effect

    # Get an effect
    def get_effect(
            self,
            name
    ) -> 'EffectBase':
        """Return an effect by name.

        Parameters
        ----------
        name : str
            Effect identifier.

        Returns
        -------
        EffectBase
            Effect instance if available.
        """
        # from pixel_prism.effects.effect_base import EffectBase
        # return self.effects.get(name)
        return None
    # end def get_effect

    # Add object
    def add_object(
            self,
            name,
            obj
    ):
        """Register an object in the scene object registry.

        Parameters
        ----------
        name : str
            Object identifier.
        obj : object
            Object instance.
        """
        self.objects[name] = obj
    # end def add_object

    # Add objets
    def add(
            self,
            **kwargs: object
    ):
        """Add objects and auto-register their animable transitions.

        Parameters
        ----------
        **kwargs : object
            Mapping of object names to instances.
        """
        for name, obj in kwargs.items():
            # Add object
            self.add_object(name, obj)

            # Find all animable objects
            animable_objects = find_animable_mixins(obj)

            # Add animations
            for animable in animable_objects:
                for animator in animable.animable_registry:
                    self.animate(animator)
                # end for
            # end if
        # end for
    # end def add

    # Get object
    def get_object(
            self,
            name
    ) -> object:
        """Return a previously registered object.

        Parameters
        ----------
        name : str
            Object identifier.

        Returns
        -------
        object
            Registered object.

        Raises
        ------
        ValueError
            If the object is missing.
        """
        if name not in self.objects:
            raise ValueError(f"Object not found: {name}")
        # end if
        return self.objects.get(name)
    # end def get_object

    # Get object
    def obj(self, name) -> object:
        """Short alias for :meth:`get_object`.

        Parameters
        ----------
        name : str
            Object identifier.

        Returns
        -------
        object
            Registered object.
        """
        return self.get_object(name)
    # end def obj

    # Remove an effect
    def remove_effect(
            self,
            name
    ):
        """Remove an effect from the registry.

        Parameters
        ----------
        name : str
            Effect identifier.
        """
        self.effects.pop(name)
    # end def remove_effect

    # Append animation
    def append_transition(self, transition):
        """Append a transition if it is not already registered.

        Parameters
        ----------
        transition : Animate
            Transition instance to append.
        """
        if transition not in self.transitions:
            self.transitions.append(transition)
        # end if
    # end def append_transition

    def animate(
            self,
            transition
    ):
        """Register one or multiple transitions.

        Parameters
        ----------
        transition : Animate | Animator | list[Animate]
            Transition object, animator container, or list of transitions.
        """
        if isinstance(transition, Animate):
            self.append_transition(transition)
        elif isinstance(transition, list):
            for trans in transition:
                if isinstance(trans, Animate):
                    self.append_transition(transition)
                # end if
            # end for
        elif isinstance(transition, Animator):
            for anim in transition.animations:
                self.append_transition(anim)
            # end for
        # end if
    # end def animate

    def apply_transitions(
            self,
            t
    ):
        """Apply all registered transitions at time ``t``.

        Parameters
        ----------
        t : float
            Current timeline time in seconds.
        """
        for transition in self.transitions:
            transition.update(t)
        # end for
    # end def apply_transitions

    def process_frame(
            self,
            image_canvas: ImageCanvas,
            t: float,
            frame_number: int
    ):
        """Render one frame.

        Parameters
        ----------
        image_canvas : ImageCanvas
            Canvas for current frame content.
        t : float
            Current timeline time in seconds.
        frame_number : int
            Current frame index.

        Returns
        -------
        ImageCanvas
            Updated canvas.

        Raises
        ------
        NotImplementedError
            Raised by default; subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method!")
    # end def process_frame

    def display_layers(
            self,
            image_canvas
    ):
        """Display all canvas layers with Matplotlib for debugging.

        Parameters
        ----------
        image_canvas : ImageCanvas
            Canvas containing render layers.
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
    # end def display_layers

    def save_debug_layers(
            self,
            image_canvas,
            frame_number
    ):
        """Save per-layer debug previews for one frame.

        Parameters
        ----------
        image_canvas : ImageCanvas
            Canvas containing render layers.
        frame_number : int
            Current frame index.
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
    # end def save_debug_layers

    def compose_video(
            self,
            viewer=None
    ):
        """Compose the final video by iterating over all frames.

        Parameters
        ----------
        viewer : AnimationViewer | None, default=None
            Optional interactive viewer used to preview rendering.
        """
        fps = self.fps
        width = self.width
        height = self.height

        # Open the video capture
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(
            self.output_video,
            fourcc,
            fps,
            (width, height)
        )
        if not out.isOpened():
            raise ValueError(f"Could not open the output video file for writing: {self.output_video}")
        # end if

        logger.info(f"Input video: {self.input_video}, FPS: {fps}, width: {width}, height: {height}")
        logger.info(f"Output video: {self.output_video}, fourcc: {fourcc}, out: {out}")

        # Frame count
        frame_count = self.frame_count

        # Normalize debug frame configuration
        if isinstance(self.debug_frames, bool):
            debug_frame_set = set()
        elif isinstance(self.debug_frames, int):
            debug_frame_set = {self.debug_frames}
        elif isinstance(self.debug_frames, (list, tuple, set)):
            debug_frame_set = set(self.debug_frames)
        else:
            debug_frame_set = set()
        # end if

        # Reusable compositing buffer
        render_buffer = PixelImage.fill(width, height, (0, 0, 0, 255))
        static_render_buffer = PixelImage.fill(width, height, (0, 0, 0, 255))

        # Static layer cache
        static_cache_enabled = True
        static_cache_image: Optional[PixelImage] = None
        static_cache_signature: Optional[tuple] = None
        static_canvas = ImageCanvas(width, height)
        dynamic_canvas = ImageCanvas(width, height)

        # Optional reusable canvas when there is no input video
        reusable_canvas = ImageCanvas(width, height) if not self.cap else None

        # Writer pipeline (encode on a dedicated thread)
        frame_queue = queue.Queue(maxsize=8)
        queue_sentinel = object()
        writer_errors: list[BaseException] = []

        def _writer_worker() -> None:
            """Write encoded frames from the queue to output video."""
            try:
                while True:
                    frame_payload = frame_queue.get()
                    try:
                        if frame_payload is queue_sentinel:
                            break
                        # end if
                        out.write(frame_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
                    finally:
                        frame_queue.task_done()
                    # end try
                # end while
            except BaseException as exc:
                writer_errors.append(exc)
            # end try
        # end def _writer_worker

        writer_thread = threading.Thread(target=_writer_worker, daemon=True)
        writer_thread.start()

        try:
            # Process each frame of the video
            with tqdm(total=frame_count, desc="Processing video") as pbar:
                # Frame number
                frame_number = 0

                # Get the first frame or create a blank frame
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
                    # Check if the viewer has stopped
                    if viewer and viewer.stop:
                        break
                    # end if

                    # Create canvas from a frame
                    if self.cap:
                        image_canvas = ImageCanvas.from_numpy(frame, add_alpha=True)
                    else:
                        assert reusable_canvas is not None
                        reusable_canvas.layers.clear()
                        image_canvas = reusable_canvas
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

                    # Compose final image (with optional static layer caching)
                    if static_cache_enabled:
                        static_layers = [
                            layer for layer in image_canvas.layers
                            if getattr(layer, "is_static", False)
                        ]
                        dynamic_layers = [
                            layer for layer in image_canvas.layers
                            if not getattr(layer, "is_static", False)
                        ]

                        if static_layers:
                            static_signature = tuple(
                                (
                                    layer.name,
                                    id(layer.image),
                                    layer.active,
                                    layer.blend_mode
                                )
                                for layer in static_layers
                            )

                            if (
                                static_cache_image is None
                                or static_cache_signature != static_signature
                            ):
                                static_canvas.layers = static_layers
                                static_cache_image = RenderEngine.render(
                                    static_canvas,
                                    output_buffer=static_render_buffer
                                )
                                static_cache_signature = static_signature
                            # end if

                            if dynamic_layers:
                                dynamic_canvas.layers = dynamic_layers
                                final_frame = RenderEngine.render(
                                    dynamic_canvas,
                                    output_buffer=render_buffer,
                                    base_image=static_cache_image
                                )
                            else:
                                final_frame = static_cache_image
                            # end if
                        else:
                            final_frame = RenderEngine.render(
                                image_canvas,
                                output_buffer=render_buffer
                            )
                        # end if
                    else:
                        final_frame = RenderEngine.render(
                            image_canvas,
                            output_buffer=render_buffer
                        )
                    # end if

                    # Ensure the final frame is the correct size and type
                    assert final_frame.data.shape[:2] == (height, width), "Final frame size mismatch"
                    assert final_frame.data.dtype == np.uint8, "Final frame data type mismatch"

                    # Save as RGB (copy decouples renderer buffer from encoder thread)
                    encoded_frame = final_frame.data[:, :, :3].copy()
                    frame_queue.put(encoded_frame)

                    if writer_errors:
                        raise RuntimeError("Video writer thread failed.") from writer_errors[0]
                    # end if

                    # Save frames
                    if self.save_frames:
                        frame_path = os.path.join(self.save_frames_path, f'frame_{frame_number}.png')
                        final_frame.save(frame_path)
                    # end if

                    # Display the frame in the viewer
                    if viewer:
                        viewer.update_image(final_frame.data[:, :, :3])
                        viewer.update_progress((frame_number / frame_count) * 100)
                    # end if

                    # Save debug layers
                    if frame_number in debug_frame_set:
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
        finally:
            # Drain writer queue and terminate writer thread
            frame_queue.put(queue_sentinel)
            writer_thread.join()

            # Release the video capture and writer
            if self.cap:
                self.cap.release()
            # end if
            out.release()

            if writer_errors:
                raise RuntimeError("Video writer thread failed.") from writer_errors[0]
            # end if

            # If the video was successfully generated, open it with mpv
            if not viewer or not viewer.stop:
                subprocess.run(["mpv", self.output_video])
            # end if

            # Close the viewer if it exists
            if viewer:
                viewer.stop = True
                viewer.on_close()
            # end if
        # end try
    # end def compose_video

    # endregion PUBLIC

# end class Animation
