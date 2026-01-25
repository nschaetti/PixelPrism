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

#
# Description: Base class for creating animations.
# Subclasses should implement the `process_frame` method.
#

# Imports
from typing import Optional
import os
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
from pixelprism.animate import Animate, Animator
from pixelprism.animate.able import AnimableMixin
from pixelprism.base.imagecanvas import ImageCanvas
from pixelprism.render_engine import RenderEngine
from pixelprism.utils import setup_logger


# Setup logger
logger = setup_logger(__name__)


def find_animable_mixins(obj, visited=None) -> list:
    """
    Find recursively all objects that are instances of AnimableMixin in the given object and its sub-attributes.

    Args:
        obj (object): Object to inspect.
        visited (set): Set of visited objects to avoid infinite recursion.

    Returns:
        list: List of AnimableMixin objects found.
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
        return hasattr(obj.__class__, '_attrs_to_inspect') and obj.__class__.is_animeclass()
    # end is_animeclass

    # Get animeclass attributes
    def animeclass_attributes(obj):
        return obj.__class__.animeclass_attributes()
    # end animaclass_attributes

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
# end find_animable_mixins


class AnimationViewer(tk.Tk):
    """
    A simple viewer for animations.
    """

    def __init__(self, animation):
        """
        Initialize the animation viewer.
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
    # end __init__

    def stop_animation(self):
        """
        Stop the animation.
        """
        self.stop = True
    # end stop_animation

    def on_close(self):
        """
        Close the animation viewer.
        """
        self.stop = True
        self.destroy()

        # Wait for the thread to finish if it exists
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # end if
    # end on_close

    def update_image(self, image):
        """
        Update the image on the canvas.
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
    # end update_image

    def update_progress(self, value):
        """
        Update the progress bar.
        """
        self.progress['value'] = value
        self.update_idletasks()  # Refresh the GUI
    # end update_progress

    def run(self):
        """
        Run the animation viewer.
        """
        self.thread = threading.Thread(target=self.animation.compose_video, args=(self,))
        self.thread.start()
        self.mainloop()

        # Ensure the thread is finished before exiting
        if self.thread.is_alive():
            self.thread.join()
        # end if
    # end run

# end AnimationViewer


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
            **kwargs: Additional keyword arguments
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

    # region PUBLIC

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
    ) -> 'EffectBase':
        """
        Get an effect from the animation.

        Args:
            name (str): Name of the effect

        Returns:
            EffectBase: Effect object
        """
        from pixelprism.effects.effect_base import EffectBase
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

    # Add objets
    def add(
            self,
            **kwargs: object
    ):
        """
        Add objects to the animation.

        Args:
            **kwargs: Objects to add to the animation
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
    # end add

    # Get object
    def get_object(
            self,
            name
    ) -> object:
        """
        Get a widget from the animation.

        Args:
            name (str): Name of the widget

        Returns:
            object: Object
        """
        if name not in self.objects:
            raise ValueError(f"Object not found: {name}")
        # end if
        return self.objects.get(name)
    # end get_object

    # Get object
    def obj(self, name) -> object:
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

    # Append animation
    def append_transition(self, transition):
        """
        Append a transition to the animation.

        Args:
            transition (Transition): Transition object
        """
        if transition not in self.transitions:
            self.transitions.append(transition)
        # end if
    # end append_transition

    def animate(
            self,
            transition
    ):
        """
        Add a transition to the animation.

        Args:
            transition (Transition): Transition object
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
            self,
            viewer=None
    ):
        """
        Compose the final video by applying `process_frame` to each frame.

        Args:
            viewer (AnimationViewer): Animation viewer
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

        try:
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
                    # Check if the viewer has stopped
                    if viewer and viewer.stop:
                        break
                    # end if

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
                    assert final_frame.data.input_shape[:2] == (height, width), "Final frame size mismatch"
                    assert final_frame.data.dtype == np.uint8, "Final frame data type mismatch"

                    # Save as RGB
                    out.write(final_frame.data[:, :, :3])

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
        finally:
            # Release the video capture and writer
            if self.cap:
                self.cap.release()
            # end if
            out.release()

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
    # end compose_video

    # endregion PUBLIC

# end Animation
