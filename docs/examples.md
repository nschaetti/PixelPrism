# Examples

This page contains examples of how to use PixelPrism for various video and image manipulation tasks.

## Basic Animation

Here's a simple example of creating a basic animation:

```python
import numpy as np
from pixel_prism import VideoComposer

class SimpleAnimation:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def render_frame(self, frame_number, time, delta_time):
        # Create a gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate a moving gradient based on time
        for y in range(self.height):
            for x in range(self.width):
                r = int(127 + 127 * np.sin(x * 0.01 + time))
                g = int(127 + 127 * np.sin(y * 0.01 + time * 0.7))
                b = int(127 + 127 * np.sin((x + y) * 0.01 + time * 1.3))
                frame[y, x] = [r, g, b]
                
        return frame

# Create a video composer
composer = VideoComposer(
    output_path="gradient_animation.mp4",
    duration=5.0,
    fps=30,
    width=640,
    height=480,
    animation_class=SimpleAnimation
)

# Create the video
composer.create_video()
```

## Adding Effects

You can add effects to your animations:

```python
from pixel_prism import VideoComposer
from pixel_prism.effects import BlurEffect, VignetteEffect

class EffectsAnimation:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.blur = BlurEffect(radius=5)
        self.vignette = VignetteEffect(strength=0.7)
        
    def render_frame(self, frame_number, time, delta_time):
        # Create a base frame (similar to previous example)
        frame = create_base_frame(self.width, self.height, time)
        
        # Apply effects
        frame = self.blur.apply(frame)
        frame = self.vignette.apply(frame)
                
        return frame

# Create a video composer
composer = VideoComposer(
    output_path="effects_animation.mp4",
    duration=5.0,
    fps=30,
    width=640,
    height=480,
    animation_class=EffectsAnimation
)

# Create the video
composer.create_video()
```

## Processing Existing Videos

You can also apply effects to existing videos:

```python
from pixel_prism import VideoComposer
from pixel_prism.effects import ColorGradingEffect

class VideoProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.color_grading = ColorGradingEffect(
            contrast=1.2,
            brightness=1.1,
            saturation=1.3
        )
        
    def render_frame(self, frame_number, time, delta_time, input_frame):
        # Apply color grading to the input frame
        processed_frame = self.color_grading.apply(input_frame)
        return processed_frame

# Create a video composer with an input video
composer = VideoComposer(
    input_path="input_video.mp4",
    output_path="processed_video.mp4",
    animation_class=VideoProcessor
)

# Process the video
composer.create_video()
```

## More Examples

For more examples, check out the [examples directory](https://github.com/nschaetti/PixelPrism/tree/main/examples) in the GitHub repository.