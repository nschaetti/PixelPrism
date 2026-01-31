# 🦊 PixelPrism — Where Maths meet Art

PixelPrism is a Python library that applies advanced visual effects to videos and images.

## Installation

You can install PixelPrism using pip:

```bash
pip install pixelprism
```

For development installation, clone the repository and install in development mode:

```bash
git clone https://github.com/nschaetti/PixelPrism.git
cd PixelPrism
pip install -e .
```

## Usage

### Command Line Interface

PixelPrism provides a command-line interface for applying effects to videos:

```bash
pixelprism output.mp4 --class-file my_animation.py --class-name MyAnimation --duration 10 --fps 30
```

Required arguments:
- `output`: Path to save the output video file
- `--class-file`: Path to the file containing your custom animation class
- `--class-name`: Name of the custom animation class to use

Optional arguments:
- `--input`: Path to an input video file (if not provided, generates animation from scratch)
- `--duration`: Duration of the animation in seconds (required if no input video is provided)
- `--fps`: Frames per second of the animation
- `--width`: Width of the output video (default: 1920)
- `--height`: Height of the output video (default: 1080)
- `--display`: Display the video while processing
- `--save-frames`: Save individual frames to disk

### Python API

You can also use PixelPrism as a library in your Python code:

```python
from pixelprism import VideoComposer
from my_animation import MyAnimation

# Create a video composer
composer = VideoComposer(
    output_path="output.mp4",
    duration=10,
    fps=30,
    width=1920,
    height=1080,
    animation_class=MyAnimation
)

# Create the video
composer.create_video()
```

## License

This project is licensed under the GNU General Public License v3 - see the LICENSE file for details.
