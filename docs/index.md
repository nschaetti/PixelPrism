# PixelPrism

A Python library that applies advanced visual effects to videos and images.

## Installation

You can install PixelPrism using pip:

```bash
pip install pixelprism
```

For development or to include documentation tools:

```bash
pip install pixelprism[docs]
```

## Features

PixelPrism provides a comprehensive set of tools for video and image manipulation:

- Apply visual effects to videos and images
- Create animations with a flexible API
- Compose multiple effects together
- Render high-quality output

## Quick Start

Here's a simple example of how to use PixelPrism:

```python
from pixelprism import VideoComposer
from your_animation_file import YourCustomAnimation

# Create a video composer
composer = VideoComposer(
    output_path="output.mp4",
    duration=5.0,
    fps=30,
    width=1920,
    height=1080,
    animation_class=YourCustomAnimation
)

# Create the video
composer.create_video()
```

## Documentation

For detailed documentation, please see the [API Reference](reference/) section.

Math-focused guides:

- [Math Sampling](guides/math_sampling.md)
- [Math Statistics](guides/math_statistics.md)
- [Math Graphs](guides/math_graphs.md)
- [Math Machine Learning](guides/math_machine_learning.md)

## Examples

Check out the [Examples](examples.md) page for more usage examples.

## License

PixelPrism is licensed under the GNU General Public License v3 (GPLv3).
