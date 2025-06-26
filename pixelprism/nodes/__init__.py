"""
Pixel Prism Nodes
================

This subpackage provides nodes for image processing and manipulation in the Pixel Prism framework.
Nodes are modular components that can be connected together to create complex image processing pipelines.

Available Nodes
--------------
- :class:`~pixelprism.nodes.ContourFinding`: Find contours in an image
- :class:`~pixelprism.nodes.SelectChannel`: Select a specific channel from an image
- :class:`~pixelprism.nodes.GrayScale`: Convert an image to grayscale
- :class:`~pixelprism.nodes.VectorsToString`: Convert vectors to a string representation
- :class:`~pixelprism.nodes.DrawPolygon`: Draw a polygon on an image

The nodes are organized into categories:
- Core nodes for basic image processing
- Utility nodes for common operations
- Visualization nodes for displaying and rendering

Each node has a defined set of inputs and outputs, allowing them to be connected
in a workflow. Nodes can be used individually or combined to create complex
image processing pipelines.
"""

from .nodes import ContourFinding
from .utils import SelectChannel, GrayScale
from .visualisation import VectorsToString, DrawPolygon


__all__ = [
    "ContourFinding",
    # Utils
    "SelectChannel",
    "GrayScale",
    "VectorsToString",
    # Draw
    "DrawPolygon"
]
