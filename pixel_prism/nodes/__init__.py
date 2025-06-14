"""
Pixel Prism Nodes
================

This subpackage provides nodes for image processing and manipulation in the Pixel Prism framework.
Nodes are modular components that can be connected together to create complex image processing pipelines.

Available Nodes
--------------
- :class:`~pixel_prism.nodes.ContourFinding`: Find contours in an image
- :class:`~pixel_prism.nodes.SelectChannel`: Select a specific channel from an image
- :class:`~pixel_prism.nodes.GrayScale`: Convert an image to grayscale
- :class:`~pixel_prism.nodes.VectorsToString`: Convert vectors to a string representation
- :class:`~pixel_prism.nodes.DrawPolygon`: Draw a polygon on an image

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
