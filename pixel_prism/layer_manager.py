
"""
Layer Manager Module
===================

This module provides the LayerManager class for managing image layers and effect groups.
"""

import cv2
import numpy as np

from .primitives import ImageLayer
from .effects import EffectGroup


class LayerManager:
    """
    A manager for image layers and effect groups.

    The LayerManager allows you to create, manipulate, and combine multiple image layers,
    as well as apply effects to them. It supports various blending modes and effect pipelines.

    Attributes:
        layers (list): List of ImageLayer objects
        effect_groups (list): List of EffectGroup objects
    """

    def __init__(self):
        """
        Initialize the layer manager.
        """
        self.layers = []
        self.effect_groups = []

    def add_layer(self, name, image: object = None):
        """
        Add a new layer to the layer manager.

        Args:
            name (str): Name of the layer
            image: Image data for the layer. If None, an empty layer is created.
        """
        self.layers.append(ImageLayer(name, image))

    def remove_layer(self, name):
        """
        Remove a layer from the layer manager.

        Args:
            name (str): Name of the layer to remove
        """
        self.layers = [layer for layer in self.layers if layer.name != name]

    def get_layer(self, name) -> 'ImageLayer':
        """
        Get a layer by name.

        Args:
            name (str): Name of the layer to retrieve

        Returns:
            ImageLayer: The layer with the specified name, or None if not found
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def add_effect_group(self, name, effects=None):
        """
        Add a new effect group to the layer manager.

        Args:
            name (str): Name of the effect group
            effects (list, optional): List of effects to include in the group. Defaults to None.
        """
        self.effect_groups.append(EffectGroup(name, effects))

    def get_effect_group(self, name) -> 'EffectGroup':
        """
        Get an effect group by name.

        Args:
            name (str): Name of the effect group to retrieve

        Returns:
            EffectGroup: The effect group with the specified name, or None if not found
        """
        for group in self.effect_groups:
            if group.name == name:
                return group
        return None

    def apply_effect_to_layer(self, effect: object, layer_name):
        """
        Apply an effect to a layer.

        Args:
            effect: The effect to apply
            layer_name (str): Name of the layer to apply the effect to
        """
        layer = self.get_layer(layer_name)
        if layer is not None:
            layer.apply_effect(effect)

    def apply_effect_group_to_layer(self, group_name, layer_name):
        """
        Apply an effect group to a layer.

        Args:
            group_name (str): Name of the effect group to apply
            layer_name (str): Name of the layer to apply the effect group to
        """
        group = self.get_effect_group(group_name)
        layer = self.get_layer(layer_name)
        if group is not None and layer is not None:
            layer.image = group.apply(layer.image)

    def merge_layers(self, layer1_name, layer2_name, mode='addition', output_layer_name='merged'):
        """
        Merge two layers using the specified blend mode.

        Args:
            layer1_name (str): Name of the first layer
            layer2_name (str): Name of the second layer
            mode (str, optional): Blend mode to use. Options are 'addition', 'multiply', 
                                 'screen', or 'overlay'. Defaults to 'addition'.
            output_layer_name (str, optional): Name of the output layer. Defaults to 'merged'.

        Raises:
            ValueError: If an unsupported blend mode is specified
        """
        layer1 = self.get_layer(layer1_name)
        layer2 = self.get_layer(layer2_name)

        if layer1 is not None and layer2 is not None:
            if mode == 'addition':
                merged_image = cv2.add(layer1.image, layer2.image)
            elif mode == 'multiply':
                merged_image = cv2.multiply(layer1.image, layer2.image, scale=1/255.0)
            elif mode == 'screen':
                inverted_image1 = 255 - layer1.image
                inverted_image2 = 255 - layer2.image
                merged_image = 255 - cv2.multiply(inverted_image1, inverted_image2, scale=1/255.0)
            elif mode == 'overlay':
                merged_image = np.where(layer2.image > 128, 255 - 2 * (255 - layer2.image) * (255 - layer1.image) / 255, 2 * layer1.image * layer2.image / 255)
                merged_image = merged_image.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported blend mode: {mode}")

            self.add_layer(output_layer_name, merged_image)

    def apply_pipeline(self, pipeline, input_layer_name, output_layer_name):
        """
        Apply a pipeline of effects to a layer and store the result in a new layer.

        Args:
            pipeline (list): List of effects to apply in sequence
            input_layer_name (str): Name of the input layer
            output_layer_name (str): Name of the output layer
        """
        layer = self.get_layer(input_layer_name)
        if layer is not None:
            image = layer.image
            for effect in pipeline:
                image = effect.apply(image)
            self.add_layer(output_layer_name, image)
