

import cv2

from .primitives import ImageLayer
from .effects import EffectGroup


class LayerManager:
    def __init__(self):
        self.layers = []
        self.effect_groups = []

    def add_layer(self, name, image=None):
        self.layers.append(ImageLayer(name, image))

    def remove_layer(self, name):
        self.layers = [layer for layer in self.layers if layer.name != name]

    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def add_effect_group(self, name, effects=None):
        self.effect_groups.append(EffectGroup(name, effects))

    def get_effect_group(self, name):
        for group in self.effect_groups:
            if group.name == name:
                return group
        return None

    def apply_effect_to_layer(self, effect, layer_name):
        layer = self.get_layer(layer_name)
        if layer is not None:
            layer.apply_effect(effect)

    def apply_effect_group_to_layer(self, group_name, layer_name):
        group = self.get_effect_group(group_name)
        layer = self.get_layer(layer_name)
        if group is not None and layer is not None:
            layer.image = group.apply(layer.image)

    def merge_layers(self, layer1_name, layer2_name, mode='addition', output_layer_name='merged'):
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
        layer = self.get_layer(input_layer_name)
        if layer is not None:
            image = layer.image
            for effect in pipeline:
                image = effect.apply(image)
            self.add_layer(output_layer_name, image)



