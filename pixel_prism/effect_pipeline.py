"""
Effect Pipeline Module
=====================

This module provides the EffectPipeline class for applying a sequence of effects to images.
"""

import time


class EffectPipeline:
    """
    A pipeline for applying a sequence of effects to images.

    The EffectPipeline allows you to chain multiple effects together and apply them
    sequentially to an image. It also tracks performance statistics for each effect.

    Attributes:
        effects (list): List of effect objects to apply
        time_stats (dict): Dictionary to store timing statistics for each effect
    """

    def __init__(self, effects=[]):
        """
        Initialize the effect pipeline.

        Args:
            effects (list, optional): Initial list of effects to add to the pipeline. Defaults to [].
        """
        self.effects = effects
        self.time_stats = {}

    def add_effect(self, effect: object):
        """
        Add an effect to the pipeline.

        Args:
            effect: The effect object to add to the pipeline. Must have an apply() method.
        """
        self.effects.append(effect)

    def apply(self, image: object, **kwargs: object) -> object:
        """
        Apply all effects in the pipeline to the image.

        This method applies each effect in the pipeline sequentially to the image,
        and tracks the time taken by each effect.

        Args:
            image: The input image to process.
            **kwargs: Additional keyword arguments to pass to each effect's apply method.

        Returns:
            The processed image after applying all effects.
        """
        total_start = time.time()
        for effect in self.effects:
            start = time.time()
            image = effect.apply(image, **kwargs)
            end = time.time()
            effect_name = effect.__class__.__name__
            if effect_name not in self.time_stats:
                self.time_stats[effect_name] = 0
            self.time_stats[effect_name] += end - start
        self.time_stats["Total"] = time.time() - total_start
        return image

    def print_stats(self):
        """
        Print performance statistics for each effect in the pipeline.

        This method prints the total processing time and the time taken by each effect,
        along with the percentage of total time each effect consumed.
        """
        total_time = self.time_stats.pop("Total", None)
        if total_time:
            print(f"Total processing time: {total_time:.2f} seconds")
            for effect, time_taken in self.time_stats.items():
                percentage = (time_taken / total_time) * 100
                print(f"{effect}: {time_taken:.2f} seconds ({percentage:.2f}%)")
