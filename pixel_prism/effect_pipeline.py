
import time


class EffectPipeline:
    def __init__(self, effects=[]):
        self.effects = effects
        self.time_stats = {}

    def add_effect(self, effect):
        self.effects.append(effect)

    def apply(self, image, **kwargs):
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
        total_time = self.time_stats.pop("Total", None)
        if total_time:
            print(f"Total processing time: {total_time:.2f} seconds")
            for effect, time_taken in self.time_stats.items():
                percentage = (time_taken / total_time) * 100
                print(f"{effect}: {time_taken:.2f} seconds ({percentage:.2f}%)")