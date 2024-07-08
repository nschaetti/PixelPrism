

class EffectBase:
    def apply(self, image, **kwargs):
        raise NotImplementedError("Subclasses should implement this method!")

