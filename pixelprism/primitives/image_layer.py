

class ImageLayer:
    def __init__(self, name, image=None):
        self.name = name
        self.image = image

    def apply_effect(self, effect):
        if self.image is not None:
            self.image = effect.apply(self.image)
