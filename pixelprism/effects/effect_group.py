

class EffectGroup:
    def __init__(
            self,
            name,
            effects=None
    ):
        """
        Initialize the effect group with a name and a list of effects

        Args:
            name (str): Name of the effect group
            effects (list): List of effects to apply
        """
        self.name = name
        self.effects = effects if effects is not None else []
    # end __init__

    def add_effect(
            self,
            effect
    ):
        """
        Add an effect to the effect group

        Args:
            effect (EffectBase): Effect to add to the effect group

        Returns:
            None
        """
        self.effects.append(effect)
    # end add_effect

    def apply(
            self,
            image
    ):
        """
        Apply all effects in the effect group to the image

        Args:
            image (np.ndarray): Image to apply the effects to
        """
        for effect in self.effects:
            image = effect.apply(image)
        # end for
        return image
    # end apply

# end EffectGroup
