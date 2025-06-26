

from pixelprism.base.image import Image


class EffectBase:
    """
    Base class for effects
    """

    def apply(
            self,
            image: Image,
            **kwargs
    ) -> Image:
        """
        Apply the effect to the image

        Args:
            image (Image): Image to apply the effect to
            kwargs: Additional keyword arguments

        Returns:
            Image: Image with the effect applied
        """
        raise NotImplementedError("Subclasses should implement this method!")
    # end apply

# end EffectBase

