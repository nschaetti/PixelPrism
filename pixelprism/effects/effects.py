

import cv2
import numpy as np
from tqdm import tqdm
import argparse

from pixelprism.effects.effect_base import EffectBase
import pixelprism.effects.functional as F
from pixelprism.base.image import Image


class GlowEffect(EffectBase):
    """
    Glow effect that applies a glow to the image.
    """

    def __init__(
            self,
            blur_strength=5,
            intensity=0.5,
            blend_mode='screen'
    ):
        """
        Initialize the glow effect with the blur strength, intensity, and blend mode.

        Args:
            blur_strength (int): Strength of the Gaussian blur
            intensity (float): Intensity of the glow
            blend_mode (str): Blend mode for the glow effect
        """
        self.blur_strength = blur_strength
        self.intensity = intensity
        self.blend_mode = blend_mode
    # end __init__

    def apply(
            self,
            image: Image,
            **kwargs
    ) -> Image:
        """
        Apply the glow effect to the image.

        Args:
            image (Image): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        return F.simple_glow(
            image,
            intensity=self.intensity,
            blur_strength=self.blur_strength,
            blend_mode=self.blend_mode
        )
    # end apply

# end GlowEffect


class AdvancedTVEffect(EffectBase):

    def __init__(
            self,
            pixel_width=10,
            pixel_height=10,
            border_strength=2,
            border_color=(0, 0, 0),
            corner_radius=2,
            blur_kernel_size=5,
            vertical_shift=0
    ):
        """
        Initialize the TV effect with the pixel dimensions and border properties

        Args:
            pixel_width (int): Width of the pixel
            pixel_height (int): Height of the pixel
            border_strength (int): Strength of the border
            border_color (tuple): Color of the border
            corner_radius (int): Radius of the corner
            blur_kernel_size (int): Kernel size for Gaussian blur
            vertical_shift (int): Vertical shift for odd rows
        """
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.border_strength = border_strength
        self.border_color = border_color
        self.corner_radius = corner_radius
        self.blur_kernel_size = blur_kernel_size
        self.vertical_shift = vertical_shift
        self.overlay = None
    # end __init__

    def create_overlay(
            self,
            shape
    ):
        """
        Create the overlay for the TV effect

        Args:
            shape (tuple): Shape of the image (height, width, channels)
        """
        self.overlay = F.create_tv_overlay(
            shape,
            pixel_width=self.pixel_width,
            pixel_height=self.pixel_height,
            vertical_shift=self.vertical_shift,
            border_color=self.border_color,
            border_strength=self.border_strength,
            corner_radius=self.corner_radius,
            blur_kernel_size=self.blur_kernel_size
        )
    # end create_overlay

    def apply(
            self,
            image: Image,
            **kwargs
    ) -> Image:
        """
        Apply the TV effect to the image

        Args:
            image (Image): Image to apply the effect to
            kwargs: Additional keyword arguments

        Returns:
            Image: Image with the TV effect applied
        """
        if self.overlay is None:
            self.create_overlay(image.shape)
        # end if
        tv_effect = image * self.overlay
        return tv_effect.astype(np.uint8)
    # end apply

# end AdvancedTVEffect


class ChromaticAberrationEffect(EffectBase):

    def apply(self, image, speed=1.0, **kwargs):
        shift = int(speed * 5)
        b, g, r = cv2.split(image)
        rows, cols = b.shape
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        b_shifted = cv2.warpAffine(b, M, (cols, rows))
        g_shifted = cv2.warpAffine(g, M, (cols, rows))
        chrom_aberration = cv2.merge((b_shifted, g_shifted, r))
        return chrom_aberration


class BlurEffect(EffectBase):
    def __init__(self, blur_strength=5):
        self.blur_strength = blur_strength

    def apply(self, image, **kwargs):
        # Appliquer un flou gaussien
        blurred_image = cv2.GaussianBlur(image, (self.blur_strength * 2 + 1, self.blur_strength * 2 + 1), 0)
        return blurred_image


class LenticularDistortionEffect(EffectBase):
    def __init__(self, strength=0.00001):
        self.strength = strength

    def apply(self, image, **kwargs):
        rows, cols, ch = image.shape
        # Create mesh grid
        x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        # Calculate distance from center
        r = np.sqrt(x**2 + y**2)
        # Apply barrel distortion
        factor = 1 + self.strength * r**2
        map_x = x * factor
        map_y = y * factor
        # Scale map_x and map_y to image size
        map_x = ((map_x + 1) * cols - 1) / 2
        map_y = ((map_y + 1) * rows - 1) / 2
        # Apply remap
        distorted_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
        return distorted_image
