

import cv2
import numpy as np
from tqdm import tqdm
import argparse

from pixel_prism.effects.effect_base import EffectBase


class GlowEffect(EffectBase):
    def __init__(self, blur_strength=5, intensity=0.5, blend_mode='screen'):
        self.blur_strength = blur_strength
        self.intensity = intensity
        self.blend_mode = blend_mode

    def apply(self, image, **kwargs):
        # Appliquer un flou gaussien
        blurred_image = cv2.GaussianBlur(image, (self.blur_strength * 2 + 1, self.blur_strength * 2 + 1), 0)

        # Appliquer le mode de fusion
        if self.blend_mode == 'addition':
            glow_image = cv2.addWeighted(image, 1, blurred_image, self.intensity, 0)
        elif self.blend_mode == 'multiply':
            glow_image = cv2.multiply(image, blurred_image)
            glow_image = cv2.addWeighted(image, 1, glow_image, self.intensity - 1, 0)
        elif self.blend_mode == 'overlay':
            glow_image = np.where(blurred_image > 128, 255 - 2 * (255 - blurred_image) * (255 - image) / 255, 2 * image * blurred_image / 255)
            glow_image = cv2.addWeighted(image, 1, glow_image.astype(np.uint8), self.intensity, 0)
        else:  # screen is the default blend mode
            inverted_image = 255 - image
            inverted_blur = 255 - blurred_image
            screen_image = cv2.multiply(inverted_image, inverted_blur, scale=1/255.0)
            screen_image = 255 - screen_image
            glow_image = cv2.addWeighted(image, 1, screen_image, self.intensity, 0)

        return glow_image


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
        height, width, _ = shape
        overlay = np.ones((height, width, 3), dtype=np.float32)

        for j_i, j in enumerate(range(0, width, self.pixel_width)):
            if j_i % 2 == 0:
                shift = 0
            else:
                shift = self.vertical_shift
            # end if

            for i in range(-shift, height, self.pixel_height):
                y = i
                if y > height:
                    continue

                sub_img = overlay[y:y + self.pixel_height, j:j + self.pixel_width]

                # Dessiner les bords du pixel
                cv2.line(sub_img, (0, 0), (self.pixel_width, 0), self.border_color, self.border_strength)
                cv2.line(sub_img, (0, 0), (0, self.pixel_height), self.border_color, self.border_strength)
                cv2.line(sub_img, (self.pixel_width, 0), (self.pixel_width, self.pixel_height), self.border_color, self.border_strength)
                cv2.line(sub_img, (0, self.pixel_height), (self.pixel_width, self.pixel_height), self.border_color, self.border_strength)

                # Casser les coins
                if self.corner_radius > 0:
                    cv2.line(sub_img, (0, 0), (self.corner_radius, self.corner_radius), self.border_color, self.border_strength)
                    cv2.line(sub_img, (self.pixel_width, 0), (self.pixel_width - self.corner_radius, self.corner_radius), self.border_color, self.border_strength)
                    cv2.line(sub_img, (0, self.pixel_height), (self.corner_radius, self.pixel_height - self.corner_radius), self.border_color, self.border_strength)
                    cv2.line(sub_img, (self.pixel_width, self.pixel_height), (self.pixel_width - self.corner_radius, self.pixel_height - self.corner_radius), self.border_color, self.border_strength)
                # end if
            # end for
        # end for

        if self.blur_kernel_size > 1:
            overlay = cv2.GaussianBlur(overlay, (self.blur_kernel_size, self.blur_kernel_size), 0)
        # end if

        self.overlay = overlay
    # end create_overlay

    def apply(self, image, **kwargs):
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
