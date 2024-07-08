

# Imports
import cv2
import numpy as np

from pixel_prism.effects.effect_base import EffectBase


class ChromaticSpatialShiftEffect(EffectBase):
    def __init__(self, shift_r=(5, 0), shift_g=(-5, 0), shift_b=(0, 5)):
        self.shift_r = shift_r
        self.shift_g = shift_g
        self.shift_b = shift_b

    def apply(self, image, **kwargs):
        b, g, r = cv2.split(image)
        rows, cols = b.shape

        M_r = np.float32([[1, 0, self.shift_r[0]], [0, 1, self.shift_r[1]]])
        M_g = np.float32([[1, 0, self.shift_g[0]], [0, 1, self.shift_g[1]]])
        M_b = np.float32([[1, 0, self.shift_b[0]], [0, 1, self.shift_b[1]]])

        r_shifted = cv2.warpAffine(r, M_r, (cols, rows))
        g_shifted = cv2.warpAffine(g, M_g, (cols, rows))
        b_shifted = cv2.warpAffine(b, M_b, (cols, rows))

        return cv2.merge((b_shifted, g_shifted, r_shifted))
    # end apply
# end ChromaticSpatialShiftEffect


class ChromaticTemporalPersistenceEffect(EffectBase):
    """
    Temporal persistence effect that blends the current frame with previous frames
    """

    def __init__(
            self,
            persistence_r=5,
            persistence_g=5,
            persistence_b=5
    ):
        """
        Initialize the temporal persistence effect with the number of frames to blend

        Args:
            persistence_r (int): Number of frames to blend for the red channel
            persistence_g (int): Number of frames to blend for the green channel
            persistence_b (int): Number of frames to blend for the blue channel
        """
        self.persistence_r = persistence_r
        self.persistence_g = persistence_g
        self.persistence_b = persistence_b
        self.prev_frames_r = []
        self.prev_frames_g = []
        self.prev_frames_b = []
    # end __init__

    def apply(
            self,
            image,
            **kwargs
    ):
        """
        Apply the temporal persistence effect to the image

        Args:
            image (np.ndarray): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        b, g, r = cv2.split(image)

        # Add current frame to the list of previous frames
        self.prev_frames_r.append(r)
        self.prev_frames_g.append(g)
        self.prev_frames_b.append(b)

        if len(self.prev_frames_r) > self.persistence_r + 1:
            self.prev_frames_r.pop(0)
        # end if

        if len(self.prev_frames_g) > self.persistence_g + 1:
            self.prev_frames_g.pop(0)
        # end if

        if len(self.prev_frames_b) > self.persistence_b + 1:
            self.prev_frames_b.pop(0)
        # end if

        # Calcul weighted average of previous frames
        blended_r = np.zeros_like(r, dtype=np.float32)
        blended_g = np.zeros_like(g, dtype=np.float32)
        blended_b = np.zeros_like(b, dtype=np.float32)

        weight_r = 1.0 / len(self.prev_frames_r) if len(self.prev_frames_r) > 0 else 0
        weight_g = 1.0 / len(self.prev_frames_g) if len(self.prev_frames_g) > 0 else 0
        weight_b = 1.0 / len(self.prev_frames_b) if len(self.prev_frames_b) > 0 else 0

        for frame in self.prev_frames_r:
            blended_r += frame * weight_r
            weight_r *= 0.9
        # end for

        for frame in self.prev_frames_g:
            blended_g += frame * weight_g
            weight_g *= 0.9
        # end for

        for frame in self.prev_frames_b:
            blended_b += frame * weight_b
            weight_b *= 0.9
        # end for

        # Merge channels and return the blended image
        blended = cv2.merge((
            blended_b.astype(np.uint8),
            blended_g.astype(np.uint8),
            blended_r.astype(np.uint8)
        ))

        return blended
    # end apply

# end ChromaticTemporalPersistenceEffect

