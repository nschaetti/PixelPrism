#
# Description: Module to apply color effects to images
#


# Imports
import numpy as np


def apply_lut(
        image,
        lut,
        size
):
    """
    Apply a 3D LUT to an image

    Args:
        image (np.ndarray): Image to apply the LUT to
        lut (np.ndarray): 3D LUT to apply
        size (int): Size of the LUT
    """
    # Normaliser l'image
    image = image / 255.0

    # Appliquer la LUT
    index = (image * (size - 1)).astype(int)
    result = lut[index[:, :, 0], index[:, :, 1], index[:, :, 2]]

    # Remettre l'image à l'échelle de 0 à 255
    return (result * 255).astype(np.uint8)
# end apply_lut

