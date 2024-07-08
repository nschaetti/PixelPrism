import numpy as np
import cv2


from pixel_prism.effects.effect_base import EffectBase


def load_cube_lut(
        file_path
):
    """
    Load a 3D LUT from a .cube file

    Args:
        file_path (str): Path to the .cube file
    """
    # Load the LUT file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # end with

    title = ''
    size = 0
    lut = []

    # Parse the LUT file
    for line in lines:
        if line.startswith('#'):
            continue
        # end if

        if 'TITLE' in line:
            title = line.split()[1]
        # end if

        if 'LUT_3D_SIZE' in line:
            size = int(line.split()[1])
        # end if

        if len(line.split()) == 3:
            lut.append([float(x) for x in line.split()])
        # end if
    # end for

    # Convert the LUT to a numpy array
    lut = np.array(lut)
    lut = lut.reshape((size, size, size, 3))

    return lut, size
# end load_cube_lut


class LUTEffect(EffectBase):

    def __init__(self, lut_path):
        self.lut, self.size = load_cube_lut(lut_path)
    # end __init__

    def apply(self, image, **kwargs):
        # Normaliser l'image
        image = image / 255.0
        lut = self.lut

        # Appliquer la LUT
        index = (image * (self.size - 1)).astype(int)
        result = lut[index[:, :, 0], index[:, :, 1], index[:, :, 2]]

        # Remettre l'image à l'échelle de 0 à 255
        result = (result * 255).astype(np.uint8)
        return result
    # end apply

# end LUTEffect

