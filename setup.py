
# Imports
from setuptools import setup, find_packages

setup(
    name="pixelprism",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pixelprism=pixel_prism.pixel_prism:main',
        ],
    },
    install_requires=[
        "opencv-python",
        "matplotlib",
        "tqdm",
    ],
)
