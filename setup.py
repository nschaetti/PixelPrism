# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Imports
from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pixelprism",
    version="0.1.0",
    author="Nils Schaetti",
    author_email="nils.schaetti@gmail.com",
    description="A Python library that applies advanced visual effects to videos and images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nschaetti/PixelPrism",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'pixelprism=pixelprism:main',
        ],
    },
    install_requires=[
        "numpy",
        "tqdm",
        "pillow",
        "requests",
        "svgpathtools",
        "lxml",
        "colorama",
        "matplotlib",
        "opencv-python",
        "pycairo",
        "torch",
        "scikit-image",
        "click",
    ],
)
