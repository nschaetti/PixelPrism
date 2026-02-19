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
import subprocess
import os
import cairo
from lxml import etree


def render_latex_to_svg(
        latex,
        output_path: str = 'latex.svg'
):
    """
    Render a LaTeX equation to an SVG file.

    Args:
        latex (str): LaTeX equation to render
        output_path (str): Path to save the SVG file
    """
    # LaTeX document template
    latex_template = r"""
    \documentclass[preview,border=1pt]{standalone}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    \usepackage{graphicx}
    \begin{document}
    $$
    {latex}
    $$
    \end{document}
    """

    # Create temporary directory for LaTeX rendering
    temp_dir = "tmp_latex"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # end if

    # Create LaTeX file
    latex_content = latex_template.replace("{latex}", latex)
    latex_file_path = os.path.join(temp_dir, "equation.tex")
    with open(latex_file_path, "w") as f:
        f.write(latex_content)
    # end with

    # Run pdflatex to generate DVI
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-format=dvi",
            "-output-directory",
            temp_dir,
            latex_file_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dvi_path = os.path.join(temp_dir, "equation.dvi")
    subprocess.run(
        ["dvisvgm", dvi_path, "-o", output_path, "--no-fonts"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Clean up temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    # end for
    os.rmdir(temp_dir)
# end render_latex_to_svg
