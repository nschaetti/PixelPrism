
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
    subprocess.run(["pdflatex", "-output-format=dvi", "-output-directory", temp_dir, latex_file_path])
    dvi_path = os.path.join(temp_dir, "equation.dvi")
    subprocess.run(["dvisvgm", dvi_path, "-o", output_path, "--no-fonts"])

    # Clean up temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    # end for
    os.rmdir(temp_dir)
# end render_latex_to_svg

