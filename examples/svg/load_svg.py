

from pixel_prism.drawing import VectorGraphics


# Create a vector graphic from svg
icon = VectorGraphics.from_svg('examples/svg/latex2.svg')

# Print the vector graphics
print(icon)

