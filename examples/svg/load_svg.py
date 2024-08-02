

from pixel_prism.data import VectorGraphicsData


# Create a vector graphic from svg
icon = VectorGraphicsData.from_svg('examples/svg/latex2.svg')

# Print the vector graphics
print(icon)

