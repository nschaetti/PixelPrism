

from pixel_prism.utils import parse_svg

# Parse the SVG file
paths = parse_svg('latex.svg')

for path in paths:
    print(path)
# end for
