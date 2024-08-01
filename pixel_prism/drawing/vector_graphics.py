

class VectorGraphics:
    def __init__(self):
        self._lines = []

    def add_line(self, line):
        self._lines.append(line)

    def draw(self):
        for line in self._lines:
            line.draw()
    # end draw

# end VectorGraphics

