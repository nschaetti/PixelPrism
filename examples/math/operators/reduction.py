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
# Copyright (C) 2025 Pixel Prism
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
#

import pixelprism.math as pm
import pixelprism.math.functional as R
import pixelprism.math.render as render


# Variables
x = pm.var("x", dtype=pm.DType.FLOAT32, shape=(2, 2))
y = pm.var("y", dtype=pm.DType.FLOAT32, shape=(2, 2))

# Math equations
z1 = R.sum(x, axis=0)
z2 = R.sum(y, axis=1)
z3 = z1 + z2

# Show latex
print(f"Latex of z1: {render.to_latex(z1)}")
print(f"Latex of z2: {render.to_latex(z2)}")
print(f"Latex of z3: {render.to_latex(z3)}")

# Set value and evaluate
with pm.new_context():
    pm.set_value("x", [[1.0, 2.0], [3.0, 4.0]])
    pm.set_value("y", [[5.0, 6.0], [7.0, 8.0]])
    print(f"Equation evaluation z1: {z1.eval()}")
    print(f"Equation evaluation z2: {z2.eval()}")
    print(f"Equation evaluation z3: {z3.eval()}")
# end with

