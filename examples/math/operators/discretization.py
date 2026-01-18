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
import pixelprism.math.functional as D
import pixelprism.math.render as render


# Variables
x = pm.var("x", dtype=pm.DType.FLOAT32, shape=())
y = pm.var("y", dtype=pm.DType.FLOAT32, shape=())

# Math equations
z1 = D.ceil(x + y)
z2 = D.floor(x + y)
z3 = D.round(x + y)
z4 = D.trunc(x + y)
z5 = D.sign(x + y)
z6 = D.rint(x + y)
z7 = D.clip(x + y, 0.0, 1.0)


# Show latex
print(f"Latex of z1: {render.to_latex(z1)}")
print(f"Latex of z2: {render.to_latex(z2)}")
print(f"Latex of z3: {render.to_latex(z3)}")
print(f"Latex of z4: {render.to_latex(z4)}")
print(f"Latex of z5: {render.to_latex(z5)}")
print(f"Latex of z6: {render.to_latex(z6)}")
print(f"Latex of z7: {render.to_latex(z7)}")

# Set value and evaluate
with pm.new_context():
    pm.set_value("x", 2.4)
    pm.set_value("y", 2.2)
    print(f"Equation evaluation z1: {z1.eval()}")
    print(f"Equation evaluation z2: {z2.eval()}")
    print(f"Equation evaluation z3: {z3.eval()}")
    print(f"Equation evaluation z4: {z4.eval()}")
    print(f"Equation evaluation z5: {z5.eval()}")
    print(f"Equation evaluation z6: {z6.eval()}")
    print(f"Equation evaluation z7: {z7.eval()}")
# end with
