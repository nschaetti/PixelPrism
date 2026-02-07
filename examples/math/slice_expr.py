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
from pixelprism.math.math_expr import SliceExpr


def main():
    # Build a slice expression using integers
    seq_slice = SliceExpr.create(start=1, stop=7, step=2)
    print("Slice expression:", seq_slice)
    print("Native slice:", seq_slice.as_slice)

    # Build a slice expression from constants
    start = pm.const("slice_start", data=0, dtype=pm.DType.Z)
    stop = pm.const("slice_stop", data=4, dtype=pm.DType.Z)
    const_slice = SliceExpr.create(start=start, stop=stop)
    print("Constant-backed slice:", const_slice)
    print("Slice values:", const_slice.start_value, const_slice.stop_value, const_slice.step_value)

    # Convert a Python slice to SliceExpr
    py_slice = slice(None, 5, None)
    expr_from_slice = SliceExpr.from_slice(py_slice)
    print("From python slice:", expr_from_slice)


if __name__ == "__main__":
    main()
