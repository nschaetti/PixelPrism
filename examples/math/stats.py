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

from __future__ import annotations

import numpy as np

import pixelprism.math as pm
import pixelprism.math.functional.stats as S


def main() -> None:
    x = pm.var("stats_x", dtype=pm.DType.R, shape=(5, 2))
    ddof = pm.var("stats_ddof", dtype=pm.DType.Z, shape=())
    loc = pm.var("stats_loc", dtype=pm.DType.R, shape=())
    scale = pm.var("stats_scale", dtype=pm.DType.R, shape=())

    z = S.zscore(x, axis=0, ddof=ddof)
    cov = S.cov(x, rowvar=False, ddof=ddof)
    corr = S.corr(x, rowvar=False)
    normal_sample = S.normal(shape=(3, 3), loc=loc, scale=scale)

    with pm.new_context():
        pm.set_value("stats_x", np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 6.0], [4.0, 8.0], [5.0, 9.0]], dtype=np.float32))
        pm.set_value("stats_ddof", 1)
        pm.set_value("stats_loc", 2.0)
        pm.set_value("stats_scale", 0.5)

        print("zscore:\n", z.eval().value)
        print("cov:\n", cov.eval().value)
        print("corr:\n", corr.eval().value)
        print("normal sample:\n", normal_sample.eval().value)
    # end with
# end def main


if __name__ == "__main__":
    main()
