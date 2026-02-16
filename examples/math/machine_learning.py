# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####

from __future__ import annotations

import numpy as np

import pixelprism.math as pm


def main() -> None:
    try:
        # Perceptron
        x = pm.var("ml_x", dtype=pm.DType.R, shape=(4, 2))
        y_bin = pm.var("ml_y_bin", dtype=pm.DType.Z, shape=(4,))

        perceptron = pm.Perceptron(max_iter=200, learning_rate=1.0, shuffle=False)
        perceptron.fit(x, y_bin)

        # Decision tree multiclass
        x_tree = pm.var("ml_x_tree", dtype=pm.DType.R, shape=(6, 2))
        y_tree = pm.var("ml_y_tree", dtype=pm.DType.Z, shape=(6,))
        tree = pm.DecisionTreeClassifier(max_depth=3, criterion="entropy")
        tree.fit(x_tree, y_tree)

        # Linear SVC
        svc = pm.SVC(c=1.0, max_iter=2000)
        svc.fit(x, y_bin)

        with pm.new_context():
            x_bin = np.array([[2.0, 2.0], [1.0, 1.0], [-1.0, -1.0], [-2.0, -2.0]], dtype=np.float32)
            yb = np.array([1, 1, -1, -1], dtype=np.int64)
            pm.set_value("ml_x", x_bin)
            pm.set_value("ml_y_bin", yb)

            x_mul = np.array([[-2.0, -2.0], [-1.0, -1.0], [0.0, 0.0], [0.5, 0.5], [2.0, 2.0], [2.5, 2.5]], dtype=np.float32)
            ym = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
            pm.set_value("ml_x_tree", x_mul)
            pm.set_value("ml_y_tree", ym)

            print("Perceptron predict:", perceptron.predict(x).eval().value)
            print("Perceptron theta:", perceptron.theta_.eval().value)
            print("Tree predict:", tree.predict(x_tree).eval().value)
            print("Tree classes:", tree.classes_.eval().value)
            print("SVC scores:", svc.decision_function(x).eval().value)
            print("SVC predict:", svc.predict(x).eval().value)
    except ImportError as exc:
        print("Machine learning examples require scikit-learn:", exc)
    # end try
# end def main


if __name__ == "__main__":
    main()
