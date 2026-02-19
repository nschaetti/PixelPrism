import numpy as np
import pytest

import pixelprism.math as pm
import pixelprism.math.functional.machine_learning as ML


pytest.importorskip("sklearn")


def test_perceptron_operators_fit_predict_boundary():
    x_np = np.array(
        [
            [2.0, 2.0],
            [1.5, 1.0],
            [-1.5, -1.0],
            [-2.0, -2.0],
        ],
        dtype=np.float32,
    )
    y_np = np.array([1, 1, -1, -1], dtype=np.int64)

    x = pm.const("perc_x", data=x_np, dtype=pm.DType.R)
    y = pm.const("perc_y", data=y_np, dtype=pm.DType.Z)

    theta = ML.fit(x, y, max_iter=100, learning_rate=1.0, shuffle=False)
    pred = ML.predict(x, theta)
    boundary = ML.decision_boundary(theta)

    pred_val = pred.eval().value
    theta_val = theta.eval().value
    boundary_val = boundary.eval().value
    np.testing.assert_array_equal(pred_val, y_np)
    np.testing.assert_allclose(boundary_val, theta_val, rtol=1e-7, atol=1e-7)
# end test_perceptron_operators_fit_predict_boundary


def test_perceptron_operators_recompute_when_context_changes():
    x = pm.var("perc_var_x", dtype=pm.DType.R, shape=(4, 2))
    y = pm.var("perc_var_y", dtype=pm.DType.Z, shape=(4,))
    theta = ML.fit(x, y, max_iter=50, learning_rate=1.0, shuffle=False)

    with pm.new_context():
        pm.set_value("perc_var_x", np.array([[2.0, 2.0], [1.0, 1.0], [-1.0, -1.0], [-2.0, -2.0]], dtype=np.float32))
        pm.set_value("perc_var_y", np.array([1, 1, -1, -1], dtype=np.int64))
        theta_a = theta.eval().value
    # end with

    with pm.new_context():
        pm.set_value("perc_var_x", np.array([[2.0, -2.0], [1.0, -1.0], [-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32))
        pm.set_value("perc_var_y", np.array([-1, -1, 1, 1], dtype=np.int64))
        theta_b = theta.eval().value
    # end with

    assert theta_a.shape == theta_b.shape == (3,)
    assert not np.allclose(theta_a, theta_b)
# end test_perceptron_operators_recompute_when_context_changes


def test_decision_tree_multiclass_gini_and_entropy():
    x_np = np.array(
        [
            [-2.0, -2.0],
            [-1.5, -1.0],
            [0.2, 0.0],
            [0.5, 0.4],
            [2.0, 2.0],
            [2.5, 1.5],
        ],
        dtype=np.float32,
    )
    y_np = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    x = pm.const("tree_x", data=x_np, dtype=pm.DType.R)
    y = pm.const("tree_y", data=y_np, dtype=pm.DType.Z)

    tree_gini = ML.tree_fit(x, y, max_depth=3, criterion="gini")
    pred_gini = ML.tree_predict(x, tree_gini)
    np.testing.assert_array_equal(pred_gini.eval().value, y_np)

    tree_entropy = ML.tree_fit(x, y, max_depth=3, criterion="entropy")
    pred_entropy = ML.tree_predict(x, tree_entropy)
    np.testing.assert_array_equal(pred_entropy.eval().value, y_np)

    classes = ML.tree_classes(y).eval().value
    np.testing.assert_array_equal(classes, np.array([0, 1, 2], dtype=np.int64))
# end test_decision_tree_multiclass_gini_and_entropy


def test_decision_tree_recompute_when_context_changes():
    x = pm.var("tree_var_x", dtype=pm.DType.R, shape=(6, 2))
    y = pm.var("tree_var_y", dtype=pm.DType.Z, shape=(6,))

    tree = ML.tree_fit(x, y, max_depth=2, criterion="gini")

    with pm.new_context():
        pm.set_value(
            "tree_var_x",
            np.array([[-2.0, -2.0], [-1.0, -1.0], [0.0, 0.0], [0.5, 0.5], [2.0, 2.0], [2.5, 2.5]], dtype=np.float32),
        )
        pm.set_value("tree_var_y", np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
        tree_a = tree.eval().value
    # end with

    with pm.new_context():
        pm.set_value(
            "tree_var_x",
            np.array([[-2.0, 2.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], dtype=np.float32),
        )
        pm.set_value("tree_var_y", np.array([2, 2, 1, 1, 0, 0], dtype=np.int64))
        tree_b = tree.eval().value
    # end with

    assert tree_a.shape == tree_b.shape
    assert not np.allclose(tree_a, tree_b)
# end test_decision_tree_recompute_when_context_changes


def test_svm_linear_fit_predict_and_decision_function():
    x_np = np.array(
        [
            [2.0, 2.0],
            [1.5, 1.0],
            [-1.5, -1.0],
            [-2.0, -2.0],
        ],
        dtype=np.float32,
    )
    y_np = np.array([1, 1, -1, -1], dtype=np.int64)
    x = pm.const("svm_x", data=x_np, dtype=pm.DType.R)
    y = pm.const("svm_y", data=y_np, dtype=pm.DType.Z)

    theta = ML.svm_fit(x, y, c=1.0, max_iter=2000)
    pred = ML.svm_predict(x, theta)
    score = ML.svm_decision_function(x, theta)

    pred_val = pred.eval().value
    score_val = score.eval().value
    np.testing.assert_array_equal(pred_val, y_np)
    assert score_val.shape == (4,)
    assert np.all(np.sign(score_val) == y_np)
# end test_svm_linear_fit_predict_and_decision_function
