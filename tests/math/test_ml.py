import numpy as np
import pytest

import pixelprism.math as pm


pytest.importorskip("sklearn")


def test_perceptron_sklearn_like_api_and_attrs():
    x = pm.var("ml_x", dtype=pm.DType.R, shape=(4, 2))
    y = pm.var("ml_y", dtype=pm.DType.Z, shape=(4,))

    clf = pm.Perceptron(max_iter=80, learning_rate=1.0, shuffle=False)
    clf.fit(x, y)

    pred_expr = clf.predict(x)
    theta_expr = clf.theta_
    coef_expr = clf.coef_
    intercept_expr = clf.intercept_
    classes_expr = clf.classes_
    boundary_expr = clf.decision_boundary()

    with pm.new_context():
        x_np = np.array(
            [
                [2.0, 2.0],
                [1.0, 1.0],
                [-1.0, -1.0],
                [-2.0, -2.0],
            ],
            dtype=np.float32,
        )
        y_np = np.array([1, 1, -1, -1], dtype=np.int64)
        pm.set_value("ml_x", x_np)
        pm.set_value("ml_y", y_np)

        pred_val = pred_expr.eval().value
        theta_val = theta_expr.eval().value
        coef_val = coef_expr.eval().value
        intercept_val = intercept_expr.eval().value
        classes_val = classes_expr.eval().value
        boundary_val = boundary_expr.eval().value
    # end with

    np.testing.assert_array_equal(pred_val, y_np)
    assert theta_val.shape == (3,)
    assert coef_val.shape == (2,)
    assert np.asarray(intercept_val).shape == ()
    np.testing.assert_array_equal(classes_val, np.array([-1, 1], dtype=np.int64))
    np.testing.assert_allclose(boundary_val, theta_val, rtol=1e-7, atol=1e-7)
# end test_perceptron_sklearn_like_api_and_attrs


def test_decision_tree_classifier_api_multiclass():
    x = pm.var("ml_tree_x", dtype=pm.DType.R, shape=(6, 2))
    y = pm.var("ml_tree_y", dtype=pm.DType.Z, shape=(6,))

    clf = pm.DecisionTreeClassifier(max_depth=3, criterion="entropy")
    clf.fit(x, y)

    pred_expr = clf.predict(x)
    tree_expr = clf.tree_
    classes_expr = clf.classes_

    with pm.new_context():
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
        pm.set_value("ml_tree_x", x_np)
        pm.set_value("ml_tree_y", y_np)

        pred_val = pred_expr.eval().value
        tree_val = tree_expr.eval().value
        classes_val = classes_expr.eval().value
    # end with

    np.testing.assert_array_equal(pred_val, y_np)
    assert tree_val.ndim == 2 and tree_val.shape[1] == 6
    np.testing.assert_array_equal(classes_val, np.array([0, 1, 2], dtype=np.int64))
# end test_decision_tree_classifier_api_multiclass


def test_svc_api_binary():
    x = pm.var("ml_svm_x", dtype=pm.DType.R, shape=(4, 2))
    y = pm.var("ml_svm_y", dtype=pm.DType.Z, shape=(4,))

    clf = pm.SVC(c=1.0, max_iter=2000)
    clf.fit(x, y)

    pred_expr = clf.predict(x)
    score_expr = clf.decision_function(x)
    theta_expr = clf.theta_
    coef_expr = clf.coef_
    intercept_expr = clf.intercept_
    classes_expr = clf.classes_

    with pm.new_context():
        x_np = np.array([[2.0, 2.0], [1.0, 1.0], [-1.0, -1.0], [-2.0, -2.0]], dtype=np.float32)
        y_np = np.array([1, 1, -1, -1], dtype=np.int64)
        pm.set_value("ml_svm_x", x_np)
        pm.set_value("ml_svm_y", y_np)

        pred_val = pred_expr.eval().value
        score_val = score_expr.eval().value
        theta_val = theta_expr.eval().value
        coef_val = coef_expr.eval().value
        intercept_val = intercept_expr.eval().value
        classes_val = classes_expr.eval().value
    # end with

    np.testing.assert_array_equal(pred_val, y_np)
    assert score_val.shape == (4,)
    assert theta_val.shape == (3,)
    assert coef_val.shape == (2,)
    assert np.asarray(intercept_val).shape == ()
    np.testing.assert_array_equal(classes_val, np.array([-1, 1], dtype=np.int64))
# end test_svc_api_binary
