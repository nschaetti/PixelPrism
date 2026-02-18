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
# Copyright (C) 2026 Pixel Prism

import numpy as np
import numpy.testing as npt
import pytest

import pixelprism.math as pm
from pixelprism.math.geometry import Point2, Line2, Affine2


def test_point2_vars_builds_named_scalar_variables():
    p = Point2.vars("p")

    assert p.x.name == "p.x"
    assert p.y.name == "p.y"
    assert p.x.shape.dims == ()
    assert p.y.shape.dims == ()
# end def test_point2_vars_builds_named_scalar_variables


def test_line2_vars_builds_named_endpoint_variables():
    line = Line2.vars("line")

    assert line.start.x.name == "line.start.x"
    assert line.start.y.name == "line.start.y"
    assert line.end.x.name == "line.end.x"
    assert line.end.y.name == "line.end.y"
# end def test_line2_vars_builds_named_endpoint_variables


def test_point2_rejects_non_scalar_coordinate_expressions():
    vector_expr = pm.var("v", dtype=pm.DType.R, shape=(2,))

    with pytest.raises(TypeError):
        Point2(vector_expr, 0.0)
    # end with
# end def test_point2_rejects_non_scalar_coordinate_expressions


def test_line_point_at_half_returns_midpoint():
    line = Line2(Point2(0.0, 0.0), Point2(2.0, 4.0))
    mid = line.point_at(0.5)

    npt.assert_allclose(mid.eval().value, np.array([1.0, 2.0]), rtol=0.0, atol=1e-9)
# end def test_line_point_at_half_returns_midpoint


def test_line_length_matches_3_4_5_triangle():
    line = Line2(Point2(0.0, 0.0), Point2(3.0, 4.0))

    npt.assert_allclose(line.length().eval().value, np.array(5.0), rtol=0.0, atol=1e-9)
# end def test_line_length_matches_3_4_5_triangle


def test_constant_point_rotation_around_center():
    p = Point2(3.0, 2.0)
    center = Point2(2.0, 2.0)
    q = p.rotated(np.pi / 2.0, center=center)

    npt.assert_allclose(q.eval().value, np.array([2.0, 3.0]), rtol=0.0, atol=1e-8)
# end def test_constant_point_rotation_around_center


def test_variable_point_and_angle_rotation():
    p = Point2.vars("p")
    center = Point2(2.0, 2.0)
    theta = pm.var("theta", dtype=pm.DType.R, shape=())
    q = p.rotated(theta, center=center)

    with pm.new_context():
        pm.set_value("p.x", 3.0)
        pm.set_value("p.y", 2.0)
        pm.set_value("theta", np.pi / 2.0)
        npt.assert_allclose(q.eval().value, np.array([2.0, 3.0]), rtol=0.0, atol=1e-8)
    # end with
# end def test_variable_point_and_angle_rotation


def test_affine_translation_applies_to_point():
    transform = Affine2.translation(10.0, -3.0)
    p = Point2(1.5, 2.5)
    q = transform.apply_point(p)

    npt.assert_allclose(q.eval().value, np.array([11.5, -0.5]), rtol=0.0, atol=1e-9)
# end def test_affine_translation_applies_to_point


def test_affine_compose_applies_other_then_self():
    rotation = Affine2.rotation(np.pi / 2.0)
    translation = Affine2.translation(10.0, 0.0)
    composed = translation.compose(rotation)

    p = Point2(1.0, 0.0)
    q = composed.apply_point(p)

    npt.assert_allclose(q.eval().value, np.array([10.0, 1.0]), rtol=0.0, atol=1e-8)
# end def test_affine_compose_applies_other_then_self


def test_affine_inverse_round_trip_point():
    transform = Affine2.translation(2.0, -5.0).compose(Affine2.scale(3.0, 4.0))
    inv = transform.inverse()
    p = Point2(1.25, -0.5)

    back = inv.apply_point(transform.apply_point(p))
    npt.assert_allclose(back.eval().value, p.eval().value, rtol=0.0, atol=1e-8)
# end def test_affine_inverse_round_trip_point


def test_line_transformed_transforms_both_endpoints():
    line = Line2(Point2(0.0, 0.0), Point2(2.0, 0.0))
    transform = Affine2.translation(1.0, 2.0)
    out = line.transformed(transform)

    npt.assert_allclose(out.start.eval().value, np.array([1.0, 2.0]), rtol=0.0, atol=1e-9)
    npt.assert_allclose(out.end.eval().value, np.array([3.0, 2.0]), rtol=0.0, atol=1e-9)
# end def test_line_transformed_transforms_both_endpoints


def test_affine_matrix_shapes():
    transform = Affine2.identity()

    assert transform.matrix2x3().shape.dims == (2, 3)
    assert transform.matrix3().shape.dims == (3, 3)
# end def test_affine_matrix_shapes
