
import pixelprism.math as pm
import pixelprism.math.functional as F


def test_expression_01():
    x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
    y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))
    n = pm.var("n", dtype=pm.DType.Z, shape=())

    my_eq = F.pow(x + (1 + y) - 12.0, n)

    with pm.new_context():
        pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
        pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
        pm.set_value("n", pm.tensor(2, dtype=pm.DType.Z))
        assert my_eq.eval() == pm.tensor([[25.0, 9.0], [1.0, 1.0]], dtype=pm.DType.R)
    # end with
# end def test_expression_01


def test_expression_02():
    x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
    y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))
    n = pm.var("n", dtype=pm.DType.Z, shape=())

    my_eq = F.cbrt(F.log10(x) + (1 + y) - 12.0)

    with pm.new_context():
        pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
        pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
        pm.set_value("n", pm.tensor(2, dtype=pm.DType.Z))
        assert my_eq.eval() == pm.tensor([[-1.8171205928321394, -1.674946310302838], [-1.5215955610977152, -1.3384827223871292]], dtype=pm.DType.R)
    # end with
# end def test_expression_02


def test_expression_03():
    x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
    y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))

    my_eq = F.floor(F.sin(x) + F.cos(y))

    with pm.new_context():
        pm.set_value("x", pm.tensor([[1, 2], [3, 4]], dtype=pm.DType.R))
        pm.set_value("y", pm.tensor([[5, 6], [7, 8]], dtype=pm.DType.R))
        assert my_eq.eval() == pm.tensor(
            [[1.0, 1.0], [0.0, -1.0]],
            dtype=pm.DType.R
        )
    # end with
# end def test_expression_03

