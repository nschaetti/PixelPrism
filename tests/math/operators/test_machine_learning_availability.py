from pixelprism.math.operators import operator_registry
from pixelprism.math.operators.machine_learning import SKLEARN_AVAILABLE


def test_basic_ml_operator_registration_depends_on_sklearn():
    names = [
        "perceptron_train",
        "perceptron_predict",
        "decision_tree_train",
        "decision_tree_predict",
        "svm_train",
        "svm_predict",
    ]
    for name in names:
        assert operator_registry.has(name) is SKLEARN_AVAILABLE
    # end for
# end test_basic_ml_operator_registration_depends_on_sklearn
