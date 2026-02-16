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
"""Machine-learning operators (perceptron and helpers)."""

from __future__ import annotations

from abc import ABC
from typing import Optional, Sequence, Union

import numpy as np

try:
    from sklearn.linear_model import Perceptron as SklearnPerceptron
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
    from sklearn.svm import LinearSVC as SklearnLinearSVC
    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc

from ..build import as_expr
from ..dtype import DType, to_numpy
from ..math_node import MathNode
from ..shape import Shape
from ..tensor import Tensor
from ..typing import MathExpr
from .algorithmic import register_algorithm, get_algorithm
from .base import Operands, OperatorBase, ParametricOperator, operator_registry


ScalarParameter = Union[MathExpr, int, float]


__all__ = [
    "MachineLearningOperator",
    "PerceptronTrain",
    "PerceptronPredict",
    "PerceptronDecisionBoundary",
    "PerceptronCoefficients",
    "PerceptronIntercept",
    "PerceptronClasses",
    "DecisionTreeTrain",
    "DecisionTreePredict",
    "DecisionTreeClasses",
    "SVMTrain",
    "SVMPredict",
    "SVMDecisionFunction",
    "SVMClasses",
    "SKLEARN_AVAILABLE",
]


def _ensure_scalar_parameter(value: ScalarParameter, name: str) -> MathExpr:
    expr = as_expr(value)
    if expr.rank != 0:
        raise ValueError(f"{name} must be a scalar expression.")
    # end if
    return expr
# end def _ensure_scalar_parameter


def _eval_scalar_parameter(expr: MathExpr, name: str) -> float:
    value = np.asarray(expr.eval().value)
    if value.shape != ():
        raise ValueError(f"{name} must evaluate to a scalar.")
    # end if
    return float(value.item())
# end def _eval_scalar_parameter


def _eval_int_parameter(expr: MathExpr, name: str) -> int:
    value = _eval_scalar_parameter(expr, name)
    if not float(value).is_integer():
        raise ValueError(f"{name} must evaluate to an integer.")
    # end if
    return int(value)
# end def _eval_int_parameter


def _flatten_targets(y: np.ndarray, name: str) -> np.ndarray:
    if y.ndim == 1:
        return y
    # end if
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    # end if
    raise ValueError(f"{name} expects y with shape (n,) or (n, 1), got {y.shape}")
# end def _flatten_targets


def _algo_perceptron_train(
        x: np.ndarray,
        y: np.ndarray,
        *,
        max_iter: int,
        learning_rate: float,
        shuffle: bool,
        seed: Optional[int],
        fit_intercept: bool,
) -> np.ndarray:
    _require_sklearn()
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = _flatten_targets(np.asarray(y, dtype=np.float64), "perceptron_train")

    if x_arr.ndim != 2:
        raise ValueError(f"perceptron_train expects x rank-2, got shape={x_arr.shape}")
    # end if
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(f"perceptron_train sample mismatch: {x_arr.shape[0]} != {y_arr.shape[0]}")
    # end if

    unique = set(np.unique(y_arr).tolist())
    if unique != {-1.0, 1.0}:
        raise ValueError(f"perceptron_train expects binary labels in {{-1, +1}}, got {sorted(unique)}")
    # end if

    if max_iter <= 0:
        raise ValueError("perceptron_train requires max_iter > 0")
    # end if
    if learning_rate <= 0:
        raise ValueError("perceptron_train requires learning_rate > 0")
    # end if

    model = SklearnPerceptron(
        max_iter=max_iter,
        tol=None,
        eta0=learning_rate,
        fit_intercept=fit_intercept,
        shuffle=shuffle,
        random_state=seed,
    )
    model.fit(x_arr, y_arr)
    w = np.asarray(model.coef_[0], dtype=np.float64)
    b = float(model.intercept_[0]) if fit_intercept else 0.0
    return np.concatenate([w, np.asarray([b], dtype=np.float64)])
# end def _algo_perceptron_train


def _algo_perceptron_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError(f"perceptron_predict expects x rank-2, got shape={x_arr.shape}")
    # end if
    if theta_arr.shape[0] != x_arr.shape[1] + 1:
        raise ValueError(
            f"perceptron_predict expects theta shape ({x_arr.shape[1] + 1},), got {theta_arr.shape}"
        )
    # end if
    w = theta_arr[:-1]
    b = theta_arr[-1]
    scores = x_arr @ w + b
    y_pred = np.where(scores >= 0.0, 1, -1)
    return y_pred.astype(np.int64)
# end def _algo_perceptron_predict


def _impurity(y: np.ndarray, criterion: str) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    if criterion == "gini":
        return float(1.0 - np.sum(probs ** 2))
    # end if
    safe = probs[probs > 0.0]
    return float(-np.sum(safe * np.log2(safe)))
# end def _impurity


def _majority_label(y: np.ndarray) -> int:
    labels, counts = np.unique(y, return_counts=True)
    return int(labels[int(np.argmax(counts))])
# end def _majority_label


def _algo_decision_tree_train(
        x: np.ndarray,
        y: np.ndarray,
        *,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        criterion: str,
) -> np.ndarray:
    _require_sklearn()
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = _flatten_targets(np.asarray(y, dtype=np.float64), "decision_tree_train")

    if x_arr.ndim != 2:
        raise ValueError(f"decision_tree_train expects x rank-2, got shape={x_arr.shape}")
    # end if
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(f"decision_tree_train sample mismatch: {x_arr.shape[0]} != {y_arr.shape[0]}")
    # end if
    if not np.allclose(y_arr, np.round(y_arr)):
        raise ValueError("decision_tree_train expects integer class labels")
    # end if

    y_int = np.asarray(np.round(y_arr), dtype=np.int64)
    n_samples, n_features = x_arr.shape
    if max_depth < 1:
        raise ValueError("decision_tree_train requires max_depth >= 1")
    # end if
    if min_samples_split < 2:
        raise ValueError("decision_tree_train requires min_samples_split >= 2")
    # end if
    if min_samples_leaf < 1:
        raise ValueError("decision_tree_train requires min_samples_leaf >= 1")
    # end if
    if criterion not in {"gini", "entropy"}:
        raise ValueError("decision_tree_train criterion must be 'gini' or 'entropy'")
    # end if

    model = SklearnDecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    )
    model.fit(x_arr, y_int)

    max_nodes = (2 ** (max_depth + 1)) - 1
    tree = np.full((max_nodes, 6), -1.0, dtype=np.float64)

    sk_tree = model.tree_
    n_nodes = int(sk_tree.node_count)
    for node_idx in range(min(max_nodes, n_nodes)):
        feature = int(sk_tree.feature[node_idx])
        threshold = float(sk_tree.threshold[node_idx]) if feature >= 0 else 0.0
        left = int(sk_tree.children_left[node_idx])
        right = int(sk_tree.children_right[node_idx])
        values = sk_tree.value[node_idx][0]
        label = int(model.classes_[int(np.argmax(values))])
        is_leaf = 1.0 if left == right else 0.0

        tree[node_idx, 0] = float(feature)
        tree[node_idx, 1] = threshold
        tree[node_idx, 2] = float(left if left >= 0 else -1)
        tree[node_idx, 3] = float(right if right >= 0 else -1)
        tree[node_idx, 4] = float(label)
        tree[node_idx, 5] = is_leaf
    # end for

    return tree
# end def _algo_decision_tree_train


def _algo_decision_tree_predict(x: np.ndarray, tree: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    tree_arr = np.asarray(tree, dtype=np.float64)

    if x_arr.ndim != 2:
        raise ValueError(f"decision_tree_predict expects x rank-2, got shape={x_arr.shape}")
    # end if
    if tree_arr.ndim != 2 or tree_arr.shape[1] != 6:
        raise ValueError(f"decision_tree_predict expects tree shape (n_nodes, 6), got {tree_arr.shape}")
    # end if

    n_samples = x_arr.shape[0]
    n_nodes = tree_arr.shape[0]
    out = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        node = 0
        while True:
            if node < 0 or node >= n_nodes:
                out[i] = int(np.round(tree_arr[0, 4]))
                break
            # end if
            feature = int(np.round(tree_arr[node, 0]))
            threshold = float(tree_arr[node, 1])
            left_child = int(np.round(tree_arr[node, 2]))
            right_child = int(np.round(tree_arr[node, 3]))
            label = int(np.round(tree_arr[node, 4]))
            is_leaf = int(np.round(tree_arr[node, 5]))

            if is_leaf == 1 or feature < 0:
                out[i] = label
                break
            # end if
            if feature >= x_arr.shape[1]:
                out[i] = label
                break
            # end if

            node = left_child if x_arr[i, feature] <= threshold else right_child
        # end while
    # end for

    return out
# end def _algo_decision_tree_predict


def _algo_svm_train(
        x: np.ndarray,
        y: np.ndarray,
        *,
        c: float,
        max_iter: int,
        tol: float,
        fit_intercept: bool,
        seed: Optional[int],
) -> np.ndarray:
    _require_sklearn()
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = _flatten_targets(np.asarray(y, dtype=np.float64), "svm_train")

    if x_arr.ndim != 2:
        raise ValueError(f"svm_train expects x rank-2, got shape={x_arr.shape}")
    # end if
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(f"svm_train sample mismatch: {x_arr.shape[0]} != {y_arr.shape[0]}")
    # end if

    unique = set(np.unique(y_arr).tolist())
    if unique != {-1.0, 1.0}:
        raise ValueError(f"svm_train expects binary labels in {{-1, +1}}, got {sorted(unique)}")
    # end if
    if c <= 0.0:
        raise ValueError("svm_train requires c > 0")
    # end if
    if max_iter <= 0:
        raise ValueError("svm_train requires max_iter > 0")
    # end if
    if tol <= 0.0:
        raise ValueError("svm_train requires tol > 0")
    # end if

    model = SklearnLinearSVC(
        C=c,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        random_state=seed,
        dual="auto",
    )
    model.fit(x_arr, y_arr)
    w = np.asarray(model.coef_[0], dtype=np.float64)
    b = float(model.intercept_[0]) if fit_intercept else 0.0
    return np.concatenate([w, np.asarray([b], dtype=np.float64)])
# end def _algo_svm_train


def _algo_svm_decision_function(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError(f"svm_decision_function expects x rank-2, got shape={x_arr.shape}")
    # end if
    if theta_arr.shape[0] != x_arr.shape[1] + 1:
        raise ValueError(f"svm_decision_function expects theta shape ({x_arr.shape[1] + 1},), got {theta_arr.shape}")
    # end if
    w = theta_arr[:-1]
    b = theta_arr[-1]
    return (x_arr @ w) + b
# end def _algo_svm_decision_function


def _algo_svm_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    scores = _algo_svm_decision_function(x, theta)
    return np.where(scores >= 0.0, 1, -1).astype(np.int64)
# end def _algo_svm_predict


def _require_sklearn() -> None:
    if SKLEARN_AVAILABLE:
        return
    # end if
    raise ImportError(
        "scikit-learn is required for basic ML operators. "
        "Install it to use perceptron/decision-tree/svm operators."
    ) from SKLEARN_IMPORT_ERROR
# end def _require_sklearn


if SKLEARN_AVAILABLE:
    register_algorithm("perceptron_train", _algo_perceptron_train)
    register_algorithm("perceptron_predict", _algo_perceptron_predict)
    register_algorithm("decision_tree_train", _algo_decision_tree_train)
    register_algorithm("decision_tree_predict", _algo_decision_tree_predict)
    register_algorithm("svm_train", _algo_svm_train)
    register_algorithm("svm_decision_function", _algo_svm_decision_function)
    register_algorithm("svm_predict", _algo_svm_predict)


class MachineLearningOperator(OperatorBase, ParametricOperator, ABC):
    """Base class for machine learning operators."""

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class MachineLearningOperator


class PerceptronTrain(MachineLearningOperator):
    """Fit binary perceptron and return hyperplane parameters theta=[w..., b]."""

    NAME = "perceptron_train"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(
            self,
            max_iter: ScalarParameter = 1000,
            learning_rate: ScalarParameter = 1.0,
            shuffle: bool = True,
            seed: Optional[int] = None,
            fit_intercept: bool = True,
            dtype: DType = DType.R,
    ):
        max_iter_expr = _ensure_scalar_parameter(max_iter, f"{self.NAME} max_iter")
        lr_expr = _ensure_scalar_parameter(learning_rate, f"{self.NAME} learning_rate")
        super().__init__(
            max_iter=max_iter_expr,
            learning_rate=lr_expr,
            shuffle=shuffle,
            seed=seed,
            fit_intercept=fit_intercept,
            dtype=dtype,
        )
        self._max_iter = max_iter_expr
        self._learning_rate = lr_expr
        self._shuffle = bool(shuffle)
        self._seed = seed
        self._fit_intercept = bool(fit_intercept)
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return (
            self._max_iter.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._learning_rate.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            max_iter: ScalarParameter = 1000,
            learning_rate: ScalarParameter = 1.0,
            shuffle: bool = True,
            seed: Optional[int] = None,
            fit_intercept: bool = True,
            dtype: DType = DType.R,
    ) -> bool:
        return isinstance(shuffle, bool) and (seed is None or isinstance(seed, int)) and isinstance(fit_intercept, bool)
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, y = operands
        if x.rank != 2:
            return False
        # end if
        if y.rank not in {1, 2}:
            return False
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        x, _ = operands
        return Shape.vector(x.shape[1] + 1)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        y = np.asarray(operands[1].eval().value)
        max_iter = _eval_int_parameter(self._max_iter, f"{self.NAME} max_iter")
        lr = _eval_scalar_parameter(self._learning_rate, f"{self.NAME} learning_rate")
        theta = get_algorithm("perceptron_train")(
            x,
            y,
            max_iter=max_iter,
            learning_rate=lr,
            shuffle=self._shuffle,
            seed=self._seed,
            fit_intercept=self._fit_intercept,
        )
        return Tensor(data=np.asarray(theta, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PerceptronTrain does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return (
            f"{self.NAME}(max_iter={self._max_iter.name}, learning_rate={self._learning_rate.name}, "
            f"shuffle={self._shuffle}, fit_intercept={self._fit_intercept}, dtype={self._dtype.name})"
        )
    # end def __str__

# end class PerceptronTrain


class PerceptronPredict(MachineLearningOperator):
    """Predict binary classes {-1,+1} from x and theta."""

    NAME = "perceptron_predict"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.Z):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.Z) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, theta = operands
        return x.rank == 2 and theta.rank == 1
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        x, _ = operands
        return Shape.vector(x.shape[0])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        theta = np.asarray(operands[1].eval().value)
        pred = get_algorithm("perceptron_predict")(x, theta)
        return Tensor(data=np.asarray(pred, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PerceptronPredict does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class PerceptronPredict


class PerceptronDecisionBoundary(MachineLearningOperator):
    """Return hyperplane parameters theta=[w..., b]."""

    NAME = "perceptron_decision_boundary"
    ARITY = 1
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.R):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.R) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].rank == 1
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return operands[0].shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        theta = np.asarray(operands[0].eval().value)
        return Tensor(data=np.asarray(theta, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PerceptronDecisionBoundary does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class PerceptronDecisionBoundary


class PerceptronCoefficients(MachineLearningOperator):
    """Extract perceptron coefficients w from theta=[w..., b]."""

    NAME = "perceptron_coefficients"
    ARITY = 1
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.R):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.R) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].rank == 1
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(operands[0].shape[0] - 1)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        theta = np.asarray(operands[0].eval().value)
        w = theta[:-1]
        return Tensor(data=np.asarray(w, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PerceptronCoefficients does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class PerceptronCoefficients


class PerceptronIntercept(MachineLearningOperator):
    """Extract perceptron intercept b from theta=[w..., b]."""

    NAME = "perceptron_intercept"
    ARITY = 1
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.R):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.R) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].rank == 1
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.scalar()
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        theta = np.asarray(operands[0].eval().value)
        b = theta[-1]
        return Tensor(data=np.asarray(b, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PerceptronIntercept does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class PerceptronIntercept


class PerceptronClasses(MachineLearningOperator):
    """Return class labels used by the perceptron classifier: [-1, +1]."""

    NAME = "perceptron_classes"
    ARITY = 0
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.Z):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.Z) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(2)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        return Tensor(data=np.asarray([-1, 1], dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PerceptronClasses does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class PerceptronClasses


class DecisionTreeTrain(MachineLearningOperator):
    """Train a multiclass decision tree and return serialized tree matrix."""

    NAME = "decision_tree_train"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(
            self,
            max_depth: ScalarParameter = 3,
            min_samples_split: ScalarParameter = 2,
            min_samples_leaf: ScalarParameter = 1,
            criterion: str = "gini",
            dtype: DType = DType.R,
    ):
        max_depth_expr = _ensure_scalar_parameter(max_depth, f"{self.NAME} max_depth")
        split_expr = _ensure_scalar_parameter(min_samples_split, f"{self.NAME} min_samples_split")
        leaf_expr = _ensure_scalar_parameter(min_samples_leaf, f"{self.NAME} min_samples_leaf")
        super().__init__(
            max_depth=max_depth_expr,
            min_samples_split=split_expr,
            min_samples_leaf=leaf_expr,
            criterion=criterion,
            dtype=dtype,
        )
        self._max_depth = max_depth_expr
        self._min_samples_split = split_expr
        self._min_samples_leaf = leaf_expr
        self._criterion = str(criterion)
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return (
            self._max_depth.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._min_samples_split.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._min_samples_leaf.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            max_depth: ScalarParameter = 3,
            min_samples_split: ScalarParameter = 2,
            min_samples_leaf: ScalarParameter = 1,
            criterion: str = "gini",
            dtype: DType = DType.R,
    ) -> bool:
        return criterion in {"gini", "entropy"}
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, y = operands
        if x.rank != 2:
            return False
        # end if
        if y.rank not in {1, 2}:
            return False
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        max_depth = _eval_int_parameter(self._max_depth, f"{self.NAME} max_depth")
        max_nodes = (2 ** (max_depth + 1)) - 1
        return Shape.matrix(max_nodes, 6)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        y = np.asarray(operands[1].eval().value)
        max_depth = _eval_int_parameter(self._max_depth, f"{self.NAME} max_depth")
        min_samples_split = _eval_int_parameter(self._min_samples_split, f"{self.NAME} min_samples_split")
        min_samples_leaf = _eval_int_parameter(self._min_samples_leaf, f"{self.NAME} min_samples_leaf")
        tree = get_algorithm("decision_tree_train")(
            x,
            y,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=self._criterion,
        )
        return Tensor(data=np.asarray(tree, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("DecisionTreeTrain does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return (
            f"{self.NAME}(max_depth={self._max_depth.name}, min_samples_split={self._min_samples_split.name}, "
            f"min_samples_leaf={self._min_samples_leaf.name}, criterion={self._criterion}, dtype={self._dtype.name})"
        )
    # end def __str__

# end class DecisionTreeTrain


class DecisionTreePredict(MachineLearningOperator):
    """Predict multiclass labels from x and serialized tree model."""

    NAME = "decision_tree_predict"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.Z):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.Z) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, tree = operands
        return x.rank == 2 and tree.rank == 2 and tree.shape[1] == 6
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        x, _ = operands
        return Shape.vector(x.shape[0])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        tree = np.asarray(operands[1].eval().value)
        pred = get_algorithm("decision_tree_predict")(x, tree)
        return Tensor(data=np.asarray(pred, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("DecisionTreePredict does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class DecisionTreePredict


class DecisionTreeClasses(MachineLearningOperator):
    """Return sorted unique class labels from y."""

    NAME = "decision_tree_classes"
    ARITY = 1
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.Z):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.Z) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].rank in {1, 2}
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        y = operands[0]
        n = y.shape[0] if y.rank == 1 else (y.shape[0] * y.shape[1])
        return Shape.vector(n)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        y = np.asarray(operands[0].eval().value)
        classes = np.unique(y.reshape(-1))
        if not np.allclose(classes, np.round(classes)):
            raise ValueError("decision_tree_classes expects integer class labels")
        # end if
        return Tensor(data=np.asarray(classes, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("DecisionTreeClasses does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class DecisionTreeClasses


class SVMTrain(MachineLearningOperator):
    """Train binary linear SVM and return hyperplane parameters theta=[w..., b]."""

    NAME = "svm_train"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(
            self,
            c: ScalarParameter = 1.0,
            max_iter: ScalarParameter = 1000,
            tol: ScalarParameter = 1e-4,
            fit_intercept: bool = True,
            seed: Optional[int] = None,
            dtype: DType = DType.R,
    ):
        c_expr = _ensure_scalar_parameter(c, f"{self.NAME} c")
        max_iter_expr = _ensure_scalar_parameter(max_iter, f"{self.NAME} max_iter")
        tol_expr = _ensure_scalar_parameter(tol, f"{self.NAME} tol")
        super().__init__(
            c=c_expr,
            max_iter=max_iter_expr,
            tol=tol_expr,
            fit_intercept=fit_intercept,
            seed=seed,
            dtype=dtype,
        )
        self._c = c_expr
        self._max_iter = max_iter_expr
        self._tol = tol_expr
        self._fit_intercept = bool(fit_intercept)
        self._seed = seed
        self._dtype = dtype
        _require_sklearn()
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return (
            self._c.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._max_iter.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._tol.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            c: ScalarParameter = 1.0,
            max_iter: ScalarParameter = 1000,
            tol: ScalarParameter = 1e-4,
            fit_intercept: bool = True,
            seed: Optional[int] = None,
            dtype: DType = DType.R,
    ) -> bool:
        return SKLEARN_AVAILABLE and isinstance(fit_intercept, bool) and (seed is None or isinstance(seed, int))
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, y = operands
        return x.rank == 2 and y.rank in {1, 2}
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        x, _ = operands
        return Shape.vector(x.shape[1] + 1)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        y = np.asarray(operands[1].eval().value)
        theta = get_algorithm("svm_train")(
            x,
            y,
            c=_eval_scalar_parameter(self._c, f"{self.NAME} c"),
            max_iter=_eval_int_parameter(self._max_iter, f"{self.NAME} max_iter"),
            tol=_eval_scalar_parameter(self._tol, f"{self.NAME} tol"),
            fit_intercept=self._fit_intercept,
            seed=self._seed,
        )
        return Tensor(data=np.asarray(theta, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("SVMTrain does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return (
            f"{self.NAME}(c={self._c.name}, max_iter={self._max_iter.name}, tol={self._tol.name}, "
            f"fit_intercept={self._fit_intercept}, dtype={self._dtype.name})"
        )
    # end def __str__

# end class SVMTrain


class SVMDecisionFunction(MachineLearningOperator):
    """Compute linear SVM decision score for each sample."""

    NAME = "svm_decision_function"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.R):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.R) -> bool:
        return SKLEARN_AVAILABLE
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, theta = operands
        return x.rank == 2 and theta.rank == 1
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(operands[0].shape[0])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        score = get_algorithm("svm_decision_function")(
            np.asarray(operands[0].eval().value),
            np.asarray(operands[1].eval().value),
        )
        return Tensor(data=np.asarray(score, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("SVMDecisionFunction does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class SVMDecisionFunction


class SVMPredict(MachineLearningOperator):
    """Predict binary classes {-1,+1} from x and SVM theta."""

    NAME = "svm_predict"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.Z):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.Z) -> bool:
        return SKLEARN_AVAILABLE
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, theta = operands
        return x.rank == 2 and theta.rank == 1
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(operands[0].shape[0])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        pred = get_algorithm("svm_predict")(
            np.asarray(operands[0].eval().value),
            np.asarray(operands[1].eval().value),
        )
        return Tensor(data=np.asarray(pred, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("SVMPredict does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class SVMPredict


class SVMClasses(MachineLearningOperator):
    """Return class labels used by binary linear SVM: [-1, +1]."""

    NAME = "svm_classes"
    ARITY = 0
    IS_VARIADIC = False

    def __init__(self, dtype: DType = DType.Z):
        super().__init__(dtype=dtype)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(cls, dtype: DType = DType.Z) -> bool:
        return SKLEARN_AVAILABLE
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(2)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        return Tensor(data=np.asarray([-1, 1], dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("SVMClasses does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(dtype={self._dtype.name})"
    # end def __str__

# end class SVMClasses


if SKLEARN_AVAILABLE:
    operator_registry.register(PerceptronTrain)
    operator_registry.register(PerceptronPredict)
    operator_registry.register(PerceptronDecisionBoundary)
    operator_registry.register(PerceptronCoefficients)
    operator_registry.register(PerceptronIntercept)
    operator_registry.register(PerceptronClasses)
    operator_registry.register(DecisionTreeTrain)
    operator_registry.register(DecisionTreePredict)
    operator_registry.register(DecisionTreeClasses)
    operator_registry.register(SVMTrain)
    operator_registry.register(SVMPredict)
    operator_registry.register(SVMDecisionFunction)
    operator_registry.register(SVMClasses)
