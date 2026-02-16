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
"""Sklearn-like machine-learning facade for symbolic math expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .dtype import DType
from .typing import MathExpr
from .functional import machine_learning as ML
from .operators.machine_learning import SKLEARN_AVAILABLE


ScalarLikeExpr = Union[MathExpr, int, float]


__all__ = [
    "Perceptron",
    "DecisionTreeClassifier",
    "SVC",
]


@dataclass
class Perceptron:
    """Binary perceptron model with sklearn-like API."""

    max_iter: ScalarLikeExpr = 1000
    learning_rate: ScalarLikeExpr = 1.0
    shuffle: bool = True
    seed: Optional[int] = None
    fit_intercept: bool = True
    param_dtype: DType = DType.R
    pred_dtype: DType = DType.Z

    def __post_init__(self) -> None:
        _ensure_ml_available()
        self._theta: Optional[MathExpr] = None
        self._classes: MathExpr = ML.classes(dtype=self.pred_dtype)
    # end def __post_init__

    def fit(self, x: MathExpr, y: MathExpr) -> "Perceptron":
        self._theta = ML.fit(
            x,
            y,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            shuffle=self.shuffle,
            seed=self.seed,
            fit_intercept=self.fit_intercept,
            dtype=self.param_dtype,
        )
        return self
    # end def fit

    def predict(self, x: MathExpr) -> MathExpr:
        if self._theta is None:
            raise ValueError("Perceptron must be fitted before predict().")
        # end if
        return ML.predict(x, self._theta, dtype=self.pred_dtype)
    # end def predict

    def decision_boundary(self) -> MathExpr:
        if self._theta is None:
            raise ValueError("Perceptron must be fitted before decision_boundary().")
        # end if
        return ML.decision_boundary(self._theta, dtype=self.param_dtype)
    # end def decision_boundary

    @property
    def theta_(self) -> MathExpr:
        if self._theta is None:
            raise ValueError("Perceptron is not fitted; theta_ is unavailable.")
        # end if
        return self._theta
    # end def theta_

    @property
    def coef_(self) -> MathExpr:
        return ML.coefficients(self.theta_, dtype=self.param_dtype)
    # end def coef_

    @property
    def intercept_(self) -> MathExpr:
        return ML.intercept(self.theta_, dtype=self.param_dtype)
    # end def intercept_

    @property
    def classes_(self) -> MathExpr:
        return self._classes
    # end def classes_


@dataclass
class DecisionTreeClassifier:
    """Multiclass decision tree classifier with sklearn-like API."""

    max_depth: ScalarLikeExpr = 3
    min_samples_split: ScalarLikeExpr = 2
    min_samples_leaf: ScalarLikeExpr = 1
    criterion: str = "gini"
    model_dtype: DType = DType.R
    pred_dtype: DType = DType.Z

    def __post_init__(self) -> None:
        _ensure_ml_available()
        self._tree: Optional[MathExpr] = None
        self._classes: Optional[MathExpr] = None
    # end def __post_init__

    def fit(self, x: MathExpr, y: MathExpr) -> "DecisionTreeClassifier":
        self._tree = ML.tree_fit(
            x,
            y,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            dtype=self.model_dtype,
        )
        self._classes = ML.tree_classes(y, dtype=self.pred_dtype)
        return self
    # end def fit

    def predict(self, x: MathExpr) -> MathExpr:
        if self._tree is None:
            raise ValueError("DecisionTreeClassifier must be fitted before predict().")
        # end if
        return ML.tree_predict(x, self._tree, dtype=self.pred_dtype)
    # end def predict

    @property
    def tree_(self) -> MathExpr:
        if self._tree is None:
            raise ValueError("DecisionTreeClassifier is not fitted; tree_ is unavailable.")
        # end if
        return self._tree
    # end def tree_

    @property
    def classes_(self) -> MathExpr:
        if self._classes is None:
            raise ValueError("DecisionTreeClassifier is not fitted; classes_ is unavailable.")
        # end if
        return self._classes
    # end def classes_


@dataclass
class SVC:
    """Linear binary SVM classifier with sklearn-like API."""

    c: ScalarLikeExpr = 1.0
    max_iter: ScalarLikeExpr = 1000
    tol: ScalarLikeExpr = 1e-4
    fit_intercept: bool = True
    seed: Optional[int] = None
    param_dtype: DType = DType.R
    pred_dtype: DType = DType.Z

    def __post_init__(self) -> None:
        _ensure_ml_available()
        self._theta: Optional[MathExpr] = None
        self._classes: MathExpr = ML.svm_classes(dtype=self.pred_dtype)
    # end def __post_init__

    def fit(self, x: MathExpr, y: MathExpr) -> "SVC":
        self._theta = ML.svm_fit(
            x,
            y,
            c=self.c,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            seed=self.seed,
            dtype=self.param_dtype,
        )
        return self
    # end def fit

    def decision_function(self, x: MathExpr) -> MathExpr:
        if self._theta is None:
            raise ValueError("SVC must be fitted before decision_function().")
        # end if
        return ML.svm_decision_function(x, self._theta, dtype=self.param_dtype)
    # end def decision_function

    def predict(self, x: MathExpr) -> MathExpr:
        if self._theta is None:
            raise ValueError("SVC must be fitted before predict().")
        # end if
        return ML.svm_predict(x, self._theta, dtype=self.pred_dtype)
    # end def predict

    @property
    def theta_(self) -> MathExpr:
        if self._theta is None:
            raise ValueError("SVC is not fitted; theta_ is unavailable.")
        # end if
        return self._theta
    # end def theta_

    @property
    def coef_(self) -> MathExpr:
        return ML.coefficients(self.theta_, dtype=self.param_dtype)
    # end def coef_

    @property
    def intercept_(self) -> MathExpr:
        return ML.intercept(self.theta_, dtype=self.param_dtype)
    # end def intercept_

    @property
    def classes_(self) -> MathExpr:
        return self._classes
    # end def classes_


def _ensure_ml_available() -> None:
    if SKLEARN_AVAILABLE:
        return
    # end if
    raise ImportError("scikit-learn is required for pixelprism.math.ml basic models")
# end def _ensure_ml_available
