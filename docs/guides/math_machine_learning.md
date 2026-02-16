# Math Machine Learning

PixelPrism exposes basic ML operators and sklearn-like facades for symbolic workflows.

## Dependency Note

Basic ML operators depend on `scikit-learn`.

- If available: ML operators are registered and usable.
- If unavailable: calling ML APIs raises a clear `ImportError`.

## Perceptron (Binary `{-1, +1}`)

Object API:

```python
import pixelprism.math as pm

x = pm.var("x", dtype=pm.DType.R, shape=(4, 2))
y = pm.var("y", dtype=pm.DType.Z, shape=(4,))

clf = pm.Perceptron(max_iter=200, learning_rate=1.0)
clf.fit(x, y)

pred = clf.predict(x)
theta = clf.theta_
boundary = clf.decision_boundary()
```

## Decision Tree (Multiclass)

Supports `criterion="gini"` and `criterion="entropy"`.

```python
import pixelprism.math as pm

x = pm.var("x", dtype=pm.DType.R, shape=(6, 2))
y = pm.var("y", dtype=pm.DType.Z, shape=(6,))

tree = pm.DecisionTreeClassifier(max_depth=3, criterion="entropy")
tree.fit(x, y)
pred = tree.predict(x)
```

## SVC (Linear)

Current SVM support is linear only.

```python
import pixelprism.math as pm

x = pm.var("x", dtype=pm.DType.R, shape=(4, 2))
y = pm.var("y", dtype=pm.DType.Z, shape=(4,))

svc = pm.SVC(c=1.0, max_iter=2000)
svc.fit(x, y)

scores = svc.decision_function(x)
pred = svc.predict(x)
```

## Functional ML API

In `pixelprism.math.functional.machine_learning`:

- Perceptron: `fit`, `predict`, `decision_boundary`, `coefficients`, `intercept`, `classes`
- Decision tree: `tree_fit`, `tree_predict`, `tree_classes`
- SVM: `svm_fit`, `svm_predict`, `svm_decision_function`, `svm_classes`

All outputs remain symbolic expressions and are re-evaluated from the current context.
