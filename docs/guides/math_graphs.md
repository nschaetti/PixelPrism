# Math Graphs

PixelPrism provides graph primitives and adjacency-based graph operators that integrate
with symbolic math expressions.

## Graph Primitives

Use `pixelprism.math` graph helpers:

- `graph(...)` for undirected simple graphs
- `digraph(...)` for directed simple graphs
- `graph_from_edges(...)`
- `graph_from_adjacency(...)`

```python
import pixelprism.math as pm

g = pm.digraph(n_nodes=4, edges=[(0, 1), (1, 2), (0, 3)], allow_cycles=False)
print(g.is_cyclic())
print(g.topological_sort())
```

## Expression Dependency Graph

Any `MathExpr` can be exported as a graph view:

- `expression_to_graph(expr)`
- `expression_to_adjacency(expr)`
- `graph_to_dot(graph)`

```python
import pixelprism.math as pm
import pixelprism.math.functional as F

x = pm.var("x", dtype=pm.DType.R, shape=())
y = pm.var("y", dtype=pm.DType.R, shape=())
expr = F.mul(F.add(x, y), y)

view = pm.expression_to_graph(expr)
print(view.graph.n_nodes, view.graph.n_edges)
print(pm.graph_to_dot(view.graph))
```

## Graph Operators (Adjacency Input)

In `pixelprism.math.functional.graph`:

- `degree`, `in_degree`, `out_degree`
- `laplacian` (normalized or unnormalized)
- `is_cyclic`
- `topological_sort`

```python
import numpy as np
import pixelprism.math as pm
import pixelprism.math.functional.graph as G

adj = pm.const("adj", data=np.array([[0, 1], [0, 0]], dtype=np.int32), dtype=pm.DType.Z)
print(G.out_degree(adj).eval().value)
print(G.topological_sort(adj).eval().value)
```
