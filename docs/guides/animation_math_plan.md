# Plan d'architecture : animation pilotee par MathExpr

## Objectif

Passer d'un modele d'animation imperative (transitions appliquees directement aux objets) a un modele **data-driven** :

- les proprietes des objets sont liees a des variables/expressions,
- les keyframes pilotent ces variables dans le temps,
- le rendu evalue tout a l'instant `t`.

Cela unifie :

- animation par keyframes,
- animation par formule (`MathExpr`),
- API simple pour l'utilisateur (`move()`, `to()`, `at()`).

---

## Carte des sous-packages (cible)

### `pixelprism.math`

Responsabilites :

- moteur symbolique (`MathExpr`, `var`, `const`, operateurs),
- contexte d'evaluation (`Context`),
- rendu LaTeX/SVG des expressions si besoin.

Ne gere pas la timeline ni les keyframes.

### `pixelprism.animation.timeline`

Responsabilites :

- gestion du temps,
- `Timeline`, `Track[T]`, `Keyframe[T]`, easing/interpolation,
- calcul de valeurs animees a partir de `t`.

### `pixelprism.animation.binding`

Responsabilites :

- interface entre proprietes drawables et source de valeur,
- conteneur de `Binding[T]` et ses implementations.

Implementations typiques :

- `ConstantBinding[T]`,
- `TrackBinding[T]`,
- `ExprBinding[T]`,
- `ComputedBinding[T]` (optionnel).

### `pixelprism.animation.runtime`

Responsabilites :

- boucle frame-by-frame,
- injection de `t` et des variables animees dans le `math.Context`,
- evaluation de tous les bindings,
- appel du rendu de la scene.

### `pixelprism.drawing.types`

Responsabilites :

- types geometriques forts pour l'API :
  - 1D : `S1`,
  - 2D : `V2`,
  - 3D : `V3`,
  - primitives : `Line2`, `Rect`, `Bezier2`, `Bezier3`.
- validation shape/dtype,
- conversion explicite (tuple -> `V2`, etc.).

### `pixelprism.drawing.shapes`

Responsabilites :

- objets dessinables (`Line`, `Circle`, `Path`, `Bezier`, ...),
- proprietes stockees en `Binding[...]`,
- dessin base sur des valeurs deja resolues.

### `pixelprism.scene`

Responsabilites :

- agregation des objets,
- ordre de rendu, groupes et visibilite,
- facade haut niveau (`scene.add`, `scene.animate`, ...).

### `pixelprism.compat.animate_legacy` (transitoire)

Responsabilites :

- compatibilite ascendante avec l'API actuelle,
- adaptation de `move()`, `fadein()`, etc. vers tracks + bindings.

---

## `Binding[T]` : definition et role

`Binding[T]` represente la maniere d'obtenir une valeur de type `T` a l'instant courant.

Interface conceptuelle :

```python
class Binding[T]:
    def eval(self, frame_ctx) -> T:
        ...
```

### Ce que `Binding[T]` unifie

- constante,
- keyframes via un track,
- expression symbolique (`MathExpr`) dependant de `t` ou d'autres variables.

Les drawables ne connaissent plus la source. Ils demandent seulement la valeur finale du bon type.

### Exemples de proprietes typees

- `line.start: Binding[V2]`
- `line.end: Binding[V2]`
- `line.width: Binding[S1]`
- `line.opacity: Binding[S1]`

### Regles de conception

- typage strict par propriete,
- conversion explicite en sortie,
- evaluation sans effet de bord,
- erreurs claires (shape invalide, variable manquante, dtype incompatible).

---

## Flux d'execution d'une frame

1. fixer `t`,
2. evaluer les tracks,
3. injecter les valeurs dans `math.Context`,
4. evaluer tous les `Binding[T]`,
5. dessiner la scene avec les valeurs resolues.

---

## API utilisateur visee

```python
scene = Scene(duration=6, fps=60)

line = Line2D(start=(0, 0), end=(1, 0), width=2.0)

line.start.move(to=(2, 1), dur=2)
line.end.at(0, (1, 0)).at(3, (3, 2))

amp = scene.var("amp", 0.2)
scene.animate(amp).to(1.5, dur=3)

line.end.bind_expr(v2(cos(t()) * amp, sin(t()) * amp))
```

---

## Plan de migration

1. Introduire `Timeline/Track/Keyframe` + runtime minimal.
2. Introduire `Binding[T]` et brancher 2-3 shapes (`Line`, `Circle`, `Path`).
3. Ajouter API sugar (`move`, `to`, `at`, `fade`).
4. Ajouter compat legacy.
5. Etendre a toutes les primitives + tests E2E + docs.
