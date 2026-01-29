#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#


# Imports
import sys
import importlib
import pkgutil

import pixelprism.math as m


def _import_all_submodules(pkg):
    """
    Imports recursively all the subpackages.
    """
    if not hasattr(pkg, "__path__"):
        return
    # end if
    prefix = pkg.__name__ + "."
    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__, prefix):
        importlib.import_module(name)
        # optionnel: descendre récursivement
        try:
            subpkg = sys.modules[name]
            _import_all_submodules(subpkg)
        except KeyError:
            pass
        # end try
    # end for
# end _import_all_submodules


def _resolve_operator_class(package, class_name: str) -> Type:
    """
    Tente de trouver la classe `class_name` dans le package `package`
    ou l’un de ses sous-modules déjà importés.
    """
    # 1) direct sur le package (si la classe y est "re-exportée")
    if hasattr(package, class_name):
        return getattr(package, class_name)

    # 2) chercher dans tous les modules chargés appartenant au package
    pkg_prefix = package.__name__ + "."
    for mod_name, mod in list(sys.modules.items()):
        if isinstance(mod_name, str) and mod_name.startswith(pkg_prefix) and mod is not None:
            if hasattr(mod, class_name):
                return getattr(mod, class_name)

    # 3) rien trouvé
    raise AttributeError(f"Classe opérateur introuvable: {class_name}")



def add(
        operand1: m.MathNode,
        operand2: m.MathNode
) -> m.MathOperator:
    """
    Add two math_old expressions.

    Args:
        operand1: First math_old expression.
        operand2: Second math_old expression.

    Returns:
        A new math_old expression.
    """
    # Get class names of the two operands
    operand1_class_name = operand1.__class__.__name__
    operand2_class_name = operand2.__class__.__name__

    # Compute name of the operator
    operator_class_name = f"{operand1_class_name}To{operand2_class_name}Addition"

    # Search for this class in pixelprism.math_old (m)
    # ...
# end add

