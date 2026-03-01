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

"""Effect pipeline orchestration utilities.

This module defines :class:`EffectPipeline`, a lightweight container that
applies effects sequentially and reports timing statistics.
"""

# Imports
from __future__ import annotations

import time
from typing import Any, cast


class EffectPipeline:
    """Apply a sequence of effects to input frames.

    Parameters
    ----------
    effects : list[object] | None, default=None
        Initial list of effect objects. Each effect is expected to implement
        ``apply(image, **kwargs)``.

    Attributes
    ----------
    effects : list[object]
        Ordered list of effects in the pipeline.
    time_stats : dict[str, float]
        Cumulative runtime per effect name and total runtime.
    """

    def __init__(self, effects: list[object] | None = None):
        """Initialize a new effect pipeline.

        Parameters
        ----------
        effects : list[object] | None, default=None
            Initial effect list.
        """
        self.effects = list(effects) if effects is not None else []
        self.time_stats: dict[str, float] = {}
    # end def __init__

    def add_effect(self, effect: object) -> None:
        """Append an effect to the pipeline.

        Parameters
        ----------
        effect : object
            Effect object exposing an ``apply`` method.
        """
        self.effects.append(effect)
    # end def add_effect

    def apply(self, image: object, **kwargs: object) -> object:
        """Apply all effects to an image.

        Parameters
        ----------
        image : object
            Input frame.
        **kwargs : object
            Additional keyword arguments forwarded to each effect.

        Returns
        -------
        object
            Processed frame.
        """
        total_start = time.time()

        for effect in self.effects:
            effect_obj = effect
            start = time.time()
            image = cast(Any, effect_obj).apply(image, **kwargs)
            end = time.time()

            effect_name = effect_obj.__class__.__name__
            if effect_name not in self.time_stats:
                self.time_stats[effect_name] = 0.0
            # end if
            self.time_stats[effect_name] += end - start
        # end for

        self.time_stats["Total"] = time.time() - total_start
        return image
    # end def apply

    def print_stats(self) -> None:
        """Print cumulative timing statistics for the pipeline."""
        total_time = self.time_stats.pop("Total", None)
        if total_time:
            print(f"Total processing time: {total_time:.2f} seconds")
            for effect, time_taken in self.time_stats.items():
                percentage = (time_taken / total_time) * 100
                print(f"{effect}: {time_taken:.2f} seconds ({percentage:.2f}%)")
            # end for
        # end if
    # end def print_stats

# end class EffectPipeline
