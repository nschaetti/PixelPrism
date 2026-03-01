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

"""Root Click command group for PixelPrism.

The root group exposes high-level workflows:

- ``animate``: render procedural animations.
- ``fx``: apply post-processing effects on existing videos.
"""

# Imports
import click

from .animate import animate
from .fx import fx


__all__ = ["cli"]


@click.group(no_args_is_help=True)
def cli() -> None:
    """Entry point for the PixelPrism command-line interface.

    Notes
    -----
    The command group is configured with ``no_args_is_help=True`` so running
    ``python -m pixelprism`` without a subcommand shows help instead of failing.
    """
    # No action is required here. Subcommands handle the behavior.
    pass
# end def cli


cli.add_command(animate)
cli.add_command(fx)
