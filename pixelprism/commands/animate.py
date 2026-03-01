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

"""Click command for animation generation.

This module exposes the ``animate`` command used to render a custom
``Animation`` class into a video file.
"""

# Imports
from __future__ import annotations

from typing import Any, cast

import importlib.util
import importlib

import click

try:
    RichConsole = importlib.import_module("rich.console").Console
    RichPanel = importlib.import_module("rich.panel").Panel
    RichTable = importlib.import_module("rich.table").Table
    _HAS_RICH = True
except ModuleNotFoundError:
    RichConsole = None
    RichPanel = None
    RichTable = None
    _HAS_RICH = False
# end try

from ..video_composer import VideoComposer


if _HAS_RICH and RichConsole is not None:
    console = RichConsole()
else:
    console = None
# end if


def _load_class_from_file(file_path: str, class_name: str) -> type:
    """Load a class object from a Python file.

    Parameters
    ----------
    file_path : str
        Path to the Python file containing the class declaration.
    class_name : str
        Name of the class to retrieve from the loaded module.

    Returns
    -------
    type
        Loaded class type.

    Raises
    ------
    click.ClickException
        If the module specification cannot be created.
    AttributeError
        If ``class_name`` does not exist in the loaded module.
    """
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(
            f"Unable to load module specification from '{file_path}'."
        )
    # end if

    spec_obj = spec
    loader = spec_obj.loader
    if loader is None:
        raise click.ClickException(
            f"Unable to load module from '{file_path}' (missing loader)."
        )
    # end if

    module = importlib.util.module_from_spec(spec_obj)
    loader.exec_module(module)  # pyright: ignore[reportOptionalMemberAccess]
    return getattr(module, class_name)
# end def _load_class_from_file


def _bool_arg(value: str | bool) -> bool:
    """Parse a CLI boolean value.

    Parameters
    ----------
    value : str | bool
        Input value supplied by Click.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    # end if
    return value.lower() in ("yes", "true", "t", "1")
# end def _bool_arg


def _parse_kwargs(kwargs: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``key=value`` pairs from CLI options.

    Parameters
    ----------
    kwargs : tuple[str, ...]
        Sequence of key-value entries received from ``--kwargs``.

    Returns
    -------
    dict[str, str]
        Parsed dictionary forwarded to the animation class constructor.

    Raises
    ------
    click.BadParameter
        If an entry does not respect the ``key=value`` format.
    """
    parsed: dict[str, str] = {}
    for item in kwargs:
        if "=" not in item:
            raise click.BadParameter(
                "Each --kwargs entry must use the format key=value."
            )
        # end if
        key, value = item.split("=", 1)
        parsed[key] = value
    # end for
    return parsed
# end def _parse_kwargs


def _display_animation_parameters(
    output: str,
    input_path: str | None,
    display: bool,
    debug_frames: tuple[int, ...],
    class_file: str,
    class_name: str,
    duration: float | None,
    fps: int | None,
    width: int,
    height: int,
    save_frames: bool | None,
    kwargs_dict: dict[str, str],
) -> None:
    """Render an elegant Rich summary of animation parameters.

    Parameters
    ----------
    output : str
        Output video path.
    input_path : str | None
        Optional input video path.
    display : bool
        Whether the viewer is enabled.
    debug_frames : tuple[int, ...]
        Frame indices selected for debug snapshots.
    class_file : str
        Python file containing the animation class.
    class_name : str
        Name of the animation class.
    duration : float | None
        Output duration in seconds.
    fps : int | None
        Output frame rate.
    width : int
        Output width in pixels.
    height : int
        Output height in pixels.
    save_frames : bool | None
        Whether intermediate frames are saved.
    kwargs_dict : dict[str, str]
        Additional constructor keyword arguments.
    """
    if not _HAS_RICH or console is None:
        click.echo("PixelPrism Animation")
        click.echo(f"  Animation class: {class_name} ({class_file})")
        click.echo(f"  Output: {output}")
        click.echo(f"  Input: {input_path or 'none'}")
        click.echo(f"  Duration: {duration if duration is not None else 'auto'}")
        click.echo(f"  FPS: {fps if fps is not None else 'source/default'}")
        click.echo(f"  Resolution: {width}x{height}")
        click.echo(f"  Display: {'yes' if display else 'no'}")
        click.echo(f"  Save frames: {save_frames if save_frames is not None else 'auto'}")
        click.echo(
            "  Debug frames: "
            + (", ".join(str(frame) for frame in debug_frames) if debug_frames else "none")
        )
        click.echo(
            "  Extra kwargs: "
            + (
                ", ".join(
                    f"{key}={value}" for key, value in sorted(kwargs_dict.items())
                )
                if kwargs_dict
                else "none"
            )
        )
        return
    # end if

    assert console is not None
    assert RichTable is not None
    assert RichPanel is not None

    table_class = cast(Any, RichTable)
    panel_class = cast(Any, RichPanel)

    table = table_class(show_header=False, box=None, pad_edge=False)
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="white")

    debug_frames_value = ", ".join(str(frame) for frame in debug_frames)
    if not debug_frames_value:
        debug_frames_value = "none"
    # end if

    kwargs_value = ", ".join(
        f"{key}={value}" for key, value in sorted(kwargs_dict.items())
    )
    if not kwargs_value:
        kwargs_value = "none"
    # end if

    table.add_row("Animation class", f"{class_name} ({class_file})")
    table.add_row("Output", output)
    table.add_row("Input", input_path or "none")
    table.add_row("Duration", str(duration) if duration is not None else "auto")
    table.add_row("FPS", str(fps) if fps is not None else "source/default")
    table.add_row("Resolution", f"{width}x{height}")
    table.add_row("Display", "yes" if display else "no")
    table.add_row("Save frames", str(save_frames) if save_frames is not None else "auto")
    table.add_row("Debug frames", debug_frames_value)
    table.add_row("Extra kwargs", kwargs_value)

    panel = panel_class(
        table,
        title="[bold magenta]PixelPrism Animation[/bold magenta]",
        border_style="magenta",
    )
    console.print(panel)
# end def _display_animation_parameters


@click.command("animate", help="Render a custom animation class to a video file.")
@click.argument("output", type=click.Path())
@click.option("--input", "input_path", type=click.Path(exists=True), help="Path to an optional input video.")
@click.option("--display", is_flag=True, help="Display the video while processing.")
@click.option("--debug-frames", type=int, multiple=True, help="Frame indices for debug layer snapshots.")
@click.option("--class-file", required=True, type=click.Path(exists=True), help="Path to the file containing the animation class.")
@click.option("--class-name", required=True, help="Name of the animation class.")
@click.option("--duration", type=float, help="Duration of the output video in seconds.")
@click.option("--fps", type=int, help="Frames per second.")
@click.option("--width", type=int, default=1920, show_default=True, help="Output width.")
@click.option("--height", type=int, default=1080, show_default=True, help="Output height.")
@click.option("--save-frames", type=_bool_arg, help="Whether to save intermediate frames.")
@click.option("--kwargs", multiple=True, help="Additional animation constructor kwargs as key=value.")
def animate(
    output: str,
    input_path: str | None,
    display: bool,
    debug_frames: tuple[int, ...],
    class_file: str,
    class_name: str,
    duration: float | None,
    fps: int | None,
    width: int,
    height: int,
    save_frames: bool | None,
    kwargs: tuple[str, ...],
) -> None:
    """Render a procedural animation.

    Parameters
    ----------
    output : str
        Output video path.
    input_path : str | None
        Optional input video path used by some animation classes.
    display : bool
        Whether to run the viewer during rendering.
    debug_frames : tuple[int, ...]
        Frame numbers for which debug layers are saved.
    class_file : str
        Python file that defines the custom animation class.
    class_name : str
        Name of the custom animation class to load.
    duration : float | None
        Video duration in seconds when no input video is provided.
    fps : int | None
        Frames-per-second target.
    width : int
        Output width in pixels.
    height : int
        Output height in pixels.
    save_frames : bool | None
        Optional flag to persist rendered frames on disk.
    kwargs : tuple[str, ...]
        Additional key-value arguments passed to the animation constructor.

    Raises
    ------
    click.UsageError
        If ``duration`` is missing while no input video is supplied.
    """
    if not input_path and not duration:
        raise click.UsageError(
            "Duration (--duration) is required if no input video is specified."
        )
    # end if

    custom_animation_class = _load_class_from_file(class_file, class_name)
    kwargs_dict = _parse_kwargs(kwargs)

    _display_animation_parameters(
        output=output,
        input_path=input_path,
        display=display,
        debug_frames=debug_frames,
        class_file=class_file,
        class_name=class_name,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        save_frames=save_frames,
        kwargs_dict=kwargs_dict,
    )

    duration_arg: Any = duration
    fps_arg: Any = fps
    debug_frames_arg: Any = debug_frames
    save_frames_arg: Any = save_frames

    composer = VideoComposer(
        input_path=input_path,
        output_path=output,
        duration=duration_arg,  # pyright: ignore[reportArgumentType]
        fps=fps_arg,  # pyright: ignore[reportArgumentType]
        width=width,
        height=height,
        animation_class=custom_animation_class,
        debug_frames=debug_frames_arg,  # pyright: ignore[reportArgumentType]
        save_frames=save_frames_arg,  # pyright: ignore[reportArgumentType]
        viewer=display,
        **kwargs_dict,
    )

    composer.create_video()
# end def animate
