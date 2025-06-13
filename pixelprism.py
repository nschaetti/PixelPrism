
# Imports
import importlib.util
import click

from pixel_prism import VideoComposer
from pixel_prism.utils import setup_logger


# Setup logger
logger = setup_logger(__name__)


def load_class_from_file(
        file_path,
        class_name
):
    """
    Load a class from a file.

    Args:
        file_path (str): Path to the file containing the class
        class_name (str): Name of the class to load
    """
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
# end load_class_from_file


# Boolean argument
def bool_arg(value):
    """
    Parse a boolean argument from the command line.

    Args:
        value (str): Value to parse

    Returns:
        bool: Parsed boolean value
    """
    if type(value) is bool:
        return value
    # end if
    return value.lower() in ("yes", "true", "t", "1")
# end bool_arg


@click.command(help="Apply visual effects to videos using PixelPrism.")
@click.argument("output", type=click.Path())
@click.option("--input", type=click.Path(exists=True), help="Path to the input video file")
@click.option("--display", is_flag=True, help="Display the video while processing")
@click.option("--debug-frames", type=int, multiple=True, help="List of frame numbers to debug")
@click.option("--class-file", required=True, type=click.Path(exists=True), help="Path to the file containing the CustomAnimation class")
@click.option("--class-name", required=True, help="Name of the CustomAnimation class to use")
@click.option("--duration", type=float, help="Duration of the animation in seconds.")
@click.option("--fps", type=int, help="Frames per second of the animation.")
@click.option("--width", type=int, default=1920, help="Width of the output video.")
@click.option("--height", type=int, default=1080, help="Height of the output video.")
@click.option("--save-frames", type=bool_arg, help="Save the frames to disk.")
@click.option("--kwargs", multiple=True, help="Additional keyword arguments for the CustomAnimation class in key=value format")
def main(
        output,
        input,
        display,
        debug_frames,
        class_file,
        class_name,
        duration,
        fps,
        width,
        height,
        save_frames,
        kwargs
):
    """
    Main entry point for the PixelPrism command-line interface.
    """
    # Show version
    logger.info("PixelPrism v0.1.0")

    # Check that duration is provided if no input video is given
    if not input and not duration:
        raise click.UsageError("Duration (--duration) is required if no input video is specified.")

    # Load the CustomAnimation class from the specified file
    CustomAnimationClass = load_class_from_file(class_file, class_name)

    # Parse keyword arguments
    kwargs_dict = {}
    if kwargs:
        for kwarg in kwargs:
            key, value = kwarg.split('=')
            kwargs_dict[key] = value

    # Log additional parameters
    logger.info(f"CustomAnimation class: {class_name}")
    logger.info(f"Duration: {duration}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Width: {width}")
    logger.info(f"Height: {height}")
    logger.info(f"Save frames: {save_frames}")
    logger.info(f"Keyword arguments: {kwargs_dict}")

    # Create video composer
    composer = VideoComposer(
        input_path=input,
        output_path=output,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        animation_class=CustomAnimationClass,
        debug_frames=debug_frames,
        save_frames=save_frames,
        viewer=display,
        **kwargs_dict
    )

    # Create video
    composer.create_video()
# end main


if __name__ == "__main__":
    main()  # This will invoke the click command
# end if
