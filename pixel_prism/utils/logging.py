#
# This file contains the CustomFormatter class which is a custom formatter for the logging module.
#

# Imports
import logging
import os
from colorama import init, Fore, Style

# Initialize colorama
init()


class CustomFormatter(logging.Formatter):
    """Custom Formatter for adding colors and justifying the filename"""

    # Define color for different log levels
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + "%(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "%(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(levelname)s: %(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + "%(levelname)s: %(message)s" + Style.RESET_ALL,
    }

    def format(
            self,
            record
    ):
        """
        Format the log record
        """
        # Get the original format
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatted_message = formatter.format(record)

        # Add filename right justified
        filename = os.path.basename(record.pathname)
        justified_filename = f"{filename:>30}"  # Adjust the width as needed

        return f"{formatted_message} {justified_filename}"
    # end format

# end CustomFormatter


def setup_logger(name):
    """
    Setup logger with custom formatter and color.
    Args:
        name (str): The name of the logger.
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = CustomFormatter()
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger
# end setup_logger

# Example usage:
# logger = setup_logger(__name__)
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")

