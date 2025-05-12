import logging
from termcolor import colored


class ColouredFormatter(logging.Formatter):
    """
    A logging formatter that highlights specified text.

    :author: D. Kirby
    """

    def __init__(self, *args, highlights: dict = {}, **kwargs):
        """
        Initialize the formatter.

        Example usage:
        >>> formatter = ColouredFormatter('%(levelname)-8s| %(message)s', highlights={'INFO': 'green'})

        :param highlights: The dictionary of words and colours to highlight them, defaults to {}
        :type highlights: dict, optional

        The args and kwargs are passed to the superclass. See [`logging.Formatter`](https://docs.python.org/3/library/logging.html#logging.Formatter) for more information.
        """
        super().__init__(*args, **kwargs)

        # Precompute the replacements
        self.replacer = {}
        for text, color in highlights.items():
            self.replacer[text] = colored(text, color)

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        for text, highlight in self.replacer.items():
            out = out.replace(text, highlight)
        return out


# Setup basic logging
def attach_basic_colour_formatter():
    formatter = ColouredFormatter(
        "%(levelname)-8s| %(message)s",
        highlights={
            # Log levels
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "EXCEPTION": "red",
            "CRITICAL": "red",
            # Misc
            "EVENT": "blue",
        },
    )
    logging.getLogger().handlers[0].setFormatter(formatter)
