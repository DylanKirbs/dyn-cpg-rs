import logging
from termcolor import colored
import re


class ColouredFormatter(logging.Formatter):
    """
    A logging formatter that highlights specified text.

    :author: D. Kirby
    """

    def __init__(self, *args, highlights: dict = {}, **kwargs):
        """
        Initialize the formatter.

        Example usage:
        >>> formatter = ColouredFormatter('%(levelname)-8s | %(message)s', highlights={r'INFO': 'green'})

        :param highlights: The dictionary of regular expressions and colours to highlight them, defaults to {}
        :type highlights: dict, optional

        The args and kwargs are passed to the superclass. See [`logging.Formatter`](https://docs.python.org/3/library/logging.html#logging.Formatter) for more information.
        """
        super().__init__(*args, **kwargs)

        self.highlights = {
            re.compile(rf"({text})", re.IGNORECASE): colour
            for text, colour in highlights.items()
        }

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        for exp, colour in self.highlights.items():
            out = re.sub(exp, lambda m: colored(m.group(0), colour), out)
        return out


# Setup basic logging
def attach_basic_colour_formatter():
    formatter = ColouredFormatter(
        "%(levelname)-8s | %(message)s",
        highlights={
            # Log levels
            r"DEBUG": "cyan",
            r"INFO": "green",
            r"WARNING": "yellow",
            r"ERROR|EXCEPTION|CRITICAL": "red",
            # Misc
            r"EVENT": "blue",
            r" \S+ -> {.*}": "blue",
            r"\S+ NOT YET IMPLEMENTED": "yellow",
        },
    )
    logging.getLogger().handlers[0].setFormatter(formatter)
