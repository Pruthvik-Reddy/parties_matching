"""Backward-compatible entry point for complete current suggestions.

Use ``python -m suggestions.current`` for the clearer module name. Unlike the
earlier demo, neither entry point limits the number of exported suggestions.
"""

from .current import main


if __name__ == "__main__":
    main()
