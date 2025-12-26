"""
CHAOS module entry point.

Allows running as: python -m chaos mydata.ms [options]
"""

from .cli import main

if __name__ == "__main__":
    main()
