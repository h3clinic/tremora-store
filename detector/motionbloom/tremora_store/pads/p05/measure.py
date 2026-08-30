"""Entry point for a measurement process.

Runs the timing and exits.  It never reads the table back, so the read-back's
allocation is never resident alongside four open representations and a
three-hour run.
"""

from __future__ import annotations

from .audit import measure_main

if __name__ == "__main__":
    raise SystemExit(measure_main())
