"""Entry point for a summarizer process.

A fresh process that reads a finished timing table and produces the run's
evidence.  Nothing here was resident while the timing ran.
"""

from __future__ import annotations

from .audit import summarize_main

if __name__ == "__main__":
    raise SystemExit(summarize_main())
