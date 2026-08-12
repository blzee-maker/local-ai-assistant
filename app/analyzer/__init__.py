"""Read-only disk analysis: duplicates, format integrity, and idle storage.

Nothing in this package writes to, moves, or deletes a user's files.
"""
from .service import ScanReport, human_bytes, run_scan

__all__ = ["ScanReport", "run_scan", "human_bytes"]
