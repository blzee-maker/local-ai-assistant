#!/usr/bin/env python
"""Convenience launcher: `python assistant.py chat`.

Identical to `python -m app.cli`; this just saves the module flag.
"""
from app.cli.main import entrypoint

if __name__ == "__main__":
    entrypoint()
