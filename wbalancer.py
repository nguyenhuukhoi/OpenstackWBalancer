#!/usr/bin/env python3
"""Supported source-checkout entrypoint; installed entrypoints use the same CLI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from compute_maintenance.cli.entry import balance

if __name__ == "__main__":
    balance()
