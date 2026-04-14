"""Convenience script to run project test suite."""

from __future__ import annotations

import sys

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "tests"]))
