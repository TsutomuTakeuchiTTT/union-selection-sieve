#!/usr/bin/env python3
"""Translate a JSON configuration into confirmatory command-line arguments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from union_selection_sieve.cli import confirmatory


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _arguments(config: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = _flag(key)
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
        else:
            args.extend([flag, str(value)])
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parsed = parser.parse_args()
    payload = json.loads(parsed.config.read_text(encoding="utf-8"))
    raise SystemExit(confirmatory(_arguments(payload)))
