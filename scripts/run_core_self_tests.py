#!/usr/bin/env python3
"""Execute the analytic/operator self-tests embedded in the validated core."""
from __future__ import annotations
import json
from union_selection_sieve import run_self_tests

if __name__ == "__main__":
    result = run_self_tests()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result.get("all_passed", False) else 1)
