"""Generate the checked-in RNG oracle fixture with a real Lua interpreter.

Usage:
    uv run --no-sync python scripts/generate_fixtures/run_rng_oracle.py \
        --lua-executable luajit --seed TESTSEED
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_SCRIPT = PROJECT_ROOT / "scripts" / "lua_rng_oracle.lua"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lua-executable", required=True)
    parser.add_argument("--seed", default="TESTSEED")
    args = parser.parse_args()

    completed = subprocess.run(
        [args.lua_executable, str(ORACLE_SCRIPT), args.seed],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    wrapper = json.loads(completed.stdout)
    fixture = wrapper["seeds"][0]
    fixture["lua_version"] = wrapper["lua_version"]
    fixture["note"] = wrapper["note"]

    output_path = FIXTURE_DIR / f"rng_oracle_{args.seed}.json"
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(output_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
