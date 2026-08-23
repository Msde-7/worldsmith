"""Run every offline test suite and print one summary.

    python tests/run_all.py

The in-game verification (tools/verify_in_game.py) is deliberately not included:
it downloads nothing but does need a Java 25 runtime and a Minecraft server jar,
and it accepts Mojang's EULA in a scratch directory.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    ("conformance vs deepslate + the JVM", "test_against_deepslate.py"),
    ("engine, validator and packs", "test_engine.py"),
]


def main() -> int:
    failed = 0
    for label, script in SUITES:
        print(f"--- {label} ({script})")
        started = time.time()
        result = subprocess.run([sys.executable, os.path.join(HERE, script)],
                                cwd=os.path.dirname(HERE), capture_output=True, text=True)
        out = (result.stdout or "").strip()
        print(out)
        if result.returncode:
            failed += 1
            err = (result.stderr or "").strip()
            if err:
                print(err[-2000:])
        print(f"    {time.time() - started:.1f}s\n")
    print("ALL SUITES PASSED" if not failed else f"{failed} SUITE(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
