"""Boot the generated world's server and run commands against it.

The quickest way to find out why a structure is not appearing is to ask the game
to place one and read what it says back.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from worldsmith.play import ensure_runtime          # noqa: E402


def run(work: Path, commands: list[str], settle: float = 6.0) -> list[str]:
    runtime = ensure_runtime("26.2")
    proc = subprocess.Popen(
        [str(runtime.java), "-Xmx2G", "-jar", str(runtime.jar), "nogui"],
        cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        if 'for help, type "help"' in line.lower():
            break
    threading.Thread(target=lambda: [lines.append(t.rstrip()) for t in proc.stdout],
                     daemon=True).start()
    for command in commands:
        print(f"> {command}", flush=True)
        proc.stdin.write(command + "\n")
        proc.stdin.flush()
        time.sleep(settle)
    proc.stdin.write("save-all flush\n")
    proc.stdin.flush()
    time.sleep(4)
    proc.stdin.write("stop\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
    return lines


if __name__ == "__main__":
    work = Path(sys.argv[1])
    commands = sys.argv[2:]
    output = run(work, commands)
    start = next((i for i, line in enumerate(output)
                  if 'for help, type "help"' in line.lower()), 0)
    for line in output[start:]:
        print(line)
