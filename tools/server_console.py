"""Boot the generated world's server and run commands against it.

The quickest way to find out why a structure is not appearing is to ask the game
to place one and read what it says back.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from worldsmith.play import (SERVER_READY, drain, ensure_runtime,  # noqa: E402
                             send, shut_down, wait_for_ready)


def run(work: Path, commands: list[str], settle: float = 6.0,
        start_timeout: int = 900) -> list[str]:
    """Boot the server in `work` and feed it commands. The directory needs the
    world in it; the EULA and properties are written here if they are missing,
    so this also works on one no other tool has been through."""
    work.mkdir(parents=True, exist_ok=True)
    eula = work / "eula.txt"
    if not eula.is_file():
        eula.write_text("eula=true\n", encoding="utf-8")
    properties = work / "server.properties"
    if not properties.is_file():
        properties.write_text("level-name=world\nonline-mode=false\n"
                              "max-tick-time=-1\nspawn-protection=0\n",
                              encoding="utf-8")

    runtime = ensure_runtime("26.2")
    proc = subprocess.Popen(
        [str(runtime.java), "-Xmx2G", "-jar", str(runtime.jar), "nogui"],
        cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    try:
        if not wait_for_ready(proc, lines, start_timeout):
            raise SystemExit("server never reported ready:\n" + "\n".join(lines[-40:]))
        drain(proc, lines)
        for command in commands:
            print(f"> {command}", flush=True)
            send(proc, command)
            time.sleep(settle)
        send(proc, "save-all flush")
        time.sleep(4)
    finally:
        shut_down(proc)
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python tools/server_console.py <server dir> [command ...]")
        return 2
    output = run(Path(argv[0]), argv[1:])
    start = next((i for i, line in enumerate(output)
                  if SERVER_READY in line.lower()), 0)
    for line in output[start:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
