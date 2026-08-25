"""Vendor the worldgen data that mcmeta cannot ship, out of the server jar.

Two things worldsmith needs are not in a datapack anywhere, because the game
builds them in Java at startup:

**The preset biome tables.** A dimension may name a preset rather than list its
biomes, which is what copying vanilla's overworld gives you:

    "biome_source": {"type": "minecraft:multi_noise", "preset": "minecraft:overworld"}

OverworldBiomeBuilder assembles that table in code, so mcmeta's copy of
multi_noise_biome_source_parameter_list/overworld.json is the preset name again
and there is nothing to place. Without it such a pack previews as bare terrain.

**The features.** Trees are what make a jungle look like a jungle, and the
biomes name them, but the placed and configured features behind those names are
registry objects rather than files. `render --decorate` walks them to work out
how much canopy a biome ends up under.

    python tools/extract_worldgen_data.py [--version 26.2]

The server jar can print both. It needs a Java runtime, and both are cached in
.runtime/ and shared with `worldsmith play`. Everything written here is Mojang's
data like the rest of vanilla/<version>/, so re-vendoring from mcmeta puts the
stubs back and this has to run again.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worldsmith.play import RUNTIME, ensure_runtime  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRESETS = "worldgen/multi_noise_biome_source_parameter_list"
FEATURES = ("worldgen/placed_feature", "worldgen/configured_feature")
PARAMS = ("temperature", "humidity", "continentalness", "erosion", "depth", "weirdness", "offset")


def generate(version: str) -> Path:
    """Run the data generator. Returns its output directory."""
    runtime = ensure_runtime(version)
    # The bundled server unpacks libraries/ and versions/ into the working
    # directory, so run it under .runtime/ and not in the repository root.
    work = RUNTIME / "datagen"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "generated"
    print(f"    running the {version} data generator ...", flush=True)
    proc = subprocess.run(
        [str(runtime.java), "-DbundlerMainClass=net.minecraft.data.Main",
         "-jar", str(runtime.jar), "--reports", "--server", "--output", str(out)],
        cwd=work, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write((proc.stdout or "")[-4000:])
        sys.stderr.write((proc.stderr or "")[-4000:])
        raise SystemExit(f"the data generator exited {proc.returncode}")
    return out


def write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size


def vendor_presets(out: Path, dest: Path) -> None:
    """The biome tables, from the Biome Parameters report."""
    reports = out / "reports" / "biome_parameters" / "minecraft"
    if not reports.is_dir():
        raise SystemExit(f"the generator wrote no biome parameters under {out}")
    for path in sorted(reports.glob("*.json")):
        entries = json.loads(path.read_text(encoding="utf-8")).get("biomes")
        if not isinstance(entries, list) or not entries:
            raise SystemExit(f"{path.stem}: the report has no 'biomes' list")
        for i, entry in enumerate(entries):
            params = entry.get("parameters") if isinstance(entry, dict) else None
            if not isinstance(params, dict) or not entry.get("biome"):
                raise SystemExit(f"{path.stem}: entry {i} is not a biome with parameters")
            missing = [p for p in PARAMS if p not in params]
            if missing:
                raise SystemExit(f"{path.stem}: entry {i} is missing {', '.join(missing)}")
        # vanilla's own format, so it loads as an ordinary registry entry and a
        # pack can ship one of its own the same way
        size = write(dest / PRESETS / path.name, {"biomes": entries})
        unique = len({e["biome"] for e in entries})
        print(f"  {path.stem}: {len(entries)} entries, {unique} biomes, {size / 1024:.0f} KB")


def vendor_features(out: Path, dest: Path) -> None:
    """The placed and configured features, from the server generators."""
    for category in FEATURES:
        source = out / "data" / "minecraft" / category
        if not source.is_dir():
            raise SystemExit(f"the generator wrote no {category} under {out}")
        total = count = 0
        for path in sorted(source.rglob("*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            total += write(dest / category / path.relative_to(source), obj)
            count += 1
        print(f"  {category.split('/')[-1]}: {count} files, {total / 1024:.0f} KB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="26.2")
    args = ap.parse_args()

    dest = ROOT / "vanilla" / args.version / "data" / "minecraft"
    if not dest.is_dir():
        raise SystemExit(f"no vendored vanilla data at {dest}")

    out = generate(args.version)
    vendor_presets(out, dest)
    vendor_features(out, dest)
    print(f"-> {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
