"""Check the placement model against a real server.

worldsmith.placement claims to know where the game will put a build, and which
sites the game will reject. This puts that claim in front of Minecraft.

The probe is a datapack with no dimension of its own: one build, spread as
densely as the game allows, dropped into an ordinary vanilla world. That is both
the strictest test of the biome predicate (vanilla has sixty-odd biomes and the
probe accepts ten of them) and the plainest use of worldsmith's build half on
its own, with the terrain half not involved at all.

Like tools/verify_in_game.py this downloads a server jar and a Java runtime into
.runtime/ on first use, and accepts Mojang's EULA there.

    python tools/verify_placement.py                 # a one block build
    python tools/verify_placement.py --size 64       # a big one
    python tools/verify_placement.py --reuse         # compare the last run again
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from worldsmith import play as play_mod, structures                    # noqa: E402
from worldsmith.anvil import read_world                                # noqa: E402
from worldsmith.climate import BiomeSource                             # noqa: E402
from worldsmith.pack import PackWriter                                 # noqa: E402
from worldsmith.placement import set_sites, survey                     # noqa: E402
from worldsmith.registry import Registries                             # noqa: E402
from worldsmith.voxel import Grid                                      # noqa: E402
from worldsmith.world import World                                     # noqa: E402

NS = "probe"
BUILD = f"{NS}:build"
SET = f"{NS}:builds"
# ten of the sixty-odd overworld biomes, so most sites are rejected and the
# biome predicate is what decides
BIOMES = ["minecraft:plains", "minecraft:forest", "minecraft:birch_forest",
          "minecraft:dark_forest", "minecraft:taiga", "minecraft:savanna",
          "minecraft:desert", "minecraft:snowy_plains", "minecraft:jungle",
          "minecraft:swamp"]


def build_probe(root: Path, size: int, spacing: int, separation: int) -> int:
    """A build of `size` blocks square, one block tall, sunk to sit on the ground."""
    if root.exists():
        shutil.rmtree(root)
    writer = PackWriter(root, "placement probe", "26.2")
    writer.mcmeta()
    grid = Grid(size, 1, size)
    grid.fill(0, 0, 0, size - 1, 0, size - 1, "minecraft:gold_block")
    sink = -1
    structures.add(writer, BUILD, grid, BIOMES, sink=sink)
    writer.add("structure_set", SET,
               structures.spread(BUILD, spacing=spacing, separation=separation,
                                 salt=770193))
    return sink


def compare(pack: Path, world: Path, seed: int, sink: int, size: int) -> int:
    registries = Registries.load([str(pack)])
    model = World.create(registries, "minecraft:overworld", seed)
    source = BiomeSource.from_json(
        {"type": "minecraft:multi_noise", "preset": "minecraft:overworld"}, registries)

    built, evaluated = set(), set()
    for key, chunk in read_world(world):
        evaluated.add(key)
        if BUILD in ((chunk.get("structures") or {}).get("starts") or {}):
            built.add(key)
    if not evaluated:
        raise SystemExit(f"no chunks under {world}")

    xs = [c[0] for c in evaluated]
    zs = [c[1] for c in evaluated]
    box = (min(xs) * 16, min(zs) * 16, max(xs) * 16 + 15, max(zs) * 16 + 15)
    found = [s for s in set_sites(registries, SET, seed, *box)
             if (s.chunk_x, s.chunk_z) in evaluated]
    reports = survey(model, source, found, seed=seed, biomes=BIOMES, sink=sink,
                     size=(size, size), step=8)

    predicted = {(r.site.chunk_x, r.site.chunk_z) for r in reports if r.accepted}
    missed = sorted(built - predicted)
    invented = sorted(predicted - built)
    agree = len(found) - len(missed) - len(invented)
    print(f"{len(evaluated)} chunks generated, {len(found)} sites in range")
    print(f"  server built  {len(built)}")
    print(f"  model expects {len(predicted)}")
    print(f"  agreement {agree}/{len(found)} sites "
          f"({100.0 * agree / max(1, len(found)):.2f}%)")

    by_chunk = {(r.site.chunk_x, r.site.chunk_z): r for r in reports}
    for label, group in (("built, model said no", missed),
                         ("model said yes, not built", invented)):
        if not group:
            continue
        print(f"  {label}: {len(group)}")
        for chunk in group[:12]:
            report = by_chunk.get(chunk)
            detail = (f"model biome {report.biome}, surface y {report.surface_y}"
                      if report else "not a site in the model at all")
            print(f"      {chunk}  {detail}")
    return 0 if not missed and not invented else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1, help="build footprint in blocks")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--spacing", type=int, default=4)
    parser.add_argument("--separation", type=int, default=1)
    parser.add_argument("--radius", type=int, default=384)
    parser.add_argument("--pregen", type=int, default=120)
    parser.add_argument("--reuse", action="store_true", help="skip building the world")
    args = parser.parse_args()

    pack = ROOT / "packs" / "_placement_probe"
    work = play_mod.RUNTIME / "verify" / "placement"
    sink = build_probe(pack, args.size, args.spacing, args.separation)
    if not args.reuse:
        print(f"probing with a {args.size}x{args.size} build, spacing {args.spacing}, "
              f"separation {args.separation}, seed {args.seed}")
        runtime = play_mod.ensure_runtime("26.2")
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        spawn = play_mod.Viewpoint(0, 0, 80, None, "probe")
        play_mod.generate_world(runtime, work, pack, args.seed, spawn,
                                args.radius, "creative", args.pregen)
    return compare(pack, work / "world", args.seed, sink, args.size)


if __name__ == "__main__":
    raise SystemExit(main())
