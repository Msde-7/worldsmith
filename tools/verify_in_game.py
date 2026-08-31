"""Ground truth: generate a world with the real Minecraft server, then compare
what it stored with what the worldsmith engine predicted.

Two passes. The heightmaps say the terrain is the right shape. The surface
blocks say the surface rules put the right thing on top of it, which is the half
that used to go unmeasured: a pack can match the game column for column on
height and still be banded and cliffed wrong.

This is the only check that proves the whole chain: the datapack loads, the game
accepts every field, and the world the engine drew is the world the game
builds.

    python tools/verify_in_game.py packs/basalt_spires
    python tools/verify_in_game.py packs/red_canyons --sample 200

The server jar and Java runtime come from the same .runtime cache `worldsmith
play` uses and are downloaded on first use; --java and --jar override them.
Running the server writes an eula.txt, which accepts Mojang's EULA on your
behalf, so this is opt-in and everything lands in a scratch directory you can
delete.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldsmith.anvil import (read_region, region_dir, section_blocks,
                             unpack_heightmap)
from worldsmith.climate import BiomeSource
from worldsmith.play import (RUNTIME, as_overworld, drain, ensure_runtime,
                              send, shut_down, wait_for_ready)
from worldsmith.registry import Pack, Registries
from worldsmith.scene import build_scene
from worldsmith.terrain import sample_terrain
from worldsmith.world import World

# the block pass builds a scene per chunk, so it looks at fewer than the
# heightmap pass, which reads one array
BLOCK_CHUNKS = 24


def is_decorated(registries: Registries, pack: Pack) -> bool:
    """Whether the pack's biomes place features, which land on top of the surface."""
    for ident in pack.data["biome"]:
        biome = registries.get("biome", ident) or {}
        if any(step for step in (biome.get("features") or [])):
            return True
    return False


def run_server(java: Path, jar: Path, work: Path, pack: Path, seed: int,
               start_timeout: int = 900, generate_timeout: int = 420) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    (work / "eula.txt").write_text("eula=true\n", encoding="utf-8")
    (work / "server.properties").write_text("\n".join([
        f"level-seed={seed}", "level-name=world", "online-mode=false", "max-tick-time=-1",
        "sync-chunk-writes=true", "view-distance=10", "simulation-distance=4",
        "spawn-protection=0", "generate-structures=false", "allow-nether=false",
        "enable-jmx-monitoring=false", "",
    ]), encoding="utf-8")
    datapacks = work / "world" / "datapacks"
    datapacks.mkdir(parents=True, exist_ok=True)
    target = datapacks / pack.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(pack, target)

    print(f"starting {jar.name} with {pack.name} ...")
    proc = subprocess.Popen(
        [str(java), "-Xmx2G", "-jar", str(jar), "nogui"],
        cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    started = time.time()
    lines: list[str] = []
    try:
        _drive_server(proc, work, lines, started, start_timeout, generate_timeout)
    finally:
        shut_down(proc)
        (work / "server.log").write_text("\n".join(lines), encoding="utf-8")
    return work / "world"


def _drive_server(proc, work: Path, lines: list[str], started: float,
                  start_timeout: int, generate_timeout: int) -> None:
    if not wait_for_ready(proc, lines, start_timeout,
                          ("error", "failed", "exception"),
                          lambda text: print("  " + text)):
        raise SystemExit("server never reported ready:\n" + "\n".join(lines[-40:]))
    print(f"  server up in {time.time() - started:.0f}s; force-loading chunks")
    drain(proc, lines)

    # Modern servers barely pre-generate anything, so ask explicitly. forceload
    # takes at most 256 chunks per command, hence the 256-block tiles.
    for cz in range(-16, 16, 16):
        for cx in range(-16, 16, 16):
            send(proc, f"forceload add {cx * 16} {cz * 16} "
                       f"{(cx + 15) * 16 + 15} {(cz + 15) * 16 + 15}")

    deadline = time.time() + generate_timeout
    stable, last = 0, -1
    while time.time() < deadline:
        time.sleep(6)
        send(proc, "save-all flush")
        time.sleep(4)
        # resolved every pass: the game only writes the nested layout once it
        # has a chunk to save, so this directory does not exist yet at the start
        region_root = region_dir(work / "world")
        size = (sum(f.stat().st_size for f in region_root.glob("*.mca"))
                if region_root.is_dir() else 0)
        print(f"  region data: {size / 1024:.0f} KB")
        if size == last and size > 0:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        last = size

    send(proc, "save-all flush")
    time.sleep(3)


def compare(world_dir: Path, pack: Path, seed: int, sample: int = 0) -> int:
    registries = Registries.load([str(pack)])
    dimension = registries.get("dimension", "minecraft:overworld")
    generator = dimension.get("generator") or {}
    engine = World.create(registries, generator["settings"], seed)
    try:
        source = BiomeSource.from_json(generator.get("biome_source") or {}, registries)
    except ValueError as exc:
        print(f"note: {exc}; skipping the surface block pass")
        source = None
    decorated = is_decorated(registries, Pack(pack))

    region_root = region_dir(world_dir)
    files = sorted(region_root.glob("*.mca"))
    if not files:
        raise SystemExit(f"no region files under {region_root}")

    compared = skipped = total = exact = 0
    block_chunks = block_seen = block_match = 0
    diffs: list[int] = []
    worst: list[tuple[int, int, int, int]] = []
    block_worst: list[tuple] = []
    for path in files:
        if sample and compared >= sample:
            break
        chunks = read_region(path)
        # A chunk only carries its final heightmap once it reaches `full`.
        # Including half-generated ones is what makes the numbers look worse
        # than the engine actually is.
        ready = {pos: nbt for pos, nbt in chunks.items()
                 if str(nbt.get("Status", "")).split(":")[-1] == "full"}
        skipped += len(chunks) - len(ready)
        print(f"  {path.name}: {len(ready)} finished chunks of {len(chunks)}")
        for (cx, cz), nbt in sorted(ready.items()):
            if sample and compared >= sample:
                break
            packed = (nbt.get("Heightmaps") or {}).get("OCEAN_FLOOR")
            if packed is None or len(packed) == 0:
                continue
            game = unpack_heightmap(np.asarray(packed), engine.noise.height) + engine.noise.min_y
            terrain = sample_terrain(engine, cx * 16, cz * 16, 16, 16, step=1)
            # OCEAN_FLOOR stores the y of the first non-solid block, relative to min_y
            mine = terrain.surface_y + 1
            delta = (mine - game).astype(np.int64)
            compared += 1
            total += delta.size
            exact += int((delta == 0).sum())
            diffs.extend(delta.ravel().tolist())
            if np.abs(delta).max() > 0:
                iz, ix = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
                worst.append((cx * 16 + int(ix), cz * 16 + int(iz),
                              int(mine[iz, ix]), int(game[iz, ix])))

            if source is not None and block_chunks < BLOCK_CHUNKS:
                block_chunks += 1
                tops = game - 1
                layers = section_blocks(nbt, int(tops.min()), int(tops.max()))
                scene = build_scene(engine, source, cx * 16, cz * 16, 16, 16, step=1)
                ours = np.array(scene.palette, dtype=object)[scene.surface_block]
                for iz in range(16):
                    for ix in range(16):
                        y = int(game[iz, ix]) - 1          # first non-solid, less one
                        layer = layers.get(y)
                        if layer is None or y < engine.sea_level - 1:
                            continue                        # the engine paints fluid under water
                        want, got = str(layer[iz, ix]), str(ours[iz, ix])
                        block_seen += 1
                        if want == got:
                            block_match += 1
                        elif len(block_worst) < 6:
                            block_worst.append((cx * 16 + ix, cz * 16 + iz, y, got, want))
    if total == 0:
        raise SystemExit("no finished chunks with heightmaps in the generated region files")

    arr = np.array(diffs)
    print()
    print(f"chunks compared  : {compared}")
    print(f"columns compared : {total}")
    print(f"exact matches    : {exact} ({exact / total * 100:.3f}%)")
    print(f"within 1 block   : {int((np.abs(arr) <= 1).sum())} ({(np.abs(arr) <= 1).mean() * 100:.3f}%)")
    print(f"mean |error|     : {np.abs(arr).mean():.4f} blocks")
    print(f"max  |error|     : {np.abs(arr).max()} blocks")
    if skipped:
        print(f"chunks skipped   : {skipped} (still generating)")
    if worst:
        print("worst columns (x, z, engine, game):")
        for entry in sorted(worst, key=lambda e: -abs(e[2] - e[3]))[:8]:
            print(f"    {entry}")

    blocks_ok = True
    if block_seen:
        rate = block_match / block_seen
        print()
        print(f"surface blocks   : {block_match} of {block_seen} ({rate * 100:.3f}%) "
              f"over {block_chunks} chunks")
        if decorated:
            print("                   this pack places features, which the game puts on top of "
                  "the surface and the engine never draws, so some of the gap is theirs")
        else:
            # the rules are deterministic, so anything short of agreement is a
            # disagreement. Heights carry aquifer ties, blocks do not.
            blocks_ok = rate > 0.999
        for x, z, y, got, want in block_worst:
            print(f"    ({x}, {z}) y={y}  engine {got}  game {want}")

    return 0 if exact / total > 0.99 and blocks_ok else 1


def stage(pack: Path, registries: Registries, dimension: str | None, work: Path) -> Path:
    """The server only generates the overworld, so the dimension under test is
    written out as minecraft:overworld first."""
    if registries.origin("dimension", "minecraft:overworld") == pack.name:
        dim_id = "minecraft:overworld"
    else:
        custom = [i for i in registries.ids("dimension")
                  if registries.origin("dimension", i) == pack.name]
        dim_id = dimension or (custom[0] if len(custom) == 1 else None)
        if dim_id is None:
            raise SystemExit(f"pick a dimension with --dimension: {', '.join(custom) or 'none found'}")
    return as_overworld(pack, dim_id, registries, work / "pack")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mc-version", default="26.2")
    ap.add_argument("--java", default=None, help="java binary (default: the cached runtime)")
    ap.add_argument("--jar", default=None, help="server jar (default: the cached runtime)")
    ap.add_argument("--work", default=None, help="scratch directory (default: .runtime/verify/<pack>)")
    ap.add_argument("--dimension", default=None,
                    help="dimension to generate as the overworld (default: the pack's only one)")
    ap.add_argument("--sample", type=int, default=0,
                    help="stop after this many chunks instead of comparing every one")
    ap.add_argument("--reuse", action="store_true", help="compare an existing world without regenerating")
    args = ap.parse_args(argv)

    pack = Path(args.pack).resolve()
    work = Path(args.work) if args.work else RUNTIME / "verify" / pack.name
    if not args.reuse and work.exists():
        shutil.rmtree(work)
    staged = stage(pack, Registries.load([str(pack)]), args.dimension, work)
    if args.reuse:
        return compare(work / "world", staged, args.seed, args.sample)

    runtime = None if (args.java and args.jar) else ensure_runtime(args.mc_version)
    java = Path(args.java) if args.java else runtime.java
    jar = Path(args.jar) if args.jar else runtime.jar
    return compare(run_server(java, jar, work, staged, args.seed), staged, args.seed, args.sample)


if __name__ == "__main__":
    raise SystemExit(main())
