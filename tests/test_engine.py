"""Internal consistency tests that do not need the deepslate golden files."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import worldsmith.noise as noise_mod
from worldsmith.climate import BiomeSource, unreachable_biomes
from worldsmith.density import prepare
from worldsmith.pack import scaffold
from worldsmith.registry import Registries
from worldsmith.terrain import base_height, cell_interpolated, sample_terrain
from worldsmith.validate import ERROR, Validator, validate_path
from worldsmith.world import World

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

failures: list[str] = []
checks = 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(f"{name}: {detail}")


def test_kernel_matches_numpy():
    """The numba kernel and the numpy path must agree bit for bit."""
    world = World.create(Registries.load(), "minecraft:overworld", 99)
    rng = np.random.default_rng(4)
    for noise_id in ("minecraft:continentalness", "minecraft:jagged", "minecraft:aquifer_barrier"):
        n = world.get_noise(noise_id)
        x = rng.uniform(-30000, 30000, 257)[None, :]
        z = rng.uniform(-30000, 30000, 257)[None, :]
        y = np.arange(-64, 320, 7, dtype=float)[:, None]
        noise_mod.USE_KERNEL[0] = True
        fast = n.sample(x, y, z)
        noise_mod.USE_KERNEL[0] = False
        slow = n.sample(x, y, z)
        noise_mod.USE_KERNEL[0] = True
        check(f"kernel == numpy for {noise_id}", np.array_equal(fast, slow),
              f"max diff {np.max(np.abs(fast - slow)):g}")


def test_sampling_modes_agree():
    """Where the root of final_density is an `interpolated` node the cheap lattice
    scan is exact and must equal the per-block scan. Vanilla is not such a world,
    because caves are min-ed in on top, so it must be detected as needing the
    per-block scan."""
    for pack, settings, expect_lattice in (
            (os.path.join(ROOT, "packs", "basalt_spires"), "spires:basalt_spires", True),
            (os.path.join(ROOT, "packs", "red_canyons"), "canyons:red_canyons", True),
            (None, "minecraft:overworld", False)):
        registries = Registries.load([pack] if pack else [])
        world = World.create(registries, settings, 2024)
        node = world.router["final_density"]
        prepare(node)
        check(f"{settings} lattice-exact == {expect_lattice}",
              cell_interpolated(node) == expect_lattice)
        if not expect_lattice:
            continue
        lattice = sample_terrain(world, -300, 220, 24, 24, step=7, sampling="lattice")
        block = sample_terrain(world, -300, 220, 24, 24, step=7, sampling="block")
        same = np.array_equal(lattice.surface_y, block.surface_y)
        diff = int(np.abs(lattice.surface_y - block.surface_y).max()) if not same else 0
        check(f"{settings} lattice == per-block scan", same, f"max diff {diff}")


def test_determinism():
    world_a = World.create(Registries.load(), "minecraft:overworld", 777)
    world_b = World.create(Registries.load(), "minecraft:overworld", 777)
    world_c = World.create(Registries.load(), "minecraft:overworld", 778)
    a = base_height(world_a, np.arange(0, 320, 20), np.full(16, 40))
    b = base_height(world_b, np.arange(0, 320, 20), np.full(16, 40))
    c = base_height(world_c, np.arange(0, 320, 20), np.full(16, 40))
    check("same seed gives identical terrain", np.array_equal(a, b))
    check("different seed gives different terrain", not np.array_equal(a, c))


def test_vanilla_validates_clean():
    registries = Registries.load()
    validator = Validator(registries, registries.packs[0])
    findings = validator.validate_pack()
    errors = [f for f in findings if f.level == ERROR]
    check("vanilla data has no validation errors", not errors,
          "; ".join(f.format() for f in errors[:3]))


def test_validator_catches_breakage():
    """Feed the validator known-bad JSON and require the specific complaint."""
    cases = [
        ("unknown density type",
         ("data/bad/worldgen/density_function/x.json", {"type": "minecraft:add_two", "argument": 1}),
         "unknown density function type"),
        ("misspelled field",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:clamp", "input": 1, "minimum": 0, "max": 1}),
         "has no field 'minimum'"),
        ("dangling reference",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:add", "argument1": "bad:nope", "argument2": 0}),
         "unknown density function"),
        ("spline order",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:spline", "spline": {"coordinate": "minecraft:y", "points": [
              {"location": 0.5, "value": 1}, {"location": 0.1, "value": 2}]}}),
         "strictly increase"),
        ("interval_select arity",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:interval_select", "input": "minecraft:y",
           "thresholds": [0.0, 1.0], "functions": [1, 2]}),
         "len(functions) == len(thresholds) + 1"),
        ("bad block id",
         ("data/bad/worldgen/noise_settings/s.json", {"default_block": {"Name": "minecraft:stoen"}}),
         "unknown block"),
        ("numeric block property",
         ("data/bad/worldgen/noise_settings/s.json",
          {"default_fluid": {"Name": "minecraft:water", "Properties": {"level": 0}}}),
         "must be a *string*"),
        ("sky_color moved to attributes",
         ("data/bad/worldgen/biome/b.json",
          {"temperature": 0.5, "downfall": 0.5, "has_precipitation": True,
           "effects": {"water_color": "#3f76e4", "sky_color": "#78a7ff"},
           "spawners": {}, "spawn_costs": {}, "carvers": [], "features": []}),
         "not a biome effect"),
    ]
    tmp = tempfile.mkdtemp(prefix="worldsmith-test-")
    try:
        for label, (rel, obj), expected in cases:
            root = Path(tmp) / label.replace(" ", "_")
            scaffold(root, "bad", "bad", template="basic")
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj), encoding="utf-8")
            _, findings = validate_path(root)
            messages = " | ".join(f"{f.message} {f.hint or ''}" for f in findings)
            check(f"validator catches: {label}", expected in messages,
                  f"expected {expected!r}, got: {messages[:220]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unreachable_biome_detection():
    entries = [
        {"biome": "a", "parameters": {"temperature": [-1, 1], "humidity": [-1, 1],
                                      "continentalness": [-1, 1], "erosion": [-1, 1],
                                      "depth": [0, 1], "weirdness": [-1, 1], "offset": 0.0}},
        # strictly inside a's box and never closer -> can never win
        {"biome": "b", "parameters": {"temperature": [-0.1, 0.1], "humidity": [-0.1, 0.1],
                                      "continentalness": [-0.1, 0.1], "erosion": [-0.1, 0.1],
                                      "depth": [0, 1], "weirdness": [-0.1, 0.1], "offset": 0.4}},
    ]
    source = BiomeSource.from_json({"type": "minecraft:multi_noise", "biomes": entries})
    dead = unreachable_biomes(source, samples=20000)
    check("detects a biome that can never win", dead == ["b"], str(dead))


def test_packs_generate():
    packs = sorted(p for p in (Path(ROOT) / "packs").iterdir() if (p / "pack.mcmeta").is_file())
    check("there are packs to test", bool(packs))
    for pack in packs:
        _, findings = validate_path(pack)
        errors = [f for f in findings if f.level == ERROR]
        check(f"{pack.name} validates", not errors, "; ".join(f.format() for f in errors[:2]))
        registries = Registries.load([str(pack)])
        dims = [i for i in registries.ids("dimension")
                if registries.origin("dimension", i) != registries.packs[0].name]
        for dim_id in dims:
            generator = (registries.get("dimension", dim_id) or {}).get("generator") or {}
            world = World.create(registries, generator["settings"], 12345)
            terrain = sample_terrain(world, -128, -128, 16, 16, step=16)
            stats = terrain.stats()
            check(f"{pack.name}/{dim_id} generates terrain",
                  not stats.get("empty") and stats.get("void_fraction", 1.0) < 0.98, str(stats))


def test_aquifer_levels():
    """Fluid levels against ones read out of a world the real server generated.

    Both numbers this pins were wrong in the obvious reference implementation:
    the level ladder needs the factor of 10, and the surface sampling returns on
    the sample that reaches above the surface rather than on the centre.
    """
    from worldsmith.aquifer import Aquifer

    with open(os.path.join(HERE, "golden", "aquifer_levels.json"), encoding="utf-8") as f:
        golden = json.load(f)
    world = World.create(Registries.load(), golden["settings"], golden["seed"])
    aquifer = Aquifer(world)
    centres = np.array([s["centre"] for s in golden["samples"]], dtype=np.int64)
    want = np.array([s["level"] for s in golden["samples"]], dtype=np.int64)
    got, _ = aquifer._compute_status(centres[:, 0], centres[:, 1], centres[:, 2])
    check("aquifer fluid levels match the generated world", np.array_equal(got, want),
          f"{list(zip(want.tolist(), got.tolist()))[:6]}")


def test_aquifer_barrier():
    """The stone wall an aquifer boundary leaves behind, block for block.

    At this column the server wrote water down to y 28, four blocks of stone,
    then an air pocket: exactly the barrier the engine used to miss.
    """
    from worldsmith.aquifer import Aquifer
    from worldsmith.density import Ctx

    world = World.create(Registries.load(), "minecraft:overworld", 12345)
    aquifer = Aquifer(world)
    node = world.router["final_density"]
    prepare(node)
    x, z = -159, -266
    ys = np.arange(30, 21, -1, dtype=np.int64)
    ctx = Ctx(np.array([[float(x)]]), ys[:, None].astype(float), np.array([[float(z)]]))
    density = np.ravel(np.broadcast_to(np.asarray(node.eval(ctx), float), (len(ys), 1)))
    pressure = aquifer.pressure(np.full(len(ys), x), ys, np.full(len(ys), z))
    solid = (density + pressure) > 0
    want = np.array([y in (24, 25, 26, 27) for y in ys])
    check("aquifer barrier lands on the blocks the game placed", np.array_equal(solid, want),
          f"solid at {ys[solid].tolist()}, expected [27, 26, 25, 24]")

    check("aquifers only apply where the pack asks for them",
          not World.create(Registries.load([os.path.join(ROOT, "packs", "basalt_spires")]),
                           "spires:basalt_spires", 1).aquifers_enabled)


def test_platform_paths():
    """`play` has to find the right runtime, saves folder and launcher on each OS."""
    import platform

    from worldsmith import play

    real = (play.WINDOWS, play.MACOS, platform.machine)
    cases = [
        ("win32", "AMD64", "windows/x64", ".minecraft"),
        ("darwin", "arm64", "mac/aarch64", "Application Support"),
        ("darwin", "x86_64", "mac/x64", "Application Support"),
        ("linux", "x86_64", "linux/x64", ".minecraft"),
        ("linux", "aarch64", "linux/aarch64", ".minecraft"),
    ]
    try:
        for name, machine, want_url, want_dir in cases:
            play.WINDOWS = name == "win32"
            play.MACOS = name == "darwin"
            platform.machine = lambda m=machine: m
            url = play.adoptium_url(25)
            check(f"{name}/{machine} downloads the right runtime", want_url in url, url)
            saves = str(play.minecraft_dir())
            check(f"{name} looks for saves in the right place", want_dir in saves, saves)
            check(f"{name} knows a launcher to try", bool(play.launcher_candidates()))
    finally:
        play.WINDOWS, play.MACOS, platform.machine = real


def main():
    test_kernel_matches_numpy()
    test_sampling_modes_agree()
    test_determinism()
    test_vanilla_validates_clean()
    test_validator_catches_breakage()
    test_unreachable_biome_detection()
    test_packs_generate()
    test_aquifer_levels()
    test_aquifer_barrier()
    test_platform_paths()
    print(f"{checks - len(failures)}/{checks} checks passed")
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
