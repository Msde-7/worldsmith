"""Conformance test: the engine against deepslate, the library behind
misode.github.io/worldgen, and against the JVM itself.

The engine targets Minecraft rather than deepslate, and deepslate deviates from
the JVM in four places, so every comparison runs twice:

  COMPAT["deepslate"] = True   must match deepslate exactly.
  COMPAT["deepslate"] = False  must differ only in the known places.

The four deviations, each pinned to real JVM output in test_jvm_semantics():

  1. Mth.getSeed: deepslate does not wrap at int32/int64.
  2. LegacyRandom.nextDouble: deepslate uses next(30) * 2^-30.
  3. interval_select: deepslate truncates `functions` to len(thresholds),
     dropping the top bucket.
  4. CubicSpline: deepslate keeps JSON point values, and the coordinate, as
     doubles where Java rounds both to float.

Regenerate the golden files with `npm install && node oracle.mjs && node
oracle_df.mjs` in tools/oracle.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import worldsmith.density as density
from worldsmith.density import Ctx, DensityCompiler
from worldsmith.jrandom import JavaRandom, Xoroshiro, get_seed, get_seed_np
from worldsmith.noise import BlendedNoise
from worldsmith.registry import Registries
from worldsmith.terrain import base_height
from worldsmith.world import World

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = json.load(open(os.path.join(HERE, "golden", "deepslate_golden.json")))
DF_GOLDEN = json.load(open(os.path.join(HERE, "golden", "deepslate_density_functions.json")))
SEED = int(GOLDEN["seed"])

ROUTER_NAME_MAP = {
    "finalDensity": "final_density", "preliminarySurfaceLevel": "preliminary_surface_level",
    "veinToggle": "vein_toggle", "veinRidged": "vein_ridged", "veinGap": "vein_gap",
    "fluidLevelFloodedness": "fluid_level_floodedness", "fluidLevelSpread": "fluid_level_spread",
}

# density functions expected to differ from deepslate when we follow the JVM
KNOWN_DEVIATIONS = {
    "minecraft:overworld/caves/entrances",                # interval_select top bucket
    "minecraft:overworld_large_biomes/offset",            # spline float rounding
    "minecraft:overworld_large_biomes/depth",
    "minecraft:overworld_large_biomes/sloped_cheese",
}

# Printed by a real JVM. These are the ground truth, not deepslate.
JVM_GET_SEED = {
    (0, 0, 0): 0,
    (1, 2, 3): -33674130277896,
    (-500, 60, 1200): 122051578588483,
    (123456, -40, -654321): 30295037624989,
    (1000000, 0, -1000000): 52603072207110,
}
JVM_LEGACY_12345_DOUBLES = [0.3618031071604718, 0.932993485288541, 0.8330913489710237]
JVM_LEGACY_12345_LONGS = [6674089274190705457, -1236052134575208584, -3078921119283744887]
JVM_RANDOM0_NEXTINT = -1155484576
JVM_RANDOM0_NEXTDOUBLE = 0.730967787376657

failures: list[str] = []
checks = 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(f"{name}: {detail}")
    return ok


def coord_arrays(coords):
    return (np.array([[c[0] for c in coords]], dtype=float),
            np.array([[c[1] for c in coords]], dtype=float),
            np.array([[c[2] for c in coords]], dtype=float))


def test_jvm_semantics():
    """Overflow behaviour and java.util.Random, checked against real JVM output."""
    for (x, y, z), want in JVM_GET_SEED.items():
        check(f"jvm getSeed{(x, y, z)}", get_seed(x, y, z) == want, f"{get_seed(x, y, z)} != {want}")
        vec = int(get_seed_np(np.array([x]), np.array([y]), np.array([z]))[0])
        check(f"jvm getSeed vectorised{(x, y, z)}", vec == want, f"{vec} != {want}")
    r = JavaRandom(0)
    check("jvm Random(0).nextInt", r.next_int() == JVM_RANDOM0_NEXTINT)
    r = JavaRandom(0)
    check("jvm Random(0).nextDouble", abs(r.next_double() - JVM_RANDOM0_NEXTDOUBLE) < 1e-17)
    r = JavaRandom(12345)
    got = [r.next_double() for _ in range(3)]
    check("jvm Random(12345).nextDouble x3",
          all(abs(a - b) < 1e-17 for a, b in zip(got, JVM_LEGACY_12345_DOUBLES)), str(got))
    r = JavaRandom(12345)
    got = [r.next_long() for _ in range(3)]
    check("jvm Random(12345).nextLong x3", got == JVM_LEGACY_12345_LONGS, str(got))


def test_rng_streams():
    for key, exp in GOLDEN["rng"]["streams"].items():
        kind, seed = key.split(":")
        seed = int(seed)
        r = Xoroshiro.create(seed) if kind == "xoroshiro" else JavaRandom(seed)
        longs = [str(r.next_long()) for _ in range(6)]
        doubles = [r.next_double() for _ in range(6)]
        ints = [r.next_int(256 - i) for i in range(6)]
        check(f"rng {key} longs", longs == exp["longs"], f"{longs} != {exp['longs']}")
        check(f"rng {key} ints", ints == exp["ints"], f"{ints} != {exp['ints']}")
        err = max(abs(a - b) for a, b in zip(doubles, exp["doubles"]))
        if kind == "xoroshiro":
            check(f"rng {key} doubles", err < 1e-15, f"max err {err:g}")
        else:
            # deviation 2: deepslate approximates; we match the JVM (asserted above)
            check(f"rng {key} doubles (legacy, approx)", err < 1e-7, f"max err {err:g}")


def test_positional_randoms():
    pos = Xoroshiro.create(SEED).fork_positional()
    for name, exp in GOLDEN["rng"]["fromHashOf"].items():
        r = pos.from_hash_of(name)
        got = [r.next_double(), r.next_double(), r.next_int(256)]
        ok = abs(got[0] - exp[0]) < 1e-15 and abs(got[1] - exp[1]) < 1e-15 and got[2] == exp[2]
        check(f"fromHashOf {name}", ok, f"{got} != {exp}")
    # deviation 1: at(x,y,z) matches only where deepslate's missing wraparound
    # cannot bite, i.e. the origin.
    r = pos.at(0, 0, 0)
    exp = GOLDEN["rng"]["at"]["0,0,0"]
    check("at(0,0,0)", abs(r.next_double() - exp[0]) < 1e-15)


def test_noise():
    world = World.create(Registries.load(), "minecraft:overworld", SEED)
    coords = GOLDEN["noise"]["coords"]
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    zs = np.array([c[2] for c in coords])
    for nid, exp in GOLDEN["noise"]["values"].items():
        if nid.startswith("blended:"):
            n = BlendedNoise(Xoroshiro.create(SEED).fork_positional().from_hash_of("minecraft:terrain"),
                             0.25, 0.125, 80, 160, 8)
        else:
            n = world.get_noise(nid)
        got = np.asarray(n.sample(xs, ys, zs), dtype=float)
        err = float(np.max(np.abs(got - np.array(exp["samples"]))))
        check(f"noise {nid}", err < 1e-12, f"max err {err:g}")
        check(f"noise {nid} maxValue", abs(n.max_value - exp["maxValue"]) < 1e-9)


def _density_diffs(compat: bool) -> dict[str, float]:
    density.COMPAT["deepslate"] = compat
    world = World.create(Registries.load(), "minecraft:overworld", SEED)
    compiler = DensityCompiler(world)
    xs, ys, zs = coord_arrays(DF_GOLDEN["coords"])
    diffs = {}
    for did, want in DF_GOLDEN["values"].items():
        if isinstance(want, dict):
            continue
        got = np.broadcast_to(np.asarray(compiler.compile_ref(did).eval(Ctx(xs, ys, zs)), float), xs.shape)[0]
        diffs[did] = float(np.max(np.abs(got - np.array(want, float))))
    density.COMPAT["deepslate"] = False
    return diffs


def test_density_functions():
    exact = _density_diffs(compat=True)
    bad = {k: v for k, v in exact.items() if v > 1e-12}
    check("all 35 vanilla density functions match deepslate in compat mode", not bad, str(bad))

    native = _density_diffs(compat=False)
    differing = {k for k, v in native.items() if v > 1e-12}
    check("JVM mode differs from deepslate only where documented",
          differing == KNOWN_DEVIATIONS,
          f"unexpected={sorted(differing - KNOWN_DEVIATIONS)} missing={sorted(KNOWN_DEVIATIONS - differing)}")


def test_routers():
    density.COMPAT["deepslate"] = True
    registries = Registries.load()
    try:
        for settings_id, block in GOLDEN["router"].items():
            if "error" in block:
                continue
            world = World.create(registries, settings_id, SEED)
            xs, ys, zs = coord_arrays(block["coords"])
            ctx = Ctx(xs, ys, zs)
            for js_name, values in block["values"].items():
                name = ROUTER_NAME_MAP.get(js_name, js_name)
                node = world.router.get(name)
                if node is None:
                    continue
                got = np.broadcast_to(np.asarray(node.eval(ctx), dtype=float), xs.shape)[0]
                want = np.array(values, dtype=float)
                err = float(np.max(np.abs(got - want)))
                check(f"router {settings_id}/{name}", err < 1e-9, f"max err {err:g}")
    finally:
        density.COMPAT["deepslate"] = False


def test_heightmaps():
    """End to end: our surface height against deepslate's getBaseHeight."""
    world = World.create(Registries.load(), GOLDEN["heightmap"]["settings"], SEED)
    columns = GOLDEN["heightmap"]["columns"]
    xs = np.array([c[0] for c in columns])
    zs = np.array([c[1] for c in columns])
    want = np.array([c[2] for c in columns])
    got = base_height(world, xs, zs)
    check("heightmap columns match deepslate", np.array_equal(got, want),
          f"{list(zip(want.tolist(), got.tolist()))}")


def main():
    test_jvm_semantics()
    test_rng_streams()
    test_positional_randoms()
    test_noise()
    test_density_functions()
    test_routers()
    test_heightmaps()
    print(f"{checks - len(failures)}/{checks} checks passed")
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
