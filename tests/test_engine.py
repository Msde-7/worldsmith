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
from worldsmith.climate import (BiomeSource, assign_biomes, climate_target,
                                unreachable_biomes)
from worldsmith.density import Ctx, prepare
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


def test_biome_tags():
    """Biome tags are how a structure finds its biomes.

    Get one wrong and the game says nothing at all: the tag is empty, and the
    world generates without a single village in it. So the tags have to be read,
    and every way of getting one wrong has to be caught.
    """
    caught = {
        "unknown biome": ({"values": ["bad:no_such_biome"]}, "unknown biome"),
        "vanilla typo": ({"values": ["minecraft:plian"]}, "did you mean 'plains'?"),
        "no values": ({"replace": False}, "tag has no 'values'"),
        "values not a list": ({"values": "bad:plains_like"}, "'values' must be a list"),
        "empty tag": ({"values": []}, "does nothing"),
        "empty tag that does not replace": ({"replace": False, "values": []}, "does nothing"),
        "misspelled field": ({"values": [], "value": []}, "tag has no field 'value'"),
        "replace not a bool": ({"replace": "false", "values": []}, "must be true or false"),
        "listed twice": ({"values": ["bad:plains_like"] * 2}, "listed twice"),
        "same id twice, spelled differently":
            ({"values": ["plains", "minecraft:plains"]}, "listed twice"),
        "dangling tag ref": ({"values": ["#bad:nope"]}, "unknown biome tag"),
        "entry is a number": ({"values": [7]}, "entry must be a biome id"),
    }
    # a false positive here would be worse than a miss: it would train people to
    # ignore the output
    silent = {
        "plain id": {"replace": False, "values": ["bad:plains_like"]},
        "unqualified id": {"values": ["plains"]},
        # vanilla's own tags are not vendored, so they cannot be resolved
        "vanilla tag ref": {"values": ["#minecraft:is_overworld"]},
        "another pack's tag": {"values": ["#other:whatever"]},
        "optional and absent": {"values": [{"id": "bad:not_here", "required": False}]},
        # replace + empty is how a vanilla tag gets switched off, not a mistake
        "empty tag that replaces": {"replace": True, "values": []},
        # '#x' is a tag reference and 'x' is a biome: not the same entry twice
        "tag ref beside a biome of that name":
            {"values": ["#minecraft:plains", "minecraft:plains"]},
        "object form": {"values": [{"id": "bad:plains_like", "required": True}]},
    }

    def findings_for(root, obj):
        scaffold(root, "bad", "bad", template="basic")
        path = root / "data/bad/tags/worldgen/biome/has_structure/village_plains.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")
        _, findings = validate_path(root)
        return [f for f in findings if f.where.startswith("biome_tag/")]

    tmp = tempfile.mkdtemp(prefix="worldsmith-tags-")
    try:
        for label, (obj, expected) in caught.items():
            found = findings_for(Path(tmp) / f"bad_{label.replace(' ', '_')}", obj)
            messages = " | ".join(f"{f.message} {f.hint or ''}" for f in found)
            check(f"tag validator catches: {label}", expected in messages,
                  f"expected {expected!r}, got: {messages[:200] or 'nothing'}")
        for label, obj in silent.items():
            found = findings_for(Path(tmp) / f"ok_{label.replace(' ', '_')}", obj)
            check(f"tag validator accepts: {label}", not found,
                  "; ".join(f.format() for f in found))

        # and the tags have to reach the registry in the first place, under ids
        # that keep their sub-directory
        root = Path(tmp) / "loads"
        findings_for(root, {"replace": False, "values": ["bad:plains_like"]})
        tags = Registries.load([str(root)]).data["biome_tag"]
        check("tag files load under their full id",
              "bad:has_structure/village_plains" in tags, str(sorted(tags)))
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

    # a biome may claim several boxes, and one of them winning is enough. Vanilla
    # hands out dozens per biome, so counting per entry called almost all of them
    # dead; b keeps a losing box here and still generates.
    full = {"humidity": [-1, 1], "continentalness": [-1, 1], "erosion": [-1, 1],
            "depth": [0, 1], "weirdness": [-1, 1], "offset": 0.0}
    several = [
        {"biome": "a", "parameters": {**full, "temperature": [-1, -0.5]}},
        entries[1],
        {"biome": "b", "parameters": {**full, "temperature": [0.5, 1]}},
    ]
    source = BiomeSource.from_json({"type": "minecraft:multi_noise", "biomes": several})
    dead = unreachable_biomes(source, samples=20000)
    check("a biome with one winning box out of two is alive", dead == [], str(dead))


def test_preset_biome_source():
    """A pack that starts from vanilla's overworld has to get vanilla's biomes.

    The table behind minecraft:overworld is assembled in Java, so mcmeta ships
    the preset name back as the whole file and worldsmith used to give up and
    preview such a pack as bare terrain. tools/extract_biome_parameters.py reads
    the real one out of the server jar; this checks it is vendored and lands.
    """
    registries = Registries.load()
    table = registries.get("multi_noise_biome_source_parameter_list", "minecraft:overworld") or {}
    check("the overworld preset table is vendored", bool(table.get("biomes")),
          "run python tools/extract_biome_parameters.py")
    if not table.get("biomes"):
        return
    source = BiomeSource.from_json(
        {"type": "minecraft:multi_noise", "preset": "minecraft:overworld"}, registries)
    check("the preset resolves to a multi_noise source", source.kind == "multi_noise")
    # vanilla gives one biome as many as a hundred boxes, and they have to fold
    # down, or every histogram and legend counts the same biome several times
    check("entries fold down to unique biomes",
          len(source.biomes) == len(set(source.biomes))
          and source.mins.shape[0] > len(source.biomes),
          f"{source.mins.shape[0]} entries, {len(source.biomes)} biomes")
    check("the ordinary overworld biomes are in it",
          {"minecraft:plains", "minecraft:ocean", "minecraft:jungle"} <= set(source.biomes),
          str(sorted(source.biomes)[:6]))

    world = World.create(registries, "minecraft:overworld", 12345)
    line = np.arange(-2048, 2048, 64, dtype=np.int64)
    xs = np.repeat(line, len(line))
    zs = np.tile(line, len(line))
    index = assign_biomes(source, climate_target(world, xs, zs, np.full(len(xs), 64)))
    check("every index lands inside the biome list",
          int(index.min()) >= 0 and int(index.max()) < len(source.biomes))
    placed = len(set(index.tolist()))
    check("the preset places many different biomes", placed >= 15,
          f"only {placed} over {len(xs)} columns")

    # and with nothing to resolve against it has to say so rather than guess
    try:
        BiomeSource.from_json({"type": "minecraft:multi_noise", "preset": "minecraft:overworld"})
        check("an unresolvable preset is an error", False, "it was accepted")
    except ValueError as exc:
        check("an unresolvable preset is an error", "preset" in str(exc), str(exc))


def test_biome_search_paths_agree():
    """The numba search and the numpy fallback have to pick the same biome.

    Ties go to whichever entry comes first, so both have to add the seven
    parameters in the same order or they disagree along every box boundary.
    """
    from worldsmith.climate import _nearest_box_numpy
    rng = np.random.default_rng(7)
    for k, n in ((12, 4000), (200, 3000)):
        lo = rng.uniform(-1, 1, size=(k, 7))
        source = BiomeSource(kind="multi_noise", biomes=[f"b{i}" for i in range(k)],
                             mins=np.ascontiguousarray(lo),
                             maxs=np.ascontiguousarray(lo + rng.uniform(0, 0.8, size=(k, 7))),
                             entry_biome=np.arange(k, dtype=np.int32))
        target = rng.uniform(-1.2, 1.5, size=(n, 7))
        check(f"the two searches agree over {k} boxes",
              np.array_equal(assign_biomes(source, target),
                             source.entry_biome[_nearest_box_numpy(source, target)]))
    # identical boxes are an exact tie, which the first entry has to win
    same = BiomeSource(kind="multi_noise", biomes=list("abcde"),
                       mins=np.zeros((5, 7)), maxs=np.full((5, 7), 0.5),
                       entry_biome=np.arange(5, dtype=np.int32))
    picked = set(assign_biomes(same, rng.uniform(-1, 1, size=(2000, 7))).tolist())
    check("an exact tie goes to the first entry", picked == {0}, str(picked))


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


def test_caves_and_decoration():
    """`new --caves` has to actually carve, and it has to carve where vanilla does.

    Cutting the cave layer in outside the `interpolated` node looks equivalent
    and is not: a world built that way matched the real game on 87% of columns,
    against 100% for the shape vanilla uses, aquifers off on both sides.
    """
    from worldsmith.pack import decoration_of
    from worldsmith.templates import (AQUIFER_ROUTER, CAVE_ENTRANCES, CAVE_NOODLE,
                                      with_caves)

    tmp = tempfile.mkdtemp(prefix="worldsmith-caves-")
    try:
        root = Path(tmp) / "caves"
        scaffold(root, "caves", "caves", caves=True, like="minecraft:plains")
        _, findings = validate_path(root)
        check("a caves pack validates", not [f for f in findings if f.level == ERROR],
              "; ".join(f.format() for f in findings if f.level == ERROR))

        settings = json.loads((root / "data/caves/worldgen/noise_settings/caves.json")
                              .read_text(encoding="utf-8"))
        final = settings["noise_router"]["final_density"]
        check("noodles are cut outside the interpolation", final["argument2"] == CAVE_NOODLE)
        check("the cave layer is cut inside the interpolation, as vanilla does",
              final["argument1"]["type"] == "minecraft:interpolated"
              and final["argument1"]["argument"]["argument2"] == CAVE_ENTRANCES)
        try:
            with_caves({"type": "minecraft:squeeze", "argument": {}})
            wrong_node = False
        except ValueError:
            wrong_node = True
        check("carving anything but the interpolated node is refused", wrong_node)

        # caves without aquifers flood: 5% of the cave volume stays air, against
        # 91% with them. Turning the flag on and leaving the four router fields
        # at 0 is the other half of the trap, and gives sheets of water instead.
        check("caves bring aquifers with them", settings["aquifers_enabled"])
        check("and the four aquifer router fields are actually written",
              all(settings["noise_router"][f] == AQUIFER_ROUTER[f] for f in AQUIFER_ROUTER))

        biome = json.loads((root / "data/caves/worldgen/biome/plains_like.json")
                           .read_text(encoding="utf-8"))
        check("--like brings the trees and mobs along",
              bool(biome["carvers"]) and any(biome["features"]) and bool(biome["spawners"]))

        world = World.create(Registries.load([str(root)]), "caves:caves", 12345)
        node = world.router["final_density"]
        prepare(node)
        ys = np.arange(-30.0, 60.0)
        found = False
        for x, z in ((-291.0, -288.0), (100.0, 40.0), (-40.0, 220.0)):
            ctx = Ctx(np.array([[x]]), ys[:, None], np.array([[z]]))
            solid = np.ravel(np.broadcast_to(np.asarray(node.eval(ctx), float),
                                             (len(ys), 1))) > 0
            found |= any(not solid[i] and solid[i - 2] and solid[i + 2]
                         for i in range(2, len(ys) - 2))
        check("the caves are actually hollow", found)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    check("decoration comes from the biome asked for",
          decoration_of("minecraft:desert")["carvers"] != [])


def test_grass_tint_and_block_colours():
    """The preview has to show the colour the game will show.

    A biome's effects.grass_color overrides the temperature/downfall colormap,
    so a pack that tints its turf was previewed in vanilla green: the render
    disagreed with the game about the single most visible thing in it.
    """
    import numpy as np

    from worldsmith.climate import BiomeSource
    from worldsmith.colors import BLOCK_COLORS, grass_color
    from worldsmith.render import render_map
    from worldsmith.scene import build_scene

    # every key has to be a real block, or it silently renders magenta for ever
    ids = set(json.loads((Path(ROOT) / "vanilla" / "26.2" / "blocks.json")
                         .read_text(encoding="utf-8"))["blocks"])
    unknown = sorted(k for k in BLOCK_COLORS if f"minecraft:{k}" not in ids)
    check("every block colour names a real block", not unknown, str(unknown))
    # and the other way: a short extraction would send whole biomes magenta
    uncoloured = sorted(b for b in ids if b.split(":")[-1] not in BLOCK_COLORS)
    check("every block has a colour", not uncoloured,
          f"{len(uncoloured)} without one: {uncoloured[:8]}")

    tinted, plain = (0x2B, 0x8C, 0xD9), None       # a blue no colormap would produce
    tmp = tempfile.mkdtemp(prefix="worldsmith-tint-")
    try:
        for label, tint in (("tinted", tinted), ("untinted", plain)):
            root = Path(tmp) / label
            scaffold(root, "tint", "tint", template="basic")
            path = root / "data/tint/worldgen/biome/plains_like.json"
            biome = json.loads(path.read_text(encoding="utf-8"))
            if tint is not None:
                biome["effects"]["grass_color"] = "#%02X%02X%02X" % tint
            path.write_text(json.dumps(biome), encoding="utf-8")

            registries = Registries.load([str(root)])
            world = World.create(registries, "tint:tint", 12345)
            source = BiomeSource.from_json(
                registries.get("dimension", "tint:tint")["generator"]["biome_source"])
            # most of the basic template is ocean, so look until a window has turf
            for x0, z0 in ((-1024, 512), (512, 512), (2048, -2048), (0, 0)):
                scene = build_scene(world, source, x0, z0, 32, 32, step=4)
                grass = np.array([n.split(":")[-1] == "grass_block" for n in scene.palette],
                                 dtype=bool)[np.asarray(scene.surface_block).ravel()]
                if grass.any():
                    break
            check(f"{label}: found a sample with grass in it", grass.any())
            if not grass.any():
                continue
            pixels = np.asarray(render_map(scene, scale=1, shade=False)).reshape(-1, 3)
            used = {tuple(int(v) for v in p) for p in pixels[grass]}
            expected = tint if tint is not None else grass_color(
                biome["temperature"], biome["downfall"])
            check(f"{label} grass renders as {expected}", used == {expected}, str(sorted(used)[:4]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_play_generates_structures():
    """`play` must ask the server for structures.

    The server bakes generate-structures into the world at creation, so a world
    built with it off has no village, temple or monument in it and never will,
    however correct the pack's biome tags are. This was off once: a 17,897-chunk
    world generated 0 structures; with it on, 8,100 chunks generated 66.
    """
    from worldsmith.play import server_properties

    props = server_properties(seed=7, gamemode="creative")
    lines = dict(line.split("=", 1) for line in props.splitlines() if "=" in line)
    check("play turns structure generation on", lines.get("generate-structures") == "true",
          str(lines.get("generate-structures")))
    check("play passes the seed through", lines.get("level-seed") == "7", str(lines))
    check("play passes the gamemode through", lines.get("gamemode") == "creative", str(lines))


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
    test_biome_tags()
    test_unreachable_biome_detection()
    test_preset_biome_source()
    test_biome_search_paths_agree()
    test_packs_generate()
    test_aquifer_levels()
    test_aquifer_barrier()
    test_caves_and_decoration()
    test_grass_tint_and_block_colours()
    test_play_generates_structures()
    test_platform_paths()
    print(f"{checks - len(failures)}/{checks} checks passed")
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
