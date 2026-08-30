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
        ("spline point without a derivative",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:spline", "spline": {"coordinate": "minecraft:y", "points": [
              {"location": 0, "value": 0}, {"location": 1, "value": 1}]}}),
         "needs a 'derivative'"),
        ("noise without its xz_scale",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:noise", "noise": "minecraft:cave_layer", "y_scale": 1}),
         "missing required field 'xz_scale'"),
        ("old_blended_noise without its smear",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:old_blended_noise", "xz_scale": 1, "y_scale": 1,
           "xz_factor": 80, "y_factor": 160}),
         "missing required field 'smear_scale_multiplier'"),
        ("find_top_surface without its bounds",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:find_top_surface", "density": "minecraft:y", "cell_height": 8}),
         "missing required field 'upper_bound'"),
        ("density function type the game removed",
         ("data/bad/worldgen/density_function/x.json",
          {"type": "minecraft:weird_scaled_sampler", "input": "minecraft:y",
           "noise": "minecraft:cave_layer", "rarity_value_mapper": "type_1"}),
         "is not a density function type in 26.2"),
        ("bad block id",
         ("data/bad/worldgen/noise_settings/s.json", {"default_block": {"Name": "minecraft:stoen"}}),
         "unknown block"),
        ("numeric block property",
         ("data/bad/worldgen/noise_settings/s.json",
          {"default_fluid": {"Name": "minecraft:water", "Properties": {"level": 0}}}),
         "must be a *string*"),
        ("surface condition missing a required key",
         ("data/bad/worldgen/noise_settings/s.json",
          {"surface_rule": {"type": "minecraft:condition",
                            "if_true": {"type": "minecraft:water", "offset": -1,
                                        "surface_depth_multiplier": 0},
                            "then_run": {"type": "minecraft:block",
                                         "result_state": {"Name": "minecraft:stone"}}}}),
         "missing required field 'add_stone_depth'"),
        ("misspelled surface condition field",
         ("data/bad/worldgen/noise_settings/s.json",
          {"surface_rule": {"type": "minecraft:condition",
                            "if_true": {"type": "minecraft:y_above", "ancho": {"absolute": 0},
                                        "surface_depth_multiplier": 0, "add_stone_depth": False},
                            "then_run": {"type": "minecraft:block",
                                         "result_state": {"Name": "minecraft:stone"}}}}),
         "did you mean 'anchor'?"),
        ("noise_settings missing a mandatory key",
         ("data/bad/worldgen/noise_settings/s.json", {"sea_level": 63}),
         "missing required field 'surface_rule'"),
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
    preview such a pack as bare terrain. tools/extract_worldgen_data.py reads
    the real one out of the server jar; this checks it is vendored and lands.
    """
    registries = Registries.load()
    table = registries.get("multi_noise_biome_source_parameter_list", "minecraft:overworld") or {}
    check("the overworld preset table is vendored", bool(table.get("biomes")),
          "run python tools/extract_worldgen_data.py")
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


def test_canopy_from_features():
    """A jungle has to come out under canopy and a desert bare.

    The cover is read off the biome's own feature list, through the placed and
    configured features tools/extract_worldgen_data.py vendors. If those go
    missing the walk quietly finds nothing to place, so the numbers are pinned
    here rather than only the shape.
    """
    from worldsmith.canopy import canopy_for
    names = ["minecraft:desert", "minecraft:plains", "minecraft:savanna", "minecraft:forest",
             "minecraft:dark_forest", "minecraft:taiga", "minecraft:cherry_grove",
             "minecraft:birch_forest", "minecraft:stony_peaks"]
    table = canopy_for(Registries.load(), names)
    cover = {n.split(":")[-1]: c for n, c in zip(names, table.cover)}
    leaves = {n.split(":")[-1]: leaf for n, leaf in zip(names, table.leaves)}
    check("nothing grows in a desert", cover["desert"] == 0.0 and leaves["desert"] is None)
    check("nothing grows on stony peaks", cover["stony_peaks"] == 0.0)
    check("plains are open ground", cover["plains"] < 0.05, f"{cover['plains']:.1%}")
    check("a savanna is scattered", 0.01 < cover["savanna"] < 0.30, f"{cover['savanna']:.1%}")
    check("a forest is mostly canopy", 0.5 < cover["forest"] <= 1.0, f"{cover['forest']:.1%}")
    check("a dark forest is closed canopy", cover["dark_forest"] > 0.95, f"{cover['dark_forest']:.1%}")
    # the leaf block only changes the colour for the ones the game does not tint
    # from the biome, so those are the ones worth pinning
    check("a taiga is spruce", leaves["taiga"] == "minecraft:spruce_leaves", str(leaves["taiga"]))
    check("a birch forest is birch", leaves["birch_forest"] == "minecraft:birch_leaves",
          str(leaves["birch_forest"]))
    check("a cherry grove is cherry", leaves["cherry_grove"] == "minecraft:cherry_leaves",
          str(leaves["cherry_grove"]))


def test_canopy_cover_matches_the_field():
    """Thresholding the field at a cover fraction has to cover that fraction.

    The cutoff comes from a fixed sample of the field rather than the window in
    front of it, so this checks it holds a long way from where it was taken.
    """
    from worldsmith.canopy import canopy_field, cover_cutoff
    line = np.arange(-40000, -40000 + 600 * 7, 7, dtype=np.float64)
    xs, zs = np.meshgrid(line, line + 1234)
    field = canopy_field(xs.ravel(), zs.ravel(), 99)
    for want in (0.05, 0.30, 0.76, 0.95):
        got = float((field <= cover_cutoff(want)).mean())
        check(f"cover {want:.0%} lands within two points", abs(got - want) < 0.02, f"got {got:.1%}")
    # panning must slide the same wood across the view, not redraw it
    shifted = canopy_field(xs.ravel() + 7.0, zs.ravel(), 99)
    same = canopy_field(np.array([1234.0, -99.0]), np.array([-7.0, 4321.0]), 99)
    check("the field is stable in absolute coordinates",
          np.allclose(same, canopy_field(np.array([1234.0, -99.0]), np.array([-7.0, 4321.0]), 99))
          and not np.allclose(shifted, field))


def test_decoration_is_paint_only():
    """--decorate is paint on the map view and must not reach anything else.

    The engine's claim is that the picture matches the game column for column,
    and the canopy is an estimate, so it is allowed to change the map image and
    nothing else.
    """
    from worldsmith.canopy import canopy_for
    from worldsmith.render import render_biomes, render_map
    from worldsmith.scene import build_scene
    registries = Registries.load()
    source = BiomeSource.from_json(
        {"type": "minecraft:multi_noise", "preset": "minecraft:overworld"}, registries)
    world = World.create(registries, "minecraft:overworld", 4242)
    for x0, z0 in ((3000, -1500), (-900, 2400), (0, 0)):
        scene = build_scene(world, source, x0, z0, 48, 48, step=4)
        if canopy_for(registries, scene.biomes).any():
            break
    check("found a window with something growing in it",
          canopy_for(registries, scene.biomes).any())

    blocks, heights = scene.surface_block.copy(), scene.height.copy()
    biomes_before = np.asarray(render_biomes(scene, scale=1))
    plain = np.asarray(render_map(scene, scale=1))
    decorated = np.asarray(render_map(scene, scale=1, decorate=True))
    check("the canopy changes the map", not np.array_equal(plain, decorated))
    check("the canopy leaves the surface blocks alone", np.array_equal(blocks, scene.surface_block))
    check("the canopy leaves the heights alone", np.array_equal(heights, scene.height))
    check("the canopy stays out of the biome view",
          np.array_equal(biomes_before, np.asarray(render_biomes(scene, scale=1))))
    # water is not ground, so nothing may be painted over the sea
    sea = np.array([n.split(":")[-1] in ("water", "flowing_water") for n in scene.palette],
                   dtype=bool)[scene.surface_block]
    check("nothing grows on water", np.array_equal(plain[sea], decorated[sea]))


def test_surface_blocks_against_the_game():
    """The blocks a real server wrote, column for column.

    Everything else here checks the engine against deepslate or against itself,
    and neither of those covers surface.py: the conformance suite stops at the
    heightmap. That left the surface rules unmeasured, and two of them were
    wrong. tests/golden/surface_blocks.json is the top block of every column in
    six chunks of packs/red_canyons, generated by Minecraft 26.2 at seed 12345,
    which pins the clay band table and the steep condition to the real thing.
    """
    from worldsmith.scene import build_scene
    golden = json.loads((Path(HERE) / "golden" / "surface_blocks.json").read_text(encoding="utf-8"))
    pack = Path(ROOT) / golden["pack"]
    registries = Registries.load([str(pack)])
    dimension = registries.get("dimension", golden["dimension"])
    if dimension is None:                      # the pack ships its own dimension id
        ids = [i for i in registries.ids("dimension") if registries.origin("dimension", i) != "vanilla-26.2"]
        dimension = registries.get("dimension", ids[0])
    world = World.create(registries, dimension["generator"]["settings"], golden["seed"])
    source = BiomeSource.from_json(dimension["generator"]["biome_source"], registries)

    total = agree = waterline = 0
    worst = []
    for entry in golden["chunks"]:
        cx, cz = entry["chunk"]
        scene = build_scene(world, source, cx * 16, cz * 16, 16, 16, step=1)
        mine = np.array(scene.palette, dtype=object)[scene.surface_block]
        for i, (want, top) in enumerate(zip(entry["blocks"], entry["tops"])):
            if top < world.sea_level - 1:
                continue
            z, x = divmod(i, 16)
            got = str(mine[z, x])
            total += 1
            waterline += top == world.sea_level - 1
            if got == want:
                agree += 1
            elif len(worst) < 4:
                worst.append(f"({cx * 16 + x},{cz * 16 + z}) y={top} engine {got} game {want}")
    check("the surface blocks are the game's surface blocks", agree == total,
          f"{agree}/{total} match; {'; '.join(worst)}")
    # a column whose top block sits exactly at sea_level - 1 is dry, not submerged
    check("the sample pins the waterline", waterline > 0, "no column at sea_level - 1")


def test_clay_bands_against_the_game():
    """The 192 terracotta layers, as the game built them.

    generateBands is two easy translation slips away from a different table:
    vanilla steps the orange bands with a for loop whose own i++ runs on top of
    the i += inside it, and its band width is minSize + nextInt(3). Getting
    either wrong shifts the random stream and repaints every mesa. The indices
    below were read back out of a generated world.
    """
    from worldsmith.surface import SurfaceSystem
    golden = json.loads((Path(HERE) / "golden" / "surface_blocks.json").read_text(encoding="utf-8"))
    registries = Registries.load([str(Path(ROOT) / golden["pack"])])
    ids = [i for i in registries.ids("dimension") if registries.origin("dimension", i) != "vanilla-26.2"]
    dimension = registries.get("dimension", golden["dimension"]) or registries.get("dimension", ids[0])
    world = World.create(registries, dimension["generator"]["settings"], golden["seed"])
    bands = SurfaceSystem(world).clay_bands()
    check("there are 192 bands", len(bands) == 192, str(len(bands)))
    # a handful of indices the world pinned down, spread across the table
    expected = {39: "orange_terracotta", 41: "light_gray_terracotta", 42: "white_terracotta",
                43: "light_gray_terracotta", 47: "orange_terracotta", 51: "brown_terracotta",
                61: "yellow_terracotta", 67: "orange_terracotta", 104: "orange_terracotta",
                137: "orange_terracotta"}
    bad = {i: bands[i].split(":")[-1] for i, name in expected.items()
           if bands[i].split(":")[-1] != name}
    check("the band table is the game's band table", not bad, str(bad))


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


def test_template_round_trip():
    """A structure template the game will not read is a silent failure: the
    build simply is not there. So the writer is checked against its own reader,
    and against the DataVersion the vendored jar declares."""
    from worldsmith.voxel import Grid, data_version, parse_block, read_nbt

    grid = Grid(3, 2, 3)
    grid.fill(0, 0, 0, 2, 0, 2, "minecraft:stone_bricks")
    grid.set(1, 1, 1, "minecraft:oak_stairs[facing=north,half=top]")
    grid.set(0, 1, 0, "minecraft:chest[facing=west]",
             {"id": "minecraft:chest", "LootTable": "minecraft:chests/simple_dungeon"})

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.nbt"
        grid.save(path)
        root = read_nbt(path)
        back = Grid.load(path)

    check("template keeps its size", list(root["size"]) == [3, 2, 3], str(root["size"]))
    check("template writes every set block", len(root["blocks"]) == grid.filled(),
          f"{len(root['blocks'])} vs {grid.filled()}")
    check("template carries the vendored DataVersion",
          root["DataVersion"] == data_version(), str(root["DataVersion"]))
    check("reading a template back gives the same blocks",
          back.counts() == grid.counts(), f"{back.counts()} vs {grid.counts()}")
    check("block entities survive the round trip",
          back.block_entities.get((0, 1, 0), {}).get("LootTable")
          == "minecraft:chests/simple_dungeon", str(back.block_entities))

    for spec in ("minecraft:chain", "minecraft:oak_stairs[facing=up]",
                 "minecraft:oak_stairs[nonsense=1]"):
        try:
            parse_block(spec)
            check(f"bad state {spec} is refused", False, "accepted")
        except ValueError:
            check(f"bad state {spec} is refused", True)


def test_structure_files():
    """The four files a build needs, and the two defaults that are painful to
    get wrong: a modern pool element would ignore the air a build hollows its
    rooms with, and no heightmap projection would leave it at y=0."""
    from worldsmith.pack import PackWriter
    from worldsmith.registry import Registries
    from worldsmith.structures import add, rotate_xz, spread
    from worldsmith.voxel import Grid

    grid = Grid(4, 3, 4)
    grid.fill(0, 0, 0, 3, 0, 3, "minecraft:stone_bricks")

    with tempfile.TemporaryDirectory() as tmp:
        writer = PackWriter(Path(tmp) / "p", "test")
        writer.mcmeta()
        add(writer, "test:hut", grid, ["minecraft:plains"], sink=-1)
        writer.add("structure_set", "test:huts",
                   spread("test:hut", spacing=16, separation=7, salt=1))
        registries = Registries.load([writer.root], include_vanilla=False)

        check("the template lands where the game looks for it",
              (writer.root / "data/test/structure/hut.nbt").is_file())
        check("the registry finds the template",
              "test:hut" in registries.templates, str(registries.templates))
        structure = registries.get("structure", "test:hut")
        pool = registries.get("template_pool", "test:hut")
        placed = registries.get("structure_set", "test:huts")
        check("the structure points at its pool",
              structure["start_pool"] == "test:hut", str(structure))
        check("the structure projects to the surface",
              structure["project_start_to_heightmap"] == "WORLD_SURFACE_WG", str(structure))
        check("the pool element keeps air",
              pool["elements"][0]["element"]["element_type"]
              == "minecraft:legacy_single_pool_element", str(pool))
        check("the pool points at the template",
              pool["elements"][0]["element"]["location"] == "test:hut", str(pool))
        check("the set places the structure",
              placed["structures"][0]["structure"] == "test:hut", str(placed))

    check("a full turn is the identity",
          rotate_xz(*rotate_xz(1, 0, 4, 4, "CLOCKWISE_180"), 4, 4, "CLOCKWISE_180") == (1, 0))
    check("a quarter turn moves the corner",
          rotate_xz(0, 0, 4, 4, "CLOCKWISE_90") == (3, 0),
          str(rotate_xz(0, 0, 4, 4, "CLOCKWISE_90")))


def test_placement_geometry():
    """The placement model is checked against a server by
    tools/verify_placement.py; these are the parts that can be pinned without
    one: one site per region, inside the spacing minus separation window, the
    box a rotated build covers, and the exclusion zone."""
    from worldsmith.pack import PackWriter
    from worldsmith.placement import (Site, footprint, set_sites, site_rotation,
                                      sites)
    from worldsmith.registry import Registries
    from worldsmith.structures import spread

    placement = {"spacing": 8, "separation": 3, "salt": 12345}
    found = sites(placement, seed=99, x0=0, z0=0, x1=8 * 16 * 4 - 1, z1=8 * 16 * 4 - 1)
    check("one site per region", len(found) == 16, str(len(found)))
    windows = {(s.chunk_x // 8, s.chunk_z // 8) for s in found}
    check("the sites are one per region, not several in one",
          len(windows) == len(found), f"{len(windows)} regions for {len(found)} sites")
    offsets = [(s.chunk_x % 8, s.chunk_z % 8) for s in found]
    check("every site is inside spacing minus separation",
          all(0 <= a < 5 and 0 <= b < 5 for a, b in offsets), str(offsets[:4]))
    check("the same seed gives the same sites",
          [(s.chunk_x, s.chunk_z) for s in sites(placement, 99, 0, 0, 500, 500)]
          == [(s.chunk_x, s.chunk_z) for s in sites(placement, 99, 0, 0, 500, 500)])
    check("a different salt moves them",
          [(s.chunk_x, s.chunk_z) for s in sites(dict(placement, salt=7), 99, 0, 0, 500, 500)]
          != [(s.chunk_x, s.chunk_z) for s in found])

    site = Site(2, 3)                       # anchored at block (32, 48)
    check("an unturned build grows from its anchor",
          footprint(site, 64, 64, "NONE") == (32, 48, 95, 111))
    check("a quarter turn grows the other way",
          footprint(site, 64, 64, "CLOCKWISE_90") == (-31, 48, 32, 111),
          str(footprint(site, 64, 64, "CLOCKWISE_90")))
    check("a half turn grows back and left",
          footprint(site, 64, 64, "CLOCKWISE_180") == (-31, -15, 32, 48))
    check("a one block build is its anchor whatever the turn",
          all(footprint(site, 1, 1, r) == (32, 48, 32, 48)
              for r in ("NONE", "CLOCKWISE_90", "CLOCKWISE_180", "COUNTERCLOCKWISE_90")))
    check("the turn is stable for a site",
          site_rotation(99, 2, 3) == site_rotation(99, 2, 3))

    with tempfile.TemporaryDirectory() as tmp:
        writer = PackWriter(Path(tmp) / "p", "test")
        writer.mcmeta()
        writer.add("structure_set", "test:wide", spread("test:a", spacing=8, separation=3, salt=1))
        writer.add("structure_set", "test:near",
                   spread("test:b", spacing=8, separation=3, salt=1,
                          exclusion=("test:wide", 6)))
        registries = Registries.load([writer.root], include_vanilla=False)
        wide = set_sites(registries, "test:wide", 5, 0, 0, 2000, 2000)
        near = set_sites(registries, "test:near", 5, 0, 0, 2000, 2000)
        check("the same salt without an exclusion zone gives the same sites",
              {(s.chunk_x, s.chunk_z) for s in wide} ==
              {(s.chunk_x, s.chunk_z) for s in sites(
                  {"spacing": 8, "separation": 3, "salt": 1}, 5, 0, 0, 2000, 2000)})
        check("an exclusion zone clears the sites it covers", not near,
              f"{len(near)} sites survived an exclusion of the same set")


def test_structure_validation():
    """Every one of these loads into the game without a word and then never
    places anything, which is exactly the failure check exists to catch."""
    from worldsmith.pack import PackWriter
    from worldsmith.registry import Registries
    from worldsmith.structures import pool, spread, structure
    from worldsmith.validate import Validator
    from worldsmith.voxel import Grid

    def findings(build_pack):
        with tempfile.TemporaryDirectory() as tmp:
            writer = PackWriter(Path(tmp) / "p", "test")
            writer.mcmeta()
            build_pack(writer)
            registries = Registries.load([writer.root])
            found = Validator(registries, registries.packs[-1]).validate_pack()
            return [f"{f.level} {f.where} {f.message}" for f in found]

    hollow = Grid(4, 4, 4)
    hollow.fill(0, 0, 0, 3, 3, 3, "minecraft:stone_bricks")
    hollow.fill(1, 1, 1, 2, 2, 2, "minecraft:air")

    def good(writer):
        writer.add_template("test:hut", hollow)
        writer.add("structure", "test:hut", structure("test:hut", ["minecraft:plains"]))
        writer.add("template_pool", "test:hut", pool("test:hut"))
        writer.add("structure_set", "test:huts", spread("test:hut", spacing=8,
                                                        separation=3, salt=1))
    check("a sound pack of builds is clean", findings(good) == [], str(findings(good)))

    def missing_pool(writer):
        good(writer)
        writer.add("structure", "test:hut", structure("test:nowhere", ["minecraft:plains"]))
    check("a start_pool that does not exist is caught",
          any("start_pool" in f for f in findings(missing_pool)), str(findings(missing_pool)))

    def missing_template(writer):
        good(writer)
        writer.add("template_pool", "test:hut", pool("test:absent"))
    check("a pool element with no template is caught",
          any("no template" in f for f in findings(missing_template)),
          str(findings(missing_template)))

    def modern_element(writer):
        good(writer)
        broken = pool("test:hut")
        broken["elements"][0]["element"]["element_type"] = "minecraft:single_pool_element"
        writer.add("template_pool", "test:hut", broken)
    check("the element type that throws air away is caught",
          any("ignores the air" in f for f in findings(modern_element)),
          str(findings(modern_element)))

    def bad_spread(writer):
        good(writer)
        writer.add("structure_set", "test:huts", spread("test:hut", spacing=4,
                                                        separation=9, salt=1))
    check("separation at or above spacing is caught",
          any("separation" in f for f in findings(bad_spread)), str(findings(bad_spread)))

    def bad_exclusion(writer):
        good(writer)
        writer.add("structure_set", "test:huts",
                   spread("test:hut", spacing=8, separation=3, salt=1,
                          exclusion=("test:absent_set", 4)))
    check("an exclusion zone naming nothing is caught",
          any("exclusion_zone" in f for f in findings(bad_exclusion)),
          str(findings(bad_exclusion)))

    def no_biomes(writer):
        good(writer)
        writer.add("structure", "test:hut", structure("test:hut", []))
    check("a structure with no biomes is caught",
          any("biomes is empty" in f for f in findings(no_biomes)), str(findings(no_biomes)))

    def bad_step(writer):
        good(writer)
        writer.add("structure", "test:hut",
                   structure("test:hut", ["minecraft:plains"], step="whenever"))
    check("a step the game does not have is caught",
          any("generation step" in f for f in findings(bad_step)), str(findings(bad_step)))

    def unplaced_biome(writer):
        good(writer)
        writer.add("structure", "test:hut", structure("test:hut", ["minecraft:end_barrens"]))
        writer.add("dimension", "test:d", {
            "type": "minecraft:overworld",
            "generator": {"type": "minecraft:noise", "settings": "minecraft:overworld",
                          "biome_source": {"type": "minecraft:fixed",
                                           "biome": "minecraft:plains"}}})
    check("a build keyed to a biome this dimension never places is caught",
          any("none of these biomes are placed" in f for f in findings(unplaced_biome)),
          str(findings(unplaced_biome)))


def test_set_reports():
    """A set can hold several builds, and the game picks one per site. The
    choice is checked against a server by tools/verify_placement.py; this pins
    that it is stable, that weights are honoured, and that each site is measured
    against the build that will actually stand on it."""
    from worldsmith.climate import BiomeSource
    from worldsmith.pack import PackWriter
    from worldsmith.placement import Site, chosen_build, set_reports
    from worldsmith.registry import Registries
    from worldsmith.structures import add, spread
    from worldsmith.voxel import Grid
    from worldsmith.world import World

    small, large = Grid(4, 2, 4), Grid(48, 2, 48)
    small.fill(0, 0, 0, 3, 0, 3, "minecraft:stone_bricks")
    large.fill(0, 0, 0, 47, 0, 47, "minecraft:stone_bricks")

    entries = [{"structure": "test:small", "weight": 1}, {"structure": "test:large", "weight": 3}]
    picks = [chosen_build(11, Site(x, z), entries) for x in range(20) for z in range(20)]
    check("the choice is stable for a site",
          chosen_build(11, Site(3, 4), entries) == chosen_build(11, Site(3, 4), entries))
    check("every pick is one of the set's builds",
          set(picks) <= {"test:small", "test:large"}, str(set(picks)))
    share = picks.count("test:large") / len(picks)
    check("weight 3 of 4 wins about three quarters of the sites",
          0.65 < share < 0.85, f"{share:.2f}")

    with tempfile.TemporaryDirectory() as tmp:
        writer = PackWriter(Path(tmp) / "p", "test")
        writer.mcmeta()
        add(writer, "test:small", small, ["minecraft:plains"], sink=-1)
        add(writer, "test:large", large, ["minecraft:plains"], sink=-1)
        writer.add("structure_set", "test:both",
                   spread({"test:small": 1, "test:large": 3}, spacing=8, separation=3, salt=5))
        registries = Registries.load([writer.root])
        world = World.create(registries, "minecraft:overworld", 7)
        source = BiomeSource.from_json(
            {"type": "minecraft:multi_noise", "preset": "minecraft:overworld"}, registries)
        reports = set_reports(registries, world, source, "test:both", 7, -256, -256, 255, 255)

    # 512 blocks is 32 chunks, which is 4 by 4 regions of 8
    check("every site is reported", len(reports) == 16, str(len(reports)))
    check("each site names the build that will stand on it",
          all(r.build in ("test:small", "test:large") for r in reports))
    for report in reports:
        side = report.box[2] - report.box[0] + 1
        want = 48 if report.build == "test:large" else 4
        check(f"the footprint is the build's own size ({report.build.split(':')[-1]})",
              side == want, f"{side} for {report.build}")
    check("the ground is measured, not guessed",
          all(r.high >= r.low and r.surface_y > 0 for r in reports))
    # the build's own y=0 lands at the surface plus its start_height, which was
    # -1 for both of these
    check("the floor lands where start_height puts it",
          all(r.floor_y == r.surface_y - 1 for r in reports),
          str([(r.floor_y, r.surface_y) for r in reports[:3]]))

    stand = Grid(5, 6, 5)
    stand.fill(0, 0, 0, 4, 2, 4, "minecraft:stone")
    stand.fill(1, 3, 1, 3, 5, 3, "minecraft:air")
    check("a standing spot is solid with head room above it",
          stand.standing_spot() == (2, 2, 2), str(stand.standing_spot()))
    check("a solid block with nothing placed above it is not a standing spot",
          Grid(3, 3, 3).standing_spot() is None)


def test_template_validation():
    """A template the game cannot use is quieter than a broken JSON file: the
    structure still generates, and leaves an empty box where the build was."""
    from worldsmith.pack import PackWriter
    from worldsmith.registry import Registries
    from worldsmith.structures import pool, spread, structure
    from worldsmith.validate import Validator
    from worldsmith.voxel import Grid, data_version, write_nbt

    sound = {"size": [2, 2, 2], "entities": [],
             "blocks": [{"state": 0, "pos": [0, 0, 0]}, {"state": 0, "pos": [1, 1, 1]}],
             "palette": [{"Name": "minecraft:stone_bricks"}],
             "DataVersion": data_version()}

    def findings(template=None, *, wire=True, extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            writer = PackWriter(Path(tmp) / "p", "test")
            writer.mcmeta()
            if template is None:
                writer.add_template("test:hut", Grid(2, 2, 2))
            else:
                write_nbt(template, writer.root / "data/test/structure/hut.nbt")
            if wire:
                writer.add("structure", "test:hut", structure("test:hut", ["minecraft:plains"]))
                writer.add("template_pool", "test:hut", pool("test:hut"))
                writer.add("structure_set", "test:huts",
                           spread("test:hut", spacing=8, separation=3, salt=1))
            if extra:
                extra(writer)
            registries = Registries.load([writer.root])
            found = Validator(registries, registries.packs[-1]).validate_pack()
            return [f"{f.level} {f.where} {f.message}" for f in found]

    check("a sound template is clean", findings(sound) == [], str(findings(sound)))

    bad_block = dict(sound, palette=[{"Name": "minecraft:chain"}])   # renamed in 26.2
    check("a block the version does not have is caught",
          any("unknown block" in f for f in findings(bad_block)), str(findings(bad_block)))

    bad_property = dict(sound, palette=[{"Name": "minecraft:oak_stairs",
                                        "Properties": {"nonsense": "north"}}])
    check("a property the block does not have is caught",
          any("no property" in f for f in findings(bad_property)), str(findings(bad_property)))

    outside = dict(sound, blocks=[{"state": 0, "pos": [0, 0, 0]}, {"state": 0, "pos": [9, 0, 0]}])
    check("a block outside the declared size is caught",
          any("outside the declared size" in f for f in findings(outside)), str(findings(outside)))

    dangling = dict(sound, blocks=[{"state": 7, "pos": [0, 0, 0]}])
    check("a block pointing outside the palette is caught",
          any("outside the palette" in f for f in findings(dangling)), str(findings(dangling)))

    newer = dict(sound, DataVersion=data_version() + 100)
    check("a template from a newer version is an error",
          any("DataVersion" in f and f.startswith("ERROR") for f in findings(newer)),
          str(findings(newer)))
    older = dict(sound, DataVersion=data_version() - 100)
    check("a template from an older version is a warning",
          any("DataVersion" in f and f.startswith("WARNING") for f in findings(older)),
          str(findings(older)))

    empty = dict(sound, blocks=[])
    check("a template that places nothing is caught",
          any("places no blocks" in f for f in findings(empty)), str(findings(empty)))

    zero = dict(sound, size=[0, 2, 2])
    check("a template with no size is caught",
          any("three positive numbers" in f for f in findings(zero)), str(findings(zero)))

    check("a template nothing refers to is worth a note",
          any("no template pool refers" in f for f in findings(sound, wire=False)),
          str(findings(sound, wire=False)))

    def not_a_template(writer):
        (writer.root / "data/test/structure/rubbish.nbt").write_bytes(b"not gzip at all")
    check("a file that is not a template is caught",
          any("not a readable structure template" in f
              for f in findings(sound, extra=not_a_template)),
          str(findings(sound, extra=not_a_template)))


def test_build_overlay_and_determinism():
    """Two small promises: the same build writes the same bytes, so rebuilding a
    pack does not churn the diff, and the overlay draws a build's box where the
    build actually is, in map pixels rather than blocks."""
    from PIL import Image

    from worldsmith.draw import KEPT, REJECTED, mark_builds
    from worldsmith.placement import Site, SiteReport
    from worldsmith.voxel import Grid

    def make():
        grid = Grid(3, 2, 3)
        grid.fill(0, 0, 0, 2, 0, 2, "minecraft:stone_bricks")
        grid.set(1, 1, 1, "minecraft:chest[facing=north]",
                 {"id": "minecraft:chest", "LootTable": "minecraft:chests/simple_dungeon"})
        return grid

    with tempfile.TemporaryDirectory() as tmp:
        first = make().save(Path(tmp) / "a.nbt").read_bytes()
        second = make().save(Path(tmp) / "b.nbt").read_bytes()
    check("the same build writes the same bytes", first == second,
          f"{len(first)} vs {len(second)}")

    def report(box, accepted):
        return SiteReport(site=Site(0, 0), build="test:b", rotation="NONE", box=box,
                          biome="minecraft:plains", floor_y=64, surface_y=64,
                          low=64, high=64, water=0.0, accepted=accepted)

    # a 256 block map drawn at 4 blocks per pixel and 2 pixels per cell
    image = Image.new("RGB", (128, 128), (0, 0, 0))
    mark_builds(image, [report((64, 32, 95, 63), True)], x0=0, z0=0, step=4, scale=2)
    pixels = image.load()
    check("the overlay draws the box where the build is",
          pixels[32, 16] == KEPT and pixels[47, 31] == KEPT,
          f"{pixels[32, 16]} {pixels[47, 31]}")
    check("the overlay leaves the middle of the box alone",
          pixels[40, 24] == (0, 0, 0), str(pixels[40, 24]))
    check("the overlay leaves the rest of the map alone",
          pixels[10, 10] == (0, 0, 0) and pixels[100, 100] == (0, 0, 0))

    faint = Image.new("RGB", (128, 128), (0, 0, 0))
    mark_builds(faint, [report((64, 32, 95, 63), False)], x0=0, z0=0, step=4, scale=2)
    check("a rejected site is drawn differently", faint.load()[32, 16] == REJECTED,
          str(faint.load()[32, 16]))

    off = Image.new("RGB", (128, 128), (0, 0, 0))
    mark_builds(off, [report((-900, -900, -800, -800), True)], x0=0, z0=0, step=4, scale=2)
    check("a build off the map is not drawn", off.getbbox() is None)


def test_packed_longs():
    """Block states, biomes and heightmaps are all read through unpack_longs, so
    it is checked against the plain loop it replaced, at every width the game
    uses and across the sign boundary."""
    from worldsmith.anvil import unpack_longs

    def plain(packed, bits, count):
        per_long, mask = 64 // bits, (1 << bits) - 1
        out, i = [], 0
        for word in np.asarray(packed).astype(np.uint64):
            for slot in range(per_long):
                if i >= count:
                    break
                out.append((int(word) >> (slot * bits)) & mask)
                i += 1
        return np.array(out, dtype=np.int64)

    rng = np.random.default_rng(11)
    agreed = 0
    for bits in (1, 2, 3, 4, 5, 6, 7, 9, 12, 15):
        for count in (64, 256, 4096):
            words = rng.integers(-(2 ** 63), 2 ** 63 - 1,
                                 (count + 64 // bits - 1) // (64 // bits), dtype=np.int64)
            agreed += int(np.array_equal(unpack_longs(words, bits, count),
                                         plain(words, bits, count)))
    check("unpacking agrees with the plain loop at every width", agreed == 30, str(agreed))
    check("a section of 4 bit states unpacks to 4096 entries",
          len(unpack_longs(np.zeros(256, dtype=np.int64), 4, 4096)) == 4096)


def test_scaffold_with_build():
    """`new --with-build` has to produce a pack that loads, validates and has a
    build the game would place, because it is the starting point every build
    gets changed from."""
    from worldsmith.pack import scaffold
    from worldsmith.registry import Registries
    from worldsmith.validate import ERROR, Validator
    from worldsmith.voxel import Grid

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "p"
        scaffold(root, "demo", "demo", with_build=True)
        registries = Registries.load([root])
        findings = Validator(registries, registries.packs[-1]).validate_pack()
        errors = [f.format() for f in findings if f.level == ERROR]
        check("a scaffolded pack with a build has no errors", not errors, str(errors))
        check("the build is registered", "demo:hut" in registries.templates,
              str(registries.templates))
        for category in ("structure", "template_pool", "structure_set"):
            check(f"the scaffold writes the {category}",
                  bool(registries.packs[-1].data[category]),
                  str(registries.packs[-1].data[category]))
        grid = Grid.load(registries.templates["demo:hut"])
        check("the scaffolded build is hollow", grid.name_at(4, 6, 4) == "air",
              grid.name_at(4, 6, 4))
        check("the scaffolded build has somewhere to stand",
              grid.standing_spot() is not None)
        structure = registries.get("structure", "demo:hut")
        check("the scaffolded build is keyed to a biome the pack places",
              set(structure["biomes"]) & set(registries.packs[-1].data["biome"]),
              str(structure["biomes"]))


def test_build_on_site():
    """The build drawn on the ground it will stand on: the template's blocks at
    the height the game puts them, with the terrain under and around them."""
    from worldsmith.climate import BiomeSource
    from worldsmith.pack import scaffold
    from worldsmith.placement import build_on_site
    from worldsmith.registry import Registries
    from worldsmith.voxel import Grid
    from worldsmith.world import World

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "p"
        scaffold(root, "demo", "demo", with_build=True)
        registries = Registries.load([root])
        world = World.create(registries, "demo:demo", 4242)
        source = BiomeSource.from_json(
            registries.get("dimension", "demo:demo")["generator"]["biome_source"], registries)
        grid, report = build_on_site(registries, world, source, "demo:hut", 4242, margin=6)
        template = Grid.load(registries.templates["demo:hut"])

    check("the site is one the game would keep", report.accepted)
    check("the box is the build's own size",
          report.box[2] - report.box[0] + 1 == template.sx, str(report.box))
    check("the drawing is the build plus its margin",
          grid.sx == template.sx + 12 and grid.sz == template.sz + 12,
          f"{grid.sx}x{grid.sz}")
    names = {spec.split("[")[0] for _, spec in grid.items()}
    check("the build's own blocks are in the drawing",
          "minecraft:stone_bricks" in names and "minecraft:oak_planks" in names, str(names))
    check("there is ground under it as well as the build",
          "minecraft:stone" in names, str(names))
    solid = sum(1 for _, spec in grid.items())
    check("the drawing is not empty", solid > template.filled(), str(solid))


def test_cli_smoke():
    """Every build command, run the way a person runs it.

    The library is tested above; this is the wiring. Both bugs found by hand
    while writing these commands were wiring: an argument that was never added
    to a parser, and a string that was broken in a branch nothing had run."""
    import contextlib
    import io

    from worldsmith.cli import main
    from worldsmith.pack import PackWriter, scaffold
    from worldsmith.structures import pool, structure

    def run(argv):
        """Quietly: these commands are chatty and the suite prints its own summary."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return main(argv)
        except SystemExit as exc:
            return exc.code

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "p"
        scaffold(root, "demo", "demo", with_build=True)
        out = Path(tmp) / "out.png"
        runs = [
            ["check", str(root), "--no-smoke"],
            ["build", str(root)],
            ["build", str(root), "--id", "demo:hut", "--out", str(out), "--plan", "4,6"],
            ["build", str(root), "--id", "demo:hut", "--out", str(out), "--site", "0",
             "--margin", "4", "--seed", "77"],
            ["sites", str(root), "--area", "512", "--limit", "3"],
            ["sites", str(root), "--set", "demo:huts", "--area", "256"],
            ["render", str(root), "--builds", "--size", "128", "--step", "8",
             "--views", "map", "--out", str(Path(tmp) / "map.png")],
            ["reference", "builds"],
        ]
        for argv in runs:
            code = run(argv)
            check(f"worldsmith {argv[0]} {' '.join(argv[2:4])} runs", code == 0, str(code))
        check("build --id wrote a drawing", out.is_file())
        check("build --plan wrote the slices too",
              out.with_name(out.stem + "_plan.png").is_file())

        # a placement worldsmith does not model, and a tag it cannot expand:
        # both are things a pack from elsewhere will have, and neither may crash
        writer = PackWriter(root, "demo")
        writer.add("structure_set", "demo:rings", {
            "structures": [{"structure": "demo:hut", "weight": 1}],
            "placement": {"type": "minecraft:concentric_rings", "distance": 32,
                          "spread": 3, "count": 128}})
        writer.add("structure", "demo:tagged", structure("demo:hut", "#demo:nowhere"))
        writer.add("template_pool", "demo:tagged", pool("demo:hut"))
        writer.add("structure_set", "demo:tagged", {
            "structures": [{"structure": "demo:tagged", "weight": 1}],
            "placement": {"type": "minecraft:random_spread", "spacing": 16,
                          "separation": 7, "salt": 3}})
        check("a placement that is not modelled is reported, not raised",
              run(["sites", str(root), "--area", "256"]) == 0)
        check("a tag that cannot be expanded is reported, not raised",
              run(["render", str(root), "--builds", "--size", "128", "--step", "8",
                   "--views", "map", "--out", str(Path(tmp) / "m2.png")]) == 0)

        for argv, why in ((["build", str(root), "--id", "demo:nope"], "an unknown build"),
                          (["sites", str(root), "--set", "demo:nope"], "an unknown set"),
                          (["build", str(root), "--id", "demo:hut", "--plan", "999",
                            "--out", str(out)], "a plan level past the top")):
            code = run(argv)
            check(f"{why} is refused with a message", isinstance(code, str) or code != 0,
                  str(code))


def test_biome_tags_resolve():
    """A structure usually names its biomes with a tag, and vanilla's tags name
    each other, so "where will this build go" needs them expanded rather than
    waved through."""
    from worldsmith.registry import Registries

    registries = Registries.load()
    forest = registries.biome_set("#minecraft:is_forest")
    check("a tag expands to the biomes it lists",
          forest and "minecraft:birch_forest" in forest, str(forest))
    ocean = registries.biome_set("#minecraft:is_ocean")
    check("a tag that names another tag expands through it",
          ocean and "minecraft:deep_frozen_ocean" in ocean, str(ocean))
    check("a list of ids and tags together expands to both",
          registries.biome_set(["minecraft:desert", "#minecraft:is_forest"])
          == forest | {"minecraft:desert"})
    check("a tag that is not here expands to nothing rather than a guess",
          registries.biome_set("#minecraft:no_such_tag") is None)
    check("an id without a namespace is still an id",
          registries.biome_set(["plains"]) == {"minecraft:plains"})
    check("a tag entry written as an object is read too",
          registries.biome_set([{"id": "minecraft:plains", "required": False}])
          == {"minecraft:plains"})

    from worldsmith.pack import PackWriter
    from worldsmith.structures import pool, spread, structure
    from worldsmith.validate import Validator
    from worldsmith.voxel import Grid

    with tempfile.TemporaryDirectory() as tmp:
        writer = PackWriter(Path(tmp) / "p", "test")
        writer.mcmeta()
        hut = Grid(2, 2, 2)
        hut.fill(0, 0, 0, 1, 0, 1, "minecraft:stone_bricks")
        writer.add_template("test:hut", hut)
        writer.add("structure", "test:hut", structure("test:hut", "#minecraft:is_forest"))
        writer.add("template_pool", "test:hut", pool("test:hut"))
        writer.add("structure_set", "test:huts",
                   spread("test:hut", spacing=8, separation=3, salt=1))
        loaded = Registries.load([writer.root])
        clean = [f.format() for f in Validator(loaded, loaded.packs[-1]).validate_pack()]
        check("a build keyed to a real tag validates", not clean, str(clean))

        writer.add("structure", "test:hut", structure("test:hut", "#minecraft:not_a_tag"))
        loaded = Registries.load([writer.root])
        found = [f.format() for f in Validator(loaded, loaded.packs[-1]).validate_pack()]
        check("a tag nothing defines is called out",
              any("biome tag" in f for f in found), str(found))


def test_shapes():
    """The geometry every build is made of. Each of these was written once by
    hand for a build and got something subtly wrong the first time."""
    from worldsmith.shapes import (crenellate, cylinder, fill, gable_roof, hollow_box,
                                   perimeter, ring_cells, speckle, stair_flight)
    from worldsmith.voxel import Grid

    mix = [(3, "minecraft:stone"), (1, "minecraft:cobblestone")]
    picks = [speckle(x, 0, z, mix) for x in range(40) for z in range(40)]
    check("speckle only returns blocks from the mix",
          set(picks) <= {"minecraft:stone", "minecraft:cobblestone"}, str(set(picks)))
    check("speckle answers the same for the same block",
          speckle(3, 4, 5, mix) == speckle(3, 4, 5, mix))
    share = picks.count("minecraft:stone") / len(picks)
    check("speckle follows the weights", 0.68 < share < 0.82, f"{share:.2f}")

    grid = Grid(9, 6, 9)
    fill(grid, 0, 0, 0, 8, 0, 8, lambda x, y, z: speckle(x, y, z, mix))
    check("fill takes a function as well as a block",
          {grid.name_at(x, 0, z) for x in range(9) for z in range(9)}
          == {"stone", "cobblestone"})

    hollow_box(grid, 0, 1, 0, 8, 4, 8, "minecraft:stone_bricks")
    check("a hollow box has walls", grid.name_at(0, 2, 4) == "stone_bricks")
    check("a hollow box is hollow", grid.name_at(4, 2, 4) == "air")
    check("a hollow box keeps its corners", grid.name_at(8, 4, 8) == "stone_bricks")
    check("a hollow box leaves one block of wall",
          grid.name_at(1, 2, 4) == "air" and grid.name_at(0, 2, 4) == "stone_bricks")

    round_grid = Grid(21, 4, 21)
    cylinder(round_grid, 10.5, 10.5, 8.0, 0, 3, "minecraft:stone", inner="minecraft:air")
    check("a cylinder is solid at its edge", round_grid.name_at(10, 1, 3) == "stone")
    check("a cylinder is hollow inside", round_grid.name_at(10, 1, 10) == "air")
    check("a cylinder stops at its radius", round_grid.name_at(10, 1, 0) == "")
    edge = ring_cells(10.5, 10.5, 8.0, size=(21, 21))
    check("a ring is a closed run of cells", len(edge) > 30, str(len(edge)))
    check("a ring runs around, not across",
          all(abs(a[0] - b[0]) <= 3 and abs(a[1] - b[1]) <= 3
              for a, b in zip(edge, edge[1:])), "the ring jumps")

    walk = perimeter(0, 0, 8, 8)
    check("a perimeter visits every edge cell once",
          len(walk) == len(set(walk)) == 32, f"{len(walk)} cells, {len(set(walk))} unique")
    check("a perimeter is a closed walk",
          all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
              for a, b in zip(walk, walk[1:] + walk[:1])), "the walk jumps")

    top = Grid(9, 2, 9)
    crenellate(top, walk, 1, "minecraft:stone")
    solid = [top.name_at(x, 1, z) == "stone" for x, z in walk]
    check("crenellations are two on and one off",
          solid[:6] == [True, True, False, True, True, False], str(solid[:6]))
    check("crenellations leave the gaps open", solid.count(False) == len(walk) // 3,
          f"{solid.count(False)} gaps in {len(walk)}")

    roof = Grid(9, 12, 9)
    ridge = gable_roof(roof, 0, 1, 8, 7, 6, "spruce")
    # nine wide closes in five courses, so the ridge is four above the eaves
    check("a gable roof closes to a ridge", ridge == 10, str(ridge))
    check("a gable roof is stairs on the outside",
          roof.name_at(0, 6, 4) == "spruce_stairs", roof.name_at(0, 6, 4))
    check("a gable roof is hollow under its pitch", roof.name_at(4, 6, 4) == "air",
          roof.name_at(4, 6, 4))
    check("a gable roof has a ceiling under it",
          roof.name_at(4, 5, 4) == "spruce_planks", roof.name_at(4, 5, 4))
    check("a gable roof leaves a roof space",
          all(roof.name_at(4, level, 4) == "air" for level in (6, 7, 8, 9)),
          str([roof.name_at(4, level, 4) for level in (6, 7, 8, 9)]))

    steps = Grid(12, 12, 12)
    end = stair_flight(steps, 1, 1, 1, "south", 6)
    check("a flight arrives where it says", end == (1, 7, 7), str(end))
    check("a flight climbs one block per step",
          steps.name_at(1, 3, 3) == "stone_brick_stairs", steps.name_at(1, 3, 3))
    check("a flight clears the space above it", steps.name_at(1, 5, 3) == "air")
    try:
        stair_flight(steps, 1, 1, 1, "up", 3)
        check("a flight refuses a direction that is not one", False, "accepted 'up'")
    except ValueError:
        check("a flight refuses a direction that is not one", True)


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
    test_canopy_from_features()
    test_canopy_cover_matches_the_field()
    test_decoration_is_paint_only()
    test_surface_blocks_against_the_game()
    test_clay_bands_against_the_game()
    test_packs_generate()
    test_aquifer_levels()
    test_aquifer_barrier()
    test_caves_and_decoration()
    test_grass_tint_and_block_colours()
    test_play_generates_structures()
    test_platform_paths()
    test_template_round_trip()
    test_structure_files()
    test_placement_geometry()
    test_structure_validation()
    test_set_reports()
    test_template_validation()
    test_build_overlay_and_determinism()
    test_packed_longs()
    test_scaffold_with_build()
    test_build_on_site()
    test_cli_smoke()
    test_biome_tags_resolve()
    test_shapes()
    print(f"{checks - len(failures)}/{checks} checks passed")
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
