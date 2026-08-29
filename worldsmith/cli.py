"""worldsmith: write Minecraft terrain in JSON and look at it before loading it.

    worldsmith new     packs/mesa --namespace mesa --name mesa
    worldsmith check   packs/mesa
    worldsmith render  packs/mesa --dimension mesa:mesa --out renders/mesa.png
    worldsmith probe   packs/mesa --dimension mesa:mesa --at 100 64 -200
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import play as play_mod
from .anvil import read_box, region_dir, structure_starts
from .climate import PARAM_NAMES, BiomeSource, assign_biomes, climate_target
from .density import Ctx, prepare
from .draw import mark_builds, render_iso, render_plan
from .pack import export_zip, scaffold
from .placement import Site, set_reports
from .reference import reference_text
from .registry import Registries
from .render import contact_sheet, render_biomes, render_height, render_map, render_section
from .scene import build_scene
from .structures import rotate_xz
from .terrain import cell_interpolated, sample_terrain
from .validate import ERROR, INFO, WARNING, Validator
from .voxel import Grid
from .world import World


def _resolve_dimension(registries: Registries, dimension: str | None):
    ids = registries.ids("dimension")
    custom = [i for i in ids if registries.origin("dimension", i) != registries.packs[0].name]
    if dimension is None:
        if len(custom) == 1:
            dimension = custom[0]
        elif custom:
            raise SystemExit("pack defines several dimensions; pick one with --dimension: "
                             + ", ".join(custom))
        else:
            raise SystemExit("pack defines no dimension; use --settings to render a noise_settings directly")
    obj = registries.get("dimension", dimension)
    if obj is None:
        raise SystemExit(f"unknown dimension '{dimension}' (have: {', '.join(ids)})")
    return dimension, obj


def _settings_biomes(registries, settings_id: str, biome: str | None) -> BiomeSource:
    """Biomes for a bare --settings render, which has no dimension to ask.

    Vanilla's own settings borrow the multi-noise preset of the same name, so
    `--settings minecraft:overworld` previews the biomes the game would place
    rather than one flat colour. Anything else, or an explicit --biome, is fixed.
    """
    if biome is None and registries.get("multi_noise_biome_source_parameter_list", settings_id):
        try:
            return BiomeSource.from_json(
                {"type": "minecraft:multi_noise", "preset": settings_id}, registries)
        except ValueError as exc:
            print(f"note: {exc}", file=sys.stderr)
    return BiomeSource.from_json({"type": "minecraft:fixed",
                                  "biome": biome or "minecraft:plains"})


def _load(args) -> tuple[World, BiomeSource, str]:
    paths = [args.pack] if getattr(args, "pack", None) else []
    registries = Registries.load(paths, version=args.version)
    if getattr(args, "settings", None):
        settings_id = args.settings
        source = _settings_biomes(registries, settings_id, getattr(args, "biome", None))
        label = settings_id
    else:
        dim_id, dimension = _resolve_dimension(registries, getattr(args, "dimension", None))
        generator = dimension.get("generator") or {}
        settings_id = generator.get("settings")
        if not isinstance(settings_id, str):
            raise SystemExit(f"{dim_id}: generator.settings must be a noise_settings id")
        try:
            source = BiomeSource.from_json(generator.get("biome_source") or {}, registries)
        except ValueError as exc:
            print(f"note: {exc}", file=sys.stderr)
            source = BiomeSource.from_json({"type": "minecraft:fixed", "biome": "minecraft:plains"})
        label = dim_id
    return World.create(registries, settings_id, args.seed), source, label


def _value_at(node, x, y, z) -> float:
    ctx = Ctx(np.array([[float(x)]]), np.array([[float(y)]]), np.array([[float(z)]]))
    return float(np.ravel(np.broadcast_to(np.asarray(node.eval(ctx), float), (1, 1)))[0])


def cmd_new(args):
    root = Path(args.directory)
    if root.exists() and any(root.iterdir()) and not args.force:
        raise SystemExit(f"{root} is not empty (use --force to overwrite)")
    try:
        writer = scaffold(root, args.namespace, args.name, template=args.template,
                          description=args.description, version=args.version,
                          replace_overworld=args.replace_overworld,
                          caves=args.caves, like=args.like)
    except ValueError as exc:            # unknown template or --like biome
        raise SystemExit(str(exc))
    print(f"created {root} ({len(writer.written)} files)")
    for rel in writer.relative():
        print("  ", rel)
    print(f"\nnext:  worldsmith render {root} --dimension {args.namespace}:{args.name}")
    return 0


def cmd_check(args):
    registries = Registries.load([args.pack], version=args.version)
    validator = Validator(registries, registries.packs[-1], args.version)
    findings = validator.validate_pack()
    counts = {level: sum(1 for f in findings if f.level == level) for level in (ERROR, WARNING, INFO)}
    if findings:
        print(validator.report())
        print()
    print(f"{counts[ERROR]} errors, {counts[WARNING]} warnings, {counts[INFO]} notes")

    if counts[ERROR] == 0 and not args.no_smoke:
        # A pack can be perfectly well-formed and still generate nothing.
        try:
            world, _, label = _load(args)
            stats = sample_terrain(world, -256, -256, 48, 48, step=16).stats()
            print(f"\nsmoke test ({label}, seed {args.seed}, 768x768 blocks):")
            if stats.get("empty") or stats.get("void_fraction", 0) > 0.98:
                print("  ERROR: nothing generated, the whole sample is air.")
                print("         final_density is never > 0 here; check the sign of depth/offset.")
                counts[ERROR] += 1
            else:
                print(f"  surface y {stats['min_y']}..{stats['max_y']} "
                      f"(median {stats['median_y']:.0f}), relief {stats['relief']}")
                print(f"  land {stats['land_fraction'] * 100:.0f}%  "
                      f"water {stats['water_fraction'] * 100:.0f}%  "
                      f"void {stats['void_fraction'] * 100:.0f}%")
                if stats["max_y"] >= world.noise.max_y - world.cell_height:
                    print("  WARNING: terrain reaches the top of the world, so the sky may be "
                          "solid stone; check the top slide in final_density.")
                if stats["relief"] <= 2:
                    print("  WARNING: the surface is almost perfectly flat.")
        except Exception as exc:
            print(f"\nsmoke test could not run: {exc}")
    return 1 if counts[ERROR] else 0


def _site_reports(args, world, source, x0, z0, span):
    """Every build site in a rendered area, for the overlay."""
    if not getattr(args, "pack", None):
        return []
    registries = Registries.load([args.pack], version=args.version)
    reports = []
    for set_id in sorted(registries.packs[-1].data["structure_set"]):
        reports += set_reports(registries, world, source, set_id, args.seed,
                               x0, z0, x0 + span - 1, z0 + span - 1)
    return reports


def cmd_render(args):
    started = time.time()
    world, source, label = _load(args)
    step = args.step
    n = max(8, args.size // step)
    x0 = args.center[0] - (n * step) // 2
    z0 = args.center[1] - (n * step) // 2

    known_views = ("map", "height", "biomes", "section")
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = [v for v in views if v not in known_views]
    if unknown:
        raise SystemExit(f"unknown view(s) {', '.join(unknown)}; --views takes "
                         f"{', '.join(known_views)}")
    if args.columns < 1:
        raise SystemExit("--columns must be at least 1")
    scene = build_scene(world, source, x0, z0, n, n, step=step)
    stats = scene.terrain.stats()

    panels = []
    if "map" in views:
        decorate = getattr(args, "decorate", False)
        surface = render_map(scene, scale=args.scale, decorate=decorate)
        note = ""
        if getattr(args, "builds", False):
            reports = _site_reports(args, world, source, x0, z0, n * step)
            mark_builds(surface, reports, x0, z0, step, args.scale)
            kept = sum(1 for r in reports if r.accepted)
            note = f"   {kept} of {len(reports)} build sites kept"
        panels.append((f"surface  {n * step}x{n * step} blocks @ ({args.center[0]},{args.center[1]})  "
                       f"1px = {step}b" + ("   canopy estimated from the biomes' features"
                                           if decorate else "") + note,
                       surface))
    if "height" in views:
        panels.append((f"elevation  y {stats['min_y']}..{stats['max_y']}  contours every 16",
                       render_height(scene.terrain, scale=args.scale)))
    if "biomes" in views and source.kind == "multi_noise" and len(set(scene.biomes)) > 1:
        panels.append((f"biomes  ({len(set(scene.biomes))} defined)", render_biomes(scene, scale=args.scale)))
    if "section" in views:
        length = n * step
        sect_step = max(1, length // 720)          # about 720px wide, block-accurate when it fits
        section = render_section(world, x0, args.center[1], length, scale=1,
                                 step=sect_step, y_step=1)
        panels.append((f"cross-section  z={args.center[1]}  x={x0}..{x0 + length}"
                       f"   1px = {sect_step}b wide, 1 block tall", section,
                       section.width > n * args.scale))

    footer = [
        f"seed {args.seed}   sea level {world.sea_level}   world {world.noise.min_y}..{world.noise.max_y}"
        f"   cell {world.cell_width}x{world.cell_height}",
        f"surface y {stats['min_y']}..{stats['max_y']} (median {stats['median_y']:.0f}),"
        f" relief {stats['relief']}"
        f"   land {stats['land_fraction'] * 100:.0f}%  water {stats['water_fraction'] * 100:.0f}%"
        f"  void {stats['void_fraction'] * 100:.0f}%",
        "surface blocks: " + ", ".join(f"{b.split(':')[-1]} {f * 100:.0f}%"
                                       for b, f in scene.block_histogram()[:6]),
    ]
    if source.kind == "multi_noise":
        footer.append("biomes: " + ", ".join(f"{b.split(':')[-1]} {f * 100:.0f}%"
                                             for b, f in scene.biome_histogram()[:6]))
    sheet = contact_sheet(panels, columns=args.columns, title=f"{label}   seed {args.seed}", footer=footer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({sheet.width}x{sheet.height}, {time.time() - started:.1f}s)")
    for line in footer:
        print("  " + line)
    return 0


def cmd_probe(args):
    world, source, label = _load(args)
    x, y, z = args.at
    prepare(*world.router.values())
    print(f"{label} @ ({x}, {y}, {z})  seed {args.seed}")
    for name, node in world.router.items():
        print(f"  {name:26s} {_value_at(node, x, y, z): .6f}")
    target = climate_target(world, np.array([x]), np.array([z]), np.array([y]))
    print("  climate at this column:")
    for i, pname in enumerate(PARAM_NAMES):
        print(f"    {pname:16s} {target[0, i]: .4f}")
    if source.kind == "multi_noise":
        idx = int(assign_biomes(source, target)[0])
        print(f"  -> biome {source.biomes[idx]}")
    for ident in args.density or []:
        node = world.compiler.compile_ref(ident)
        prepare(node)
        print(f"  {ident:26s} {_value_at(node, x, y, z): .6f}")
    return 0


def cmd_column(args):
    """Print one column as text, the fastest way to see what a change did."""
    world, _, label = _load(args)
    x, z = args.at
    terrain = sample_terrain(world, x, z, 1, 1, step=1)
    print(f"{label} column ({x}, {z})  seed {args.seed}")
    print(f"  surface y = {int(terrain.surface_y[0, 0])}   sea level {world.sea_level}")
    node = world.router["final_density"]
    prepare(node)
    ys = terrain.y_levels
    if not cell_interpolated(node):
        # caves are cut in below the interpolation, so they sit between the
        # lattice rows: a cave roof falls straight through an 8-block step.
        ys = np.arange(world.noise.min_y, world.noise.max_y + 1)
        print("  every block (density is not linear in y between lattice rows)")
    ctx = Ctx(np.array([[float(x)]]), ys[:, None].astype(float), np.array([[float(z)]]))
    density = np.ravel(np.broadcast_to(np.asarray(node.eval(ctx), float), (len(ys), 1)))
    for y, d in zip(ys[::-1], density[::-1]):
        bar = "#" * int(np.clip(d, 0, 1) * 30)
        print(f"   y {int(y):>5}  {d: .4f} {'solid' if d > 0 else 'air  '} {bar}")
    return 0


def cmd_export(args):
    out = Path(args.out) if args.out else Path(args.pack).with_suffix(".zip")
    export_zip(args.pack, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print("drop it in <world>/datapacks/, or in .minecraft/saves/<world>/datapacks/")
    return 0


def cmd_play(args):
    """One or more datapacks to worlds you can walk into."""
    packs = [Path(p).resolve() for p in args.packs]
    failed = 0
    for index, pack in enumerate(packs):
        if index:
            print()
        # only open the launcher once, after the last world is in place
        launch = args.launch and index == len(packs) - 1
        failed += play_one(args, pack, launch)
    if len(packs) > 1:
        print()
        print(f"{len(packs) - failed} of {len(packs)} worlds installed; "
              "they are all in Singleplayer.")
    return 1 if failed else 0


def play_one(args, pack: Path, launch: bool) -> int:
    registries = Registries.load([str(pack)], version=args.version)
    validator = Validator(registries, registries.packs[-1], args.version)
    errors = [f for f in validator.validate_pack() if f.level == ERROR]
    if errors:
        print("this pack has errors that would stop Minecraft loading it:\n")
        for finding in errors[:10]:
            print("  " + finding.format())
        print(f"\nfix those first:  worldsmith check {pack}")
        return 1

    dim_id, dimension = _resolve_dimension(registries, args.dimension)
    settings_id = (dimension.get("generator") or {}).get("settings")
    world = World.create(registries, settings_id, args.seed)
    name = args.name or dim_id.split(":")[-1].replace("_", " ").title()

    print(f"{name}   (from {dim_id}, seed {args.seed})")
    print("  1/5  looking for somewhere worth spawning")
    spawn = None
    if args.spawn_at:
        source = BiomeSource.from_json((dimension.get("generator") or {}).get("biome_source") or {},
                                       registries)
        spawn = play_mod.viewpoint_at_build(world, source, registries, args.spawn_at, args.seed)
        if spawn is None:
            print(f"       no {args.spawn_at} within reach; falling back to the terrain")
    spawn = spawn or play_mod.pick_viewpoint(world)
    print(f"       spawn at ({spawn.x}, {spawn.y}, {spawn.z}) - {spawn.note}")

    print("  2/5  fetching the Minecraft server and a Java runtime (cached after the first run)")
    runtime = play_mod.ensure_runtime(args.mc_version)

    print("  3/5  generating the world")
    work = Path(args.work) if args.work else (play_mod.RUNTIME / "build" / pack.name)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    staged = play_mod.as_overworld(pack, dim_id, registries, work / "pack")
    world_dir = play_mod.generate_world(runtime, work, staged, args.seed, spawn,
                                        args.radius, args.gamemode, args.pregen)

    print("  4/5  installing into Minecraft")
    dest = play_mod.install_world(world_dir, name)
    zipped = play_mod.install_datapack(pack)
    print(f"       world     {dest}")
    if zipped:
        print(f"       datapack  {zipped}")

    if launch:
        print("  5/5  opening the Minecraft launcher")
        opened = play_mod.launch_minecraft()
        print("       " + (f"launched {opened}" if opened else
                           "could not find the launcher, start Minecraft yourself"))
    else:
        print("  5/5  not launching")

    print()
    print(f"  In Minecraft {args.mc_version}:  Singleplayer  ->  {name}")
    print(f"  You spawn at ({spawn.x}, {spawn.y}, {spawn.z}) in creative, with cheats on.")
    if spawn.landmark:
        lx, lz, ly = spawn.landmark
        print(f"  The big landmark is at x {lx}, z {lz}, topping out at y {ly}.")
        print(f"      /tp @s {lx} {ly + 3} {lz}      to stand on top of it")
    print(f"      /tp @s {spawn.x} {spawn.y} {spawn.z}      back to spawn")
    print("  Press F3 for coordinates, double-tap space to fly.")
    return 0


def _build_world(args, registries):
    """The dimension to survey against. A pack of builds and nothing else is a
    normal use, so fall back to the vanilla overworld rather than refusing."""
    ids = registries.ids("dimension")
    custom = [i for i in ids if registries.origin("dimension", i) != registries.packs[0].name]
    if args.dimension or custom:
        dim_id, dimension = _resolve_dimension(registries, args.dimension)
        settings = (dimension.get("generator") or {}).get("settings")
        source = BiomeSource.from_json((dimension.get("generator") or {}).get("biome_source") or {},
                                       registries)
        return World.create(registries, settings, args.seed), source, dim_id
    source = BiomeSource.from_json(
        {"type": "minecraft:multi_noise", "preset": "minecraft:overworld"}, registries)
    return World.create(registries, "minecraft:overworld", args.seed), source, "minecraft:overworld"


def cmd_build(args):
    """Draw a build, or list the ones a pack has."""
    registries = Registries.load([args.pack], version=args.version)
    templates = registries.packs[-1].templates
    if not templates:
        raise SystemExit(f"{args.pack} has no builds under data/<namespace>/structure/")
    if not args.id:
        print(f"{len(templates)} build(s) in {args.pack}")
        for ident in sorted(templates):
            grid = Grid.load(templates[ident])
            print(f"  {ident:34s} {grid.sx}x{grid.sy}x{grid.sz}  "
                  f"{grid.filled():7d} blocks  {len(grid.palette):3d} states")
        return 0
    if args.id not in templates:
        raise SystemExit(f"no build {args.id} (have: {', '.join(sorted(templates))})")

    grid = Grid.load(templates[args.id])
    name = args.id.split(":")[-1].replace("/", "_")
    out = Path(args.out) if args.out else Path("renders") / f"{name}.png"
    render_iso(grid, out, scale=args.scale, turn=args.turn, label=args.id)
    print(f"wrote {out}  ({grid.sx}x{grid.sy}x{grid.sz}, {grid.filled()} blocks)")
    if args.plan:
        levels = [int(v) for v in args.plan.split(",")]
        bad = [v for v in levels if not 0 <= v < grid.sy]
        if bad:
            raise SystemExit(f"--plan level(s) {bad} are outside 0..{grid.sy - 1}")
        plan = out.with_name(out.stem + "_plan.png")
        render_plan(grid, plan, levels, scale=max(2, args.scale // 2), label=args.id)
        print(f"wrote {plan}  (levels {', '.join(str(v) for v in levels)})")
    return 0


def cmd_sites(args):
    """Where the game will put a pack's builds, and what the ground is like."""
    registries = Registries.load([args.pack], version=args.version)
    world, source, label = _build_world(args, registries)
    sets = [args.set] if args.set else sorted(registries.packs[-1].data["structure_set"])
    if not sets:
        raise SystemExit(f"{args.pack} defines no structure sets")

    half = args.area // 2
    box = (args.center[0] - half, args.center[1] - half,
           args.center[0] + half, args.center[1] + half)
    print(f"{label}, seed {args.seed}, {args.area}x{args.area} blocks "
          f"around ({args.center[0]}, {args.center[1]})")
    for set_id in sets:
        entry = registries.get("structure_set", set_id)
        if entry is None:
            raise SystemExit(f"unknown structure set {set_id}")
        placement = entry.get("placement") or {}
        reports = set_reports(registries, world, source, set_id, args.seed, *box)
        kept = [r for r in reports if r.accepted]
        print()
        print(f"{set_id}  spacing {placement.get('spacing')}, "
              f"separation {placement.get('separation')}")
        print(f"  {len(kept)} of {len(reports)} sites kept")
        for report in sorted(reports, key=lambda r: abs(r.site.x) + abs(r.site.z))[:args.limit]:
            verdict = "kept" if report.accepted else f"no: {report.biome.split(':')[-1]}"
            print(f"    {report.build.split(':')[-1]:16s} x {report.box[0]:7d} z {report.box[1]:7d}  "
                  f"{report.biome.split(':')[-1]:14s} ground y {report.surface_y:3d}  "
                  f"relief {report.relief:3d}  water {report.water * 100:3.0f}%  {verdict}")
        if kept:
            worst = max(kept, key=lambda r: r.relief)
            print(f"  roughest ground under a kept build: {worst.relief} blocks "
                  f"under {worst.build.split(':')[-1]} at x {worst.box[0]} z {worst.box[1]}")
    return 0


def cmd_inspect(args):
    """What a generated world actually contains, which is the only record of
    what the game did rather than what the pack asked for."""
    world = Path(args.world)
    if not region_dir(world).is_dir():
        raise SystemExit(f"no region files under {world}")
    starts = structure_starts(world)
    if not starts:
        print("no structures in this world")
        return 1

    for ident, entries in sorted(starts.items()):
        unique = {tuple(e["centre"]): e for e in entries}
        print(f"{ident}: {len(unique)} placed")
        for centre, entry in sorted(unique.items())[:args.limit]:
            box = entry["box"]
            print(f"    x {centre[0]:7d} z {centre[1]:7d}   y {box[1]}..{box[4]}   "
                  f"{box[3] - box[0] + 1}x{box[5] - box[2] + 1}   {entry['rotation'].lower()}")
        if len(unique) > args.limit:
            print(f"    ... {len(unique) - args.limit} more")

    if not args.structure:
        return 0
    entries = starts.get(args.structure)
    if not entries:
        raise SystemExit(f"{args.structure} is not in this world")
    unique = sorted({tuple(e["centre"]): e for e in entries}.values(),
                    key=lambda e: abs(e["centre"][0]) + abs(e["centre"][1]))
    pick = unique[min(args.index, len(unique) - 1)]
    box = pick["box"]
    print(f"\n{args.structure} at {tuple(pick['centre'])}, turned {pick['rotation'].lower()}")

    if args.render:
        pad = args.pad
        grid = read_box(world, box[0] - pad, box[2] - pad, box[3] + pad, box[5] + pad,
                        box[1] - 4, box[4] + 2)
        out = render_iso(grid, args.render, scale=args.scale, turn=args.turn,
                         label=f"{args.structure} at x {pick['centre'][0]} "
                               f"z {pick['centre'][1]}, as generated")
        print(f"  wrote {out}  ({grid.filled()} blocks read)")

    if not args.pack:
        return 0
    registries = Registries.load([args.pack], version=args.version)
    template_id = (registries.get("structure", args.structure) or {}).get("start_pool")
    path = registries.templates.get(template_id or args.structure)
    if path is None:
        print(f"  {args.pack} has no template for {args.structure}, nothing to compare")
        return 0

    # the model predicted where this would land before the world existed, so
    # say whether it was right
    world_model, source, _ = _build_world(args, registries)
    owner = next((ident for ident in registries.ids("structure_set")
                  if any(e.get("structure") == args.structure
                         for e in (registries.get("structure_set", ident) or {}).get("structures") or [])),
                 None)
    if owner:
        site = Site(pick["chunk"][0], pick["chunk"][1])
        predicted = next((r for r in set_reports(registries, world_model, source, owner,
                                                 args.seed, site.x - 1, site.z - 1,
                                                 site.x + 1, site.z + 1)
                          if (r.site.chunk_x, r.site.chunk_z) == pick["chunk"]), None)
        if predicted is not None:
            agree = (predicted.floor_y == box[1] and predicted.rotation == pick["rotation"]
                     and predicted.box[0] == box[0] and predicted.box[1] == box[2])
            print(f"  the model said: {predicted.build.split(':')[-1]} at x {predicted.box[0]} "
                  f"z {predicted.box[1]}, floor y {predicted.floor_y}, "
                  f"{predicted.rotation.lower()}  ->  {'agrees' if agree else 'DISAGREES'}")

    template = Grid.load(path)
    placed = read_box(world, box[0], box[2], box[3], box[5], box[1], box[4])
    same, missing = 0, {}
    for x in range(template.sx):
        for y in range(template.sy):
            for z in range(template.sz):
                wanted = template.get(x, y, z)
                if wanted is None:
                    continue
                wx, wz = rotate_xz(x, z, template.sx, template.sz, pick["rotation"])
                found = placed.get(wx, y, wz) or "minecraft:air"
                if found.split("[")[0] == wanted.split("[")[0]:
                    same += 1
                else:
                    key = (wanted.split("[")[0], found.split("[")[0])
                    missing[key] = missing.get(key, 0) + 1
    total = same + sum(missing.values())
    print(f"  {same}/{total} blocks are what the template asked for "
          f"({100.0 * same / max(1, total):.2f}%)")
    for (wanted, found), n in sorted(missing.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {n:6d}  wanted {wanted.split(':')[-1]:22s} found {found.split(':')[-1]}")
    return 0


def cmd_reference(args):
    print(reference_text(args.topic))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldsmith", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, pack=True):
        if pack:
            p.add_argument("pack", help="path to the datapack directory")
        p.add_argument("--version", default="26.2", help="Minecraft version to validate against")
        p.add_argument("--seed", type=int, default=12345)
        p.add_argument("--dimension", help="dimension id to render (default: the pack's only one)")
        p.add_argument("--settings", help="render a noise_settings id directly instead of a dimension")
        p.add_argument("--biome", help="biome to assume when rendering --settings")

    p = sub.add_parser("new", help="scaffold a datapack that already loads")
    p.add_argument("directory")
    p.add_argument("--namespace", default="worldsmith")
    p.add_argument("--name", default="custom")
    p.add_argument("--template", default="basic")
    p.add_argument("--description", default="")
    p.add_argument("--version", default="26.2")
    p.add_argument("--replace-overworld", action="store_true",
                   help="also write data/minecraft/dimension/overworld.json so new worlds use this terrain")
    p.add_argument("--caves", action="store_true",
                   help="cut vanilla's cave functions into the terrain, and turn on the "
                        "aquifers that keep them from flooding (the preview shows them; "
                        "renders take the slower exact scan)")
    p.add_argument("--like", metavar="BIOME",
                   help="borrow the trees, ores, mobs and carvers of a vanilla biome, "
                        "e.g. minecraft:plains (the game places them; the preview does not)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("check", help="validate a pack and smoke-test that it generates terrain")
    common(p)
    p.add_argument("--no-smoke", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("render", help="render preview images")
    common(p)
    p.add_argument("--out", default="renders/preview.png")
    p.add_argument("--size", type=int, default=1024, help="side length in blocks")
    p.add_argument("--step", type=int, default=4, help="blocks per pixel (4 = the noise cell width)")
    p.add_argument("--scale", type=int, default=3, help="pixel scale of the output image")
    p.add_argument("--center", type=int, nargs=2, default=[0, 0], metavar=("X", "Z"))
    p.add_argument("--views", default="map,height,biomes,section")
    p.add_argument("--columns", type=int, default=2)
    p.add_argument("--builds", action="store_true",
                   help="outline where this pack's builds land")
    p.add_argument("--decorate", action="store_true",
                   help="stipple the canopy the biomes' tree features imply (an estimate, "
                        "not a placement)")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("probe", help="print every router value at one position")
    common(p)
    p.add_argument("--at", type=int, nargs=3, default=[0, 64, 0], metavar=("X", "Y", "Z"))
    p.add_argument("--density", nargs="*", help="extra density function ids to evaluate")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("column", help="print the density of one column top to bottom")
    common(p)
    p.add_argument("--at", type=int, nargs=2, default=[0, 0], metavar=("X", "Z"))
    p.set_defaults(func=cmd_column)

    p = sub.add_parser("build", help="draw a build in the pack, or list them")
    common(p)
    p.add_argument("--id", help="which build to draw (default: list them)")
    p.add_argument("--out", default=None)
    p.add_argument("--scale", type=int, default=6, help="pixels per block")
    p.add_argument("--turn", type=int, default=0, help="quarter turns to view from")
    p.add_argument("--plan", help="also draw plan slices at these heights, e.g. 10,17,24")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("sites", help="where the game will put this pack's builds")
    common(p)
    p.add_argument("--set", help="one structure set (default: all of the pack's)")
    p.add_argument("--area", type=int, default=2048, help="side length in blocks to search")
    p.add_argument("--center", type=int, nargs=2, default=[0, 0], metavar=("X", "Z"))
    p.add_argument("--limit", type=int, default=12, help="sites to list per set")
    p.set_defaults(func=cmd_sites)

    p = sub.add_parser("export", help="zip the pack for dropping into a world")
    p.add_argument("pack")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("play", help="build the world(s) and put them in Singleplayer")
    p.add_argument("packs", nargs="+", help="one or more datapack directories")
    common(p, pack=False)
    p.add_argument("--name", help="world name in the Singleplayer list")
    p.add_argument("--mc-version", default="26.2", help="Minecraft version to generate with")
    p.add_argument("--gamemode", default="creative")
    p.add_argument("--radius", type=int, default=384, help="blocks to pre-build around spawn")
    p.add_argument("--pregen", type=int, default=90, help="seconds to spend pre-building (0 to skip)")
    p.add_argument("--work", default=None)
    p.add_argument("--spawn-at", help="spawn beside this build instead of at the terrain")
    p.add_argument("--no-launch", dest="launch", action="store_false", default=True)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("inspect", help="what a generated world actually contains")
    p.add_argument("world", help="path to a world directory")
    p.add_argument("--pack", help="compare a placed build against this pack's template")
    p.add_argument("--version", default="26.2")
    p.add_argument("--seed", type=int, default=12345,
                   help="the seed the world was made with, to check the model against it")
    p.add_argument("--dimension", help="dimension id, when the pack defines several")
    p.add_argument("--structure", help="which build to look at closely")
    p.add_argument("--index", type=int, default=0, help="which one of them, nearest first")
    p.add_argument("--render", help="draw it as it stands, to this file")
    p.add_argument("--pad", type=int, default=12, help="blocks of terrain around it")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--turn", type=int, default=0)
    p.add_argument("--limit", type=int, default=8, help="positions to list per structure")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("reference", help="print the worldgen syntax reference")
    p.add_argument("topic", nargs="?", default="index")
    p.set_defaults(func=cmd_reference)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
