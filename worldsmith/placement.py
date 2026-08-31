"""Where the game will put a build, worked out before the game is asked.

A structure set's placement is deterministic: seed, salt, spacing and separation
pick one chunk in every spacing-by-spacing region, and the structure appears
there if the biome underneath it is one the structure accepts. Reproducing that
here is what lets the preview draw builds on the terrain it already draws, and
what lets a pack say whether the ground under each site is worth building on.

This is the seam between the two halves of worldsmith: terrain knows nothing
about builds, builds know nothing about terrain, and this module is where the
one is asked about the other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .climate import assign_biomes, climate_target
from .jrandom import JavaRandom, to_signed64
from .scene import build_scene
from .structures import rotate_xz
from .terrain import base_height, sample_terrain
from .voxel import Grid

ROTATIONS = ("NONE", "CLOCKWISE_90", "CLOCKWISE_180", "COUNTERCLOCKWISE_90")


@dataclass(frozen=True)
class Site:
    """A chunk the game would consider, and the block a build is anchored at.

    A jigsaw structure starts at the chunk's minimum corner, not its middle, and
    that is also where the biome is tested. Eight blocks is enough to answer with
    a different biome at a coast, which is the difference between a build
    appearing and not: measured against a server, using the middle put nine
    sites in 508 on the wrong side.
    """
    chunk_x: int
    chunk_z: int

    @property
    def x(self) -> int:
        return self.chunk_x * 16

    @property
    def z(self) -> int:
        return self.chunk_z * 16


def chunk_random(seed: int, chunk_x: int, chunk_z: int) -> JavaRandom:
    """The structure random for one chunk, seeded the way the game seeds it.
    Rotation and the choice of build are its first two draws."""
    random = JavaRandom(0)
    random.set_seed(seed)
    a, b = random.next_long(), random.next_long()
    random.set_seed(to_signed64(to_signed64(chunk_x * a) ^ to_signed64(chunk_z * b) ^ seed))
    return random


def site_rotation(seed: int, chunk_x: int, chunk_z: int) -> str:
    """The turn the game gives a build here, without generating anything."""
    return ROTATIONS[chunk_random(seed, chunk_x, chunk_z).next_int(4)]


def chosen_build(seed: int, site: Site, entries: list[dict]) -> str | None:
    """Which build of a set the game tries at this site.

    A set holds several builds with weights and picks one per site. A build the
    game then refuses is dropped and another drawn, so this is the first choice
    rather than the last word. Where the builds of a set share a biome list,
    which is the usual case, the first choice is what gets built.
    """
    weights = [(e.get("structure"), int(e.get("weight", 1))) for e in entries]
    total = sum(w for _, w in weights)
    if total <= 0:
        return None
    pick = chunk_random(seed, site.chunk_x, site.chunk_z).next_int(total)
    for ident, weight in weights:
        pick -= weight
        if pick < 0:
            return ident
    return weights[-1][0]


def footprint(site: Site, size_x: int, size_z: int, rotation: str) -> tuple[int, int, int, int]:
    """The block box a build of this size covers, once turned and anchored.

    The template is anchored at the site and turned about that corner, so the
    box grows away from it in a direction that depends on the rotation.
    """
    dx, dz = size_x - 1, size_z - 1
    corners = {"NONE": (0, 0, dx, dz),
               "CLOCKWISE_90": (-dz, 0, 0, dx),
               "CLOCKWISE_180": (-dx, -dz, 0, 0),
               "COUNTERCLOCKWISE_90": (0, -dx, dz, 0)}.get(rotation)
    if corners is None:
        raise ValueError(f"unknown rotation {rotation!r}, want one of {ROTATIONS}")
    return (site.x + corners[0], site.z + corners[1],
            site.x + corners[2], site.z + corners[3])


def box_centre(box: tuple[int, int, int, int]) -> tuple[int, int]:
    """Where the game reads the ground and the biome: the middle of the box a
    build will occupy, not the chunk it was picked in."""
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)   # Java truncates


@dataclass
class SiteReport:
    site: Site
    build: str
    rotation: str
    box: tuple[int, int, int, int]     # x0, z0, x1, z1 the build covers
    biome: str
    floor_y: int          # world y of the build's own y=0
    surface_y: int
    low: int
    high: int
    water: float          # fraction of the footprint below sea level
    accepted: bool        # the structure's biome list allows this site

    @property
    def relief(self) -> int:
        return self.high - self.low


def region_chunk(seed: int, placement: dict, region_x: int, region_z: int) -> Site:
    """Vanilla's RandomSpreadStructurePlacement, one region at a time."""
    spacing = int(placement["spacing"])
    separation = int(placement["separation"])
    salt = int(placement.get("salt", 0))
    span = spacing - separation
    random = JavaRandom(0)
    random.set_seed(region_x * 341873128712 + region_z * 132897987541 + seed + salt)
    if placement.get("spread_type", "linear") == "triangular":
        offset_x = (random.next_int(span) + random.next_int(span)) // 2
        offset_z = (random.next_int(span) + random.next_int(span)) // 2
    else:
        offset_x = random.next_int(span)
        offset_z = random.next_int(span)
    return Site(region_x * spacing + offset_x, region_z * spacing + offset_z)


def sites(placement: dict, seed: int, x0: int, z0: int, x1: int, z1: int) -> list[Site]:
    """Every site whose chunk falls in this block area."""
    kind = placement.get("type", "minecraft:random_spread")
    if kind != "minecraft:random_spread":
        raise NotImplementedError(f"{kind} placement is not modelled")
    if float(placement.get("frequency", 1.0)) < 1.0:
        raise NotImplementedError("a frequency below 1 is not modelled")
    spacing = int(placement["spacing"])
    chunks = [c // 16 for c in (x0, z0, x1, z1)]
    regions_x = range(chunks[0] // spacing, chunks[2] // spacing + 1)
    regions_z = range(chunks[1] // spacing, chunks[3] // spacing + 1)
    found = [region_chunk(seed, placement, rx, rz)
             for rx in regions_x for rz in regions_z]
    return [s for s in found
            if chunks[0] <= s.chunk_x <= chunks[2] and chunks[1] <= s.chunk_z <= chunks[3]]


def set_sites(registries, set_id: str, seed: int, x0: int, z0: int, x1: int, z1: int) -> list[Site]:
    """The sites of one structure set, with its exclusion zone applied."""
    placement = (registries.get("structure_set", set_id) or {}).get("placement") or {}
    found = sites(placement, seed, x0, z0, x1, z1)
    zone = placement.get("exclusion_zone")
    if not zone:
        return found
    reach = int(zone["chunk_count"])
    margin = reach * 16
    other = set_sites(registries, zone["other_set"], seed,
                      x0 - margin, z0 - margin, x1 + margin, z1 + margin)
    blocked = {(s.chunk_x, s.chunk_z) for s in other}
    return [s for s in found
            if not any((s.chunk_x + dx, s.chunk_z + dz) in blocked
                       for dx in range(-reach, reach + 1)
                       for dz in range(-reach, reach + 1))]


def survey(world, source, found: list[Site], *, seed: int, biomes=None, sink: int = 0,
           size: tuple[int, int] = (1, 1), step: int = 8, build: str = "",
           ground: bool = True) -> list[SiteReport]:
    """What the ground is like where each site landed, and whether the game will
    accept it.

    The biome is read where the game reads it: the middle of the box the build
    will occupy, at the WORLD_SURFACE_WG height there plus the structure's
    start_height. `floor_y` is where the build's own y=0 lands, which is that
    height less one for the ground level delta a template without jigsaw blocks
    carries; measured against 46 placed builds it is exact.

    `biomes` is the structure's biome list, already expanded. Sites outside it
    are reported rather than dropped, because "my structure never generates" is
    usually this.
    """
    if not found:
        return []
    allowed = None if biomes is None else set(biomes)
    boxes, anchors = [], []
    for site in found:
        rotation = site_rotation(seed, site.chunk_x, site.chunk_z)
        box = footprint(site, size[0], size[1], rotation)
        boxes.append((rotation, box))
        anchors.append(box_centre(box))

    xs = np.array([a[0] for a in anchors], dtype=np.int64)
    zs = np.array([a[1] for a in anchors], dtype=np.int64)
    ground_at = base_height(world, xs, zs)
    picks = assign_biomes(source, climate_target(world, xs, zs, ground_at + sink))

    reports = []
    for i, site in enumerate(found):
        rotation, box = boxes[i]
        if ground:
            cells_x = max(2, (box[2] - box[0] + 1) // step)
            cells_z = max(2, (box[3] - box[1] + 1) // step)
            heights = sample_terrain(world, box[0], box[1], cells_x, cells_z,
                                     step=step).surface_y
        else:
            # one sample_terrain per site is the whole cost of a survey, and a
            # caller that only wants to know where the builds are does not need it
            heights = np.full((1, 1), ground_at[i] - 1)
        biome = source.biomes[int(picks[i])] if source.biomes else ""
        reports.append(SiteReport(
            site=site,
            build=build,
            rotation=rotation,
            box=box,
            biome=biome,
            floor_y=int(ground_at[i]) + sink - 1,
            surface_y=int(ground_at[i]) - 1,
            low=int(heights.min()),
            high=int(heights.max()),
            water=float((heights < world.sea_level).mean()),
            accepted=allowed is None or biome in allowed,
        ))
    return reports


def owning_set_id(registries, structure_id: str) -> str | None:
    """The structure set that places this build, if one does."""
    return next((ident for ident in registries.ids("structure_set")
                 if any(entry.get("structure") == structure_id
                        for entry in (registries.get("structure_set", ident)
                                      or {}).get("structures") or [])),
                None)


def owning_set(registries, structure_id: str) -> tuple[dict, str]:
    """A build and the set that places it. Either half missing is the reason it
    never appears, so say which."""
    structure = registries.get("structure", structure_id)
    if structure is None:
        raise SystemExit(f"unknown structure {structure_id}")
    owner = owning_set_id(registries, structure_id)
    if owner is None:
        raise SystemExit(f"no structure set places {structure_id}")
    return structure, owner


def set_reports(registries, world, source, set_id: str, seed: int,
                x0: int, z0: int, x1: int, z1: int,
                ground: bool = True) -> list[SiteReport]:
    """Every site of one structure set, surveyed as the build that will stand there.

    A set can hold builds of different sizes and different biome lists, and the
    game picks one per site, so each site is measured against its own build.
    """
    entry = registries.get("structure_set", set_id)
    if entry is None:
        raise ValueError(f"unknown structure set {set_id}")
    entries = entry.get("structures") or []
    by_build: dict[str, list[Site]] = {}
    for site in set_sites(registries, set_id, seed, x0, z0, x1, z1):
        ident = chosen_build(seed, site, entries)
        if ident:
            by_build.setdefault(ident, []).append(site)

    reports = []
    for ident, group in by_build.items():
        structure = registries.get("structure", ident) or {}
        template = registries.start_template(structure)
        size = (1, 1)
        if template is not None:
            grid = Grid.load(template)
            size = (grid.sx, grid.sz)
        reports += survey(world, source, group, seed=seed,
                          biomes=registries.biome_set(structure.get("biomes")),
                          sink=int((structure.get("start_height") or {}).get("absolute", 0)),
                          size=size, build=ident, ground=ground)
    return sorted(reports, key=lambda r: (r.site.chunk_x, r.site.chunk_z))


def build_on_site(registries, world, source, structure_id: str, seed: int,
                  index: int = 0, margin: int = 16, reach: int = 4096):
    """The build standing on the ground it will actually stand on.

    Terrain is sampled around the site and the template is pasted into it at the
    height the game will put it, so the picture is the two halves together
    without generating a world. The ground the game will lay against the build
    itself is not modelled, so this is the land before the build lands on it.
    """
    if index < 0:
        raise SystemExit(f"site index must be 0 or more, got {index}")
    structure, owner = owning_set(registries, structure_id)
    kept: list[SiteReport] = []
    span = 1024
    while span <= reach and len(kept) <= index:
        kept = [r for r in set_reports(registries, world, source, owner, seed,
                                       -span, -span, span, span, ground=False)
                if r.accepted and r.build == structure_id]
        span *= 2
    if not kept:
        raise SystemExit(f"no site within {reach} blocks keeps {structure_id}")
    kept.sort(key=lambda r: abs(r.site.x) + abs(r.site.z))
    if index >= len(kept):
        raise SystemExit(f"{structure_id} has {len(kept)} site(s) within {reach} "
                         f"blocks, so there is no site {index}")
    chosen = kept[index]

    path = registries.start_template(structure)
    if path is None:
        raise SystemExit(f"{structure_id} has no template to draw, "
                         f"check the pool {structure.get('start_pool')} names one")
    template = Grid.load(path)
    # the ground was skipped while picking, so measure it for the one site chosen
    report = survey(world, source, [chosen.site], seed=seed, build=structure_id,
                    biomes=registries.biome_set(structure.get("biomes")),
                    sink=int((structure.get("start_height") or {}).get("absolute", 0)),
                    size=(template.sx, template.sz))[0]
    x0, z0 = report.box[0] - margin, report.box[1] - margin
    width = report.box[2] - report.box[0] + 1 + 2 * margin
    depth = report.box[3] - report.box[1] + 1 + 2 * margin

    scene = build_scene(world, source, x0, z0, width, depth, step=1)
    heights = scene.terrain.surface_y
    sea = world.sea_level
    floor = report.floor_y
    low = int(min(heights.min(), floor)) - 3
    high = int(max(heights.max(), sea, floor + template.sy)) + 1

    grid = Grid(width, high - low + 1, depth)
    for iz in range(depth):
        for ix in range(width):
            top = int(heights[iz, ix])
            surface = scene.palette[int(scene.surface_block[iz, ix])]
            grid.fill(ix, 0, iz, ix, min(top, high) - low - 1, iz, "minecraft:stone")
            if low <= top <= high:
                grid.set(ix, top - low, iz, surface)
            if top < sea:
                # the top water block is at sea_level - 1, as it is in game
                grid.fill(ix, max(top + 1, low) - low, iz, ix, sea - 1 - low, iz,
                          "minecraft:water[level=0]")

    for (tx, ty, tz), spec in template.items():
        wx, wz = rotate_xz(tx, tz, template.sx, template.sz, report.rotation)
        gx, gz = report.box[0] + wx - x0, report.box[1] + wz - z0
        gy = floor + ty - low
        if 0 <= gy < grid.sy:
            grid.set(gx, gy, gz, spec)
    return grid, report
