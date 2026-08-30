"""The worldgen JSON that makes the game place a build.

Four files per build: the template (voxel.Grid), the structure, a pool holding
the template, and a set saying how often it appears. `add` writes the first
three; `spread` writes the set, which can hold several structures with weights.

Two defaults here are the ones that are painful to get wrong:

* the pool element is `legacy_single_pool_element`. The modern
  `single_pool_element` ignores air in a template, so every room a build hollows
  out stays solid and every trench it digs fills in.
* `terrain_adaptation` is `beard_box`, the way the ancient city does it: the
  whole box is cleared of terrain, so a hillside cannot stand inside the build.
"""
from __future__ import annotations


def structure(pool: str, biomes, *, sink: int = 0, step: str = "surface_structures",
              adaptation: str = "beard_box", heightmap: str | None = "WORLD_SURFACE_WG",
              size: int = 1, max_distance: int = 116) -> dict:
    """`sink` is how far below the surface the template's y=0 lands, so a build
    whose ground floor sits at y=N wants sink=-(N+1)."""
    out = {
        "type": "minecraft:jigsaw",
        "biomes": biomes if isinstance(biomes, str) else list(biomes),
        "step": step,
        "terrain_adaptation": adaptation,
        "spawn_overrides": {},
        "start_pool": pool,
        "size": size,
        "start_height": {"absolute": sink},
        "max_distance_from_center": max_distance,
        "use_expansion_hack": False,
    }
    if heightmap:
        out["project_start_to_heightmap"] = heightmap
    return out


def pool(template: str) -> dict:
    """A pool of one build.

    The projection is rigid, which is the only one the rest of worldsmith
    follows: a terrain_matching build steps with the ground, so its floor is not
    one height and `sites` could not tell you what it is.
    """
    return {
        "fallback": "minecraft:empty",
        "elements": [{
            "weight": 1,
            "element": {
                "element_type": "minecraft:legacy_single_pool_element",
                "location": template,
                "processors": "minecraft:empty",
                "projection": "rigid",
            },
        }],
    }


def spread(structures, *, spacing: int, separation: int, salt: int,
           exclusion: tuple[str, int] | None = None) -> dict:
    """`structures` is an id, or {id: weight}. Spacing and separation are in
    chunks, and separation must be smaller than spacing."""
    if isinstance(structures, str):
        structures = {structures: 1}
    placement = {"type": "minecraft:random_spread", "spacing": spacing,
                 "separation": separation, "salt": salt}
    if exclusion:
        other, chunks = exclusion
        placement["exclusion_zone"] = {"other_set": other, "chunk_count": chunks}
    return {
        "structures": [{"structure": ident, "weight": weight}
                       for ident, weight in structures.items()],
        "placement": placement,
    }


def add(writer, ident: str, grid, biomes, **options):
    """Template, structure and pool for one build, all under the same id."""
    writer.add_template(ident, grid)
    writer.add("structure", ident, structure(ident, biomes, **options))
    writer.add("template_pool", ident, pool(ident))


def rotate_xz(x: int, z: int, size_x: int, size_z: int, rotation: str) -> tuple[int, int]:
    """Where a template position ends up once the game has turned the build."""
    if rotation == "NONE":
        return x, z
    if rotation == "CLOCKWISE_90":
        return size_z - 1 - z, x
    if rotation == "CLOCKWISE_180":
        return size_x - 1 - x, size_z - 1 - z
    if rotation == "COUNTERCLOCKWISE_90":
        return z, size_x - 1 - x
    raise ValueError(f"unknown rotation {rotation!r}")
