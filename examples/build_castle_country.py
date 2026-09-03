"""Castle country: rolling downs, oak woods, and flat-topped crags to build on.

The shape that matters for castles is a level place to stand. Terrain is mostly
gentle (high factor), with a rare, sharply-stepped lift where the ridge noise
peaks: a crag with a flat top and cliff sides, which is where a castle wants to
be.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from worldsmith.pack import PackWriter, decoration_of
from worldsmith.templates import (base_density_functions, base_router, biome_entry,
                                  biome_json, flat_cache, point, spline)

NS = "castle"
NAME = "castle_country"
MIN_Y, HEIGHT, SEA = -64, 384, 63


def cache(node):
    return flat_cache({"type": "minecraft:cache_2d", "argument": node})


def add(a, b):
    return {"type": "minecraft:add", "argument1": a, "argument2": b}


def mul(a, b):
    return {"type": "minecraft:mul", "argument1": a, "argument2": b}


# --- the shaping splines -----------------------------------------------------

# ground level before anything is added. offset -0.5 == sea level, 0.1 == 13 blocks
BASE_OFFSET = [
    point(-1.2, -0.80),    # deep water     y ~26
    point(-0.60, -0.66),   # open water     y ~44
    point(-0.35, -0.56),   # shallows       y ~56
    point(-0.28, -0.505),  # shore          y ~63
    point(-0.15, -0.465),  # low fields     y ~69
    point(0.20, -0.425),   # downs          y ~74
    point(1.2, -0.375),    # high downs     y ~80
]

# how much of the land shaping applies: nothing at sea, everything inland
LAND_MASK = [point(-1.2, 0.0), point(-0.35, 0.0), point(-0.28, 0.3), point(-0.10, 1.0),
             point(1.2, 1.0)]

# rolling: low erosion is hill country, high erosion is flat farmland
RELIEF = [point(-1.0, 0.115), point(-0.45, 0.06), point(0.0, 0.02),
          point(0.45, -0.005), point(1.0, -0.02)]

# the crag: a hard step where the crag noise peaks, so the sides come out steep.
# Its own noise, at ~128 block wavelength, so a crag top is about the size of a
# castle rather than the size of a county.
CRAG_NOISE = {"firstOctave": -7, "amplitudes": [1.0, 0.6, 0.3]}
CRAG_RIDGE = [point(-1.2, 0.0), point(0.30, 0.0), point(0.38, 1.0), point(1.2, 1.0)]
CRAG_LAND = [point(-1.2, 0.0), point(-0.16, 0.0), point(-0.02, 1.0), point(1.2, 1.0)]
CRAG_STEP = 0.24           # ~31 blocks of cliff

# flat farmland is flat, hill country is not, crags override both
BASE_FACTOR = [point(-1.0, 4.2), point(-0.45, 5.0), point(0.0, 5.9),
               point(0.45, 6.8), point(1.0, 7.4)]
CRAG_FACTOR = 9.0

JAGGED = [point(-1.0, 0.0), point(0.85, 0.0), point(1.0, 0.2)]


def density_functions() -> dict:
    fns = base_density_functions(NS, min_y=MIN_Y, height=HEIGHT,
                                 offset_points=BASE_OFFSET, factor_points=BASE_FACTOR,
                                 jagged_points=JAGGED)
    fns[f"{NS}:base_offset"] = fns.pop(f"{NS}:offset")
    fns.pop(f"{NS}:factor")
    # factor belongs on erosion, not on continents like the template puts it
    fns[f"{NS}:base_factor"] = cache(spline(f"{NS}:erosion", BASE_FACTOR))
    fns[f"{NS}:land_mask"] = cache(spline(f"{NS}:continents", LAND_MASK))
    fns[f"{NS}:relief"] = cache(spline(f"{NS}:erosion", RELIEF))
    fns[f"{NS}:crag_field"] = cache({"type": "minecraft:noise", "noise": f"{NS}:crag",
                                     "xz_scale": 1.0, "y_scale": 0.0})
    fns[f"{NS}:crag_mask"] = mul(cache(spline(f"{NS}:crag_field", CRAG_RIDGE)),
                                 cache(spline(f"{NS}:continents", CRAG_LAND)))
    fns[f"{NS}:offset"] = add(
        f"{NS}:base_offset",
        mul(f"{NS}:land_mask",
            add(f"{NS}:relief", mul(f"{NS}:crag_mask", CRAG_STEP))))
    fns[f"{NS}:factor"] = add(
        mul(f"{NS}:base_factor", add(1.0, mul(-1.0, f"{NS}:crag_mask"))),
        mul(CRAG_FACTOR, f"{NS}:crag_mask"))
    return fns


def surface_rule() -> dict:
    def block(name, **props):
        state = {"Name": name}
        if props:
            state["Properties"] = {k: str(v) for k, v in props.items()}
        return {"type": "minecraft:block", "result_state": state}

    def cond(if_true, then_run):
        return {"type": "minecraft:condition", "if_true": if_true, "then_run": then_run}

    surface = {"type": "minecraft:stone_depth", "offset": 0, "surface_type": "floor",
               "add_surface_depth": False, "secondary_depth_range": 0}
    under = {"type": "minecraft:stone_depth", "offset": 0, "surface_type": "floor",
             "add_surface_depth": True, "secondary_depth_range": 0}
    dry = {"type": "minecraft:water", "offset": -1, "surface_depth_multiplier": 0,
           "add_stone_depth": False}
    shallow = {"type": "minecraft:water", "offset": -6, "surface_depth_multiplier": 0,
               "add_stone_depth": False}
    steep = {"type": "minecraft:steep"}

    return {"type": "minecraft:sequence", "sequence": [
        cond({"type": "minecraft:vertical_gradient", "random_name": "minecraft:bedrock_floor",
              "true_at_and_below": {"absolute": MIN_Y},
              "false_at_and_above": {"absolute": MIN_Y + 5}},
             block("minecraft:bedrock")),
        # cliff faces stay bare rock, which is what makes a crag read as a crag
        cond(surface, cond(steep, cond(dry, block("minecraft:stone")))),
        cond(surface, cond(dry, block("minecraft:grass_block", snowy="false"))),
        cond(surface, cond(shallow, block("minecraft:sand"))),
        cond(surface, block("minecraft:gravel")),
        cond(under, cond(steep, block("minecraft:stone"))),
        cond(under, cond(dry, block("minecraft:dirt"))),
        cond(under, block("minecraft:sand")),
        cond({"type": "minecraft:vertical_gradient", "random_name": "minecraft:deepslate",
              "true_at_and_below": {"absolute": -8},
              "false_at_and_above": {"absolute": 0}},
             block("minecraft:deepslate")),
    ]}


def biomes() -> dict:
    def make(temp, downfall, like):
        return biome_json(temp, downfall, decoration=decoration_of(like))

    return {
        f"{NS}:sea": make(0.6, 0.5, "minecraft:ocean"),
        f"{NS}:shore": make(0.7, 0.4, "minecraft:beach"),
        f"{NS}:downs": make(0.75, 0.4, "minecraft:plains"),
        f"{NS}:oakwood": make(0.7, 0.7, "minecraft:forest"),
        f"{NS}:heath": make(0.5, 0.6, "minecraft:meadow"),
        f"{NS}:crag": make(0.4, 0.3, "minecraft:stony_peaks"),
    }


def biome_source() -> dict:
    """Boxes over the same values the terrain is shaped by, so the rocky biome
    lands on the crags and the woods on the rolling country."""
    # weirdness is the crag field itself (the router feeds it there), so the
    # rocky biome covers exactly the crag tops.
    crag_edge = 0.34
    entries = [
        biome_entry(f"{NS}:sea", continentalness=(-1.2, -0.35)),
        biome_entry(f"{NS}:shore", continentalness=(-0.35, -0.28)),
        biome_entry(f"{NS}:crag", continentalness=(-0.02, 1.0), weirdness=(crag_edge, 1.0)),
        # hill country (low erosion), split wooded / open by humidity
        biome_entry(f"{NS}:oakwood", continentalness=(-0.28, 1.0), erosion=(-1.0, -0.2),
                    humidity=(0.0, 1.0), weirdness=(-1.0, crag_edge)),
        biome_entry(f"{NS}:heath", continentalness=(-0.28, 1.0), erosion=(-1.0, -0.2),
                    humidity=(-1.0, 0.0), weirdness=(-1.0, crag_edge)),
        # farmland, split wooded / open by humidity
        biome_entry(f"{NS}:oakwood", continentalness=(-0.28, 1.0), erosion=(-0.2, 1.0),
                    humidity=(0.15, 1.0), weirdness=(-1.0, crag_edge)),
        biome_entry(f"{NS}:downs", continentalness=(-0.28, 1.0), erosion=(-0.2, 1.0),
                    humidity=(-1.0, 0.15), weirdness=(-1.0, crag_edge)),
    ]
    return {"type": "minecraft:multi_noise", "biomes": entries}


def build(root: Path) -> None:
    writer = PackWriter(root, "Castle country - crags, downs and oak woods", "26.2")
    writer.mcmeta()
    router = base_router(NS, min_y=MIN_Y, height=HEIGHT)
    # weirdness for the biome source is the crag field, so biomes follow the crags
    router["ridges"] = f"{NS}:crag_field"
    writer.add_all({
        "noise": {f"{NS}:crag": CRAG_NOISE},
        "density_function": density_functions(),
        "biome": biomes(),
        "noise_settings": {f"{NS}:{NAME}": {
            "sea_level": SEA,
            "disable_mob_generation": False,
            "aquifers_enabled": False,
            "ore_veins_enabled": False,
            "legacy_random_source": False,
            "default_block": {"Name": "minecraft:stone"},
            "default_fluid": {"Name": "minecraft:water", "Properties": {"level": "0"}},
            "noise": {"min_y": MIN_Y, "height": HEIGHT, "size_horizontal": 1,
                      "size_vertical": 2},
            "spawn_target": [],
            "noise_router": router,
            "surface_rule": surface_rule(),
        }},
        "dimension": {f"{NS}:{NAME}": {
            "type": "minecraft:overworld",
            "generator": {"type": "minecraft:noise", "settings": f"{NS}:{NAME}",
                          "biome_source": biome_source()},
        }},
    })
    print(f"wrote {len(writer.written)} files to {root}")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "packs" / "castle_country"))
