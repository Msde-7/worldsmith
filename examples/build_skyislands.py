"""Authoring script for packs/sky_islands: "floating islands over an endless void".

The interesting bit is `depth`. Normal terrain uses a single ramp that is
positive underground and negative in the sky, so the world has one surface. Here
depth is a *band*: two opposing y gradients min-ed together make a tent that is
positive only in a slab of sky, so terrain has a top *and* a bottom and the void
swallows everything else.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldsmith.pack import PackWriter
from worldsmith.templates import flat_cache, point, shifted_noise, spline

ROOT = Path(__file__).resolve().parent.parent / "packs" / "sky_islands"
NS = "sky"
MIN_Y, HEIGHT = -64, 384
TOP = MIN_Y + HEIGHT
SEA = MIN_Y                      # sea level at the world floor: nothing floods
BAND_CENTRE = 120                # where the islands hang
BAND_UP, BAND_DOWN = 96, 64      # how far the band reaches below / above centre

CATEGORY = {"df": "density_function", "noise": "noise", "ns": "noise_settings",
            "biome": "biome", "dim": "dimension"}
pack = PackWriter(ROOT, description="Floating sky islands over an endless void")


def w(kind, ident, obj):
    pack.add(CATEGORY[kind], f"{NS}:{ident}", obj)


def block(name):
    return {"type": "minecraft:block", "result_state": {"Name": name}}


def cond(c, r):
    return {"type": "minecraft:condition", "if_true": c, "then_run": r}


def seq(*rules):
    return {"type": "minecraft:sequence", "sequence": list(rules)}


def gradient(from_y, to_y, from_value, to_value):
    return {"type": "minecraft:y_clamped_gradient", "from_y": from_y, "to_y": to_y,
            "from_value": from_value, "to_value": to_value}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    pack.mcmeta()

    w("noise", "archipelago", {"firstOctave": -7, "amplitudes": [1.0, 0.7, 0.4, 0.2]})
    w("noise", "island_lift", {"firstOctave": -8, "amplitudes": [1.0, 0.5]})

    w("df", "continents", shifted_noise("minecraft:continentalness"))
    w("df", "erosion", shifted_noise("minecraft:erosion"))
    w("df", "ridges", shifted_noise("minecraft:ridge"))

    w("df", "archipelago", flat_cache({"type": "minecraft:noise", "noise": f"{NS}:archipelago",
                                 "xz_scale": 1.0, "y_scale": 0.0}))
    w("df", "island_lift", flat_cache({"type": "minecraft:noise", "noise": f"{NS}:island_lift",
                                 "xz_scale": 1.0, "y_scale": 0.0}))

    # Where islands exist at all. Below the threshold the offset is deeply
    # negative, which the band cannot overcome, so those columns are pure void.
    w("df", "island_mask", flat_cache(spline(f"{NS}:archipelago", [
        point(-1.20, -1.60), point(0.12, -1.25), point(0.21, -0.45),
        point(0.30, 0.02), point(0.55, 0.26), point(1.20, 0.34)])))

    # each island floats at its own height
    w("df", "lift", flat_cache(spline(f"{NS}:island_lift", [
        point(-1.0, -0.34), point(-0.3, -0.12), point(0.3, 0.12), point(1.0, 0.36)])))

    # THE BAND. Positive only between centre-BAND_UP and centre+BAND_DOWN,
    # peaking in the middle, an island slab rather than a ground plane.
    w("df", "band", {"type": "minecraft:min",
                     "argument1": gradient(BAND_CENTRE - BAND_UP, BAND_CENTRE, -1.6, 1.0),
                     "argument2": gradient(BAND_CENTRE, BAND_CENTRE + BAND_DOWN, 1.0, -1.6)})

    w("df", "depth", {"type": "minecraft:add",
                      "argument1": f"{NS}:band",
                      "argument2": {"type": "minecraft:add",
                                    "argument1": f"{NS}:island_mask",
                                    "argument2": f"{NS}:lift"}})

    # low factor: the 3D noise gets to push the surface around freely, which is
    # what breaks the slab into separate islands with ragged undersides
    w("df", "factor", flat_cache(spline(f"{NS}:archipelago", [
        point(-1.2, 1.1), point(0.1, 1.5), point(0.5, 2.4), point(1.2, 2.9)])))

    # blobbier than vanilla (small y_factor) so the undersides are lumpy
    w("df", "base_3d_noise", {"type": "minecraft:old_blended_noise", "xz_scale": 0.45, "y_scale": 0.22,
                              "xz_factor": 55.0, "y_factor": 90.0, "smear_scale_multiplier": 8.0})

    w("df", "sloped_cheese", {
        "type": "minecraft:add",
        "argument1": {"type": "minecraft:mul", "argument1": 4.0, "argument2": {
            "type": "minecraft:quarter_negative", "argument": {
                "type": "minecraft:mul", "argument1": f"{NS}:depth", "argument2": f"{NS}:factor"}}},
        "argument2": f"{NS}:base_3d_noise"})

    surface = seq(
        cond({"type": "minecraft:stone_depth", "offset": 0, "surface_type": "floor",
              "add_surface_depth": False, "secondary_depth_range": 0},
             seq(
                 cond({"type": "minecraft:steep"}, block("minecraft:stone")),
                 cond({"type": "minecraft:noise_threshold", "noise": "minecraft:surface",
                       "min_threshold": 0.35, "max_threshold": 2.0, "is_3d": False},
                      block("minecraft:coarse_dirt")),
                 block("minecraft:grass_block"),
             )),
        cond({"type": "minecraft:stone_depth", "offset": 0, "surface_type": "floor",
              "add_surface_depth": True, "secondary_depth_range": 0},
             block("minecraft:dirt")),
        # the undersides: a crust of stone, then dripping deepslate teeth
        cond({"type": "minecraft:stone_depth", "offset": 1, "surface_type": "ceiling",
              "add_surface_depth": False, "secondary_depth_range": 0},
             block("minecraft:deepslate")),
        cond({"type": "minecraft:stone_depth", "offset": 4, "surface_type": "ceiling",
              "add_surface_depth": True, "secondary_depth_range": 0},
             block("minecraft:cobbled_deepslate")),
    )

    w("ns", "sky_islands", {
        "sea_level": SEA, "disable_mob_generation": False, "aquifers_enabled": False,
        "ore_veins_enabled": False, "legacy_random_source": False,
        "default_block": {"Name": "minecraft:stone"},
        "default_fluid": {"Name": "minecraft:water", "Properties": {"level": "0"}},
        "noise": {"min_y": MIN_Y, "height": HEIGHT, "size_horizontal": 1, "size_vertical": 2},
        "spawn_target": [],
        "noise_router": {
            "barrier": 0.0, "fluid_level_floodedness": 0.0, "fluid_level_spread": 0.0, "lava": 0.0,
            "temperature": shifted_noise("minecraft:temperature"),
            "vegetation": shifted_noise("minecraft:vegetation"),
            "continents": f"{NS}:island_mask", "erosion": f"{NS}:erosion",
            "depth": f"{NS}:depth", "ridges": f"{NS}:ridges",
            "preliminary_surface_level": {"type": "minecraft:find_top_surface", "cell_height": 8,
                                          "lower_bound": MIN_Y, "upper_bound": float(TOP),
                                          "density": f"{NS}:sloped_cheese"},
            # no slides needed: the band already forbids terrain near the ceiling
            "final_density": {"type": "minecraft:interpolated", "argument": {
                "type": "minecraft:squeeze", "argument": {
                    "type": "minecraft:mul", "argument1": 0.64,
                    "argument2": f"{NS}:sloped_cheese"}}},
            "vein_toggle": 0.0, "vein_ridged": 0.0, "vein_gap": 0.0},
        "surface_rule": surface})

    def biome(temp, down, sky_color, grass=None, foliage=None):
        effects = {"water_color": "#4c6bd6"}
        if grass:
            effects["grass_color"] = grass
        if foliage:
            effects["foliage_color"] = foliage
        return {"temperature": temp, "downfall": down, "has_precipitation": down > 0,
                "effects": effects,
                "attributes": {"minecraft:visual/sky_color": sky_color,
                               "minecraft:visual/fog_color": "#b9d5ff"},
                "spawners": {}, "spawn_costs": {}, "carvers": [],
                "features": [[], [], [], [], [], [], [], [], [], [], []]}

    w("biome", "cloud_meadow", biome(0.7, 0.6, "#9fc7ff", grass="#7fc45a", foliage="#6db24a"))
    w("biome", "high_isle", biome(0.4, 0.4, "#8fb8ff", grass="#8ab86e", foliage="#78a860"))
    w("biome", "the_deep_void", biome(0.5, 0.0, "#0a0a18"))

    def entry(name, **kw):
        params = {"temperature": [-1.0, 1.0], "humidity": [-1.0, 1.0], "continentalness": [-1.5, 1.0],
                  "erosion": [-1.0, 1.0], "depth": [-1.0, 1.5], "weirdness": [-1.0, 1.0], "offset": 0.0}
        params.update({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in kw.items()})
        return {"biome": f"{NS}:{name}", "parameters": params}

    w("dim", "sky_islands", {
        "type": "minecraft:overworld",
        "generator": {"type": "minecraft:noise", "settings": f"{NS}:sky_islands",
                      "biome_source": {"type": "minecraft:multi_noise", "biomes": [
                          # continentalness is the island mask here, so the void
                          # biome covers exactly the columns with no island
                          entry("the_deep_void", continentalness=[-1.5, -0.5]),
                          entry("cloud_meadow", continentalness=[-0.5, 1.0], temperature=[0.0, 1.0]),
                          entry("high_isle", continentalness=[-0.5, 1.0], temperature=[-1.0, 0.0]),
                      ]}}})
    print(f"wrote {ROOT}")


if __name__ == "__main__":
    main()
