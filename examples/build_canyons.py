"""Authoring script for packs/red_canyons: "endless red-rock canyon lands cut by
deep slot gorges".

Kept as a script rather than hand-written JSON because the spline points are
easier to reason about as numbers with comments next to them. The output is a
plain datapack; nothing at run time depends on this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldsmith.pack import PackWriter
from worldsmith.templates import flat_cache, point, shifted_noise, spline

ROOT = Path(__file__).resolve().parent.parent / "packs" / "red_canyons"
NS = "canyons"
MIN_Y, HEIGHT, SEA = -64, 384, 40
TOP = MIN_Y + HEIGHT

CATEGORY = {
    "df": "density_function", "noise": "noise", "ns": "noise_settings",
    "biome": "biome", "dim": "dimension",
}
pack = PackWriter(ROOT, description="Red canyon lands cut by slot gorges")


def w(kind, ident, obj):
    pack.add(CATEGORY[kind], f"{NS}:{ident}", obj)


def block(name, props=None):
    state = {"Name": name}
    if props:
        state["Properties"] = props
    return {"type": "minecraft:block", "result_state": state}


def cond(c, r):
    return {"type": "minecraft:condition", "if_true": c, "then_run": r}


def seq(*rules):
    return {"type": "minecraft:sequence", "sequence": list(rules)}


def main():
    pack.mcmeta()

    # ---- noises ----------------------------------------------------------
    # gorge: the canyon network. Low frequency so the canyons are long.
    w("noise", "gorge", {"firstOctave": -8, "amplitudes": [1.0, 0.5, 0.25]})
    # a second, finer network for side branches
    w("noise", "gorge_branch", {"firstOctave": -6, "amplitudes": [1.0, 0.5]})
    # which plateau level a column sits on
    w("noise", "mesa_step", {"firstOctave": -7, "amplitudes": [1.0, 0.6, 0.3]})

    # ---- shaping ---------------------------------------------------------
    w("df", "continents", shifted_noise("minecraft:continentalness"))
    w("df", "erosion", shifted_noise("minecraft:erosion"))
    w("df", "ridges", shifted_noise("minecraft:ridge"))
    w("df", "ridges_folded", {
        "type": "minecraft:mul", "argument1": -3.0,
        "argument2": {"type": "minecraft:add", "argument1": -0.3333333333333333,
                      "argument2": {"type": "minecraft:abs", "argument": {
                          "type": "minecraft:add", "argument1": -0.6666666666666666,
                          "argument2": {"type": "minecraft:abs", "argument": f"{NS}:ridges"}}}}})

    # abs(noise) is near zero along a thin winding line, and that line is the gorge.
    for name, noise in (("gorge_dist", f"{NS}:gorge"), ("branch_dist", f"{NS}:gorge_branch")):
        w("df", name, flat_cache({"type": "minecraft:abs", "argument": {
            "type": "minecraft:noise", "noise": noise, "xz_scale": 1.0, "y_scale": 0.0}}))

    w("df", "mesa_step", flat_cache({"type": "minecraft:noise", "noise": f"{NS}:mesa_step",
                               "xz_scale": 1.0, "y_scale": 0.0}))

    # terraces: a staircase spline turns smooth noise into flat mesa tops with
    # abrupt risers between them. Flat run, steep riser, flat run...
    w("df", "mesa_terraces", flat_cache(spline(f"{NS}:mesa_step", [
        point(-1.00, 0.00), point(-0.34, 0.00),        # lowest bench
        point(-0.30, 0.055), point(-0.10, 0.055),      # +7 blocks
        point(-0.06, 0.115), point(0.16, 0.115),       # +15
        point(0.20, 0.175), point(0.44, 0.175),        # +23
        point(0.48, 0.245), point(1.00, 0.245),        # +32
    ])))

    # the cut itself: full depth on the centre line, back to zero by 0.10
    w("df", "gorge_cut", flat_cache({
        "type": "minecraft:add",
        "argument1": spline(f"{NS}:gorge_dist", [
            point(0.000, -0.62), point(0.012, -0.60), point(0.030, -0.34),
            point(0.055, -0.10), point(0.085, 0.0), point(1.0, 0.0)]),
        "argument2": spline(f"{NS}:branch_dist", [
            point(0.000, -0.30), point(0.010, -0.28), point(0.028, -0.12),
            point(0.050, 0.0), point(1.0, 0.0)])}))

    # base height: a high tableland everywhere, dipping only at the map's edges
    w("df", "base_offset", flat_cache(spline(f"{NS}:continents", [
        point(-1.2, -0.46), point(-0.3, -0.30), point(0.0, -0.20), point(0.4, -0.16), point(1.0, -0.14)])))

    w("df", "offset", flat_cache({"type": "minecraft:add",
                            "argument1": {"type": "minecraft:add",
                                          "argument1": f"{NS}:base_offset",
                                          "argument2": f"{NS}:mesa_terraces"},
                            "argument2": f"{NS}:gorge_cut"}))

    # very high factor: the surface follows the offset almost exactly, so the
    # terrace risers and gorge walls come out vertical instead of sloped.
    w("df", "factor", flat_cache(spline(f"{NS}:gorge_dist", [
        point(0.0, 7.0), point(0.05, 11.0), point(0.2, 15.0), point(1.0, 16.0)])))

    # gorge proximity as a climate parameter: +1 on the canyon floor, -1 on the rim
    w("df", "climate_gorge", flat_cache(spline(f"{NS}:gorge_dist", [
        point(0.000, 0.95), point(0.030, 0.55), point(0.060, 0.05),
        point(0.100, -0.55), point(0.300, -0.90), point(1.0, -0.95)])))
    # which mesa bench a column sits on, remapped to [-1, 1]
    w("df", "climate_bench", flat_cache(spline(f"{NS}:mesa_terraces", [
        point(0.000, -0.85), point(0.055, -0.4), point(0.115, 0.0), point(0.175, 0.45), point(0.245, 0.9)])))

    w("df", "jaggedness", 0.0)

    w("df", "depth", {"type": "minecraft:add",
                      "argument1": {"type": "minecraft:y_clamped_gradient", "from_y": MIN_Y, "to_y": TOP,
                                    "from_value": 1.5, "to_value": -1.5},
                      "argument2": f"{NS}:offset"})

    # stretched vertically (big y_factor) so cliff faces stay clean sheets of
    # rock rather than bubbling into overhangs
    w("df", "base_3d_noise", {"type": "minecraft:old_blended_noise", "xz_scale": 0.3, "y_scale": 0.08,
                              "xz_factor": 90.0, "y_factor": 280.0, "smear_scale_multiplier": 8.0})

    w("df", "sloped_cheese", {
        "type": "minecraft:add",
        "argument1": {"type": "minecraft:mul", "argument1": 4.0, "argument2": {
            "type": "minecraft:quarter_negative", "argument": {
                "type": "minecraft:mul", "argument1": f"{NS}:depth", "argument2": f"{NS}:factor"}}},
        "argument2": f"{NS}:base_3d_noise"})

    # ---- surface ---------------------------------------------------------
    surface = seq(
        cond({"type": "minecraft:vertical_gradient", "random_name": f"{NS}:bedrock_floor",
              "true_at_and_below": {"above_bottom": 0}, "false_at_and_above": {"above_bottom": 5}},
             block("minecraft:bedrock")),
        cond({"type": "minecraft:stone_depth", "offset": 0, "surface_type": "floor",
              "add_surface_depth": False, "secondary_depth_range": 0},
             seq(
                 cond({"type": "minecraft:water", "offset": -1, "surface_depth_multiplier": 0,
                       "add_stone_depth": False},
                      seq(
                          # cliff faces show the banded rock; flat ground gets sand
                          cond({"type": "minecraft:steep"}, {"type": "minecraft:bandlands"}),
                          cond({"type": "minecraft:y_above", "anchor": {"absolute": 96},
                                "surface_depth_multiplier": 0, "add_stone_depth": False},
                               block("minecraft:red_sand")),
                          cond({"type": "minecraft:noise_threshold", "noise": "minecraft:surface",
                                "min_threshold": -0.909, "max_threshold": -0.5454, "is_3d": False},
                               block("minecraft:coarse_dirt")),
                          block("minecraft:red_sand"),
                      )),
                 block("minecraft:gravel"),
             )),
        # a few blocks of red sandstone under the sand, then bands all the way down
        cond({"type": "minecraft:stone_depth", "offset": 3, "surface_type": "floor",
              "add_surface_depth": True, "secondary_depth_range": 0},
             cond({"type": "minecraft:y_above", "anchor": {"absolute": 84},
                   "surface_depth_multiplier": 0, "add_stone_depth": False},
                  block("minecraft:red_sandstone"))),
        cond({"type": "minecraft:y_above", "anchor": {"absolute": 40},
              "surface_depth_multiplier": 0, "add_stone_depth": False},
             {"type": "minecraft:bandlands"}),
        cond({"type": "minecraft:vertical_gradient", "random_name": f"{NS}:deepslate",
              "true_at_and_below": {"absolute": -8}, "false_at_and_above": {"absolute": 16}},
             block("minecraft:deepslate")),
    )

    w("ns", "red_canyons", {
        "sea_level": SEA, "disable_mob_generation": False, "aquifers_enabled": False,
        "ore_veins_enabled": False, "legacy_random_source": False,
        "default_block": {"Name": "minecraft:terracotta"},
        "default_fluid": {"Name": "minecraft:water", "Properties": {"level": "0"}},
        "noise": {"min_y": MIN_Y, "height": HEIGHT, "size_horizontal": 1, "size_vertical": 2},
        "spawn_target": [],
        "noise_router": {
            "barrier": 0.0, "fluid_level_floodedness": 0.0, "fluid_level_spread": 0.0, "lava": 0.0,
            # The router's climate fields feed biome placement only; terrain
            # reads the density functions directly. So point them at the
            # canyon geometry and biomes land exactly where the landforms are.
            "temperature": f"{NS}:climate_bench", "vegetation": shifted_noise("minecraft:vegetation"),
            "continents": f"{NS}:continents", "erosion": f"{NS}:climate_gorge",
            "depth": f"{NS}:depth", "ridges": f"{NS}:ridges",
            "preliminary_surface_level": {"type": "minecraft:find_top_surface", "cell_height": 8,
                                          "lower_bound": MIN_Y, "upper_bound": float(TOP),
                                          "density": f"{NS}:sloped_cheese"},
            "final_density": {"type": "minecraft:interpolated", "argument": {
                "type": "minecraft:squeeze", "argument": {
                    "type": "minecraft:mul", "argument1": 0.64, "argument2": {
                        "type": "minecraft:add",
                        "argument1": {"type": "minecraft:y_clamped_gradient", "from_y": TOP - 140,
                                      "to_y": TOP - 40, "from_value": 0.0, "to_value": -3.0},
                        "argument2": {"type": "minecraft:add",
                                      "argument1": {"type": "minecraft:y_clamped_gradient",
                                                    "from_y": MIN_Y, "to_y": MIN_Y + 24,
                                                    "from_value": 1.5, "to_value": 0.0},
                                      "argument2": f"{NS}:sloped_cheese"}}}}},
            "vein_toggle": 0.0, "vein_ridged": 0.0, "vein_gap": 0.0},
        "surface_rule": surface})

    def biome(temp, down, water, sky, grass=None, foliage=None):
        effects = {"water_color": water}
        if grass:
            effects["grass_color"] = grass
        if foliage:
            effects["foliage_color"] = foliage
        return {"temperature": temp, "downfall": down, "has_precipitation": down > 0,
                "effects": effects,
                "attributes": {"minecraft:visual/sky_color": sky},
                "spawners": {}, "spawn_costs": {}, "carvers": [],
                "features": [[], [], [], [], [], [], [], [], [], [], []]}

    w("biome", "high_mesa", biome(2.0, 0.0, "#4e7f9e", "#c58f5a", grass="#a08b4a", foliage="#9c8347"))
    w("biome", "gorge_floor", biome(1.6, 0.1, "#4a7f8e", "#b8825a", grass="#8f7c46", foliage="#8a7742"))
    w("biome", "painted_flats", biome(1.8, 0.0, "#4e7f9e", "#cf9a63", grass="#a89150", foliage="#a08a4c"))
    w("biome", "banded_bench", biome(2.0, 0.0, "#4e7f9e", "#d9a86e", grass="#b09a55", foliage="#a89150"))

    def entry(name, **kw):
        params = {"temperature": [-1.0, 1.0], "humidity": [-1.0, 1.0], "continentalness": [-1.2, 1.0],
                  "erosion": [-1.0, 1.0], "depth": [0.0, 1.0], "weirdness": [-1.0, 1.0], "offset": 0.0}
        params.update({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in kw.items()})
        return {"biome": f"{NS}:{name}", "parameters": params}

    w("dim", "red_canyons", {
        "type": "minecraft:overworld",
        "generator": {"type": "minecraft:noise", "settings": f"{NS}:red_canyons",
                      "biome_source": {"type": "minecraft:multi_noise", "biomes": [
                          # erosion is gorge proximity, temperature is bench height
                          entry("gorge_floor", erosion=[0.3, 1.0]),
                          entry("painted_flats", erosion=[-0.35, 0.3], temperature=[-1.0, 0.2]),
                          entry("high_mesa", erosion=[-1.0, -0.35], temperature=[-1.0, 0.2]),
                          entry("banded_bench", erosion=[-1.0, 0.3], temperature=[0.2, 1.0]),
                      ]}}})
    print(f"wrote {ROOT}")


if __name__ == "__main__":
    main()
