"""Starting points: complete, valid datapacks meant to be edited.

Every template mirrors vanilla's structure, offset and factor splines feeding a
depth that is multiplied into a 3D noise, because that is the shape the game's
own terrain has and the shape all the tuning intuition is written for.

The named density functions are the knobs:

  <ns>:offset   how high the ground sits (spline over continents)
  <ns>:factor   how stretched the terrain is: big = flat plateau,
                small = violent relief
  <ns>:depth    offset combined with the y gradient; > 0 means solid
  <ns>:jaggedness  extra high-frequency spikiness, applied near peaks
"""
from __future__ import annotations

PACK_FORMAT = {"26.2": 107, "26.1": 105, "1.21.11": 88, "1.21.8": 71, "1.21.4": 57, "1.21": 41}

# Vanilla's caves are density functions, not carvers, so a pack that borrows
# them gets caves the renderer draws exactly. `entrances` brings the spaghetti
# layer with it; `noodle` is the thin winding sort.
CAVE_ENTRANCES = "minecraft:overworld/caves/entrances"
CAVE_NOODLE = "minecraft:overworld/caves/noodle"

# Caves need aquifers or they flood: with these fields left at 0 the game fills
# every cavity below sea level with water and everything under y=-54 with lava.
# These are vanilla's own four, and the noises they name are vanilla's too.
AQUIFER_ROUTER = {
    "barrier": {"type": "minecraft:noise", "noise": "minecraft:aquifer_barrier",
                "xz_scale": 1.0, "y_scale": 0.5},
    "fluid_level_floodedness": {"type": "minecraft:noise",
                                "noise": "minecraft:aquifer_fluid_level_floodedness",
                                "xz_scale": 1.0, "y_scale": 0.67},
    "fluid_level_spread": {"type": "minecraft:noise",
                           "noise": "minecraft:aquifer_fluid_level_spread",
                           "xz_scale": 1.0, "y_scale": 0.7142857142857143},
    "lava": {"type": "minecraft:noise", "noise": "minecraft:aquifer_lava",
             "xz_scale": 1.0, "y_scale": 1.0},
}


# The JSON these helpers build is shared with the authoring scripts in tools/.


def spline(coordinate, points):
    return {"type": "minecraft:spline", "spline": {"coordinate": coordinate, "points": points}}


def point(location, value, derivative=0.0):
    return {"location": location, "value": value, "derivative": derivative}


def flat_cache(arg):
    return {"type": "minecraft:flat_cache", "argument": {"type": "minecraft:cache_2d", "argument": arg}}


def shifted_noise(noise, xz_scale=0.25):
    return flat_cache({
        "type": "minecraft:shifted_noise", "noise": noise,
        "shift_x": "minecraft:shift_x", "shift_y": 0.0, "shift_z": "minecraft:shift_z",
        "xz_scale": xz_scale, "y_scale": 0.0,
    })


def with_caves(final_density: dict) -> dict:
    """Cut vanilla's caves into a final_density, the way vanilla cuts them.

    The entrance and spaghetti layer goes inside the interpolation so the walls
    come out smooth; noodles are applied outside it, at block resolution.

    The caller has to hand over the `interpolated` node itself. Cutting the
    entrances into anything wrapped around it builds a pack the game still
    loads and still fills with caves, only in the wrong places: that shape
    matched a real server on 87% of columns against 100% for this one.

    The functions carry vanilla's own altitudes: `noodle` is gated to y -60..321
    and `entrances` ramps between y -10 and 30, so a pack that moves `min_y` or
    `height` keeps its caves at vanilla's heights. Caves also want aquifers
    (`AQUIFER_ROUTER`): without them the game floods every cavity below sea
    level, which measured 5% dry cave volume against 91% with them.
    """
    if final_density.get("type") != "minecraft:interpolated":
        raise ValueError("with_caves needs the minecraft:interpolated node, got "
                         f"{final_density.get('type')!r}")
    carved = dict(final_density)
    carved["argument"] = {"type": "minecraft:min",
                          "argument1": final_density["argument"],
                          "argument2": CAVE_ENTRANCES}
    return {"type": "minecraft:min", "argument1": carved, "argument2": CAVE_NOODLE}


def base_router(ns: str, *, min_y: int, height: int, caves: bool = False,
                aquifers: bool = False) -> dict:
    """The 15 router fields. Aquifers and ore veins are off, so their fields are
    constant 0, but they must still be present or the file is rejected."""
    top = min_y + height
    router = {
        "barrier": 0.0,
        "fluid_level_floodedness": 0.0,
        "fluid_level_spread": 0.0,
        "lava": 0.0,
        "temperature": shifted_noise("minecraft:temperature"),
        "vegetation": shifted_noise("minecraft:vegetation"),
        "continents": f"{ns}:continents",
        "erosion": f"{ns}:erosion",
        "depth": f"{ns}:depth",
        "ridges": f"{ns}:ridges",
        "preliminary_surface_level": {
            "type": "minecraft:find_top_surface",
            "cell_height": 8,
            "lower_bound": min_y,
            "upper_bound": float(top),
            "density": f"{ns}:sloped_cheese",
        },
        "final_density": {
            "type": "minecraft:interpolated",
            "argument": {
                "type": "minecraft:squeeze",
                "argument": {
                    "type": "minecraft:mul",
                    "argument1": 0.64,
                    "argument2": {
                        "type": "minecraft:add",
                        "argument1": {
                            "type": "minecraft:y_clamped_gradient",
                            "from_y": top - 24, "to_y": top,
                            "from_value": 0.0, "to_value": -1.5,
                        },
                        "argument2": {
                            "type": "minecraft:add",
                            "argument1": {
                                "type": "minecraft:y_clamped_gradient",
                                "from_y": min_y, "to_y": min_y + 24,
                                "from_value": 1.5, "to_value": 0.0,
                            },
                            "argument2": f"{ns}:sloped_cheese",
                        },
                    },
                },
            },
        },
        "vein_toggle": 0.0,
        "vein_ridged": 0.0,
        "vein_gap": 0.0,
    }
    if aquifers:
        router.update(AQUIFER_ROUTER)
    if caves:
        router["final_density"] = with_caves(router["final_density"])
    return router


def base_density_functions(ns: str, *, min_y: int, height: int,
                           offset_points, factor_points, jagged_points) -> dict:
    top = min_y + height
    fns = {
        f"{ns}:continents": shifted_noise("minecraft:continentalness"),
        f"{ns}:erosion": shifted_noise("minecraft:erosion"),
        f"{ns}:ridges": shifted_noise("minecraft:ridge"),
        # folded ridges: turns the ridge noise into the classic "peaks and
        # valleys" W shape. Same expression vanilla uses.
        f"{ns}:ridges_folded": {
            "type": "minecraft:mul", "argument1": -3.0,
            "argument2": {
                "type": "minecraft:add", "argument1": -0.3333333333333333,
                "argument2": {
                    "type": "minecraft:abs",
                    "argument": {
                        "type": "minecraft:add", "argument1": -0.6666666666666666,
                        "argument2": {"type": "minecraft:abs", "argument": f"{ns}:ridges"},
                    },
                },
            },
        },
        f"{ns}:offset": flat_cache(spline(f"{ns}:continents", offset_points)),
        f"{ns}:factor": flat_cache(spline(f"{ns}:continents", factor_points)),
        f"{ns}:depth": {
            "type": "minecraft:add",
            "argument1": {
                "type": "minecraft:y_clamped_gradient",
                "from_y": min_y, "to_y": top, "from_value": 1.5, "to_value": -1.5,
            },
            "argument2": f"{ns}:offset",
        },
        f"{ns}:base_3d_noise": {
            "type": "minecraft:old_blended_noise",
            "xz_scale": 0.25, "y_scale": 0.125,
            "xz_factor": 80.0, "y_factor": 160.0, "smear_scale_multiplier": 8.0,
        },
        f"{ns}:sloped_cheese": {
            "type": "minecraft:add",
            "argument1": {
                "type": "minecraft:mul", "argument1": 4.0,
                "argument2": {
                    "type": "minecraft:quarter_negative",
                    "argument": {
                        "type": "minecraft:mul",
                        "argument1": {
                            "type": "minecraft:add",
                            "argument1": f"{ns}:depth",
                            "argument2": flat_cache({
                                "type": "minecraft:mul",
                                "argument1": f"{ns}:jaggedness",
                                "argument2": {
                                    "type": "minecraft:half_negative",
                                    "argument": {"type": "minecraft:noise", "noise": "minecraft:jagged",
                                                 "xz_scale": 1500.0, "y_scale": 0.0},
                                },
                            }),
                        },
                        "argument2": f"{ns}:factor",
                    },
                },
            },
            "argument2": f"{ns}:base_3d_noise",
        },
    }
    fns[f"{ns}:jaggedness"] = flat_cache(spline(f"{ns}:ridges_folded", jagged_points))
    return fns


def surface_rule_simple(top_block: str, mid_block: str, deep_block: str,
                        underwater_block: str) -> dict:
    """Grass-over-dirt-over-stone, sand under water, bedrock at the bottom."""
    def block(name):
        return {"type": "minecraft:block", "result_state": {"Name": name}}

    return {
        "type": "minecraft:sequence",
        "sequence": [
            {   # bedrock floor: a ragged layer, not a flat slab
                "type": "minecraft:condition",
                "if_true": {"type": "minecraft:vertical_gradient", "random_name": "minecraft:bedrock_floor",
                            "true_at_and_below": {"above_bottom": 0},
                            "false_at_and_above": {"above_bottom": 5}},
                "then_run": block("minecraft:bedrock"),
            },
            {   # only the top few blocks of stone get a surface treatment
                "type": "minecraft:condition",
                "if_true": {"type": "minecraft:stone_depth", "offset": 0, "surface_type": "floor",
                            "add_surface_depth": False, "secondary_depth_range": 0},
                "then_run": {
                    "type": "minecraft:sequence",
                    "sequence": [
                        {
                            "type": "minecraft:condition",
                            "if_true": {"type": "minecraft:water", "offset": -1,
                                        "surface_depth_multiplier": 0, "add_stone_depth": False},
                            "then_run": block(top_block),
                        },
                        block(underwater_block),
                    ],
                },
            },
            {   # a band of dirt beneath the surface block
                "type": "minecraft:condition",
                "if_true": {"type": "minecraft:stone_depth", "offset": 0, "surface_type": "floor",
                            "add_surface_depth": True, "secondary_depth_range": 0},
                "then_run": {
                    "type": "minecraft:condition",
                    "if_true": {"type": "minecraft:water", "offset": -6,
                                "surface_depth_multiplier": -1, "add_stone_depth": True},
                    "then_run": block(mid_block),
                },
            },
            {   # deepslate takes over low down
                "type": "minecraft:condition",
                "if_true": {"type": "minecraft:vertical_gradient", "random_name": "minecraft:deepslate",
                            "true_at_and_below": {"absolute": -8},
                            "false_at_and_above": {"absolute": 8}},
                "then_run": block(deep_block),
            },
        ],
    }


def biome_json(temperature: float, downfall: float, *, water_color="#3f76e4",
               sky_color="#78a7ff", fog_color=None, water_fog_color=None,
               foliage=None, grass=None, decoration=None) -> dict:
    """26.2 shape: `effects` carries colours only; sky/fog moved to `attributes`."""
    effects = {"water_color": water_color}
    if foliage:
        effects["foliage_color"] = foliage
    if grass:
        effects["grass_color"] = grass
    attributes = {}
    if sky_color:
        attributes["minecraft:visual/sky_color"] = sky_color
    if fog_color:
        attributes["minecraft:visual/fog_color"] = fog_color
    if water_fog_color:
        attributes["minecraft:visual/water_fog_color"] = water_fog_color
    biome = {
        "temperature": temperature,
        "downfall": downfall,
        "has_precipitation": downfall > 0.0,
        "effects": effects,
        "spawners": {},
        "spawn_costs": {},
        "carvers": [],
        "features": [[], [], [], [], [], [], [], [], [], [], []],
    }
    # trees, ores, mobs and carver tunnels, borrowed wholesale from a vanilla
    # biome. The game places all of these; the preview draws none of them.
    if decoration:
        biome.update(decoration)
    if attributes:
        biome["attributes"] = attributes
    return biome


def biome_entry(biome: str, *, temperature=(-1.0, 1.0), humidity=(-1.0, 1.0),
                continentalness=(-1.2, 1.0), erosion=(-1.0, 1.0),
                depth=(0.0, 1.0), weirdness=(-1.0, 1.0), offset=0.0) -> dict:
    return {
        "biome": biome,
        "parameters": {
            "temperature": list(temperature), "humidity": list(humidity),
            "continentalness": list(continentalness), "erosion": list(erosion),
            "depth": list(depth), "weirdness": list(weirdness), "offset": offset,
        },
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def template_basic(ns: str, name: str, *, caves: bool = False, decoration=None) -> dict:
    """Continents and oceans: the readable skeleton to edit."""
    min_y, height, sea = -64, 384, 63
    # depth = (1.5 - (y - min_y) * 3 / height) + offset, so the ground sits where
    # depth crosses 0:   y = min_y + height/3 * (1.5 + offset).
    # With min_y=-64, height=384 that is  y = -64 + 128 * (1.5 + offset),
    # i.e. offset -0.5 puts the surface at sea level. Vanilla bakes the same
    # -0.50375 constant into its offset spline.
    offset_points = [
        point(-1.2, -0.78),    # deep ocean   ~y 30
        point(-0.455, -0.66),  # ocean        ~y 45
        point(-0.19, -0.56),   # shallow      ~y 58
        point(-0.11, -0.50),   # shore        ~y 64
        point(0.03, -0.44),    # near inland  ~y 72
        point(0.3, -0.30),     # mid inland   ~y 90
        point(1.0, -0.05),     # far inland   ~y 122
    ]
    # factor is the divisor of relief: large = flat, small = dramatic.
    factor_points = [
        point(-1.2, 6.5), point(-0.19, 6.0), point(-0.11, 5.0),
        point(0.03, 4.0), point(0.3, 3.0), point(1.0, 2.0),
    ]
    # jaggedness stays at 0 except on the sharpest ridges
    jagged_points = [point(-1.0, 0.0), point(0.45, 0.0), point(1.0, 0.6)]
    files = {
        "density_function": base_density_functions(ns, min_y=min_y, height=height,
                                                   offset_points=offset_points,
                                                   factor_points=factor_points,
                                                   jagged_points=jagged_points),
        "noise_settings": {f"{ns}:{name}": {
            "sea_level": sea, "disable_mob_generation": False,
            "aquifers_enabled": caves, "ore_veins_enabled": False,
            "legacy_random_source": False,
            "default_block": {"Name": "minecraft:stone"},
            "default_fluid": {"Name": "minecraft:water", "Properties": {"level": "0"}},
            "noise": {"min_y": min_y, "height": height, "size_horizontal": 1, "size_vertical": 2},
            "spawn_target": [],
            "noise_router": base_router(ns, min_y=min_y, height=height, caves=caves,
                                        aquifers=caves),
            "surface_rule": surface_rule_simple("minecraft:grass_block", "minecraft:dirt",
                                                "minecraft:deepslate", "minecraft:gravel"),
        }},
        "biome": {f"{ns}:plains_like": biome_json(0.8, 0.4, decoration=decoration)},
        "dimension": {},
    }
    files["dimension"][f"{ns}:{name}"] = {
        "type": "minecraft:overworld",
        "generator": {
            "type": "minecraft:noise",
            "settings": f"{ns}:{name}",
            "biome_source": {
                "type": "minecraft:multi_noise",
                "biomes": [biome_entry(f"{ns}:plains_like")],
            },
        },
    }
    return files


def starter_build(ns: str, biomes: list[str]) -> tuple:
    """A small stone hut, scaffolded so a pack has a working build to change.

    Deliberately plain and deliberately complete: a buried footing, a floor, a
    doorway, a window, a light and a hollow inside, which is every part of a
    build that has to be right before anything more interesting can be. It is
    also the shortest demonstration of worldsmith.shapes.
    """
    from .shapes import crenellate, fill, hollow_box, perimeter, speckle
    from .voxel import Grid

    stone = [(70, "minecraft:stone_bricks"), (15, "minecraft:cobblestone"),
             (10, "minecraft:mossy_stone_bricks"), (5, "minecraft:cracked_stone_bricks")]

    def masonry(x, y, z):
        return speckle(x, y, z, stone)

    floor, height = 4, 5                     # footing below, walls above
    grid = Grid(9, floor + height + 4, 9)
    fill(grid, 0, 0, 0, 8, floor - 1, 8, "minecraft:cobblestone")     # buried
    fill(grid, 0, floor, 0, 8, floor, 8, masonry)                     # the floor
    hollow_box(grid, 0, floor + 1, 0, 8, floor + height, 8, masonry)
    fill(grid, 0, floor + height + 1, 0, 8, floor + height + 1, 8, "minecraft:oak_planks")
    fill(grid, 1, floor + height + 2, 1, 7, floor + height + 3, 7, "minecraft:air")
    crenellate(grid, perimeter(0, 0, 8, 8), floor + height + 2, masonry)

    door = 4
    fill(grid, door, floor + 1, 8, door, floor + 2, 8, "minecraft:air")
    grid.set(door, floor + 1, 8, "minecraft:oak_door[facing=south,half=lower,hinge=left]")
    grid.set(door, floor + 2, 8, "minecraft:oak_door[facing=south,half=upper,hinge=left]")
    grid.set(0, floor + 3, 4, "minecraft:glass_pane")
    grid.set(8, floor + 3, 4, "minecraft:glass_pane")
    grid.set(4, floor + 4, 4, "minecraft:lantern[hanging=true]")
    grid.set(2, floor + 1, 2, "minecraft:chest[facing=east]",
             {"id": "minecraft:chest", "LootTable": "minecraft:chests/simple_dungeon"})
    return grid, -(floor + 1), biomes


TEMPLATES = {"basic": template_basic}
