"""Approximate top-face colours for the blocks worldgen actually places.

These are average texture colours, good enough to read terrain at a glance.
Unknown blocks render magenta on purpose: a magenta patch means the pack asked
for a block this table has never heard of, which is usually a typo.
"""
from __future__ import annotations

import colorsys

import numpy as np

MISSING = (255, 0, 220)

BLOCK_COLORS: dict[str, tuple[int, int, int]] = {
    # stone family
    "stone": (125, 125, 125), "deepslate": (77, 77, 80), "cobblestone": (127, 127, 127),
    "andesite": (136, 136, 137), "diorite": (188, 188, 189), "granite": (149, 103, 85),
    "tuff": (108, 109, 102), "calcite": (223, 222, 217), "dripstone_block": (134, 107, 92),
    "smooth_basalt": (72, 72, 80), "basalt": (80, 78, 84), "blackstone": (42, 35, 41),
    "obsidian": (21, 18, 30), "bedrock": (85, 85, 85), "gravel": (131, 127, 126),
    "netherrack": (114, 58, 57), "end_stone": (221, 223, 165), "amethyst_block": (134, 97, 197),
    "magma_block": (142, 63, 31), "sculk": (14, 28, 35),
    # dirt family
    "dirt": (134, 96, 67), "coarse_dirt": (119, 85, 59), "rooted_dirt": (144, 103, 76),
    "grass_block": (127, 178, 56), "podzol": (91, 61, 26), "mycelium": (111, 99, 100),
    "mud": (60, 55, 60), "clay": (160, 166, 179), "moss_block": (89, 109, 45),
    "farmland": (98, 66, 41), "dirt_path": (148, 122, 66),
    # sand / sandstone
    "sand": (219, 207, 163), "red_sand": (190, 102, 33), "sandstone": (216, 203, 155),
    "red_sandstone": (190, 102, 33), "soul_sand": (81, 62, 50), "soul_soil": (75, 57, 46),
    # snow / ice
    "snow_block": (249, 254, 254), "snow": (249, 254, 254), "powder_snow": (249, 254, 254),
    "ice": (145, 183, 253), "packed_ice": (141, 180, 250), "blue_ice": (116, 167, 253),
    # liquids
    "water": (57, 87, 176), "lava": (217, 104, 24),
    # nether / end vegetation blocks
    "crimson_nylium": (130, 31, 31), "warped_nylium": (22, 119, 105),
    "nether_wart_block": (114, 3, 3), "warped_wart_block": (20, 89, 89),
    "shroomlight": (241, 154, 76), "glowstone": (248, 215, 115),
    # terracotta bands
    "terracotta": (152, 94, 68), "white_terracotta": (209, 178, 161),
    "orange_terracotta": (161, 83, 37), "magenta_terracotta": (149, 88, 108),
    "light_blue_terracotta": (113, 108, 137), "yellow_terracotta": (186, 133, 35),
    "lime_terracotta": (103, 117, 52), "pink_terracotta": (161, 78, 78),
    "gray_terracotta": (57, 42, 35), "light_gray_terracotta": (135, 107, 98),
    "cyan_terracotta": (86, 91, 91), "purple_terracotta": (118, 70, 86),
    "blue_terracotta": (74, 59, 91), "brown_terracotta": (77, 51, 35),
    "green_terracotta": (76, 83, 42), "red_terracotta": (143, 61, 46),
    "black_terracotta": (37, 22, 16),
    # concrete / wool (packs love these for stylised terrain)
    "white_concrete": (207, 213, 214), "orange_concrete": (224, 97, 0),
    "magenta_concrete": (169, 48, 159), "light_blue_concrete": (35, 137, 198),
    "yellow_concrete": (240, 175, 21), "lime_concrete": (94, 168, 24),
    "pink_concrete": (213, 101, 142), "gray_concrete": (54, 57, 61),
    "light_gray_concrete": (125, 125, 115), "cyan_concrete": (21, 119, 136),
    "purple_concrete": (100, 31, 156), "blue_concrete": (44, 46, 143),
    "brown_concrete": (96, 59, 31), "green_concrete": (73, 91, 36),
    "red_concrete": (142, 32, 32), "black_concrete": (8, 10, 15),
    "white_wool": (233, 236, 236), "black_wool": (20, 21, 25),
    # ores and metal blocks
    "coal_block": (16, 15, 15), "iron_block": (220, 220, 220), "gold_block": (246, 208, 61),
    "diamond_block": (98, 219, 214), "emerald_block": (42, 203, 87), "copper_block": (192, 107, 79),
    "redstone_block": (171, 25, 6), "lapis_block": (30, 67, 140), "netherite_block": (66, 60, 62),
    "quartz_block": (235, 229, 222), "prismarine": (99, 156, 151),
    "raw_iron_block": (166, 135, 107), "raw_copper_block": (154, 91, 67), "raw_gold_block": (221, 169, 46),
    # misc
    "air": (0, 0, 0), "cave_air": (0, 0, 0), "glass": (200, 220, 235),
    "honey_block": (251, 184, 48), "slime_block": (111, 192, 91),
    "bone_block": (229, 225, 206), "ochre_froglight": (250, 240, 195),
    "verdant_froglight": (218, 235, 205), "pearlescent_froglight": (243, 220, 232),
    "mossy_cobblestone": (110, 118, 95), "cobbled_deepslate": (77, 77, 80),
    "smooth_stone": (159, 159, 159), "stone_bricks": (122, 121, 122),
    "polished_blackstone": (53, 48, 56), "gilded_blackstone": (69, 44, 39),
    "crying_obsidian": (32, 10, 60), "ancient_debris": (94, 66, 60),
    "sulfur": (214, 197, 84), "cinnabar": (166, 63, 52),
}

# grass tint anchors sampled from vanilla's grass colormap: (temperature, downfall) -> rgb
_GRASS_ANCHORS = [
    (0.00, 0.5, (128, 180, 151)),   # snowy
    (0.25, 0.8, (134, 183, 131)),   # taiga
    (0.50, 0.5, (142, 185, 113)),   # cool
    (0.80, 0.4, (145, 189, 89)),    # plains
    (0.95, 0.9, (89, 201, 60)),     # jungle
    (1.20, 0.0, (191, 183, 85)),    # savanna / dry
    (2.00, 0.0, (191, 183, 85)),    # desert
]


def grass_color(temperature: float, downfall: float) -> tuple[int, int, int]:
    """Inverse-distance blend of the anchors above; close enough to read."""
    t = float(np.clip(temperature, 0.0, 2.0))
    d = float(np.clip(downfall, 0.0, 1.0))
    weights, colors = [], []
    for at, ad, rgb in _GRASS_ANCHORS:
        dist = (at - t) ** 2 + ((ad - d) * 0.6) ** 2
        weights.append(1.0 / (dist + 1e-3))
        colors.append(rgb)
    w = np.array(weights)[:, None]
    c = (np.array(colors) * w).sum(axis=0) / w.sum()
    return tuple(int(round(v)) for v in c)


def parse_hex(value, default=(63, 118, 228)) -> tuple[int, int, int]:
    """Biome effect colours are "#rrggbb" strings or plain ints."""
    if isinstance(value, int):
        return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    if isinstance(value, str):
        text = value.lstrip("#")
        if len(text) == 6:
            try:
                return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
            except ValueError:
                pass
    return default


def block_color(name: str) -> tuple[int, int, int]:
    key = name.split(":")[-1].split("[")[0]
    return BLOCK_COLORS.get(key, MISSING)


def palette_rgb(palette: list[str]) -> np.ndarray:
    return np.array([block_color(name) for name in palette], dtype=np.uint8)


def is_missing(name: str) -> bool:
    return name.split(":")[-1].split("[")[0] not in BLOCK_COLORS


BIOME_COLORS = {
    "ocean": (0, 60, 140), "deep_ocean": (0, 40, 110), "cold_ocean": (32, 80, 150),
    "frozen_ocean": (140, 180, 220), "warm_ocean": (0, 120, 190), "lukewarm_ocean": (0, 100, 170),
    "deep_frozen_ocean": (100, 140, 190), "deep_cold_ocean": (20, 60, 130),
    "deep_lukewarm_ocean": (0, 80, 150),
    "river": (40, 100, 200), "frozen_river": (150, 200, 235),
    "beach": (238, 220, 160), "snowy_beach": (240, 245, 240), "stony_shore": (140, 140, 140),
    "plains": (140, 190, 90), "sunflower_plains": (170, 205, 90),
    "meadow": (120, 195, 120), "forest": (60, 140, 60), "birch_forest": (110, 170, 100),
    "dark_forest": (35, 90, 45), "old_growth_birch_forest": (130, 180, 110),
    "flower_forest": (110, 200, 110), "taiga": (50, 110, 90),
    "snowy_taiga": (190, 215, 210), "old_growth_pine_taiga": (45, 100, 70),
    "old_growth_spruce_taiga": (40, 95, 65),
    "jungle": (40, 160, 40), "sparse_jungle": (80, 170, 60), "bamboo_jungle": (100, 190, 60),
    "savanna": (190, 180, 90), "savanna_plateau": (175, 170, 95), "windswept_savanna": (160, 165, 100),
    "desert": (240, 220, 140), "badlands": (200, 110, 50), "eroded_badlands": (215, 130, 60),
    "wooded_badlands": (180, 130, 70),
    "swamp": (70, 105, 70), "mangrove_swamp": (60, 115, 85),
    "snowy_plains": (240, 245, 250), "ice_spikes": (200, 230, 255),
    "jagged_peaks": (235, 240, 245), "frozen_peaks": (225, 235, 245),
    "stony_peaks": (160, 155, 150), "snowy_slopes": (225, 235, 240),
    "grove": (150, 190, 175), "windswept_hills": (130, 155, 130),
    "windswept_gravelly_hills": (150, 155, 150), "windswept_forest": (95, 140, 95),
    "cherry_grove": (240, 170, 200), "pale_garden": (170, 175, 165),
    "lush_caves": (90, 170, 70), "dripstone_caves": (140, 110, 90), "deep_dark": (20, 35, 45),
    "nether_wastes": (130, 45, 40), "crimson_forest": (170, 40, 40),
    "warped_forest": (30, 130, 120), "soul_sand_valley": (95, 80, 65),
    "basalt_deltas": (70, 65, 70),
    "the_end": (220, 220, 170), "end_highlands": (200, 200, 150),
    "end_midlands": (210, 205, 160), "small_end_islands": (180, 180, 140),
    "end_barrens": (190, 190, 150), "the_void": (5, 5, 10),
}


def biome_color(name: str, index: int) -> tuple[int, int, int]:
    """Vanilla-ish colour where there is one, otherwise a stable generated hue."""
    key = name.split(":")[-1]
    if key in BIOME_COLORS:
        return BIOME_COLORS[key]
    # Deterministic, well-spread hues for custom biomes. Magenta is reserved for
    # "block colour missing", so it is skipped here.
    hue = (index * 0.618033988749895 + 0.11) % 1.0
    if 0.80 < hue < 0.92:
        hue = (hue + 0.13) % 1.0
    sat = 0.45 + 0.22 * ((index * 7) % 3) / 2.0
    val = 0.68 + 0.24 * ((index * 5) % 4) / 3.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)
