"""Top-face colours for every block, so the preview can draw what the game draws.

The values are averages of the real block textures, pulled out of the client jar
by tools/extract_block_colors.py and vendored as
vanilla/<version>/block_colors.json. They used to be typed out by hand, which
meant the table lagged behind the blocks packs actually reach for and anything
missing rendered magenta.

A handful of blocks are still set here, because the jar's texture is not what
you see in game: water and foliage are stored greyscale and tinted at runtime,
glowstone is lit far brighter than its texture, and air has no texture at all.

Unknown blocks still render magenta on purpose: a magenta patch means the pack
asked for a block that does not exist, which is usually a typo.
"""
from __future__ import annotations

import colorsys
import json

import numpy as np

from .registry import DEFAULT_VERSION, VANILLA_ROOT

MISSING = (255, 0, 220)


def _extracted(version: str = DEFAULT_VERSION) -> dict[str, tuple[int, int, int]]:
    path = VANILLA_ROOT / version / "block_colors.json"
    if not path.is_file():                    # not extracted yet for this version
        return {}
    return {name: tuple(rgb) for name, rgb in
            json.loads(path.read_text(encoding="utf-8")).items()}


# Greyscale in the jar; the game multiplies in a biome tint, and these are what
# that tint looks like at its most ordinary.
_TINTED = {
    "water": (57, 87, 176),
    "grass_block": (127, 178, 56), "short_grass": (127, 178, 56),
    "tall_grass": (127, 178, 56), "fern": (127, 178, 56), "large_fern": (127, 178, 56),
    "oak_leaves": (86, 130, 51), "jungle_leaves": (86, 130, 51),
    "acacia_leaves": (86, 130, 51), "dark_oak_leaves": (86, 130, 51),
    "mangrove_leaves": (86, 130, 51), "vine": (86, 130, 51),
    "lily_pad": (86, 130, 51), "sugar_cane": (128, 180, 80),
}

BLOCK_COLORS: dict[str, tuple[int, int, int]] = {
    **_extracted(),
    **_TINTED,
    # lit far brighter than the texture it is drawn from
    "glowstone": (248, 215, 115),
    # no texture to average
    "air": (0, 0, 0), "cave_air": (0, 0, 0), "void_air": (0, 0, 0),
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


# The same points off foliage.png, which is the colormap leaves take.
_FOLIAGE_ANCHORS = [
    (0.00, 0.5, (96, 161, 123)),    # snowy
    (0.25, 0.8, (104, 164, 100)),   # taiga
    (0.50, 0.5, (113, 167, 77)),    # cool
    (0.80, 0.4, (119, 171, 47)),    # plains
    (0.95, 0.9, (48, 187, 11)),     # jungle
    (1.20, 0.0, (174, 164, 42)),    # savanna / dry
    (2.00, 0.0, (174, 164, 42)),    # desert
]


def _blend(anchors, temperature: float, downfall: float) -> tuple[int, int, int]:
    """Inverse-distance blend of the anchors above; close enough to read."""
    t = float(np.clip(temperature, 0.0, 2.0))
    d = float(np.clip(downfall, 0.0, 1.0))
    weights, colors = [], []
    for at, ad, rgb in anchors:
        dist = (at - t) ** 2 + ((ad - d) * 0.6) ** 2
        weights.append(1.0 / (dist + 1e-3))
        colors.append(rgb)
    w = np.array(weights)[:, None]
    c = (np.array(colors) * w).sum(axis=0) / w.sum()
    return tuple(int(round(v)) for v in c)


def grass_color(temperature: float, downfall: float) -> tuple[int, int, int]:
    return _blend(_GRASS_ANCHORS, temperature, downfall)


def foliage_color(temperature: float, downfall: float) -> tuple[int, int, int]:
    """Leaf tint for the biome, which is darker and deeper than its grass."""
    return _blend(_FOLIAGE_ANCHORS, temperature, downfall)


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
