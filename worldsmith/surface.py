"""Surface rules: what block sits on top of the stone.

The game walks each column top down and applies a rule tree to every stone
block. For a top-down preview only the topmost block matters, so the tree is
evaluated once per column at stone_depth_above = 1, vectorised, with each rule
filling in only the columns no earlier rule has claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .jrandom import Xoroshiro
from .noise import NormalNoise
from .world import World

NO_WATER = -(1 << 62)

SURFACE_RULE_TYPES = {"block", "sequence", "condition"}
SURFACE_CONDITION_TYPES = {
    "above_preliminary_surface", "biome", "hole", "noise_threshold", "not",
    "steep", "stone_depth", "temperature", "vertical_gradient", "water", "y_above",
}


def vertical_anchor(obj, min_y: int, max_y: int) -> np.ndarray | int:
    if isinstance(obj, (int, float)):
        return int(obj)
    if not isinstance(obj, dict):
        return 0
    if "absolute" in obj:
        return int(obj["absolute"])
    if "above_bottom" in obj:
        return min_y + int(obj["above_bottom"])
    if "below_top" in obj:
        return max_y - 1 - int(obj["below_top"])
    return 0


@dataclass
class SurfaceContext:
    world: World
    x: np.ndarray                 # (N,) block coords
    z: np.ndarray
    y: np.ndarray                 # top solid block
    stone_depth_above: np.ndarray
    stone_depth_below: np.ndarray
    water_height: np.ndarray
    surface_depth: np.ndarray
    surface_secondary: np.ndarray
    min_surface_level: np.ndarray
    biome: np.ndarray             # (N,) index into biome_names
    biome_names: list[str]
    biome_temperature: np.ndarray
    steep: np.ndarray
    _random_cache: dict = field(default_factory=dict)

    def positional(self, name: str):
        if name not in self._random_cache:
            base = Xoroshiro.create(self.world.seed).fork_positional()
            self._random_cache[name] = base.from_hash_of(name).fork_positional()
        return self._random_cache[name]


class SurfaceSystem:
    """Compiles the surface_rule tree and evaluates it over columns."""

    def __init__(self, world: World):
        self.world = world
        self.rule = world.surface_rule
        pos = Xoroshiro.create(world.seed).fork_positional()
        self.surface_noise = NormalNoise(pos.from_hash_of("minecraft:surface"), -6, [1.0, 1.0, 1.0])
        self.secondary_noise = NormalNoise(pos.from_hash_of("minecraft:surface_secondary"),
                                           -6, [1.0, 1.0, 0.0, 1.0])
        self.random = pos
        self.palette: list[str] = []
        self._palette_index: dict[str, int] = {}
        self._clay_bands: list[str] | None = None
        self._clay_offset_noise: NormalNoise | None = None

    def block_index(self, name: str) -> int:
        if name not in self._palette_index:
            self._palette_index[name] = len(self.palette)
            self.palette.append(name)
        return self._palette_index[name]

    def surface_depth(self, x, z) -> np.ndarray:
        noise = self.surface_noise.sample(np.asarray(x, float), 0.0, np.asarray(z, float))
        jitter = self.random.at_next_double(np.asarray(x, np.int64), 0, np.asarray(z, np.int64)) * 0.25
        return np.asarray(noise) * 2.75 + 3.0 + jitter

    def clay_bands(self) -> list[str]:
        """The 192 terracotta layers the badlands `bandlands` rule cycles through."""
        if self._clay_bands is None:
            random = self.random.from_hash_of("minecraft:clay_bands")
            bands = ["minecraft:terracotta"] * 192
            i = 0
            while i < len(bands):
                i += random.next_int(5) + 1
                if i < len(bands):
                    bands[i] = "minecraft:orange_terracotta"
            self._make_bands(random, bands, 1, "minecraft:yellow_terracotta")
            self._make_bands(random, bands, 2, "minecraft:brown_terracotta")
            self._make_bands(random, bands, 1, "minecraft:red_terracotta")
            count = random.next_int_between_inclusive(9, 15)
            made = 0
            j = 0
            while made < count and j < len(bands):
                bands[j] = "minecraft:white_terracotta"
                if j > 1 and random.next_boolean():
                    bands[j - 1] = "minecraft:light_gray_terracotta"
                if j < len(bands) - 1 and random.next_boolean():
                    bands[j + 1] = "minecraft:light_gray_terracotta"
                made += 1
                j += random.next_int(16) + 4
            self._clay_bands = bands
            self._clay_offset_noise = NormalNoise(
                self.random.from_hash_of("minecraft:clay_bands_offset"), -8, [1.0])
        return self._clay_bands

    @staticmethod
    def _make_bands(random, bands, half_width, block):
        count = random.next_int_between_inclusive(6, 15)
        for _ in range(count):
            width = random.next_int_between_inclusive(1, 3) + half_width
            start = random.next_int(len(bands))
            for k in range(width):
                if start + k < len(bands) and width > 0:
                    bands[start + k] = block

    def band_at(self, ctx: SurfaceContext) -> np.ndarray:
        bands = self.clay_bands()
        offset = np.round(np.asarray(self._clay_offset_noise.sample(
            ctx.x.astype(float), 0.0, ctx.z.astype(float))) * 4.0).astype(np.int64)
        idx = (ctx.y + offset + len(bands)) % len(bands)
        lookup = np.array([self.block_index(b) for b in bands], dtype=np.int32)
        return lookup[idx]

    def evaluate(self, ctx: SurfaceContext) -> np.ndarray:
        out = np.full(ctx.x.shape, -1, dtype=np.int32)
        if self.rule is not None:
            self._apply(self.rule, ctx, np.ones(ctx.x.shape, dtype=bool), out)
        fallback = self.block_index(self.world.default_block)
        return np.where(out < 0, fallback, out)

    def _apply(self, rule, ctx: SurfaceContext, mask: np.ndarray, out: np.ndarray) -> None:
        if not np.any(mask):
            return
        t = str(rule.get("type", "")).split(":")[-1]
        if t == "sequence":
            remaining = mask & (out < 0)
            for sub in rule.get("sequence", []):
                if not np.any(remaining):
                    return
                self._apply(sub, ctx, remaining, out)
                remaining = mask & (out < 0)
            return
        if t == "condition":
            cond = self.condition(rule.get("if_true") or {}, ctx)
            self._apply(rule.get("then_run") or {}, ctx, mask & cond, out)
            return
        if t == "block":
            state = rule.get("result_state") or {}
            name = state.get("Name", self.world.default_block)
            fill = mask & (out < 0)
            out[fill] = self.block_index(name)
            return
        if t == "bandlands":
            fill = mask & (out < 0)
            out[fill] = self.band_at(ctx)[fill]
            return
        # unknown rule type: leave unset (the validator reports it)

    def condition(self, obj, ctx: SurfaceContext) -> np.ndarray:
        t = str(obj.get("type", "")).split(":")[-1]
        n = ctx.x.shape
        world = self.world
        if t == "not":
            return ~self.condition(obj.get("invert") or {}, ctx)
        if t == "biome":
            raw = obj.get("biome_is") or []
            wanted = {str(raw)} if isinstance(raw, str) else {str(b) for b in raw}
            allowed = np.array([name in wanted for name in ctx.biome_names], dtype=bool)
            return allowed[ctx.biome] if len(allowed) else np.zeros(n, dtype=bool)
        if t == "y_above":
            anchor = vertical_anchor(obj.get("anchor"), world.noise.min_y, world.noise.max_y)
            mult = float(obj.get("surface_depth_multiplier", 0))
            depth = ctx.stone_depth_above if obj.get("add_stone_depth") else 0
            return (ctx.y + depth) >= (anchor + ctx.surface_depth * mult)
        if t == "water":
            offset = float(obj.get("offset", 0))
            mult = float(obj.get("surface_depth_multiplier", 0))
            depth = ctx.stone_depth_above if obj.get("add_stone_depth") else 0
            no_water = ctx.water_height <= NO_WATER
            return no_water | ((ctx.y + depth) >= (ctx.water_height + offset + ctx.surface_depth * mult))
        if t == "stone_depth":
            offset = int(obj.get("offset", 0))
            ceiling = obj.get("surface_type") == "ceiling"
            depth = ctx.stone_depth_below if ceiling else ctx.stone_depth_above
            add = ctx.surface_depth if obj.get("add_surface_depth") else 0.0
            rng = int(obj.get("secondary_depth_range", 0))
            secondary = 0.0 if rng == 0 else (ctx.surface_secondary + 1.0) / 2.0 * rng
            return depth <= (1 + offset + add + secondary)
        if t == "noise_threshold":
            noise = self.world.get_noise(obj.get("noise"))
            y = ctx.y.astype(float) if obj.get("is_3d") else 0.0
            value = np.asarray(noise.sample(ctx.x.astype(float), y, ctx.z.astype(float)))
            return ((value >= float(obj.get("min_threshold", 0)))
                    & (value <= float(obj.get("max_threshold", 0))))
        if t == "vertical_gradient":
            name = str(obj.get("random_name", ""))
            below = vertical_anchor(obj.get("true_at_and_below"), world.noise.min_y, world.noise.max_y)
            above = vertical_anchor(obj.get("false_at_and_above"), world.noise.min_y, world.noise.max_y)
            chance = np.clip((ctx.y - below) / max(1e-9, (above - below)), 0.0, 1.0)
            chance = 1.0 - chance
            draw = ctx.positional(name).at_next_float(ctx.x.astype(np.int64), ctx.y.astype(np.int64),
                                                      ctx.z.astype(np.int64))
            return np.where(ctx.y <= below, True, np.where(ctx.y >= above, False, draw < chance))
        if t == "above_preliminary_surface":
            return ctx.y >= ctx.min_surface_level
        if t == "hole":
            return ctx.surface_depth <= 0
        if t == "steep":
            return ctx.steep
        if t == "temperature":
            return ctx.biome_temperature < 0.15
        return np.zeros(n, dtype=bool)
