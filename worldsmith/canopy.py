"""How much canopy a biome would end up under, for the map view only.

Features are placed by the game after generation, so a render shows bare ground:
a jungle and a bare plateau come out the same shade of green, which is the one
thing a screenshot of the same seed would never agree with.

This does not place trees. It reads the biome's own feature list, follows each
placed feature to the tree it configures, and works out the fraction of ground
those features are expected to cover and what colour their leaves are. The
renderer stipples that fraction over the map with a smooth field, so a dark
forest reads as closed canopy and a savanna as scattered.

It is an estimate of cover, not a simulation of placement. The game's random
source, its `would_survive` checks and its heightmaps are not consulted, and two
worlds with the same cover get the same blobs in different places. Nothing here
reaches the terrain, the block histogram or the in-game comparison: it is paint
on the preview, and `--decorate` is what turns it on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# a chunk is 16x16 columns, and feature counts are all per chunk
CHUNK_COLUMNS = 256.0

# Trees need something a sapling would survive on. The predicate that says so is
# a block tag the engine does not carry, so this is the ground it stands for.
SOIL = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "moss_block", "pale_moss_block", "mud", "muddy_mangrove_roots", "farmland",
    "sand", "red_sand", "snow_block", "powder_snow", "snow",
}


# Leaves whose texture is greyscale in the jar and coloured by the biome at
# runtime. Cherry, azalea and pale oak carry their own colour and are drawn as
# extracted. Spruce and birch are tinted by a constant rather than by the
# colormap, close enough to their biome's foliage colour to go in here.
COLORMAPPED_LEAVES = {
    "oak_leaves", "dark_oak_leaves", "jungle_leaves", "acacia_leaves",
    "mangrove_leaves", "spruce_leaves", "birch_leaves",
}


@dataclass
class Canopy:
    """Per biome: how much of the ground ends up under leaves, and which leaves."""
    cover: np.ndarray            # (B,) fraction of ground, 0..1
    leaves: list[str | None]     # (B,) the leaf block, or None where nothing grows

    def any(self) -> bool:
        return bool((self.cover > 0.005).any())


def _expected(value) -> float:
    """The mean of one of vanilla's int providers."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return 0.0
    kind = str(value.get("type", "")).split(":")[-1]
    if kind == "constant":
        return float(value.get("value", 0.0))
    if kind in ("uniform", "biased_to_bottom", "trapezoid"):
        return (float(value.get("min_inclusive", 0)) + float(value.get("max_inclusive", 0))) / 2.0
    if kind == "clamped_normal":
        return float(value.get("mean", 0.0))
    if kind == "weighted_list":
        entries = value.get("distribution") or []
        total = sum(float(e.get("weight", 0)) for e in entries)
        if total <= 0:
            return 0.0
        return sum(_expected(e.get("data")) * float(e.get("weight", 0)) for e in entries) / total
    return 0.0


def _per_chunk(placed: dict) -> float:
    """Expected tries per chunk, from the placement modifiers that change the count.

    The rest of them are filters on where a try may land, not how many there are,
    and modelling those would mean running the placement.
    """
    count = 1.0
    for modifier in placed.get("placement") or []:
        kind = str(modifier.get("type", "")).split(":")[-1]
        if kind in ("count", "count_on_every_layer"):
            count *= _expected(modifier.get("count"))
        elif kind == "rarity_filter":
            count /= max(1.0, float(modifier.get("chance", 1)))
        elif kind == "noise_threshold_count":
            count *= (_expected(modifier.get("below_noise", 0))
                      + _expected(modifier.get("above_noise", 0))) / 2.0
    return count


def _resolve(registries, ref, seen: frozenset):
    """A feature reference is an id, an inline feature, or an inline placed one."""
    if isinstance(ref, str):
        if ref in seen:
            return None, seen                       # a selector that loops back
        seen = seen | {ref}
        configured = registries.get("configured_feature", ref)
        if configured is None:
            placed = registries.get("placed_feature", ref)
            return _resolve(registries, placed.get("feature"), seen) if placed else (None, seen)
        return configured, seen
    if isinstance(ref, dict):
        if "placement" in ref and "feature" in ref:
            return _resolve(registries, ref["feature"], seen)
        return ref, seen
    return None, seen


def _trees(registries, ref, seen=frozenset(), weight=1.0, out=None) -> list[tuple[float, float, str]]:
    """Every tree a feature can place, as (share, foliage radius, leaf block).

    Selectors nest, and a dark forest is a selector of selectors, so the shares
    multiply down the branches.
    """
    out = [] if out is None else out
    if weight <= 0.0:
        return out
    feature, seen = _resolve(registries, ref, seen)
    if not isinstance(feature, dict):
        return out
    kind = str(feature.get("type", "")).split(":")[-1]
    config = feature.get("config") or {}
    if kind == "tree":
        radius = _expected((config.get("foliage_placer") or {}).get("radius", 2.0)) or 2.0
        state = (config.get("foliage_provider") or {}).get("state") or {}
        out.append((weight, radius, str(state.get("Name", "minecraft:oak_leaves"))))
    elif kind == "random_selector":
        entries = config.get("features") or []
        taken = 0.0
        for entry in entries:
            chance = float(entry.get("chance", 0.0))
            taken += chance
            _trees(registries, entry.get("feature"), seen, weight * chance, out)
        if config.get("default") is not None:
            _trees(registries, config["default"], seen, weight * max(0.0, 1.0 - taken), out)
    elif kind == "simple_random_selector":
        entries = config.get("features") or []
        for entry in entries:
            _trees(registries, entry, seen, weight / max(1, len(entries)), out)
    elif kind == "random_boolean_selector":
        _trees(registries, config.get("feature_true"), seen, weight * 0.5, out)
        _trees(registries, config.get("feature_false"), seen, weight * 0.5, out)
    return out


def canopy_for(registries, biomes: list[str]) -> Canopy:
    """Cover and leaf block for each biome in a scene."""
    cover = np.zeros(len(biomes), dtype=np.float64)
    leaves: list[str | None] = [None] * len(biomes)
    for i, ident in enumerate(biomes):
        biome = registries.get("biome", ident)
        if not isinstance(biome, dict):
            continue
        total, by_leaf = 0.0, {}
        for step in biome.get("features") or []:
            for fid in step if isinstance(step, list) else []:
                placed = registries.get("placed_feature", fid) if isinstance(fid, str) else None
                if placed is None:
                    continue
                tries = _per_chunk(placed)
                if tries <= 0.0:
                    continue
                for share, radius, leaf in _trees(registries, placed.get("feature")):
                    # a canopy is a disc of foliage, and one tree's worth of it
                    # is that area over the chunk it was rolled for
                    area = math.pi * (radius + 0.5) ** 2
                    part = tries * share * area / CHUNK_COLUMNS
                    total += part
                    by_leaf[leaf] = by_leaf.get(leaf, 0.0) + part
        if by_leaf:
            cover[i] = min(total, 1.0)
            leaves[i] = max(by_leaf, key=by_leaf.get)
    return Canopy(cover=cover, leaves=leaves)


def _hash01(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """A stable value in [0, 1) per lattice point, in absolute coordinates.

    Absolute so that panning the window slides the same forest across it rather
    than reshuffling one, and hashed rather than drawn so no array of the world
    has to exist.
    """
    x = ix.astype(np.uint64)
    z = iz.astype(np.uint64)
    k = x * np.uint64(0x9E3779B97F4A7C15) ^ z * np.uint64(0xC2B2AE3D27D4EB4F)
    k ^= np.uint64((seed * 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF)
    k ^= k >> np.uint64(33)
    k *= np.uint64(0xFF51AFD7ED558CCD)
    k ^= k >> np.uint64(29)
    return (k >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _value_noise(xs: np.ndarray, zs: np.ndarray, seed: int, cell: float) -> np.ndarray:
    """Smooth 0..1 field over absolute block coordinates."""
    u, v = xs / cell, zs / cell
    ix, iz = np.floor(u), np.floor(v)
    fx, fz = u - ix, v - iz
    sx = fx * fx * (3.0 - 2.0 * fx)
    sz = fz * fz * (3.0 - 2.0 * fz)
    ix, iz = ix.astype(np.int64), iz.astype(np.int64)
    c00 = _hash01(ix, iz, seed)
    c10 = _hash01(ix + 1, iz, seed)
    c01 = _hash01(ix, iz + 1, seed)
    c11 = _hash01(ix + 1, iz + 1, seed)
    top = c00 + (c10 - c00) * sx
    bottom = c01 + (c11 - c01) * sx
    return top + (bottom - top) * sz


def canopy_field(xs: np.ndarray, zs: np.ndarray, seed: int) -> np.ndarray:
    """The field a biome's cover fraction is thresholded against.

    Two octaves, because one octave gives round blobs and the edge of a wood is
    ragged. Values are not uniform, so cover_cutoff turns a fraction into the
    matching threshold rather than using the fraction directly.
    """
    coarse = _value_noise(xs, zs, seed, 96.0)
    fine = _value_noise(xs, zs, seed ^ 0x5EED, 28.0)
    return 0.62 * coarse + 0.38 * fine


def canopy_mottle(xs: np.ndarray, zs: np.ndarray, seed: int) -> np.ndarray:
    """Brightness variation across a canopy, roughly one clump per few blocks.

    A canopy seen from above is not a flat wash of one colour: it is crowns and
    the gaps between them. Without this a jungle and a grass field differ only
    in hue, which is the thing the decoration is meant to fix.
    """
    return _value_noise(xs, zs, seed ^ 0x1EAF, 9.0)


_SAMPLE: np.ndarray | None = None


def cover_cutoff(cover: np.ndarray | float) -> np.ndarray:
    """The field value below which a fraction `cover` of the ground lies.

    Taken from a fixed sample of the field rather than from the window being
    drawn, so the same biome keeps the same threshold and panning slides the
    same wood across the view instead of redrawing it.
    """
    global _SAMPLE
    if _SAMPLE is None:
        line = np.arange(0, 4096, 8, dtype=np.float64)
        xs, zs = np.meshgrid(line, line)
        _SAMPLE = np.sort(canopy_field(xs.ravel(), zs.ravel(), 0))
    idx = np.clip(np.asarray(cover, dtype=np.float64), 0.0, 1.0) * (len(_SAMPLE) - 1)
    return _SAMPLE[np.rint(idx).astype(np.int64)]
