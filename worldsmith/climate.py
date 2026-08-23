"""Biome placement: the multi-noise climate sampler.

Each biome entry claims a 6-dimensional box (temperature, humidity,
continentalness, erosion, depth, weirdness) plus a tie-break offset. A position
is assigned the biome whose box is nearest, so boxes never need to tile the
space, but overlapping boxes mean one of them silently never wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .density import Ctx, prepare
from .world import World

PARAM_NAMES = ("temperature", "humidity", "continentalness", "erosion", "depth", "weirdness")

# Router field feeding each climate parameter. Vegetation is humidity and ridges
# is weirdness, which is a trap when writing packs by hand.
ROUTER_FOR_PARAM = {
    "temperature": "temperature",
    "humidity": "vegetation",
    "continentalness": "continents",
    "erosion": "erosion",
    "depth": "depth",
    "weirdness": "ridges",
}


def _range(value) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if isinstance(value, dict):
        return float(value.get("min", 0.0)), float(value.get("max", 0.0))
    lo, hi = (list(value) + [0.0, 0.0])[:2]
    return float(lo), float(hi)


@dataclass
class BiomeSource:
    kind: str
    biomes: list[str]
    mins: np.ndarray | None = None      # (K, 7)
    maxs: np.ndarray | None = None

    @classmethod
    def from_json(cls, obj: dict) -> "BiomeSource":
        kind = (obj.get("type") or "minecraft:multi_noise").split(":")[-1]
        if kind == "fixed":
            return cls(kind="fixed", biomes=[obj.get("biome", "minecraft:plains")])
        if kind == "checkerboard":
            biomes = obj.get("biomes")
            biomes = biomes if isinstance(biomes, list) else [biomes]
            return cls(kind="checkerboard", biomes=[str(b) for b in biomes])
        entries = obj.get("biomes")
        if not entries:
            preset = obj.get("preset")
            raise ValueError(
                f"multi_noise biome source uses preset {preset!r}; worldsmith needs an explicit "
                "`biomes` list (the presets live in Java code, not in data)")
        names, mins, maxs = [], [], []
        for entry in entries:
            names.append(str(entry.get("biome")))
            params = entry.get("parameters") or {}
            lo, hi = [], []
            for key in PARAM_NAMES:
                a, b = _range(params.get(key, 0.0))
                lo.append(a)
                hi.append(b)
            offset = float(params.get("offset", 0.0))
            lo.append(offset)
            hi.append(offset)
            mins.append(lo)
            maxs.append(hi)
        return cls(kind="multi_noise", biomes=names,
                   mins=np.array(mins, dtype=np.float64), maxs=np.array(maxs, dtype=np.float64))


def climate_target(world: World, xs, zs, ys) -> np.ndarray:
    """Sample the six climate parameters at block coordinates.

    The game samples climate at quart resolution, so the coordinates are snapped
    to a multiple of 4 first. Returns (N, 7).
    """
    xs = (np.floor(np.asarray(xs, dtype=np.float64) / 4.0) * 4.0)[None, :]
    zs = (np.floor(np.asarray(zs, dtype=np.float64) / 4.0) * 4.0)[None, :]
    ys = (np.floor(np.asarray(ys, dtype=np.float64) / 4.0) * 4.0)[None, :]
    ctx = Ctx(xs, ys, zs)
    n = xs.shape[1]
    out = np.zeros((n, 7), dtype=np.float64)
    prepare(*[world.router[ROUTER_FOR_PARAM[p]] for p in PARAM_NAMES])
    for i, name in enumerate(PARAM_NAMES):
        node = world.router[ROUTER_FOR_PARAM[name]]
        out[:, i] = np.broadcast_to(np.asarray(node.eval(ctx), dtype=np.float64), (1, n))[0]
    return out


def assign_biomes(source: BiomeSource, target: np.ndarray) -> np.ndarray:
    """Index into source.biomes for each row of target (N, 7)."""
    if source.kind != "multi_noise":
        return np.zeros(target.shape[0], dtype=np.int32)
    lo = source.mins[:, None, :]          # (K, 1, 7)
    hi = source.maxs[:, None, :]
    t = target[None, :, :]                # (1, N, 7)
    d = np.maximum(np.maximum(lo - t, t - hi), 0.0)
    fit = np.einsum("knp,knp->kn", d, d)
    return np.argmin(fit, axis=0).astype(np.int32)


def unreachable_biomes(source: BiomeSource, samples: int = 200000, seed: int = 0) -> list[str]:
    """Entries that never win anywhere in the climate cube.

    This is the classic "my biome never spawns" bug, found without launching the
    game.
    """
    if source.kind != "multi_noise":
        return []
    rng = np.random.default_rng(seed)
    # sample the space the routers can actually produce: parameters in [-1, 1],
    # depth in [-1, 1.5] because depth exceeds 1 underground
    pts = rng.uniform(-1.0, 1.0, size=(samples, 7))
    pts[:, 4] = rng.uniform(-1.0, 1.5, size=samples)
    pts[:, 6] = 0.0
    won = np.unique(assign_biomes(source, pts))
    return [b for i, b in enumerate(source.biomes) if i not in set(won.tolist())]
