"""Biome placement: the multi-noise climate sampler.

Each biome entry claims a 6-dimensional box (temperature, humidity,
continentalness, erosion, depth, weirdness) plus a tie-break offset. A position
is assigned the biome whose box is nearest, so boxes never need to tile the
space, but overlapping boxes mean one of them silently never wins.

A source can name a preset instead of listing entries, which is how a pack that
starts from vanilla's overworld arrives. Those tables are vendored under
multi_noise_biome_source_parameter_list by tools/extract_biome_parameters.py and
they are large: 7594 entries where a hand-written pack has a dozen. That is why
the search below streams over the entries instead of broadcasting against them,
which at overworld size would ask for 28 GB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .density import Ctx, prepare
from .kernels import HAVE_NUMBA, njit, prange
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

# entries x points held at once by the fallback search, about 16 MB of doubles
BROADCAST_BUDGET = 2_000_000


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
    mins: np.ndarray | None = None      # (K, 7) boxes, one per entry
    maxs: np.ndarray | None = None
    entry_biome: np.ndarray | None = None   # (K,) index into biomes
    preset: str | None = None

    @classmethod
    def from_json(cls, obj: dict, registries=None) -> "BiomeSource":
        kind = (obj.get("type") or "minecraft:multi_noise").split(":")[-1]
        if kind == "fixed":
            return cls(kind="fixed", biomes=[obj.get("biome", "minecraft:plains")])
        if kind == "checkerboard":
            biomes = obj.get("biomes")
            biomes = biomes if isinstance(biomes, list) else [biomes]
            return cls(kind="checkerboard", biomes=[str(b) for b in biomes])
        entries = obj.get("biomes")
        preset = obj.get("preset")
        if not entries and preset:
            entries = _preset_entries(str(preset), registries)
        if not entries:
            raise ValueError("multi_noise biome source has no 'biomes' list and no 'preset'")
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
        # a biome claims as many boxes as it likes, and vanilla's overworld gives
        # some of them a hundred, so the entries are folded down to one index per
        # biome. Everything downstream counts, colours and legends by biome.
        unique = list(dict.fromkeys(names))
        at = {name: i for i, name in enumerate(unique)}
        return cls(kind="multi_noise", biomes=unique,
                   mins=np.ascontiguousarray(mins, dtype=np.float64),
                   maxs=np.ascontiguousarray(maxs, dtype=np.float64),
                   entry_biome=np.array([at[n] for n in names], dtype=np.int32),
                   preset=str(preset) if preset else None)


def _preset_entries(preset: str, registries) -> list[dict]:
    """The biome list behind a preset name.

    Vanilla assembles these in Java, so mcmeta ships the name back as the whole
    file and there is nothing to read until the extractor has run.
    """
    hint = "run tools/extract_biome_parameters.py to read it out of the server jar"
    if registries is None:
        raise ValueError(f"biome source uses preset {preset!r} and nothing was passed "
                         f"to resolve it against")
    table = registries.get("multi_noise_biome_source_parameter_list", preset)
    if table is None:
        raise ValueError(f"unknown multi-noise preset {preset!r}")
    entries = table.get("biomes")
    if not entries:
        raise ValueError(f"the vendored table for preset {preset!r} holds no biomes; {hint}")
    return entries


@njit(cache=True, parallel=True, nogil=True)
def _nearest_box(mins, maxs, target, out):
    """out[i] = the entry whose box is nearest target[i]; the first wins ties.

    A distance only grows as parameters are added, so a running total that has
    already passed the best cannot win and the rest of its box is skipped. On
    vanilla's table that drops most entries after one or two parameters.
    """
    count = mins.shape[0]
    for i in prange(target.shape[0]):
        best = np.inf
        pick = 0
        for k in range(count):
            total = 0.0
            for p in range(7):
                t = target[i, p]
                d = mins[k, p] - t
                if d < 0.0:
                    d = t - maxs[k, p]
                    if d < 0.0:
                        d = 0.0
                total += d * d
                if total >= best:
                    break
            if total < best:
                best = total
                pick = k
        out[i] = pick


def _nearest_box_numpy(source: "BiomeSource", target: np.ndarray) -> np.ndarray:
    """The same search without numba, a slab of points at a time.

    Parameters accumulate in the same order as the kernel so the two paths agree
    to the last bit, which matters because ties go to whichever entry is first.
    """
    count = source.mins.shape[0]
    step = max(1, BROADCAST_BUDGET // max(1, count))
    out = np.empty(target.shape[0], dtype=np.int32)
    for start in range(0, target.shape[0], step):
        chunk = target[start:start + step]
        fit = np.zeros((count, chunk.shape[0]), dtype=np.float64)
        for p in range(7):
            t = chunk[:, p][None, :]
            d = np.maximum(np.maximum(source.mins[:, p][:, None] - t,
                                      t - source.maxs[:, p][:, None]), 0.0)
            fit += d * d
        out[start:start + step] = np.argmin(fit, axis=0)
    return out


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
    target = np.ascontiguousarray(target, dtype=np.float64)
    if HAVE_NUMBA:
        picks = np.empty(target.shape[0], dtype=np.int32)
        _nearest_box(source.mins, source.maxs, target, picks)
    else:
        picks = _nearest_box_numpy(source, target)
    return source.entry_biome[picks]


def unreachable_biomes(source: BiomeSource, samples: int = 200000, seed: int = 0) -> list[str]:
    """Biomes that never win anywhere in the climate cube.

    This is the classic "my biome never spawns" bug, found without launching the
    game. A biome that claims several boxes only needs one of them to win, which
    is already how assign_biomes counts.
    """
    if source.kind != "multi_noise":
        return []
    rng = np.random.default_rng(seed)
    # sample the space the routers can actually produce: parameters in [-1, 1],
    # depth in [-1, 1.5] because depth exceeds 1 underground
    pts = rng.uniform(-1.0, 1.0, size=(samples, 7))
    pts[:, 4] = rng.uniform(-1.0, 1.5, size=samples)
    pts[:, 6] = 0.0
    won = set(np.unique(assign_biomes(source, pts)).tolist())
    return [b for i, b in enumerate(source.biomes) if i not in won]
