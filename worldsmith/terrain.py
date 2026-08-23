"""Turning a compiled world into terrain: heightmaps, columns and cross-sections.

The generator defines terrain on a lattice of noise cells, 4 blocks wide and 8
tall for the overworld, and trilinearly interpolates inside each cell. Sampling
on that lattice is therefore not an approximation, it is where the data lives.
Between two vertical lattice points the density is linear in y, so the exact
height of the surface is the interpolated zero crossing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aquifer import Aquifer
from .density import Ctx, Interpolated, Marker, prepare
from .world import World

# The game's rule: density above 0 is solid.
SOLID = 0.0


@dataclass
class Terrain:
    """A rectangular sample of a world."""

    world: World
    x0: int
    z0: int
    nx: int
    nz: int
    step: int
    y_levels: np.ndarray            # (M,) ascending
    surface_y: np.ndarray           # (nz, nx) highest solid block, or below-min
    solid_anywhere: np.ndarray      # (nz, nx) bool
    sampling: str = "lattice"       # "lattice" (cell corners) or "block" (exact scan)

    @property
    def sea_level(self) -> int:
        return self.world.sea_level

    @property
    def xs(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.step

    @property
    def zs(self) -> np.ndarray:
        return self.z0 + np.arange(self.nz) * self.step

    def water_depth(self) -> np.ndarray:
        return np.maximum(0, self.sea_level - 1 - self.surface_y)

    def stats(self) -> dict:
        solid = self.solid_anywhere
        s = self.surface_y[solid]
        if s.size == 0:
            return {"empty": True, "void_fraction": 1.0}
        # a column with no terrain at all is void, not ocean
        land = solid & (self.surface_y >= self.sea_level)
        water = solid & (self.surface_y < self.sea_level)
        return {
            "min_y": int(s.min()), "max_y": int(s.max()),
            "mean_y": float(s.mean()), "median_y": float(np.median(s)),
            "land_fraction": float(land.mean()),
            "water_fraction": float(water.mean()),
            "relief": int(s.max() - s.min()),
            "void_fraction": float((~solid).mean()),
        }


def column_grid(x0: int, z0: int, nx: int, nz: int, step: int):
    xs = x0 + np.arange(nx, dtype=np.float64) * step
    zs = z0 + np.arange(nz, dtype=np.float64) * step
    gx, gz = np.meshgrid(xs, zs)                      # (nz, nx)
    return gx.ravel()[None, :], gz.ravel()[None, :]   # (1, N)


def lattice_y(world: World) -> np.ndarray:
    ch = world.cell_height
    n = world.noise.height // ch
    return world.noise.min_y + np.arange(n + 1) * ch


def sample_density(world: World, node, x_flat, z_flat, y_levels, batch: int = 8192) -> np.ndarray:
    """Evaluate a node over a column grid. Returns (len(y_levels), N)."""
    y = np.asarray(y_levels, dtype=np.float64)[:, None]
    n = x_flat.shape[1]
    out = np.empty((y.shape[0], n), dtype=np.float64)
    for start in range(0, n, batch):
        end = min(n, start + batch)
        ctx = Ctx(x_flat[:, start:end], y, z_flat[:, start:end])
        value = node.eval(ctx)
        out[:, start:end] = np.broadcast_to(np.asarray(value, dtype=np.float64),
                                            (y.shape[0], end - start))
    return out


def surface_from_density(density: np.ndarray, y_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Highest solid block per column, refined by the zero crossing.

    density is (M, N) with y_levels ascending.
    """
    solid = density > SOLID
    any_solid = solid.any(axis=0)
    top_idx = solid.shape[0] - 1 - np.argmax(solid[::-1], axis=0)
    top_idx = np.where(any_solid, top_idx, 0)

    n = np.arange(density.shape[1])
    d_here = density[top_idx, n]
    y_here = y_levels[top_idx].astype(np.float64)
    has_above = top_idx < density.shape[0] - 1
    idx_above = np.minimum(top_idx + 1, density.shape[0] - 1)
    d_above = density[idx_above, n]
    y_above = y_levels[idx_above].astype(np.float64)

    denom = d_here - d_above
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom > 0, d_here / denom, 0.0)
    crossing = np.where(has_above, y_here + frac * (y_above - y_here), y_here)
    # Density is solid strictly above 0, so the top solid block is the highest
    # integer y strictly below the crossing. ceil-1 rather than floor: when the
    # crossing lands exactly on a lattice point that point has density 0, which
    # the game counts as air. Worth about 1.5% of columns.
    surface = np.ceil(crossing) - 1.0
    return surface.astype(np.int64), any_solid


def cell_interpolated(node) -> bool:
    """True when density is exactly linear in y between lattice points, so
    sampling the lattice is not an approximation.

    That holds only when the root is the interpolated node, or a pass-through
    wrapper around it. It is not enough for the interpolated node to be
    somewhere inside: anything non-linear applied on top, a min with a cave
    function or a range_choice that switches branch part-way up a cell, makes
    the value between two lattice points something other than a straight line,
    and a lattice scan can then step straight over a cave roof.

    Vanilla is min(squeeze(interpolated(...)), noodle) and therefore fails; a
    pack whose final_density is interpolated(...) passes and renders about five
    times faster.
    """
    seen: set[int] = set()

    def peel(n) -> bool:
        while True:
            if n is None or id(n) in seen:
                return False
            if isinstance(n, Interpolated):
                return True
            if isinstance(n, Marker):
                seen.add(id(n))
                n = n.arg
                continue
            return False

    return peel(node)


def _compact_cache(cache2d: dict, keep: np.ndarray, width: int) -> dict:
    """Keep the y-independent results when columns drop out of the scan."""
    out = {}
    for key, value in cache2d.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 2 and value.shape[1] == width:
                out[key] = value[:, keep]
            elif value.ndim == 1 and value.shape[0] == width:
                out[key] = value[keep]
            elif value.ndim == 0:
                out[key] = value
        elif np.isscalar(value):
            out[key] = value
    return out


def _with_aquifer(aquifer, values, xs, zs, ys) -> np.ndarray:
    """Raise the density where an aquifer barrier would turn air into stone."""
    air = values <= SOLID
    if not air.any():
        return values
    x = np.broadcast_to(xs, values.shape)[air].astype(np.int64)
    z = np.broadcast_to(zs, values.shape)[air].astype(np.int64)
    y = np.broadcast_to(ys, values.shape)[air].astype(np.int64)
    values = values.copy()
    values[air] += aquifer.pressure(x, y, z)
    return values


def scan_surface(world: World, node, x_flat, z_flat, batch: int = 2048,
                 slab: int = 48, hint: np.ndarray | None = None,
                 margin: int | None = None,
                 aquifer: Aquifer | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The game's own algorithm: evaluate every block from the top down and take
    the first with density above 0. Exact whatever the tree contains.

    Two things keep it affordable. Columns leave the scan as soon as they find
    their surface. And when a hint (the cheap lattice-scan surface) is given,
    columns are processed in descending height order and each batch starts just
    above its own tallest column instead of at the world ceiling, so an ocean
    batch never rescans 250 blocks of sky. The start height is checked to be air
    before being trusted, and any batch that starts inside rock falls back to
    scanning from the top.
    """
    n = x_flat.shape[1]
    surface = np.full(n, world.noise.min_y - 1, dtype=np.int64)
    found = np.zeros(n, dtype=bool)
    top, bottom = world.noise.max_y, world.noise.min_y
    margin = 3 * world.cell_height if margin is None else margin

    order = np.argsort(-np.asarray(hint).ravel()) if hint is not None else np.arange(n)

    for start in range(0, n, batch):
        columns = order[start:start + batch]
        if columns.size == 0:
            continue
        xs = x_flat[:, columns]
        zs = z_flat[:, columns]
        cache2d: dict = {}
        y_hi = top
        if hint is not None:
            guess = int(np.max(np.asarray(hint).ravel()[columns])) + margin
            y_hi = int(min(top, max(bottom, guess)))
            probe = Ctx(xs, np.array([[float(y_hi)]]), zs, cache2d=cache2d)
            values = np.broadcast_to(np.asarray(node.eval(probe), dtype=np.float64),
                                     (1, columns.size))
            if aquifer is not None:
                values = _with_aquifer(aquifer, values, xs, zs, np.array([[y_hi]]))
            if np.any(values > SOLID):        # started inside rock: be safe
                y_hi = top
                cache2d = {}
        while y_hi >= bottom and columns.size:
            y_lo = max(bottom, y_hi - slab + 1)
            ys = np.arange(y_hi, y_lo - 1, -1, dtype=np.float64)     # descending
            ctx = Ctx(xs, ys[:, None], zs, cache2d=cache2d)
            values = np.broadcast_to(np.asarray(node.eval(ctx), dtype=np.float64),
                                     (ys.size, columns.size))
            if aquifer is not None:
                values = _with_aquifer(aquifer, values, xs, zs, ys[:, None])
            solid = values > SOLID
            hit = solid.any(axis=0)
            if hit.any():
                first = np.argmax(solid, axis=0)
                sel = np.nonzero(hit)[0]
                surface[columns[sel]] = ys[first[sel]].astype(np.int64)
                found[columns[sel]] = True
                keep = ~hit
                width = columns.size
                columns = columns[keep]
                xs = xs[:, keep]
                zs = zs[:, keep]
                cache2d = _compact_cache(cache2d, keep, width)
            y_hi = y_lo - 1
    return surface, found


def sample_terrain(world: World, x0: int, z0: int, nx: int, nz: int, step: int = 4,
                   y_levels: np.ndarray | None = None, batch: int = 8192,
                   sampling: str = "auto") -> Terrain:
    node = world.router["final_density"]
    prepare(node)
    x_flat, z_flat = column_grid(x0, z0, nx, nz, step)
    aquifer = Aquifer(world) if world.aquifers_enabled else None
    mode = sampling
    if mode == "auto":
        mode = "lattice" if (cell_interpolated(node) and aquifer is None) else "block"

    if y_levels is None:
        y_levels = lattice_y(world)
    if mode == "lattice":
        density = sample_density(world, node, x_flat, z_flat, y_levels, batch=batch)
        surface, any_solid = surface_from_density(density, y_levels)
    else:
        # a cheap lattice pass first, purely as a starting-height hint
        approx = sample_density(world, node, x_flat, z_flat, y_levels, batch=batch)
        hint, _ = surface_from_density(approx, y_levels)
        del approx
        surface, any_solid = scan_surface(world, node, x_flat, z_flat, hint=hint,
                                          aquifer=aquifer)

    below = world.noise.min_y - 1
    surface = np.where(any_solid, surface, below)
    return Terrain(
        world=world, x0=x0, z0=z0, nx=nx, nz=nz, step=step,
        y_levels=np.asarray(y_levels),
        surface_y=surface.reshape(nz, nx),
        solid_anywhere=any_solid.reshape(nz, nx),
        sampling=mode,
    )


def base_height(world: World, xs, zs) -> np.ndarray:
    """NoiseChunkGenerator.getBaseHeight for WORLD_SURFACE_WG, water included."""
    xs = np.asarray(xs, dtype=np.float64)[None, :]
    zs = np.asarray(zs, dtype=np.float64)[None, :]
    node = world.router["final_density"]
    prepare(node)
    if cell_interpolated(node):
        y_levels = lattice_y(world)
        density = sample_density(world, node, xs, zs, y_levels)
        surface, any_solid = surface_from_density(density, y_levels)
    else:
        surface, any_solid = scan_surface(world, node, xs, zs)
    surface = np.where(any_solid, surface, world.noise.min_y - 1)
    return np.maximum(surface + 1, world.sea_level)


def cross_section(world: World, x0: int, z: int, length: int, step: int = 1,
                  y_step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertical slice along +x at fixed z. Returns (xs, ys, solid mask)."""
    node = world.router["final_density"]
    prepare(node)
    xs = x0 + np.arange(length // step) * step
    ys = np.arange(world.noise.min_y, world.noise.max_y + 1, y_step)
    x_flat = xs.astype(np.float64)[None, :]
    z_flat = np.full_like(x_flat, float(z))
    density = sample_density(world, node, x_flat, z_flat, ys)
    return xs, ys, density > SOLID
