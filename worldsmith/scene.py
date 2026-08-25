"""Ties terrain + climate + surface rules into one renderable sample."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .climate import BiomeSource, assign_biomes, climate_target
from .density import Ctx, prepare
from .surface import NO_WATER, SurfaceContext, SurfaceSystem
from .terrain import Terrain, sample_terrain
from .world import World


@dataclass
class Scene:
    world: World
    terrain: Terrain
    surface_block: np.ndarray        # (nz, nx) index into `palette`
    palette: list[str]
    biome_index: np.ndarray          # (nz, nx) index into `biomes`
    biomes: list[str]
    climate: np.ndarray              # (nz, nx, 7)
    surface_depth: np.ndarray

    @property
    def height(self) -> np.ndarray:
        return self.terrain.surface_y

    def biome_histogram(self) -> list[tuple[str, float]]:
        counts = np.bincount(self.biome_index.ravel(), minlength=len(self.biomes))
        total = max(1, counts.sum())
        pairs = [(self.biomes[i], counts[i] / total) for i in range(len(self.biomes)) if counts[i]]
        return sorted(pairs, key=lambda p: -p[1])

    def block_histogram(self) -> list[tuple[str, float]]:
        counts = np.bincount(self.surface_block.ravel(), minlength=len(self.palette))
        total = max(1, counts.sum())
        pairs = [(self.palette[i], counts[i] / total) for i in range(len(self.palette)) if counts[i]]
        return sorted(pairs, key=lambda p: -p[1])


def biome_climate_table(world: World, biome_ids: list[str]) -> np.ndarray:
    """(temperature, downfall) per biome, from the biome JSON."""
    out = np.zeros((len(biome_ids), 2), dtype=np.float64)
    for i, ident in enumerate(biome_ids):
        data = world.registries.get("biome", ident) or {}
        out[i, 0] = float(data.get("temperature", 0.5))
        out[i, 1] = float(data.get("downfall", 0.5))
    return out


def build_scene(world: World, biome_source: BiomeSource, x0: int, z0: int,
                nx: int, nz: int, step: int = 4, preliminary: bool = True,
                batch: int = 8192) -> Scene:
    terrain = sample_terrain(world, x0, z0, nx, nz, step, batch=batch)
    n = nx * nz
    xs = np.repeat(terrain.xs[None, :], nz, axis=0).ravel().astype(np.int64)
    zs = np.repeat(terrain.zs[:, None], nx, axis=1).ravel().astype(np.int64)
    surface_y = terrain.surface_y.ravel().astype(np.int64)

    climate = climate_target(world, xs, zs, surface_y)
    biome_idx = assign_biomes(biome_source, climate)
    biomes = biome_source.biomes

    system = SurfaceSystem(world)
    depth = system.surface_depth(xs, zs)
    secondary = np.asarray(system.secondary_noise.sample(xs.astype(float), 0.0, zs.astype(float)))

    sea = world.sea_level
    water_height = np.where(surface_y < sea, sea, NO_WATER).astype(np.float64)

    if preliminary and world.router.get("preliminary_surface_level") is not None:
        node = world.router["preliminary_surface_level"]
        prepare(node)
        prelim = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch):
            end = min(n, start + batch)
            ctx = Ctx(xs[None, start:end].astype(float), 0.0, zs[None, start:end].astype(float))
            prelim[start:end] = np.broadcast_to(
                np.asarray(node.eval(ctx), dtype=np.float64), (1, end - start))[0]
        min_surface = np.floor(prelim) + depth - 8.0
    else:
        min_surface = np.full(n, float(world.noise.min_y))

    # `steep` looks one block either side of the column and asks for a 4 block
    # step. It reads both axes, and the two comparisons run in opposite
    # directions: rising in +z, or falling in +x. Reproducing only the z half
    # marks 6.5% of a canyon world flat that the game builds as cliff.
    h = terrain.surface_y.astype(np.float64)
    z_lo = np.vstack([h[:1], h[:-1]])
    z_hi = np.vstack([h[1:], h[-1:]])
    x_lo = np.hstack([h[:, :1], h[:, :-1]])
    x_hi = np.hstack([h[:, 1:], h[:, -1:]])
    if step == 1:
        # the game reads its own chunk, so a lookup off the edge is clamped back
        # to the column itself rather than crossing into the neighbour
        zc = ((z0 + np.arange(nz)) & 15)[:, None]
        xc = ((x0 + np.arange(nx)) & 15)[None, :]
        z_lo = np.where(zc == 0, h, z_lo)
        z_hi = np.where(zc == 15, h, z_hi)
        x_lo = np.where(xc == 0, h, x_lo)
        x_hi = np.where(xc == 15, h, x_hi)
    # at step > 1 the neighbours are step blocks away, so scale the difference
    # back to the 2-block baseline the rule is written against
    scale = max(1, step)
    steep = ((z_hi - z_lo) / scale >= 4.0) | ((x_lo - x_hi) / scale >= 4.0)

    climate_table = biome_climate_table(world, biomes)
    base_temp = climate_table[biome_idx, 0]
    # altitude cooling, as Biome.getTemperature does above sea level
    temp = np.where(surface_y > sea,
                    base_temp - (surface_y - sea) * 0.05 / 40.0,
                    base_temp)

    ctx = SurfaceContext(
        world=world, x=xs, z=zs, y=surface_y,
        stone_depth_above=np.ones(n, dtype=np.int64),
        stone_depth_below=np.full(n, 1 << 20, dtype=np.int64),
        water_height=water_height,
        surface_depth=depth,
        surface_secondary=secondary,
        min_surface_level=min_surface,
        biome=biome_idx,
        biome_names=biomes,
        biome_temperature=temp,
        steep=steep.ravel(),
    )
    blocks = system.evaluate(ctx)

    # columns below sea level show the fluid, not the block underneath
    fluid = system.block_index(world.default_fluid)
    blocks = np.where(surface_y < sea, fluid, blocks)
    # columns with no terrain at all are void
    air = system.block_index("minecraft:air")
    blocks = np.where(terrain.solid_anywhere.ravel(), blocks, air)

    return Scene(
        world=world, terrain=terrain,
        surface_block=blocks.reshape(nz, nx), palette=system.palette,
        biome_index=biome_idx.reshape(nz, nx), biomes=biomes,
        climate=climate.reshape(nz, nx, 7), surface_depth=depth.reshape(nz, nx),
    )
