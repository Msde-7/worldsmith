"""Aquifers: underground water and lava, and the stone barriers between them.

With `aquifers_enabled` the game does not simply read final_density and call
anything below zero air. It scatters aquifer centres on a 16x12x16 grid, gives
each one a fluid and a level, and where two neighbouring aquifers disagree it
raises the density again so a wall of stone forms between them. Those barriers
are solid, so they show up in the heightmap, which is why an engine that skips
them reads the surface too low in cave country.

The shape of the algorithm, per position with density <= 0:

  * take the three nearest aquifer centres out of the twelve grid cells around
  * work out each one's fluid level, which depends on the preliminary surface
    above it, a floodedness noise and a spread noise
  * turn the disagreement between them into a pressure, softened by the barrier
    noise, and add that to the density

Everything here is vectorised over positions, and the two expensive parts, the
preliminary surface and the per-centre status, are cached the way the game
caches them per chunk.
"""
from __future__ import annotations

import numpy as np

from .density import Ctx, prepare
from .jrandom import Xoroshiro

X_SPACING, Y_SPACING, Z_SPACING = 16, 12, 16
LAVA_LEVEL = -54

# Offsets, in chunks, that a centre samples to find the surface above it. The
# order matters: (0, 0) sits eighth, so the running minimum covers the first
# seven before the centre itself can return early.
SURFACE_SAMPLING = ((-2, -1), (-1, -1), (0, -1), (1, -1), (-3, 0), (-2, 0), (-1, 0),
                    (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1))

# Java uses Integer.MIN_VALUE for "this aquifer holds no fluid at all".
NO_FLUID = np.int64(-(1 << 31))

NEIGHBOURS = tuple((ox, oy, oz) for ox in (0, 1) for oy in (-1, 0, 1) for oz in (0, 1))


def _lerp(t, a, b):
    return a + t * (b - a)


def _inverse_lerp(v, lo, hi):
    return (v - lo) / (hi - lo)


def _map(v, from_lo, from_hi, to_lo, to_hi):
    return _lerp(_inverse_lerp(v, from_lo, from_hi), to_lo, to_hi)


def _clamped_map(v, from_lo, from_hi, to_lo, to_hi):
    t = np.clip(_inverse_lerp(v, from_lo, from_hi), 0.0, 1.0)
    return _lerp(t, to_lo, to_hi)


class Aquifer:
    """The state a dimension needs to answer "is this air actually stone?"."""

    def __init__(self, world):
        self.world = world
        self.sea_level = world.sea_level
        self.default_is_lava = "lava" in world.default_fluid
        self.random = (Xoroshiro.create(world.seed).fork_positional()
                       .from_hash_of("minecraft:aquifer").fork_positional())

        self.barrier = world.router["barrier"]
        self.floodedness = world.router["fluid_level_floodedness"]
        self.spread = world.router["fluid_level_spread"]
        self.lava = world.router["lava"]
        self.surface_level = world.router["preliminary_surface_level"]
        prepare(self.barrier, self.floodedness, self.spread, self.lava, self.surface_level)

        # key -> value caches, kept sorted so lookups are a searchsorted away
        self._surface_keys = np.zeros(0, dtype=np.int64)
        self._surface_values = np.zeros(0, dtype=np.int64)
        self._status_keys = np.zeros(0, dtype=np.int64)
        self._status_levels = np.zeros(0, dtype=np.int64)
        self._status_lava = np.zeros(0, dtype=bool)

    def _eval(self, node, x, y, z) -> np.ndarray:
        """One value per position, the game's SinglePointContext."""
        shape = np.shape(x)
        ctx = Ctx(np.asarray(x, dtype=np.float64)[None, :],
                  np.asarray(y, dtype=np.float64)[None, :],
                  np.asarray(z, dtype=np.float64)[None, :])
        return np.broadcast_to(np.asarray(node.eval(ctx), dtype=np.float64), (1,) + shape)[0]

    # -- the preliminary surface, per quart-aligned column --------------------
    def preliminary_surface(self, x, z) -> np.ndarray:
        qx = (np.asarray(x, dtype=np.int64) >> 2) << 2
        qz = (np.asarray(z, dtype=np.int64) >> 2) << 2
        keys = (qx.astype(np.int64) << np.int64(32)) ^ (qz.astype(np.int64) & np.int64(0xFFFFFFFF))
        unique = np.unique(keys)
        known = np.isin(unique, self._surface_keys)
        missing = unique[~known]
        if missing.size:
            mx = (missing >> np.int64(32)).astype(np.int64)
            mz = (missing & np.int64(0xFFFFFFFF)).astype(np.int32).astype(np.int64)
            values = self._eval(self.surface_level, mx, np.zeros_like(mx), mz).astype(np.int64)
            order = np.argsort(np.concatenate([self._surface_keys, missing]))
            self._surface_keys = np.concatenate([self._surface_keys, missing])[order]
            self._surface_values = np.concatenate([self._surface_values, values])[order]
        return self._surface_values[np.searchsorted(self._surface_keys, keys)]

    # -- fluid levels ---------------------------------------------------------
    def _global_status(self, y) -> tuple[np.ndarray, np.ndarray]:
        """The dimension-wide fallback: lava at the very bottom, sea above."""
        lava = np.asarray(y) < min(LAVA_LEVEL, self.sea_level)
        level = np.where(lava, LAVA_LEVEL, self.sea_level).astype(np.int64)
        return level, lava | self.default_is_lava

    def status(self, x, y, z) -> tuple[np.ndarray, np.ndarray]:
        """(level, is_lava) for the aquifer centred at each position."""
        x = np.asarray(x, dtype=np.int64).ravel()
        y = np.asarray(y, dtype=np.int64).ravel()
        z = np.asarray(z, dtype=np.int64).ravel()
        keys = (((x & np.int64(0x3FFFFFF)) << np.int64(38))
                | ((z & np.int64(0x3FFFFFF)) << np.int64(12))
                | (y & np.int64(0xFFF)))
        unique, representative = np.unique(keys, return_index=True)
        fresh = ~np.isin(unique, self._status_keys)
        if np.any(fresh):
            take = representative[fresh]
            level, lava = self._compute_status(x[take], y[take], z[take])
            merged = np.concatenate([self._status_keys, unique[fresh]])
            order = np.argsort(merged)
            self._status_keys = merged[order]
            self._status_levels = np.concatenate([self._status_levels, level])[order]
            self._status_lava = np.concatenate([self._status_lava, lava])[order]
        at = np.searchsorted(self._status_keys, keys)
        return self._status_levels[at], self._status_lava[at]

    def _compute_status(self, x, y, z) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        z = np.asarray(z, dtype=np.int64)
        global_level, global_lava = self._global_status(y)

        level = global_level.copy()
        is_lava = np.asarray(global_lava, dtype=bool).copy()
        done = np.zeros(x.shape, dtype=bool)
        min_surface = np.full(x.shape, np.int64(1 << 40))
        near_surface = np.zeros(x.shape, dtype=bool)

        # One scan for all 13 offsets: the surface is found by walking a column
        # down, so asking for 13 columns at once costs about what one does.
        offsets = np.array(SURFACE_SAMPLING, dtype=np.int64)
        surfaces = self.preliminary_surface(
            (x[None, :] + offsets[:, 0, None] * 16).ravel(),
            (z[None, :] + offsets[:, 1, None] * 16).ravel()).reshape(len(SURFACE_SAMPLING), -1)

        for (ox, oz), surface in zip(SURFACE_SAMPLING, surfaces):
            min_surface = np.minimum(min_surface, surface)
            centred = ox == 0 and oz == 0
            if centred:
                # far enough below the surface that the global fluid wins
                early = ~done & (y - 12 > surface + 8)
                level = np.where(early, global_level, level)
                is_lava = np.where(early, global_lava, is_lava)
                done |= early
            above = y + 12 > surface + 8
            considered = ~done & (above | centred)
            if not np.any(considered):
                continue
            probe_y = surface + 8
            probe_level, probe_lava = self._global_status(probe_y)
            wet = considered & (probe_y < probe_level)      # the probe is inside fluid
            # The centre only flags that this aquifer is near the surface; it is
            # a sample reaching above the surface that hands over to the sea.
            near_surface |= wet & centred
            take = wet & above
            level = np.where(take, probe_level, level)
            is_lava = np.where(take, probe_lava, is_lava)
            done |= take

        allowed = np.where(near_surface, _clamped_map(min_surface + 8 - y, 0, 64, 1, 0), 0.0)
        floodedness = np.clip(self._eval(self.floodedness, x, y * 0.67, z), -1.0, 1.0)

        to_global = ~done & (floodedness > _map(allowed, 1, 0, -0.3, 0.8))
        level = np.where(to_global, global_level, level)
        is_lava = np.where(to_global, global_lava, is_lava)
        done |= to_global

        dry = ~done & (floodedness <= _map(allowed, 1, 0, -0.8, 0.4))
        level = np.where(dry, NO_FLUID, level)
        is_lava = np.where(dry, global_lava, is_lava)
        done |= dry

        if np.any(~done):
            grid_y = np.floor_divide(y, 40)
            spread = self._eval(self.spread, np.floor_divide(x, 16), grid_y, np.floor_divide(z, 16))
            # levels sit on a 3-block ladder from the middle of each 40-block
            # band. deepslate drops the factor of 10 here, which puts every
            # aquifer a step or two off; the blocks in a generated world say
            # this form is the right one.
            fluid_level = grid_y * 40 + 20 + (np.floor(spread * 10.0 / 3.0) * 3).astype(np.int64)
            deep = fluid_level <= -10
            fluid_lava = np.asarray(global_lava, dtype=bool).copy()
            if np.any(deep):
                lava_noise = self._eval(self.lava, np.floor_divide(x, 64),
                                        np.floor_divide(y, 40), np.floor_divide(z, 64))
                fluid_lava = fluid_lava | (deep & (np.abs(lava_noise) > 0.3))
            level = np.where(~done, np.minimum(min_surface, fluid_level), level)
            is_lava = np.where(~done, fluid_lava, is_lava)
        return level.astype(np.int64), is_lava.astype(bool)

    # -- pressure -------------------------------------------------------------
    def _location(self, cx, cy, cz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the aquifer centre for a grid cell actually sits."""
        random = self.random.at_batch(cx, cy, cz)
        return (cx * X_SPACING + random.next_int(10),
                cy * Y_SPACING + random.next_int(9),
                cz * Z_SPACING + random.next_int(10))

    def _calculate_pressure(self, y, level_a, lava_a, level_b, lava_b, barrier) -> np.ndarray:
        fluid_a = y < level_a
        fluid_b = y < level_b
        opposed = ((fluid_a & lava_a & fluid_b & ~lava_b)
                   | (fluid_a & ~lava_a & fluid_b & lava_b))

        difference = np.abs(level_a - level_b).astype(np.float64)
        average = (level_a + level_b) / 2.0
        above = y + 0.5 - average
        p = difference / 2.0 - np.abs(above)
        pressure = np.where(above > 0,
                            np.where(p > 0, p / 1.5, p / 2.5),
                            np.where(p > -3, (p + 3) / 3.0, (p + 3) / 10.0))
        # the barrier noise only softens the middle of the range
        pressure = np.where((pressure < -2) | (pressure > 2), pressure, pressure + barrier)
        pressure = np.where(difference == 0, 0.0, pressure)
        return np.where(opposed, 1.0, pressure)

    def pressure(self, x, y, z) -> np.ndarray:
        """How much to add to a non-positive density before calling it air."""
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        z = np.asarray(z, dtype=np.int64)
        shape = x.shape

        global_level, global_lava = self._global_status(y)
        in_lava_sea = global_lava & (y < global_level)

        grid_x = np.floor_divide(x - 5, X_SPACING)
        grid_y = np.floor_divide(y + 1, Y_SPACING)
        grid_z = np.floor_divide(z - 5, Z_SPACING)

        big = np.int64(1 << 40)
        mag = [np.full(shape, big) for _ in range(3)]
        loc = [[np.zeros(shape, dtype=np.int64) for _ in range(3)] for _ in range(3)]

        for ox, oy, oz in NEIGHBOURS:
            lx, ly, lz = self._location(grid_x + ox, grid_y + oy, grid_z + oz)
            dx, dy, dz = lx - x, ly - y, lz - z
            m = dx * dx + dy * dy + dz * dz
            # the game keeps the three smallest with >=, so a later tie wins
            first = mag[0] >= m
            second = ~first & (mag[1] >= m)
            third = ~first & ~second & (mag[2] >= m)
            for axis in range(3):
                loc[2][axis] = np.where(first | second, loc[1][axis],
                                        np.where(third, [lx, ly, lz][axis], loc[2][axis]))
                loc[1][axis] = np.where(first, loc[0][axis],
                                        np.where(second, [lx, ly, lz][axis], loc[1][axis]))
                loc[0][axis] = np.where(first, [lx, ly, lz][axis], loc[0][axis])
            mag[2] = np.where(first | second, mag[1], np.where(third, m, mag[2]))
            mag[1] = np.where(first, mag[0], np.where(second, m, mag[1]))
            mag[0] = np.where(first, m, mag[0])

        level1, lava1 = self.status(loc[0][0], loc[0][1], loc[0][2])
        level2, lava2 = self.status(loc[1][0], loc[1][1], loc[1][2])
        level3, lava3 = self.status(loc[2][0], loc[2][1], loc[2][2])

        similarity12 = 1.0 - np.abs(mag[1] - mag[0]) / 25.0
        similarity13 = 1.0 - np.abs(mag[2] - mag[0]) / 25.0
        similarity23 = 1.0 - np.abs(mag[2] - mag[1]) / 25.0

        barrier = self._eval(self.barrier, x, y * 0.5, z)
        p12 = self._calculate_pressure(y, level1, lava1, level2, lava2, barrier)
        p13 = self._calculate_pressure(y, level1, lava1, level3, lava3, barrier)
        p23 = self._calculate_pressure(y, level2, lava2, level3, lava3, barrier)
        combined = np.maximum(p12, np.maximum(p13 * np.maximum(0.0, similarity13),
                                              p23 * np.maximum(0.0, similarity23)))
        pressure = np.maximum(0.0, 2.0 * np.maximum(0.0, similarity12) * combined)
        pressure = np.where(similarity12 > -1, pressure, 0.0)

        # water sitting directly on the lava sea always seals
        below_level, below_lava = self._global_status(y - 1)
        on_lava = (y < level1) & ~lava1 & below_lava & (y - 1 < below_level)
        pressure = np.where(on_lava, 1.0, pressure)

        # inside the lava sea the substance is lava, never a barrier
        return np.where(in_lava_sea, 0.0, pressure)
