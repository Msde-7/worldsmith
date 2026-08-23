"""Vectorised ports of Minecraft's noise stack.

ImprovedNoise feeds PerlinNoise (octaves) feeds NormalNoise (two Perlins), plus
BlendedNoise (the pre-1.18 old_blended_noise) and the 2D simplex used by
end_islands.

Every sample() accepts numpy arrays and relies on broadcasting: pass x and z
shaped (1, N) and y shaped (M, 1) and you get an (M, N) result, while a purely
2D function (y_scale == 0) collapses to (1, N) and is therefore evaluated once
per column instead of once per block.
"""
from __future__ import annotations

import numpy as np

from . import kernels
from .jrandom import JavaRandom, Xoroshiro

# net.minecraft.world.level.levelgen.synth.SimplexNoise.GRADIENT
GRADIENT = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 1, 0], [0, -1, 1], [-1, 1, 0], [0, -1, -1],
], dtype=np.float64)
_GX = np.ascontiguousarray(GRADIENT[:, 0])
_GY = np.ascontiguousarray(GRADIENT[:, 1])
_GZ = np.ascontiguousarray(GRADIENT[:, 2])

WRAP_PERIOD = 3.3554432e7

# Turned off by the test that compares the numba kernel against the numpy path.
USE_KERNEL = [True]

Random = JavaRandom | Xoroshiro


def smoothstep(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(t, a, b):
    return a + t * (b - a)


def wrap(value):
    return value - np.floor(value / WRAP_PERIOD + 0.5) * WRAP_PERIOD


def _grad_dot(h, x, y, z):
    """SimplexNoise.gradDot over an index array."""
    idx = h & 15
    return _GX[idx] * x + _GY[idx] * y + _GZ[idx] * z


class GradientTable:
    """The shuffled permutation and sample offsets both noises start from."""

    __slots__ = ("p", "xo", "yo", "zo")

    def __init__(self, random: Random):
        self.xo = random.next_double() * 256.0
        self.yo = random.next_double() * 256.0
        self.zo = random.next_double() * 256.0
        p = list(range(256))
        for i in range(256):
            j = random.next_int(256 - i)
            p[i], p[i + j] = p[i + j], p[i]
        self.p = np.array(p, dtype=np.int64)      # already 0..255, i.e. signed byte & 0xFF

    def _P(self, i):
        return self.p[i & 0xFF]


class ImprovedNoise(GradientTable):
    """A single Perlin octave."""

    __slots__ = ()

    def sample(self, x, y, z, y_scale=0.0, y_limit=0.0):
        if kernels.HAVE_NUMBA and USE_KERNEL[0]:
            fast = kernels.sample_grid(self, x, y, z, y_scale, y_limit)
            if fast is not None:
                return fast
        return self.sample_numpy(x, y, z, y_scale, y_limit)

    def sample_numpy(self, x, y, z, y_scale=0.0, y_limit=0.0):
        x2 = x + self.xo
        y2 = y + self.yo
        z2 = z + self.zo
        xi = np.floor(x2)
        yi = np.floor(y2)
        zi = np.floor(z2)
        xf = x2 - xi
        yf = y2 - yi
        zf = z2 - zi
        xi = xi.astype(np.int64) if isinstance(xi, np.ndarray) else np.int64(xi)
        yi = yi.astype(np.int64) if isinstance(yi, np.ndarray) else np.int64(yi)
        zi = zi.astype(np.int64) if isinstance(zi, np.ndarray) else np.int64(zi)

        yy = yf
        scale_is_zero = np.isscalar(y_scale) and y_scale == 0.0
        if not scale_is_zero:
            t = np.where((y_limit >= 0) & (y_limit < yf), y_limit, yf)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.floor(t / y_scale + 1.0e-7) * y_scale
            step = np.where(np.asarray(y_scale) == 0.0, 0.0, step)
            yy = yf - step

        h = self._P(xi)
        i_ = self._P(xi + 1)
        j_ = self._P(h + yi)
        k_ = self._P(h + yi + 1)
        l_ = self._P(i_ + yi)
        m_ = self._P(i_ + yi + 1)

        n = _grad_dot(self._P(j_ + zi), xf, yy, zf)
        o = _grad_dot(self._P(l_ + zi), xf - 1.0, yy, zf)
        p_ = _grad_dot(self._P(k_ + zi), xf, yy - 1.0, zf)
        q = _grad_dot(self._P(m_ + zi), xf - 1.0, yy - 1.0, zf)
        r = _grad_dot(self._P(j_ + zi + 1), xf, yy, zf - 1.0)
        s = _grad_dot(self._P(l_ + zi + 1), xf - 1.0, yy, zf - 1.0)
        t2 = _grad_dot(self._P(k_ + zi + 1), xf, yy - 1.0, zf - 1.0)
        u = _grad_dot(self._P(m_ + zi + 1), xf - 1.0, yy - 1.0, zf - 1.0)

        a = smoothstep(xf)
        b = smoothstep(yf)
        c = smoothstep(zf)
        lo = lerp(b, lerp(a, n, o), lerp(a, p_, q))
        hi = lerp(b, lerp(a, r, s), lerp(a, t2, u))
        return lerp(c, lo, hi)


class PerlinNoise:
    """Octave stack.

    force_legacy selects the sequential seeding used by old_blended_noise and by
    legacy_random_source worlds; otherwise each octave is seeded independently
    from its own name.
    """

    def __init__(self, random: Random, first_octave: int, amplitudes, force_legacy: bool = False):
        amplitudes = list(amplitudes)
        self.amplitudes = np.array(amplitudes, dtype=np.float64)
        self.first_octave = first_octave
        self.noise_levels: list[ImprovedNoise | None] = [None] * len(amplitudes)

        if isinstance(random, Xoroshiro) and not force_legacy:
            forked = random.fork_positional()
            for i, amp in enumerate(amplitudes):
                if amp != 0.0:
                    self.noise_levels[i] = ImprovedNoise(forked.from_hash_of(f"octave_{first_octave + i}"))
        else:
            if 1 - first_octave < len(amplitudes):
                raise ValueError("positive octaves are not allowed with the legacy random source")
            for i in range(-first_octave, -1, -1):
                if i < len(amplitudes) and amplitudes[i] != 0.0:
                    self.noise_levels[i] = ImprovedNoise(random)
                else:
                    random.consume(262)

        self.lowest_freq_input_factor = 2.0 ** first_octave
        n = len(amplitudes)
        self.lowest_freq_value_factor = (2.0 ** (n - 1)) / (2.0 ** n - 1.0)
        self.max_value = self.edge_value(2.0)

    def edge_value(self, x: float) -> float:
        value = 0.0
        factor = self.lowest_freq_value_factor
        for i, noise in enumerate(self.noise_levels):
            if noise is not None:
                value += self.amplitudes[i] * x * factor
            factor /= 2.0
        return value

    def get_octave_noise(self, i: int):
        return self.noise_levels[len(self.noise_levels) - 1 - i]

    def sample(self, x, y, z, y_scale=0.0, y_limit=0.0, fix_y=False):
        value = 0.0
        input_f = self.lowest_freq_input_factor
        value_f = self.lowest_freq_value_factor
        for i, noise in enumerate(self.noise_levels):
            if noise is not None:
                yv = -noise.yo if fix_y else wrap(y * input_f)
                value = value + self.amplitudes[i] * value_f * noise.sample(
                    wrap(x * input_f), yv, wrap(z * input_f),
                    y_scale * input_f, y_limit * input_f,
                )
            input_f *= 2.0
            value_f /= 2.0
        return value


class NormalNoise:
    INPUT_FACTOR = 1.0181268882175227

    def __init__(self, random: Random, first_octave: int, amplitudes):
        self.first = PerlinNoise(random, first_octave, amplitudes)
        self.second = PerlinNoise(random, first_octave, amplitudes)
        nonzero = [i for i, a in enumerate(amplitudes) if a != 0.0]
        lo, hi = (min(nonzero), max(nonzero)) if nonzero else (0, 0)
        expected_deviation = 0.1 * (1.0 + 1.0 / (hi - lo + 1))
        self.value_factor = (1.0 / 6.0) / expected_deviation
        self.max_value = (self.first.max_value + self.second.max_value) * self.value_factor

    def sample(self, x, y, z):
        f = self.INPUT_FACTOR
        return (self.first.sample(x, y, z) + self.second.sample(x * f, y * f, z * f)) * self.value_factor


class BlendedNoise:
    """The pre-1.18 terrain generator, still the base 3D noise of every vanilla
    dimension."""

    _AMP16 = [1.0] * 16
    _AMP8 = [1.0] * 8

    def __init__(self, random: Random, xz_scale, y_scale, xz_factor, y_factor, smear_scale_multiplier):
        self.min_limit = PerlinNoise(random, -15, self._AMP16, True)
        self.max_limit = PerlinNoise(random, -15, self._AMP16, True)
        self.main = PerlinNoise(random, -7, self._AMP8, True)
        self.xz_scale = xz_scale
        self.y_scale = y_scale
        self.xz_factor = xz_factor
        self.y_factor = y_factor
        self.smear_scale_multiplier = smear_scale_multiplier
        self.xz_multiplier = 684.412 * xz_scale
        self.y_multiplier = 684.412 * y_scale
        self.max_value = self.min_limit.edge_value(self.y_multiplier + 2.0)

    def sample(self, x, y, z):
        scaled_x = x * self.xz_multiplier
        scaled_y = y * self.y_multiplier
        scaled_z = z * self.xz_multiplier
        factored_x = scaled_x / self.xz_factor
        factored_y = scaled_y / self.y_factor
        factored_z = scaled_z / self.xz_factor
        smear = self.y_multiplier * self.smear_scale_multiplier
        factored_smear = smear / self.y_factor

        value = 0.0
        factor = 1.0
        for i in range(8):
            noise = self.main.get_octave_noise(i)
            if noise is not None:
                value = value + noise.sample(
                    wrap(factored_x * factor), wrap(factored_y * factor), wrap(factored_z * factor),
                    factored_smear * factor, factored_y * factor,
                ) / factor
            factor /= 2.0
        value = (value / 10.0 + 1.0) / 2.0

        # Java short-circuits the two limit stacks per position; evaluating both
        # gives the same arithmetic because the lerp below discards the unused
        # side.
        factor = 1.0
        vmin = 0.0
        vmax = 0.0
        for i in range(16):
            xx = wrap(scaled_x * factor)
            yy = wrap(scaled_y * factor)
            zz = wrap(scaled_z * factor)
            smeared = smear * factor
            n_min = self.min_limit.get_octave_noise(i)
            if n_min is not None:
                vmin = vmin + n_min.sample(xx, yy, zz, smeared, scaled_y * factor) / factor
            n_max = self.max_limit.get_octave_noise(i)
            if n_max is not None:
                vmax = vmax + n_max.sample(xx, yy, zz, smeared, scaled_y * factor) / factor
            factor /= 2.0

        t = np.clip(value, 0.0, 1.0)
        return lerp(t, vmin / 512.0, vmax / 512.0) / 128.0


class SimplexNoise(GradientTable):
    """2D simplex, used only by end_islands."""

    __slots__ = ()

    F2 = 0.5 * (3.0 ** 0.5 - 1.0)
    G2 = (3.0 - 3.0 ** 0.5) / 6.0

    @staticmethod
    def _corner(h, x, y, z, offset):
        t = offset - x * x - y * y - z * z
        t2 = np.where(t < 0, 0.0, t) ** 2
        return t2 * t2 * _grad_dot(h, x, y, z)

    def sample_2d(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        skew = (x + y) * self.F2
        i = np.floor(x + skew).astype(np.int64)
        j = np.floor(y + skew).astype(np.int64)
        t = (i + j) * self.G2
        x0 = x - (i - t)
        y0 = y - (j - t)
        i1 = (x0 > y0).astype(np.int64)
        j1 = 1 - i1
        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2
        ii = i & 0xFF
        jj = j & 0xFF
        g0 = self._P(ii + self._P(jj)) % 12
        g1 = self._P(ii + i1 + self._P(jj + j1)) % 12
        g2 = self._P(ii + 1 + self._P(jj + 1)) % 12
        n0 = self._corner(g0, x0, y0, 0.0, 0.5)
        n1 = self._corner(g1, x1, y1, 0.0, 0.5)
        n2 = self._corner(g2, x2, y2, 0.0, 0.5)
        return 70.0 * (n0 + n1 + n2)
