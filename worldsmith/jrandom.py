"""Bit-exact ports of Minecraft's random sources.

Two generators matter for worldgen. JavaRandom is LegacyRandomSource, i.e.
java.util.Random's 48-bit LCG, used when a noise settings file sets
legacy_random_source (only the pre-1.18 presets do). Xoroshiro is
XoroshiroRandomSource, the xoroshiro128++ generator behind every modern one.

Both expose fork_positional(), which returns a factory that derives an
independent stream from a name (md5) or from a position. Noise objects are
seeded with from_hash_of(<noise id>), which is why two dimensions sharing a
world seed but using different noise ids get uncorrelated terrain.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .compat import COMPAT

MASK64 = (1 << 64) - 1
MASK48 = (1 << 48) - 1
MASK32 = (1 << 32) - 1

_SILVER_RATIO_64 = 0x6A09E667F3BCC909
_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15
_STAFFORD_1 = 0xBF58476D1CE4E5B9
_STAFFORD_2 = 0x94D049BB133111EB

DOUBLE_MULTIPLIER = 1.1102230246251565e-16   # 2**-53
FLOAT_MULTIPLIER = 1.0 / (1 << 24)


def to_signed64(v: int) -> int:
    v &= MASK64
    return v - (1 << 64) if v >= (1 << 63) else v


def to_signed32(v: int) -> int:
    v &= MASK32
    return v - (1 << 32) if v >= (1 << 31) else v


def _rotl64(v: int, k: int) -> int:
    v &= MASK64
    return ((v << k) | (v >> (64 - k))) & MASK64


def mix_stafford13(v: int) -> int:
    v &= MASK64
    v = ((v ^ (v >> 30)) * _STAFFORD_1) & MASK64
    v = ((v ^ (v >> 27)) * _STAFFORD_2) & MASK64
    return (v ^ (v >> 31)) & MASK64


def upgrade_seed_to_128bit(seed: int) -> tuple[int, int]:
    seed &= MASK64
    lo = seed ^ _SILVER_RATIO_64
    hi = (lo + _GOLDEN_RATIO_64) & MASK64
    return mix_stafford13(lo), mix_stafford13(hi)


def get_seed(x: int, y: int, z: int) -> int:
    """Mth.getSeed.

    The x multiply is an int multiply in Java and wraps at 2^31. deepslate does
    it in JS doubles and so diverges from the game once abs(x) exceeds ~686;
    this follows the game.
    """
    l = to_signed64(to_signed32(x * 3129871)) ^ to_signed64(z * 116129781) ^ to_signed64(y)
    l = to_signed64(to_signed64(to_signed64(l * l) * 42317861) + to_signed64(l * 11))
    return l >> 16                      # arithmetic shift, like Java's >> on a long


def get_seed_np(x, y, z) -> np.ndarray:
    """Vectorised get_seed, returning int64."""
    x = np.asarray(x, dtype=np.int64)
    z = np.asarray(z, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    with np.errstate(over="ignore"):
        xi = (x.astype(np.int32) * np.int32(3129871)).astype(np.int64)
        l = xi ^ (z * np.int64(116129781)) ^ y
        l = l * l * np.int64(42317861) + l * np.int64(11)
    return l >> np.int64(16)


class JavaRandom:
    _MUL = 0x5DEECE66D
    _ADD = 0xB

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.seed = (seed ^ self._MUL) & MASK48

    def _advance(self) -> None:
        self.seed = (self.seed * self._MUL + self._ADD) & MASK48

    def _next(self, bits: int) -> int:
        self._advance()
        return to_signed32(self.seed >> (48 - bits))

    def consume(self, count: int) -> None:
        for _ in range(count):
            self._advance()

    def next_int(self, bound=None) -> int:
        if bound is None:
            return self._next(32)
        if bound <= 0:
            raise ValueError("bound must be positive")
        m = bound - 1
        if (bound & m) == 0:                       # power of two
            return (bound * self._next(31)) >> 31
        while True:
            u = self._next(31)
            r = u % bound
            if to_signed32(u - r + m) >= 0:
                return r

    def next_long(self) -> int:
        return to_signed64((self._next(32) << 32) + self._next(32))

    def next_double(self) -> float:
        if COMPAT["deepslate"]:
            # deepslate draws next(30) * 2**-30 and then burns one advance;
            # java.util.Random combines a 26-bit and a 27-bit draw.
            value = self._next(30)
            self._advance()
            return value * (1.0 / (1 << 30))
        hi = self._next(26)
        lo = self._next(27)
        return ((hi << 27) + lo) * DOUBLE_MULTIPLIER

    def next_boolean(self) -> bool:
        return self._next(1) != 0

    def next_int_between_inclusive(self, lo: int, hi: int) -> int:
        return self.next_int(hi - lo + 1) + lo

    def fork_positional(self) -> "LegacyPositional":
        return LegacyPositional(self.next_long())


class Xoroshiro:
    def __init__(self, lo: int, hi: int):
        self.lo = lo & MASK64
        self.hi = hi & MASK64

    @staticmethod
    def create(seed: int) -> "Xoroshiro":
        return Xoroshiro(*upgrade_seed_to_128bit(seed))

    def _next(self) -> int:
        lo, hi = self.lo, self.hi
        value = (_rotl64((lo + hi) & MASK64, 17) + lo) & MASK64
        hi ^= lo
        self.lo = _rotl64(lo, 49) ^ hi ^ ((hi << 21) & MASK64)
        self.hi = _rotl64(hi, 28)
        return value

    def consume(self, count: int) -> None:
        for _ in range(count):
            self._next()

    def next_long(self) -> int:
        return to_signed64(self._next())

    def next_double(self) -> float:
        return (self._next() >> 11) * DOUBLE_MULTIPLIER

    def next_boolean(self) -> bool:
        return (self._next() & 1) != 0

    def next_int(self, bound=None) -> int:
        value = self._next() & MASK32
        if bound is None:
            return to_signed32(value)
        if bound <= 0:
            raise ValueError("bound must be positive")
        product = value * bound
        product_lo = product & MASK32
        if product_lo < bound:
            threshold = ((~bound & MASK32) + 1) % bound      # (2**32 - bound) % bound
            while product_lo < threshold:
                value = self._next() & MASK32
                product = value * bound
                product_lo = product & MASK32
        return product >> 32

    def next_int_between_inclusive(self, lo: int, hi: int) -> int:
        return self.next_int(hi - lo + 1) + lo

    def fork_positional(self) -> "XoroshiroPositional":
        return XoroshiroPositional(self._next(), self._next())


def _md5_longs(name: str) -> tuple[int, int]:
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:], "big")


class XoroshiroPositional:
    def __init__(self, lo: int, hi: int):
        self.lo = lo & MASK64
        self.hi = hi & MASK64

    def at(self, x: int, y: int, z: int) -> Xoroshiro:
        return Xoroshiro(get_seed(x, y, z) ^ self.lo, self.hi)

    def from_hash_of(self, name: str) -> Xoroshiro:
        lo, hi = _md5_longs(name)
        return Xoroshiro(lo ^ self.lo, hi ^ self.hi)

    def _first_draw(self, x, y, z) -> np.ndarray:
        """The first xoroshiro output at each position, vectorised."""
        lo = get_seed_np(x, y, z).astype(np.uint64) ^ np.uint64(self.lo)
        s = lo + np.uint64(self.hi)
        return ((s << np.uint64(17)) | (s >> np.uint64(47))) + lo

    def at_batch(self, x, y, z) -> "XoroshiroBatch":
        """A stream per position, seeded exactly as at(x, y, z) would be."""
        lo = get_seed_np(x, y, z).astype(np.uint64) ^ np.uint64(self.lo)
        hi = np.full(np.shape(lo), self.hi, dtype=np.uint64)
        return XoroshiroBatch(lo, hi)

    def at_next_double(self, x, y, z) -> np.ndarray:
        return (self._first_draw(x, y, z) >> np.uint64(11)).astype(np.float64) * DOUBLE_MULTIPLIER

    def at_next_float(self, x, y, z) -> np.ndarray:
        return (self._first_draw(x, y, z) >> np.uint64(40)).astype(np.float64) * FLOAT_MULTIPLIER


class XoroshiroBatch:
    """Many independent xoroshiro streams advanced in lockstep.

    The aquifer grid needs three next_int draws from a stream seeded at every
    cell it looks at, which is far too many positions to walk one at a time.
    """

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=np.uint64).copy()
        self.hi = np.asarray(hi, dtype=np.uint64).copy()

    def next(self) -> np.ndarray:
        lo, hi = self.lo, self.hi
        s = lo + hi
        value = ((s << np.uint64(17)) | (s >> np.uint64(47))) + lo
        hi = hi ^ lo
        self.lo = ((lo << np.uint64(49)) | (lo >> np.uint64(15))) ^ hi ^ (hi << np.uint64(21))
        self.hi = (hi << np.uint64(28)) | (hi >> np.uint64(36))
        return value

    def next_int(self, bound: int) -> np.ndarray:
        """Lemire's bounded draw, with the same rejection step as the game.

        The retry is taken about once in 700 million draws for the bounds the
        aquifer uses, but it has to be there or a stream can desynchronise.
        """
        value = self.next() & np.uint64(MASK32)
        product = value * np.uint64(bound)
        product_lo = product & np.uint64(MASK32)
        threshold = np.uint64(((~bound & MASK32) + 1) % bound)
        retry = product_lo < threshold
        while np.any(retry):
            redraw = self.next() & np.uint64(MASK32)
            value = np.where(retry, redraw, value)
            product = value * np.uint64(bound)
            product_lo = product & np.uint64(MASK32)
            retry = retry & (product_lo < threshold)
        return (product >> np.uint64(32)).astype(np.int64)


class LegacyPositional:
    def __init__(self, seed: int):
        self.seed = to_signed64(seed)

    def at(self, x: int, y: int, z: int) -> JavaRandom:
        return JavaRandom(get_seed(x, y, z) ^ self.seed)

    def from_hash_of(self, name: str) -> JavaRandom:
        lo, _ = _md5_longs(name)
        return JavaRandom(to_signed64(lo) ^ self.seed)
