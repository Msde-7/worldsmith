"""Compiler and vectorised evaluator for Minecraft density functions.

A density function is a pure function of a block position. The JSON tree is
compiled once into node objects and then evaluated over whole coordinate arrays.
The evaluation context keeps x and z shaped (1, N) and y shaped (M, 1), so numpy
broadcasting does the work and any purely 2D sub-tree (everything under a
flat_cache: continentalness, erosion, depth, offset, factor, jaggedness) is
computed once per column rather than once per block. That is the single biggest
reason a render takes seconds instead of hours.

Supported node types are exactly those the game accepts in 26.2, plus
weird_scaled_sampler for older packs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .compat import COMPAT
from .jrandom import JavaRandom
from .noise import BlendedNoise, NormalNoise, SimplexNoise


def f32(a):
    return np.asarray(a, dtype=np.float32)

# Node types the game knows about, used by the validator too.
DENSITY_TYPES = {
    "abs", "add", "beardifier", "blend_alpha", "blend_density", "blend_offset",
    "cache_2d", "cache_all_in_cell", "cache_once", "clamp", "constant", "cube",
    "end_islands", "find_top_surface", "flat_cache", "half_negative",
    "interpolated", "interval_select", "invert", "max", "min", "mul", "noise",
    "old_blended_noise", "quarter_negative", "range_choice", "shift", "shift_a",
    "shift_b", "shifted_noise", "spline", "square", "squeeze",
    "weird_scaled_sampler", "y_clamped_gradient",
}

# 26.2 has no such registry entry and refuses a pack that names one.
LEGACY_TYPES = {"weird_scaled_sampler"}

# Fields each type accepts (validator). Values marked D are density functions.
DENSITY_FIELDS: dict[str, dict[str, str]] = {
    "abs": {"argument": "D"},
    "square": {"argument": "D"},
    "cube": {"argument": "D"},
    "half_negative": {"argument": "D"},
    "quarter_negative": {"argument": "D"},
    "squeeze": {"argument": "D"},
    "invert": {"argument": "D"},
    "add": {"argument1": "D", "argument2": "D"},
    "mul": {"argument1": "D", "argument2": "D"},
    "min": {"argument1": "D", "argument2": "D"},
    "max": {"argument1": "D", "argument2": "D"},
    "clamp": {"input": "D", "min": "number", "max": "number"},
    "constant": {"argument": "number"},
    "y_clamped_gradient": {"from_y": "int", "to_y": "int", "from_value": "number", "to_value": "number"},
    "range_choice": {"input": "D", "min_inclusive": "number", "max_exclusive": "number",
                     "when_in_range": "D", "when_out_of_range": "D"},
    "interval_select": {"input": "D", "thresholds": "number[]", "functions": "D[]"},
    "noise": {"noise": "noise", "xz_scale": "number", "y_scale": "number"},
    "shifted_noise": {"noise": "noise", "shift_x": "D", "shift_y": "D", "shift_z": "D",
                      "xz_scale": "number", "y_scale": "number"},
    "shift": {"argument": "noise"},
    "shift_a": {"argument": "noise"},
    "shift_b": {"argument": "noise"},
    "weird_scaled_sampler": {"input": "D", "noise": "noise", "rarity_value_mapper": "string"},
    "old_blended_noise": {"xz_scale": "number", "y_scale": "number", "xz_factor": "number",
                          "y_factor": "number", "smear_scale_multiplier": "number"},
    "spline": {"spline": "spline"},
    "flat_cache": {"argument": "D"},
    "cache_2d": {"argument": "D"},
    "cache_once": {"argument": "D"},
    "cache_all_in_cell": {"argument": "D"},
    "interpolated": {"argument": "D"},
    "blend_alpha": {}, "blend_offset": {}, "blend_density": {"argument": "D"},
    "beardifier": {}, "end_islands": {},
    "find_top_surface": {"density": "D", "upper_bound": "D", "lower_bound": "int", "cell_height": "int"},
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "abs": ("argument",), "square": ("argument",), "cube": ("argument",),
    "half_negative": ("argument",), "quarter_negative": ("argument",),
    "squeeze": ("argument",), "invert": ("argument",),
    "add": ("argument1", "argument2"), "mul": ("argument1", "argument2"),
    "min": ("argument1", "argument2"), "max": ("argument1", "argument2"),
    "clamp": ("input", "min", "max"),
    "constant": ("argument",),
    "y_clamped_gradient": ("from_y", "to_y", "from_value", "to_value"),
    "range_choice": ("input", "min_inclusive", "max_exclusive", "when_in_range", "when_out_of_range"),
    "interval_select": ("input", "thresholds", "functions"),
    "noise": ("noise", "xz_scale", "y_scale"),
    "shifted_noise": ("noise", "shift_x", "shift_y", "shift_z", "xz_scale", "y_scale"),
    "shift": ("argument",), "shift_a": ("argument",), "shift_b": ("argument",),
    "weird_scaled_sampler": ("input", "noise", "rarity_value_mapper"),
    "old_blended_noise": ("xz_scale", "y_scale", "xz_factor", "y_factor",
                          "smear_scale_multiplier"),
    "spline": ("spline",),
    "flat_cache": ("argument",), "cache_2d": ("argument",), "cache_once": ("argument",),
    "cache_all_in_cell": ("argument",), "interpolated": ("argument",),
    "blend_density": ("argument",),
    "find_top_surface": ("density", "upper_bound", "lower_bound", "cell_height"),
}


class Ctx:
    """Coordinates to evaluate at, plus a per-context memo table.

    x and z are (1, N) or scalars, y is (M, 1) or a scalar, so results broadcast
    to (M, N) only where a node actually depends on y.
    """

    __slots__ = ("x", "y", "z", "cache", "cache2d")

    def __init__(self, x, y, z, cache2d=None):
        self.x = x
        self.y = y
        self.z = z
        self.cache: dict[int, Any] = {}
        # Results of y-independent nodes stay valid when only y changes, so a
        # derived context inherits them. Without this, `find_top_surface`
        # recomputes every spline once per scanned level.
        self.cache2d: dict[int, Any] = {} if cache2d is None else cache2d

    def derive(self, x=None, y=None, z=None) -> "Ctx":
        same_column = x is None and z is None
        return Ctx(self.x if x is None else x,
                   self.y if y is None else y,
                   self.z if z is None else z,
                   cache2d=self.cache2d if same_column else None)


class Node:
    """Base class for a compiled node.

    eval memoises per context, but only for the nodes mark_shared() found under
    more than one parent (the two uses of overworld/sloped_cheese, say) and for
    the cheap 2D ones. Caching everything would hold one array per node, which
    runs into gigabytes on a large render.
    """

    memo = False
    two_d = False           # value does not depend on y

    def eval(self, ctx: Ctx):
        if self.two_d:
            key = id(self)
            hit = ctx.cache2d.get(key)
            if hit is None:
                hit = self._eval(ctx)
                ctx.cache2d[key] = hit
            return hit
        if not self.memo:
            return self._eval(ctx)
        key = id(self)
        hit = ctx.cache.get(key)
        if hit is None:
            hit = self._eval(ctx)
            ctx.cache[key] = hit
        return hit

    def _eval(self, ctx: Ctx):
        raise NotImplementedError

    def children(self):
        return ()


def analyze_2d(node: "Node", memo: dict | None = None) -> bool:
    """Mark nodes whose value is independent of y."""
    memo = {} if memo is None else memo
    key = id(node)
    if key in memo:
        return memo[key]
    memo[key] = False                      # break cycles conservatively
    if isinstance(node, (Const, EndIslands)):
        result = True
    elif isinstance(node, FlatCache):
        analyze_2d(node.arg, memo)
        result = True                      # forces y = 0
    elif isinstance(node, FindTopSurface):
        analyze_2d(node.density, memo)
        analyze_2d(node.upper_bound, memo)
        result = True                      # returns a column property
    elif isinstance(node, YClampedGradient):
        result = False
    elif isinstance(node, NoiseNode):
        result = node.y_scale == 0.0
    elif isinstance(node, ShiftedNoise):
        children = all(analyze_2d(c, memo) for c in node.children())
        result = node.y_scale == 0.0 and children
    elif isinstance(node, ShiftNoise):
        result = node.kind in ("shift_a", "shift_b")
    elif isinstance(node, (OldBlendedNoise, WeirdScaledSampler)):
        for c in node.children():
            analyze_2d(c, memo)
        result = False
    else:
        kids = [analyze_2d(c, memo) for c in node.children()]
        result = all(kids) if kids else False
    memo[key] = result
    try:
        node.two_d = result
    except AttributeError:
        pass
    return result


def prepare(*roots: "Node") -> None:
    """Run both static passes before evaluating a tree."""
    memo: dict = {}
    for root in roots:
        if root is not None:
            analyze_2d(root, memo)
    mark_shared(*roots)


def mark_shared(*roots: "Node") -> int:
    """Turn on memoisation for nodes reachable more than once from roots."""
    seen: dict[int, int] = {}
    order: list[Node] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node is None:
            continue
        key = id(node)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            order.append(node)
            stack.extend(c for c in node.children() if c is not None)
    shared = 0
    for node in order:
        if seen[id(node)] > 1 and not isinstance(node, Const):
            try:
                node.memo = True
                shared += 1
            except AttributeError:      # __slots__ without a memo slot
                pass
    return shared


class Const(Node):

    def __init__(self, value: float):
        self.value = float(value)

    def eval(self, ctx):
        return self.value

    def _eval(self, ctx):
        return self.value


class Unary(Node):

    def __init__(self, op: str, arg: Node):
        self.op = op
        self.arg = arg

    def children(self):
        return (self.arg,)

    def _eval(self, ctx):
        d = self.arg.eval(ctx)
        op = self.op
        if op == "abs":
            return np.abs(d)
        if op == "square":
            return d * d
        if op == "cube":
            return d * d * d
        if op == "half_negative":
            return np.where(d > 0.0, d, d * 0.5)
        if op == "quarter_negative":
            return np.where(d > 0.0, d, d * 0.25)
        if op == "squeeze":
            c = np.clip(d, -1.0, 1.0)
            return c / 2.0 - c * c * c / 24.0
        if op == "invert":
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / d
        raise ValueError(op)


class Binary(Node):

    def __init__(self, op: str, a: Node, b: Node):
        self.op = op
        self.a = a
        self.b = b

    def children(self):
        return (self.a, self.b)

    def _eval(self, ctx):
        a = self.a.eval(ctx)
        b = self.b.eval(ctx)
        if self.op == "add":
            return a + b
        if self.op == "mul":
            return a * b
        if self.op == "min":
            return np.minimum(a, b)
        return np.maximum(a, b)


class Clamp(Node):

    def __init__(self, inp: Node, lo: float, hi: float):
        self.inp = inp
        self.lo = lo
        self.hi = hi

    def children(self):
        return (self.inp,)

    def _eval(self, ctx):
        return np.clip(self.inp.eval(ctx), self.lo, self.hi)


class YClampedGradient(Node):

    def __init__(self, from_y, to_y, from_value, to_value):
        self.from_y = float(from_y)
        self.to_y = float(to_y)
        self.from_value = float(from_value)
        self.to_value = float(to_value)

    def _eval(self, ctx):
        t = (np.asarray(ctx.y, dtype=np.float64) - self.from_y) / (self.to_y - self.from_y)
        t = np.clip(t, 0.0, 1.0)
        return self.from_value + t * (self.to_value - self.from_value)


class RangeChoice(Node):

    def __init__(self, inp, lo, hi, in_range, out_range):
        self.inp = inp
        self.lo = float(lo)
        self.hi = float(hi)
        self.in_range = in_range
        self.out_range = out_range

    def children(self):
        return (self.inp, self.in_range, self.out_range)

    def _eval(self, ctx):
        v = self.inp.eval(ctx)
        mask = (v >= self.lo) & (v < self.hi)
        # When the choice is uniform, return that branch in its own shape:
        # forcing it to the mask's shape breaks when the branch depends on more
        # axes than the input does.
        if np.all(mask):
            return self.in_range.eval(ctx)
        if not np.any(mask):
            return self.out_range.eval(ctx)
        return np.where(mask, self.in_range.eval(ctx), self.out_range.eval(ctx))


class IntervalSelect(Node):

    def __init__(self, inp, thresholds, functions):
        n = min(len(thresholds), len(functions))
        self.inp = inp
        self.thresholds = [float(t) for t in thresholds[:n]]
        self.functions = list(functions[:n + 1]) if len(functions) > n else list(functions)

    def children(self):
        return (self.inp, *self.functions)

    def _eval(self, ctx):
        v = self.inp.eval(ctx)
        functions = self.functions
        if COMPAT["deepslate"]:
            # deepslate truncates `functions` to len(thresholds), dropping the
            # final bucket. The game keeps thresholds+1 functions.
            functions = functions[:len(self.thresholds)] or functions[:1]
        result = None
        remaining = np.ones(np.shape(v), dtype=bool) if np.ndim(v) else True
        for i, threshold in enumerate(self.thresholds):
            if i >= len(functions):
                break
            sel = (v < threshold)
            take = sel & remaining if np.ndim(v) else (sel and remaining)
            if np.any(take):
                value = functions[i].eval(ctx)
                result = value if result is None else np.where(take, value, result)
            remaining = remaining & ~sel if np.ndim(v) else (remaining and not sel)
            if not np.any(remaining):
                break
        if np.any(remaining):
            value = functions[-1].eval(ctx)
            result = value if result is None else np.where(remaining, value, result)
        return result


class NoiseNode(Node):

    def __init__(self, noise: NormalNoise | None, xz_scale: float, y_scale: float):
        self.noise = noise
        self.xz_scale = float(xz_scale)
        self.y_scale = float(y_scale)

    def _eval(self, ctx):
        if self.noise is None:
            return 0.0
        y = 0.0 if self.y_scale == 0.0 else ctx.y * self.y_scale
        return self.noise.sample(ctx.x * self.xz_scale, y, ctx.z * self.xz_scale)


class ShiftedNoise(Node):

    def __init__(self, noise, xz_scale, y_scale, sx, sy, sz):
        self.noise = noise
        self.xz_scale = float(xz_scale)
        self.y_scale = float(y_scale)
        self.sx = sx
        self.sy = sy
        self.sz = sz

    def children(self):
        return (self.sx, self.sy, self.sz)

    def _eval(self, ctx):
        if self.noise is None:
            return 0.0
        xx = ctx.x * self.xz_scale + self.sx.eval(ctx)
        yy = ctx.y * self.y_scale + self.sy.eval(ctx)
        zz = ctx.z * self.xz_scale + self.sz.eval(ctx)
        return self.noise.sample(xx, yy, zz)


class ShiftNoise(Node):
    """shift, shift_a and shift_b: the noises that offset sample positions."""

    def __init__(self, noise, kind: str):
        self.noise = noise
        self.kind = kind

    def _eval(self, ctx):
        if self.noise is None:
            return 0.0
        if self.kind == "shift_a":
            x, y, z = ctx.x, 0.0, ctx.z
        elif self.kind == "shift_b":
            x, y, z = ctx.z, ctx.x, 0.0
        else:
            x, y, z = ctx.x, ctx.y, ctx.z
        return self.noise.sample(x * 0.25, y * 0.25, z * 0.25) * 4.0


class WeirdScaledSampler(Node):

    def __init__(self, inp, noise, mapper: str):
        self.inp = inp
        self.noise = noise
        self.mapper = mapper

    def children(self):
        return (self.inp,)

    def _eval(self, ctx):
        if self.noise is None:
            return 0.0
        d = self.inp.eval(ctx)
        if self.mapper == "type_1":
            rarity = np.where(d < -0.5, 0.75, np.where(d < 0.0, 1.0, np.where(d < 0.5, 1.5, 2.0)))
        else:
            rarity = np.where(d < -0.75, 0.5, np.where(d < -0.5, 0.75,
                      np.where(d < 0.5, 1.0, np.where(d < 0.75, 2.0, 3.0))))
        return rarity * np.abs(self.noise.sample(ctx.x / rarity, ctx.y / rarity, ctx.z / rarity))


class OldBlendedNoise(Node):

    def __init__(self, blended: BlendedNoise | None):
        self.blended = blended

    def _eval(self, ctx):
        if self.blended is None:
            return 0.0
        return self.blended.sample(ctx.x, ctx.y, ctx.z)


class EndIslands(Node):

    def __init__(self, seed: int):
        random = JavaRandom(seed)
        random.consume(17292)
        self.simplex = SimplexNoise(random)

    def _height(self, x, z):
        x0 = np.floor(x / 2).astype(np.int64)
        z0 = np.floor(z / 2).astype(np.int64)
        # Java/JS `%` truncates toward zero; np.remainder floors. Using the
        # wrong one mirrors the End's islands across the axes.
        x1 = np.fmod(x, 2)
        z1 = np.fmod(z, 2)
        f = np.clip(100.0 - np.sqrt(x * x + z * z) * 8.0, -100.0, 80.0)
        for i in range(-12, 13):
            for j in range(-12, 13):
                x2 = x0 + i
                z2 = z0 + j
                skip = (x2 * x2 + z2 * z2 <= 4096) | (self.simplex.sample_2d(x2, z2) >= -0.9)
                f1 = np.fmod(np.abs(x2) * 3439 + np.abs(z2) * 147, 13) + 9
                x3 = x1 + i * 2
                z3 = z1 + j * 2
                f2 = 100.0 - np.sqrt(x3 * x3 + z3 * z3) * f1
                f = np.where(skip, f, np.maximum(f, np.clip(f2, -100.0, 80.0)))
        return f

    def _eval(self, ctx):
        x = np.floor(np.asarray(ctx.x, dtype=np.float64) / 8.0)
        z = np.floor(np.asarray(ctx.z, dtype=np.float64) / 8.0)
        return (self._height(x, z) - 8.0) / 128.0


class Marker(Node):
    """blend_density, cache_once and cache_all_in_cell.

    All three are transparent here: the blender only runs at the borders of an
    upgraded world, and caching is handled by Ctx.
    """

    def __init__(self, kind: str, arg: Node):
        self.kind = kind
        self.arg = arg

    def children(self):
        return (self.arg,)

    def _eval(self, ctx):
        return self.arg.eval(ctx)


class FlatCache(Node):
    """Evaluated at the quart-aligned column corner with y = 0, like
    NoiseChunk.FlatCache. This is what makes climate parameters change in
    4-block steps."""

    def __init__(self, arg: Node):
        self.arg = arg
        self.memo = True

    def children(self):
        return (self.arg,)

    def _eval(self, ctx):
        x = np.floor(np.asarray(ctx.x, dtype=np.float64) / 4.0) * 4.0
        z = np.floor(np.asarray(ctx.z, dtype=np.float64) / 4.0) * 4.0
        return self.arg.eval(ctx.derive(x=x, y=0.0, z=z))


class Cache2D(Node):

    def __init__(self, arg: Node):
        self.arg = arg

    def children(self):
        return (self.arg,)

    def _eval(self, ctx):
        return self.arg.eval(ctx)


class Interpolated(Node):
    """Trilinear interpolation across the noise cell, as the game does.

    When every sample already sits on a cell corner, which is how the renderer
    samples by default, this collapses to a single evaluation.
    """

    def __init__(self, arg: Node, cell_width: int = 4, cell_height: int = 4):
        self.arg = arg
        self.cell_width = cell_width
        self.cell_height = cell_height

    def children(self):
        return (self.arg,)

    def _eval(self, ctx):
        w = float(self.cell_width)
        h = float(self.cell_height)
        x = np.asarray(ctx.x, dtype=np.float64)
        y = np.asarray(ctx.y, dtype=np.float64)
        z = np.asarray(ctx.z, dtype=np.float64)
        fx = np.floor(x / w) * w
        fy = np.floor(y / h) * h
        fz = np.floor(z / w) * w
        tx = (x - fx) / w
        ty = (y - fy) / h
        tz = (z - fz) / w

        on_grid_x = not np.any(tx)
        on_grid_y = not np.any(ty)
        on_grid_z = not np.any(tz)
        if on_grid_x and on_grid_y and on_grid_z:
            return self.arg.eval(ctx.derive(x=fx, y=fy, z=fz))

        def corner(dx, dy, dz):
            return self.arg.eval(ctx.derive(x=fx + dx * w, y=fy + dy * h, z=fz + dz * w))

        def lerp(t, a, b):
            return a if t is None else a + t * (b - a)

        c000, c100 = corner(0, 0, 0), (corner(1, 0, 0) if not on_grid_x else None)
        v00 = c000 if on_grid_x else lerp(tx, c000, c100)
        if on_grid_y:
            v0 = v00
        else:
            c010, c110 = corner(0, 1, 0), (corner(1, 1, 0) if not on_grid_x else None)
            v10 = c010 if on_grid_x else lerp(tx, c010, c110)
            v0 = lerp(ty, v00, v10)
        if on_grid_z:
            return v0
        c001, c101 = corner(0, 0, 1), (corner(1, 0, 1) if not on_grid_x else None)
        v01 = c001 if on_grid_x else lerp(tx, c001, c101)
        if on_grid_y:
            v1 = v01
        else:
            c011, c111 = corner(0, 1, 1), (corner(1, 1, 1) if not on_grid_x else None)
            v11 = c011 if on_grid_x else lerp(tx, c011, c111)
            v1 = lerp(ty, v01, v11)
        return lerp(tz, v0, v1)


class FindTopSurface(Node):
    """Scans down in cell_height steps for the highest y whose density is
    positive, vectorised over columns."""

    def __init__(self, density: Node, upper_bound: Node, lower_bound: int, cell_height: int):
        self.density = density
        self.upper_bound = upper_bound
        self.lower_bound = int(lower_bound)
        self.cell_height = max(1, int(cell_height))

    def children(self):
        return (self.density, self.upper_bound)

    def _eval(self, ctx):
        ch = self.cell_height
        ub = np.asarray(self.upper_bound.eval(ctx), dtype=np.float64)
        top = np.floor(ub / ch) * ch
        shape = np.broadcast_shapes(np.shape(ctx.x), np.shape(top))
        top = np.broadcast_to(top, shape)
        result = np.full(shape, float(self.lower_bound))
        found = top < self.lower_bound
        if np.all(found):
            return result
        y = float(np.max(top))
        while y >= self.lower_bound and not np.all(found):
            col = ctx.derive(y=y)
            d = np.asarray(self.density.eval(col), dtype=np.float64)
            hit = (~found) & (y <= top) & (np.broadcast_to(d, shape) > 0.0)
            result = np.where(hit, y, result)
            found = found | hit
            y -= ch
        return result


class SplineNode(Node):

    def __init__(self, spline: "CubicSpline"):
        self.spline = spline

    def children(self):
        return tuple(self.spline.all_nodes())

    def _eval(self, ctx):
        return self.spline.eval(ctx).astype(np.float64)


class CubicSpline:
    """Float32 cubic spline, matching net.minecraft.util.CubicSpline."""

    __slots__ = ("coordinate", "locations", "values", "derivatives", "constant")

    def __init__(self, constant=None, coordinate=None, locations=None, values=None, derivatives=None):
        self.constant = constant
        self.coordinate = coordinate
        self.locations = np.array(locations or [], dtype=np.float32)
        self.values = values or []
        self.derivatives = np.array(derivatives or [], dtype=np.float32)

    def all_nodes(self):
        if self.constant is not None:
            return []
        out = [self.coordinate]
        for v in self.values:
            if isinstance(v, CubicSpline):
                out.extend(v.all_nodes())
        return out

    def eval(self, ctx) -> np.ndarray:
        # Java's codec reads point values as floats; deepslate's fromJson keeps
        # them as JS doubles.
        vdtype = np.float64 if COMPAT["deepslate"] else np.float32
        if self.constant is not None:
            return np.asarray(self.constant, dtype=vdtype)
        raw = np.asarray(self.coordinate.eval(ctx), dtype=np.float64)
        # Java casts the coordinate to float before using it (the spline
        # coordinate is a ToFloatFunction); deepslate keeps it a double.
        coord = raw if COMPAT["deepslate"] else f32(raw)
        loc = self.locations
        n = len(loc) - 1
        idx = np.searchsorted(loc, coord, side="right") - 1

        vals = []
        for v in self.values:
            value = v if not isinstance(v, CubicSpline) else v.eval(ctx)
            vals.append(np.broadcast_to(np.asarray(value, dtype=vdtype), np.shape(coord)))
        vals = np.stack(vals, axis=0) if len(vals) else np.zeros((1,) + np.shape(coord), dtype=vdtype)

        def take(arr, ii):
            return np.take_along_axis(arr, ii[None, ...], axis=0)[0]

        i0 = np.clip(idx, 0, n)
        i1 = np.clip(idx + 1, 0, n)
        v0 = take(vals, i0)
        v1 = take(vals, i1)
        l0 = loc[i0]
        l1 = loc[i1]
        d0 = self.derivatives[i0]
        d1 = self.derivatives[i1]

        # cubic hermite, every step rounded to float32 exactly like Java
        with np.errstate(divide="ignore", invalid="ignore"):
            span = f32(l1 - l0)
            denom = np.where(span == 0, np.float32(1), span)
            t = f32(f32(coord - l0) / denom)
        dv = f32(v1 - v0)
        f8 = f32(f32(d0 * span) - dv)
        f9 = f32(f32(-d1 * span) + dv)
        lin = f32(v0 + f32(t * dv))
        curve = f32(f8 + f32(t * f32(f9 - f8)))
        interior = f32(lin + f32(f32(t * f32(1.0 - t)) * curve))

        below = f32(vals[0] + f32(self.derivatives[0] * f32(coord - loc[0])))
        above = f32(vals[n] + f32(self.derivatives[n] * f32(coord - loc[n])))
        out = np.where(idx < 0, below, np.where(idx >= n, above, interior))
        return out.astype(np.float32)


def parse_spline(obj, compile_fn) -> CubicSpline:
    if isinstance(obj, (int, float)):
        return CubicSpline(constant=float(obj))
    points = obj.get("points") or []
    if not points:
        return CubicSpline(constant=0.0)
    coordinate = compile_fn(obj.get("coordinate"))
    locations, values, derivatives = [], [], []
    for p in points:
        locations.append(float(p.get("location", 0.0)))
        value = p.get("value", 0.0)
        values.append(float(value) if isinstance(value, (int, float)) else parse_spline(value, compile_fn))
        derivatives.append(float(p.get("derivative", 0.0)))
    return CubicSpline(coordinate=coordinate, locations=locations, values=values, derivatives=derivatives)


class CompileError(Exception):
    pass


class DensityCompiler:
    """Turns density function JSON into Node trees.

    env must provide get_density(id) returning raw JSON, get_noise(id) returning
    a NormalNoise, plus cell_width, cell_height, max_y and seed.
    """

    def __init__(self, env):
        self.env = env
        self._by_id: dict[str, Node] = {}
        self._compiling: list[str] = []

    def compile(self, obj) -> Node:
        if isinstance(obj, (int, float)):
            return Const(float(obj))
        if isinstance(obj, str):
            return self.compile_ref(obj)
        if not isinstance(obj, dict):
            raise CompileError(f"density function must be a number, id or object, got {type(obj).__name__}")
        raw_type = obj.get("type")
        if not isinstance(raw_type, str):
            raise CompileError("density function object is missing 'type'")
        t = raw_type.split(":")[-1] if raw_type.startswith("minecraft:") else raw_type
        c = self.compile

        def num(key, default=None):
            return self._number(obj, key, default)

        if t in ("abs", "square", "cube", "half_negative", "quarter_negative", "squeeze", "invert"):
            return Unary(t, c(obj["argument"]))
        if t in ("add", "mul", "min", "max"):
            return Binary(t, c(obj["argument1"]), c(obj["argument2"]))
        if t == "constant":
            return Const(num("argument", 0.0))
        if t == "clamp":
            return Clamp(c(obj["input"]), num("min", 0.0), num("max", 1.0))
        if t == "y_clamped_gradient":
            return YClampedGradient(num("from_y", -4064), num("to_y", 4062),
                                    num("from_value", -4064.0), num("to_value", 4062.0))
        if t == "range_choice":
            return RangeChoice(c(obj["input"]), num("min_inclusive", 0.0), num("max_exclusive", 1.0),
                               c(obj["when_in_range"]), c(obj["when_out_of_range"]))
        if t == "interval_select":
            return IntervalSelect(c(obj["input"]), obj.get("thresholds") or [],
                                  [c(f) for f in (obj.get("functions") or [])])
        if t == "noise":
            return NoiseNode(self.env.get_noise(obj["noise"]), num("xz_scale", 1.0), num("y_scale", 1.0))
        if t == "shifted_noise":
            return ShiftedNoise(self.env.get_noise(obj["noise"]), num("xz_scale", 1.0), num("y_scale", 1.0),
                                c(obj["shift_x"]), c(obj["shift_y"]), c(obj["shift_z"]))
        if t in ("shift", "shift_a", "shift_b"):
            return ShiftNoise(self.env.get_noise(obj["argument"]), t)
        if t == "weird_scaled_sampler":
            return WeirdScaledSampler(c(obj["input"]), self.env.get_noise(obj["noise"]),
                                      obj.get("rarity_value_mapper", "type_1"))
        if t == "old_blended_noise":
            return OldBlendedNoise(self.env.get_blended_noise(
                num("xz_scale", 1.0), num("y_scale", 1.0), num("xz_factor", 80.0),
                num("y_factor", 160.0), num("smear_scale_multiplier", 8.0)))
        if t == "spline":
            return SplineNode(parse_spline(obj["spline"], self.compile))
        if t == "flat_cache":
            return FlatCache(c(obj["argument"]))
        if t == "cache_2d":
            return Cache2D(c(obj["argument"]))
        if t in ("cache_once", "cache_all_in_cell", "blend_density"):
            return Marker(t, c(obj["argument"]))
        if t == "interpolated":
            return Interpolated(c(obj["argument"]), self.env.cell_width, self.env.cell_height)
        if t == "blend_alpha":
            return Const(1.0)
        if t in ("blend_offset", "beardifier"):
            return Const(0.0)
        if t == "end_islands":
            return EndIslands(self.env.seed)
        if t == "find_top_surface":
            ub = obj.get("upper_bound")
            upper = c(ub) if ub is not None else Const(float(self.env.max_y))
            return FindTopSurface(c(obj["density"]), upper,
                                  int(obj.get("lower_bound", 0)), int(obj.get("cell_height", 1)))
        raise CompileError(f"unknown density function type: {raw_type}")

    @staticmethod
    def _number(obj, key, default):
        v = obj.get(key, default)
        if v is None:
            raise CompileError(f"missing numeric field '{key}'")
        if not isinstance(v, (int, float)):
            raise CompileError(f"field '{key}' must be a number, got {v!r}")
        return float(v)

    def compile_ref(self, ref: str) -> Node:
        ref = ref if ":" in ref else "minecraft:" + ref
        if ref in self._by_id:
            return self._by_id[ref]
        if ref in self._compiling:
            cycle = " -> ".join(self._compiling + [ref])
            raise CompileError(f"density function reference cycle: {cycle}")
        raw = self.env.get_density(ref)
        if raw is None:
            raise CompileError(f"unknown density function reference: {ref}")
        self._compiling.append(ref)
        try:
            node = self.compile(raw)
        finally:
            self._compiling.pop()
        self._by_id[ref] = node
        return node
