"""Optional numba kernels for the Perlin inner loop.

Profiling says about 85% of a render is ImprovedNoise.sample, and inside that it
is the eight gradient-dot gathers. numpy does those as fancy-index gathers over
the whole (levels x columns) grid; numba does them as scalar loads in a parallel
loop, which is roughly two orders of magnitude cheaper.

There are two, differing only in which axis they hand to the threads: a 3D noise
is split over its y levels, a 2D one over its columns, since it has only one row
and would otherwise run on a single core.

The kernels only handle the separable layout the renderer uses, x and z varying
along columns and y along rows. Anything else falls back to the numpy path in
worldsmith.noise, and tests/test_engine.py asserts they agree bit for bit.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap

    prange = range

GRAD_X = np.array([1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64)
GRAD_Y = np.array([1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
GRAD_Z = np.array([0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 0, -1], dtype=np.float64)


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def improved_noise_grid(p, xi, xf, zi, zf, px, px1, ys, yo, y_scale, y_limit,
                        gx, gy, gz, out):
    """out[m, n] = ImprovedNoise sample at (x[n], y[m], z[n])."""
    m_count = ys.shape[0]
    n_count = xi.shape[0]
    for mi in prange(m_count):
        yv = ys[mi] + yo
        yfl = np.floor(yv)
        yidx = np.int64(yfl)
        yf = yv - yfl
        yy = yf
        if y_scale != 0.0:
            yl = y_limit[mi]
            t = yl if (yl >= 0.0 and yl < yf) else yf
            yy = yf - np.floor(t / y_scale + 1.0e-7) * y_scale
        sy = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)
        for ni in range(n_count):
            h = px[ni]
            i2 = px1[ni]
            j2 = p[(h + yidx) & 0xFF]
            k2 = p[(h + yidx + 1) & 0xFF]
            l2 = p[(i2 + yidx) & 0xFF]
            m2 = p[(i2 + yidx + 1) & 0xFF]
            zbase = zi[ni]
            a0 = p[(j2 + zbase) & 0xFF] & 15
            a1 = p[(l2 + zbase) & 0xFF] & 15
            a2 = p[(k2 + zbase) & 0xFF] & 15
            a3 = p[(m2 + zbase) & 0xFF] & 15
            a4 = p[(j2 + zbase + 1) & 0xFF] & 15
            a5 = p[(l2 + zbase + 1) & 0xFF] & 15
            a6 = p[(k2 + zbase + 1) & 0xFF] & 15
            a7 = p[(m2 + zbase + 1) & 0xFF] & 15

            dx = xf[ni]
            dz = zf[ni]
            dx1 = dx - 1.0
            dz1 = dz - 1.0
            yy1 = yy - 1.0

            n0 = gx[a0] * dx + gy[a0] * yy + gz[a0] * dz
            n1 = gx[a1] * dx1 + gy[a1] * yy + gz[a1] * dz
            n2 = gx[a2] * dx + gy[a2] * yy1 + gz[a2] * dz
            n3 = gx[a3] * dx1 + gy[a3] * yy1 + gz[a3] * dz
            n4 = gx[a4] * dx + gy[a4] * yy + gz[a4] * dz1
            n5 = gx[a5] * dx1 + gy[a5] * yy + gz[a5] * dz1
            n6 = gx[a6] * dx + gy[a6] * yy1 + gz[a6] * dz1
            n7 = gx[a7] * dx1 + gy[a7] * yy1 + gz[a7] * dz1

            sx = dx * dx * dx * (dx * (dx * 6.0 - 15.0) + 10.0)
            sz = dz * dz * dz * (dz * (dz * 6.0 - 15.0) + 10.0)

            lo0 = n0 + sx * (n1 - n0)
            lo1 = n2 + sx * (n3 - n2)
            hi0 = n4 + sx * (n5 - n4)
            hi1 = n6 + sx * (n7 - n6)
            low = lo0 + sy * (lo1 - lo0)
            high = hi0 + sy * (hi1 - hi0)
            out[mi, ni] = low + sz * (high - low)


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def improved_noise_row(p, xs, zs, xo, zo, y, yo, y_scale, y_limit,
                       gx, gy, gz, out):
    """out[0, n] = ImprovedNoise sample at (x[n], y, z[n]), threaded over n."""
    n_count = xs.shape[0]
    yv = y + yo
    yfl = np.floor(yv)
    yidx = np.int64(yfl)
    yf = yv - yfl
    yy = yf
    if y_scale != 0.0:
        t = y_limit if (y_limit >= 0.0 and y_limit < yf) else yf
        yy = yf - np.floor(t / y_scale + 1.0e-7) * y_scale
    sy = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)
    yy1 = yy - 1.0
    for ni in prange(n_count):
        xv = xs[ni] + xo
        xfl = np.floor(xv)
        xidx = np.int64(xfl)
        zv = zs[ni] + zo
        zfl = np.floor(zv)
        h = p[xidx & 0xFF]
        i2 = p[(xidx + 1) & 0xFF]
        j2 = p[(h + yidx) & 0xFF]
        k2 = p[(h + yidx + 1) & 0xFF]
        l2 = p[(i2 + yidx) & 0xFF]
        m2 = p[(i2 + yidx + 1) & 0xFF]
        zbase = np.int64(zfl)
        a0 = p[(j2 + zbase) & 0xFF] & 15
        a1 = p[(l2 + zbase) & 0xFF] & 15
        a2 = p[(k2 + zbase) & 0xFF] & 15
        a3 = p[(m2 + zbase) & 0xFF] & 15
        a4 = p[(j2 + zbase + 1) & 0xFF] & 15
        a5 = p[(l2 + zbase + 1) & 0xFF] & 15
        a6 = p[(k2 + zbase + 1) & 0xFF] & 15
        a7 = p[(m2 + zbase + 1) & 0xFF] & 15

        dx = xv - xfl
        dz = zv - zfl
        dx1 = dx - 1.0
        dz1 = dz - 1.0

        n0 = gx[a0] * dx + gy[a0] * yy + gz[a0] * dz
        n1 = gx[a1] * dx1 + gy[a1] * yy + gz[a1] * dz
        n2 = gx[a2] * dx + gy[a2] * yy1 + gz[a2] * dz
        n3 = gx[a3] * dx1 + gy[a3] * yy1 + gz[a3] * dz
        n4 = gx[a4] * dx + gy[a4] * yy + gz[a4] * dz1
        n5 = gx[a5] * dx1 + gy[a5] * yy + gz[a5] * dz1
        n6 = gx[a6] * dx + gy[a6] * yy1 + gz[a6] * dz1
        n7 = gx[a7] * dx1 + gy[a7] * yy1 + gz[a7] * dz1

        sx = dx * dx * dx * (dx * (dx * 6.0 - 15.0) + 10.0)
        sz = dz * dz * dz * (dz * (dz * 6.0 - 15.0) + 10.0)

        lo0 = n0 + sx * (n1 - n0)
        lo1 = n2 + sx * (n3 - n2)
        hi0 = n4 + sx * (n5 - n4)
        hi1 = n6 + sx * (n7 - n6)
        low = lo0 + sy * (lo1 - lo0)
        high = hi0 + sy * (hi1 - hi0)
        out[0, ni] = low + sz * (high - low)

@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def improved_noise_points(p, xs, ys, zs, xo, yo, zo, y_scale, y_limit, gx, gy, gz, out):
    """out[0, n] where all three coordinates vary per column, as the shifts do."""
    for ni in prange(xs.shape[0]):
        xv = xs[ni] + xo
        xfl = np.floor(xv)
        xidx = np.int64(xfl)
        yv = ys[ni] + yo
        yfl = np.floor(yv)
        yidx = np.int64(yfl)
        zv = zs[ni] + zo
        zfl = np.floor(zv)
        zbase = np.int64(zfl)

        yf = yv - yfl
        yy = yf
        if y_scale != 0.0:
            t = y_limit if (y_limit >= 0.0 and y_limit < yf) else yf
            yy = yf - np.floor(t / y_scale + 1.0e-7) * y_scale
        sy = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)
        yy1 = yy - 1.0

        h = p[xidx & 0xFF]
        i2 = p[(xidx + 1) & 0xFF]
        j2 = p[(h + yidx) & 0xFF]
        k2 = p[(h + yidx + 1) & 0xFF]
        l2 = p[(i2 + yidx) & 0xFF]
        m2 = p[(i2 + yidx + 1) & 0xFF]
        a0 = p[(j2 + zbase) & 0xFF] & 15
        a1 = p[(l2 + zbase) & 0xFF] & 15
        a2 = p[(k2 + zbase) & 0xFF] & 15
        a3 = p[(m2 + zbase) & 0xFF] & 15
        a4 = p[(j2 + zbase + 1) & 0xFF] & 15
        a5 = p[(l2 + zbase + 1) & 0xFF] & 15
        a6 = p[(k2 + zbase + 1) & 0xFF] & 15
        a7 = p[(m2 + zbase + 1) & 0xFF] & 15

        dx = xv - xfl
        dz = zv - zfl
        dx1 = dx - 1.0
        dz1 = dz - 1.0

        n0 = gx[a0] * dx + gy[a0] * yy + gz[a0] * dz
        n1 = gx[a1] * dx1 + gy[a1] * yy + gz[a1] * dz
        n2 = gx[a2] * dx + gy[a2] * yy1 + gz[a2] * dz
        n3 = gx[a3] * dx1 + gy[a3] * yy1 + gz[a3] * dz
        n4 = gx[a4] * dx + gy[a4] * yy + gz[a4] * dz1
        n5 = gx[a5] * dx1 + gy[a5] * yy + gz[a5] * dz1
        n6 = gx[a6] * dx + gy[a6] * yy1 + gz[a6] * dz1
        n7 = gx[a7] * dx1 + gy[a7] * yy1 + gz[a7] * dz1

        sx = dx * dx * dx * (dx * (dx * 6.0 - 15.0) + 10.0)
        sz = dz * dz * dz * (dz * (dz * 6.0 - 15.0) + 10.0)

        lo0 = n0 + sx * (n1 - n0)
        lo1 = n2 + sx * (n3 - n2)
        hi0 = n4 + sx * (n5 - n4)
        hi1 = n6 + sx * (n7 - n6)
        low = lo0 + sy * (lo1 - lo0)
        high = hi0 + sy * (hi1 - hi0)
        out[0, ni] = low + sz * (high - low)


def pointwise(x, y, z, y_limit):
    """(xs, ys, zs, out_shape) when every coordinate is one value per column."""
    if np.ndim(y_limit) != 0:
        return None
    shapes = [np.shape(v) for v in (x, y, z)]
    rows = [s for s in shapes if s != ()]
    if not rows or any(len(s) != 2 or s[0] != 1 for s in rows) or len(set(rows)) != 1:
        return None
    n = rows[0][1]
    out = []
    for v in (x, y, z):
        a = np.asarray(v, dtype=np.float64)
        out.append(np.full(n, float(a)) if a.ndim == 0 else np.ascontiguousarray(a[0]))
    return out[0], out[1], out[2], (1, n)

def separable(x, y, z, y_limit):
    """Return (xs, ys, zs, yl, out_shape) if the call fits the grid kernel."""
    def as_columns(v):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            return a.reshape(1)
        if a.ndim == 1:
            return a
        if a.ndim == 2 and a.shape[0] == 1:
            return a[0]
        return None

    def as_rows(v):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            return a.reshape(1)
        if a.ndim == 2 and a.shape[1] == 1:
            return a[:, 0]
        return None

    xs = as_columns(x)
    zs = as_columns(z)
    ys = as_rows(y)
    if xs is None or zs is None or ys is None or xs.shape != zs.shape:
        return None
    if np.ndim(y) == 1 and np.size(y) > 1:
        return None
    yl = np.asarray(y_limit, dtype=np.float64)
    if yl.ndim == 0:
        yl = np.full(ys.shape, float(yl))
    elif yl.ndim == 2 and yl.shape[1] == 1:
        yl = yl[:, 0]
    elif yl.ndim != 1 or yl.shape != ys.shape:
        return None
    if yl.shape != ys.shape:
        return None
    out_shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
    return xs, ys, zs, yl, out_shape


def sample_grid(noise, x, y, z, y_scale, y_limit):
    """Fast path for ImprovedNoise.sample; None when the layout does not fit."""
    parts = separable(x, y, z, y_limit)
    if parts is None:
        # the shifts vary all three coordinates per column, which the grid
        # layout cannot express
        points = pointwise(x, y, z, y_limit)
        if points is None:
            return None
        xs, ys, zs, out_shape = points
        if xs.shape[0] < 512:
            return None
        out = np.empty((1, xs.shape[0]), dtype=np.float64)
        improved_noise_points(noise.p, xs, ys, zs, float(noise.xo), float(noise.yo),
                              float(noise.zo), float(y_scale), float(y_limit),
                              GRAD_X, GRAD_Y, GRAD_Z, out)
        return out.reshape(out_shape) if out.shape != out_shape else out
    xs, ys, zs, yl, out_shape = parts
    p = noise.p
    out = np.empty((ys.shape[0], xs.shape[0]), dtype=np.float64)
    # one y row is a 2D noise, and the grid kernel would have nothing to spread
    if ys.shape[0] == 1 and xs.shape[0] >= 512:
        improved_noise_row(p, np.ascontiguousarray(xs), np.ascontiguousarray(zs),
                           float(noise.xo), float(noise.zo), float(ys[0]), float(noise.yo),
                           float(y_scale), float(yl[0]), GRAD_X, GRAD_Y, GRAD_Z, out)
    else:
        xv = xs + noise.xo
        zv = zs + noise.zo
        xfl = np.floor(xv)
        zfl = np.floor(zv)
        xi = xfl.astype(np.int64)
        zi = zfl.astype(np.int64)
        improved_noise_grid(p, xi, xv - xfl, zi, zv - zfl, p[xi & 0xFF], p[(xi + 1) & 0xFF],
                            np.ascontiguousarray(ys), float(noise.yo), float(y_scale),
                            np.ascontiguousarray(yl), GRAD_X, GRAD_Y, GRAD_Z, out)
    return out.reshape(out_shape) if out.shape != out_shape else out
