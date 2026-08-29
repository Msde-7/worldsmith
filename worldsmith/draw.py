"""Drawing a build, so it can be looked at before a world is built from it.

The terrain half of worldsmith has always been able to show its work. This is
the same idea for the build half: an isometric view for the shape of a thing and
plan slices for what is inside it, both straight from the block grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .colors import block_color
from .voxel import Grid

BACKGROUND = (26, 28, 34)
LABEL = (210, 214, 224)
FADED = (150, 156, 170)
FACE_SHADE = (1.0, 0.72, 0.55)          # top, right, front


def _palette(grid: Grid) -> tuple[dict[int, tuple[int, int, int]], set[int]]:
    colors, air = {}, set()
    for index, spec in enumerate(grid.palette, start=1):
        name = spec.split("[")[0].split(":")[-1]
        colors[index] = block_color(name)
        if name == "air":
            air.add(index)
    return colors, air


def _oriented(grid: Grid, turn: int, flip: bool) -> np.ndarray:
    cells = np.rot90(grid.cells, k=turn, axes=(0, 2)) if turn else grid.cells
    return np.ascontiguousarray(cells[::-1, :, :] if flip else cells)


def render_iso(grid: Grid, path, scale: int = 4, turn: int = 0, flip: bool = False,
               label: str = "") -> Path:
    """An isometric drawing, painted back to front. `turn` is quarter turns."""
    colors, air = _palette(grid)
    cells = _oriented(grid, turn, flip)
    solid = cells != 0
    for index in air:
        solid &= cells != index

    # a block with all three faces the eye could see covered is not drawn
    hidden = np.zeros_like(solid)
    hidden[:-1, :, :] = solid[1:, :, :]
    hidden[:, :-1, :] &= solid[:, 1:, :]
    hidden[:, :, :-1] &= solid[:, :, 1:]
    hidden[-1, :, :] = hidden[:, -1, :] = hidden[:, :, -1] = False
    visible = solid & ~hidden

    sx, sy, sz = cells.shape
    w, h, v = scale, max(1, scale // 2), scale
    pad, header = 8, 14 if label else 0
    image = Image.new("RGB", ((sx + sz) * w + 2 * pad,
                              (sx + sz) * h + sy * v + 2 * pad + header), BACKGROUND)
    draw = ImageDraw.Draw(image)
    if label:
        draw.text((pad, 3), label, fill=LABEL)
    ox, oy = sz * w + pad, pad + header + sy * v

    xs, ys, zs = np.nonzero(visible)
    for i in np.argsort(xs + ys + zs):                     # far to near
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        base = colors[int(cells[x, y, z])]
        px, py = ox + (x - z) * w, oy + (x + z) * h - y * v
        top = [(px, py - v), (px + w, py + h - v), (px, py + 2 * h - v), (px - w, py + h - v)]
        right = [(px + w, py + h - v), (px + w, py + h), (px, py + 2 * h), (px, py + 2 * h - v)]
        front = [(px - w, py + h - v), (px - w, py + h), (px, py + 2 * h), (px, py + 2 * h - v)]
        for face, shade in zip((top, right, front), FACE_SHADE):
            draw.polygon(face, fill=tuple(min(255, int(c * shade)) for c in base))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def render_plan(grid: Grid, path, levels: list[int], scale: int = 4,
                label: str = "") -> Path:
    """Top down slices at the given heights: the floor plans."""
    colors, air = _palette(grid)
    pad, header, gap = 6, 16, 10
    tile_w, tile_h = grid.sx * scale, grid.sz * scale
    image = Image.new("RGB", (2 * pad + len(levels) * tile_w + (len(levels) - 1) * gap,
                              2 * pad + header + tile_h), BACKGROUND)
    draw = ImageDraw.Draw(image)
    if label:
        draw.text((pad, 3), label, fill=LABEL)
    for i, level in enumerate(levels):
        ox = pad + i * (tile_w + gap)
        draw.text((ox, header - 12), f"y={level}", fill=FADED)
        plane = grid.cells[:, level, :]
        for x in range(grid.sx):
            for z in range(grid.sz):
                index = int(plane[x, z])
                if index and index not in air:
                    draw.rectangle([ox + x * scale, header + pad + z * scale,
                                    ox + (x + 1) * scale - 1, header + pad + (z + 1) * scale - 1],
                                   fill=colors[index])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
