"""The shapes a build is made of.

Nothing here knows what it is building. A wall is a wall whether it belongs to a
tower or a barn, and getting the geometry right once is the difference between a
build being an afternoon and being a week.
"""
from __future__ import annotations

import math

from .voxel import Grid


def speckle(x: int, y: int, z: int, mix: list[tuple[int, str]]) -> str:
    """Pick a block for this position from weighted choices, the same way every
    time. A wall of one block reads as extruded; a wall of four reads as built.

    `mix` is [(weight, block), ...].
    """
    total = sum(weight for weight, _ in mix)
    h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
    h &= 0xFFFFFFFF
    h ^= h >> 13
    pick = h % total
    for weight, block in mix:
        pick -= weight
        if pick < 0:
            return block
    return mix[-1][1]


def fill(grid: Grid, x0, y0, z0, x1, y1, z1, block) -> None:
    """Like Grid.fill, but `block` may be a function of (x, y, z), so a whole
    volume can be speckled in one call."""
    if isinstance(block, str):
        grid.fill(x0, y0, z0, x1, y1, z1, block)
        return
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            for z in range(min(z0, z1), max(z0, z1) + 1):
                grid.set(x, y, z, block(x, y, z))


def hollow_box(grid: Grid, x0, y0, z0, x1, y1, z1, wall, inner="minecraft:air",
               thickness: int = 1) -> None:
    """Four walls, floor and ceiling left to the caller, and a hollow middle."""
    fill(grid, x0, y0, z0, x1, y1, z1, wall)
    if inner is not None:
        fill(grid, x0 + thickness, y0, z0 + thickness,
             x1 - thickness, y1, z1 - thickness, inner)


def cylinder(grid: Grid, cx: float, cz: float, radius: float, y0: int, y1: int,
             wall, inner=None, thickness: float = 1.5) -> None:
    """A round tower. `inner` fills what the wall encloses, or leaves it."""
    for x in range(grid.sx):
        for z in range(grid.sz):
            distance = math.hypot(x + 0.5 - cx, z + 0.5 - cz)
            if distance > radius:
                continue
            if distance > radius - thickness:
                fill(grid, x, y0, z, x, y1, z, wall)
            elif inner is not None:
                fill(grid, x, y0, z, x, y1, z, inner)


def perimeter(x0: int, z0: int, x1: int, z1: int) -> list[tuple[int, int]]:
    """The cells around a rectangle, in order, so a pattern can run along it."""
    cells = [(x, z0) for x in range(x0, x1)]
    cells += [(x1, z) for z in range(z0, z1)]
    cells += [(x, z1) for x in range(x1, x0, -1)]
    cells += [(x0, z) for z in range(z1, z0, -1)]
    return cells


def ring_cells(cx: float, cz: float, radius: float,
               thickness: float = 1.2) -> list[tuple[int, int]]:
    """The cells of a circle's edge, in order around it. Cells outside whatever
    grid they are used on are dropped there, so this needs no size."""
    found = []
    for x in range(int(cx - radius) - 1, int(cx + radius) + 2):
        for z in range(int(cz - radius) - 1, int(cz + radius) + 2):
            distance = math.hypot(x + 0.5 - cx, z + 0.5 - cz)
            if radius - thickness <= distance <= radius:
                found.append((math.atan2(z + 0.5 - cz, x + 0.5 - cx), x, z))
    return [(x, z) for _, x, z in sorted(found)]


def crenellate(grid: Grid, cells, y: int, block, solid: int = 2, gap: int = 1) -> None:
    """Merlons and embrasures along an ordered run of cells: the top of a wall."""
    period = solid + gap
    for index, (x, z) in enumerate(cells):
        if index % period < solid:
            grid.set(x, y, z, block(x, y, z) if callable(block) else block)


def gable_roof(grid: Grid, x0: int, z0: int, x1: int, z1: int, y: int,
               material: str, overhang: int = 1) -> int:
    """A pitched roof of stairs closing to a ridge along z. Returns the ridge y."""
    span = (x1 - x0 + 2) // 2
    for z in range(z0 - overhang, z1 + overhang + 1):
        for x in range(x0 + 1, x1):                       # a ceiling under the pitch
            grid.set(x, y - 1, z, f"minecraft:{material}_planks")
    for step in range(span):
        level = y + step
        for z in range(z0 - overhang, z1 + overhang + 1):
            grid.set(x0 + step, level, z, f"minecraft:{material}_stairs[facing=east]")
            grid.set(x1 - step, level, z, f"minecraft:{material}_stairs[facing=west]")
            for x in range(x0 + step + 1, x1 - step):     # the roof space stays open
                grid.set(x, level, z, "minecraft:air")
    return y + span - 1


def stair_flight(grid: Grid, x: int, y: int, z: int, facing: str, steps: int,
                 material: str = "stone_brick", width: int = 1,
                 head_room: int = 3) -> tuple[int, int, int]:
    """A straight run of stairs climbing one block per step, with the space above
    it cleared. Returns where it arrives."""
    step_x = {"east": 1, "west": -1}.get(facing, 0)
    step_z = {"south": 1, "north": -1}.get(facing, 0)
    if not (step_x or step_z):
        raise ValueError(f"facing must be north, south, east or west (got {facing!r})")
    for step in range(steps):
        for across in range(width):
            ax = x + step * step_x + (across if step_x == 0 else 0)
            az = z + step * step_z + (across if step_z == 0 else 0)
            grid.set(ax, y + step, az, f"minecraft:{material}_stairs[facing={facing}]")
            grid.fill(ax, y + step + 1, az, ax, y + step + head_room, az, "minecraft:air")
    return x + steps * step_x, y + steps, z + steps * step_z
