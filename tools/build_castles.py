"""Actual castles: curtain walls, corner towers, a gatehouse with a portcullis,
a moat with a bridge, and a keep you can walk up through.

An example of the build half of worldsmith: geometry in a voxel grid, handed to
worldsmith.structures, which writes the template and the worldgen JSON that
makes the game place it the same way it places a village.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from worldsmith import structures                             # noqa: E402
from worldsmith.draw import render_iso, render_plan           # noqa: E402
from worldsmith.pack import PackWriter                        # noqa: E402
from worldsmith.shapes import cell_hash, crenellate, fill, line, speckle  # noqa: E402
from worldsmith.voxel import Grid                             # noqa: E402

NS = "castle"

# --- the plan, in numbers ----------------------------------------------------
SIZE = 64                # footprint, matching the largest templates vanilla ships
HEIGHT = 52
GROUND = 10              # top of the courtyard floor; the ground the game sees
MOAT_IN, MOAT_OUT = 2, 7         # ring distances from the edge
WALL_OUT, WALL_IN = 10, 13       # the curtain wall band
WALL_TOP = 21            # last solid course
WALK = 22                # wall walk surface
BAILEY = WALL_IN + 1     # first courtyard block

KEEP_LO, KEEP_HI = 23, 40
KEEP_FLOORS = [GROUND, 17, 24, 31]
KEEP_ROOF = 38

TOWER_R = 6.0
TOWER_TOP = 30
GATE_X0, GATE_X1 = 30, 33        # the passage through the south wall


def dist_to_edge(x: int, z: int, lo: int, hi: int) -> int:
    """How far in from the edge of a square this column sits."""
    return min(x - lo, z - lo, hi - x, hi - z)


def dist(x: int, z: int) -> int:
    return dist_to_edge(x, z, 0, SIZE - 1)


def hashed(x: int, y: int, z: int, salt: int = 0) -> int:
    return cell_hash(x, y, z, salt) % 1000


def paving(x: int, z: int) -> str:
    """An old courtyard, and no soil for a tree to take root in. Features are
    placed after structures, so a grass bailey grows an oak wood."""
    return speckle(x, 0, z, [(520, "minecraft:cobblestone"),
                             (180, "minecraft:mossy_cobblestone"),
                             (160, "minecraft:andesite"),
                             (140, "minecraft:gravel")], salt=13)


def masonry(x: int, y: int, z: int, weathered: int = 0) -> str:
    """Stone brick with enough cobble and moss speckled through it that a wall
    reads as built rather than extruded."""
    return speckle(x, y, z, [(60 + weathered * 4, "minecraft:mossy_stone_bricks"),
                             (50 + weathered * 2, "minecraft:cracked_stone_bricks"),
                             (40 - weathered * 6, "minecraft:cobblestone"),
                             (15, "minecraft:chiseled_stone_bricks"),
                             (835, "minecraft:stone_bricks")], salt=7)


class Castle:
    def __init__(self, weathered: int = 0):
        self.g = Grid(SIZE, HEIGHT, SIZE)
        self.weathered = weathered

    # --- helpers -------------------------------------------------------------

    def masonry(self, x, y, z) -> str:
        return masonry(x, y, z, self.weathered)

    def stone(self, x0, y0, z0, x1, y1, z1) -> None:
        fill(self.g, x0, y0, z0, x1, y1, z1, self.masonry)

    def crenellate_line(self, x0, z0, x1, z1, y) -> None:
        """Merlons two wide, embrasures one wide, along a straight parapet."""
        crenellate(self.g, line(x0, z0, x1, z1), y, self.masonry)

    def crenellate_ring(self, cx, cz, radius, y, segments=14) -> None:
        for x in range(self.g.sx):
            for z in range(self.g.sz):
                d = math.hypot(x + 0.5 - cx, z + 0.5 - cz)
                if radius - 1.2 <= d <= radius:
                    angle = math.atan2(z + 0.5 - cz, x + 0.5 - cx) + math.pi
                    if int(angle / (2 * math.pi) * segments) % 2 == 0:
                        self.g.set(x, y, z, masonry(x, y, z, self.weathered))

    def stair(self, x, y, z, facing, half="bottom", material="stone_brick") -> None:
        self.g.set(x, y, z, f"minecraft:{material}_stairs[facing={facing},half={half}]")

    # --- the build -----------------------------------------------------------

    def ground_and_moat(self) -> None:
        g = self.g
        # a solid pedestal under the whole footprint: on a slope this reads as
        # the earthwork a castle is built on, and nothing is ever left floating
        g.fill(0, 0, 0, SIZE - 1, GROUND - 5, SIZE - 1, "minecraft:stone")
        g.fill(0, GROUND - 4, 0, SIZE - 1, GROUND - 1, SIZE - 1, "minecraft:dirt")
        g.fill(0, GROUND, 0, SIZE - 1, GROUND, SIZE - 1, "minecraft:grass_block[snowy=false]")

        for x in range(SIZE):
            for z in range(SIZE):
                d = dist(x, z)
                if d >= BAILEY:                       # the bailey is paved
                    g.set(x, GROUND, z, paving(x, z))
                elif MOAT_OUT + 1 <= d <= WALL_OUT - 1:   # gravel berm
                    g.set(x, GROUND, z, "minecraft:gravel")
                elif d <= MOAT_IN - 1:
                    # the outer bank is cleared ground too: a castle does not
                    # leave a wood growing against its own moat
                    g.set(x, GROUND, z, "minecraft:gravel" if hashed(x, 0, z, 5) < 700
                          else "minecraft:cobblestone")
                if MOAT_IN <= d <= MOAT_OUT:
                    g.fill(x, GROUND - 6, z, x, GROUND - 6, z, "minecraft:gravel")
                    g.fill(x, GROUND - 5, z, x, GROUND - 2, z, "minecraft:water[level=0]")
                    g.fill(x, GROUND - 1, z, x, GROUND + 14, z, "minecraft:air")
                elif d in (MOAT_IN - 1, MOAT_OUT + 1):
                    # the lip of the moat, cut back so the bank is not a cliff
                    self.stone(x, GROUND - 1, z, x, GROUND - 1, z)

    def curtain_wall(self) -> None:
        g = self.g
        for x in range(SIZE):
            for z in range(SIZE):
                d = dist(x, z)
                if WALL_OUT <= d <= WALL_IN:
                    self.stone(x, 0, z, x, WALL_TOP, z)
                    # head room over the walk, or the hillside stands in it
                    g.fill(x, WALK, z, x, WALK + 4, z, "minecraft:air")
                elif d >= BAILEY:
                    g.fill(x, GROUND + 1, z, x, WALK + 4, z, "minecraft:air")

        lo, hi = WALL_OUT, SIZE - 1 - WALL_OUT
        # the outer parapet and a kerb on the courtyard side
        for x in range(SIZE):
            for z in range(SIZE):
                d = dist(x, z)
                if d in (WALL_OUT, WALL_IN):
                    self.stone(x, WALK, z, x, WALK, z)
        self.crenellate_line(lo, lo, hi, lo, WALK + 1)
        self.crenellate_line(lo, hi, hi, hi, WALK + 1)
        self.crenellate_line(lo, lo, lo, hi, WALK + 1)
        self.crenellate_line(hi, lo, hi, hi, WALK + 1)
        # arrow loops through the outer face
        for t in range(lo + 4, hi - 3, 6):
            for y in (GROUND + 6, GROUND + 10):
                self.g.set(t, y, lo, "minecraft:air")
                self.g.set(t, y, hi, "minecraft:air")
                self.g.set(lo, y, t, "minecraft:air")
                self.g.set(hi, y, t, "minecraft:air")

    def corner_towers(self) -> None:
        g = self.g
        centre = WALL_OUT + 1.5
        for cx, cz in ((centre, centre), (SIZE - centre, centre),
                       (centre, SIZE - centre), (SIZE - centre, SIZE - centre)):
            for x in range(SIZE):
                for z in range(SIZE):
                    d = math.hypot(x + 0.5 - cx, z + 0.5 - cz)
                    if d > TOWER_R:
                        continue
                    if d > TOWER_R - 1.6:                       # the tower wall
                        self.stone(x, 0, z, x, TOWER_TOP, z)
                    else:                                        # its rooms
                        self.stone(x, 0, z, x, GROUND, z)
                        g.fill(x, GROUND + 1, z, x, TOWER_TOP, z, "minecraft:air")
            for floor in (17, 24):
                for x in range(SIZE):
                    for z in range(SIZE):
                        if math.hypot(x + 0.5 - cx, z + 0.5 - cz) <= TOWER_R - 1.6:
                            g.set(x, floor, z, "minecraft:oak_planks")
            # corbelled battlements: a course that oversails the wall below
            for x in range(SIZE):
                for z in range(SIZE):
                    d = math.hypot(x + 0.5 - cx, z + 0.5 - cz)
                    if TOWER_R - 1.9 <= d <= TOWER_R + 0.6:
                        self.stone(x, TOWER_TOP + 1, z, x, TOWER_TOP + 2, z)
                    elif d < TOWER_R - 1.9:
                        self.stone(x, TOWER_TOP + 1, z, x, TOWER_TOP + 1, z)
                        g.fill(x, TOWER_TOP + 2, z, x, TOWER_TOP + 4, z, "minecraft:air")
            self.crenellate_ring(cx, cz, TOWER_R + 0.6, TOWER_TOP + 3)
            # ladders and arrow loops
            ix, iz = int(cx), int(cz)
            for y0, y1 in ((GROUND + 1, 16), (18, 23), (25, TOWER_TOP)):
                for y in range(y0, y1 + 1):
                    g.set(ix + 2, y, iz, "minecraft:ladder[facing=east]")
            for floor in (17, 24, TOWER_TOP + 1):
                g.set(ix + 2, floor, iz, "minecraft:air")
                g.set(ix + 2, floor, iz + 1, "minecraft:air" if floor > TOWER_TOP else
                      "minecraft:oak_planks")
            for y in (GROUND + 5, GROUND + 12, GROUND + 19):
                for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    for r in range(3, 8):
                        x, z = ix + dx * r, iz + dz * r
                        if math.hypot(x + 0.5 - cx, z + 0.5 - cz) > TOWER_R - 1.7:
                            g.set(x, y, z, "minecraft:air")
                            g.set(x, y + 1, z, "minecraft:air")
            g.set(ix, TOWER_TOP + 2, iz, "minecraft:red_banner[rotation=8]")
            g.set(ix - 1, GROUND + 1, iz - 1, "minecraft:lantern[hanging=false]")

    def gatehouse(self) -> None:
        g = self.g
        z_out, z_in = SIZE - 1 - WALL_OUT, SIZE - 1 - WALL_IN     # 53 and 50
        # two flanking towers, then the passage carved straight through them
        for cx in (28.0, 35.5):
            for x in range(SIZE):
                for z in range(SIZE):
                    d = math.hypot(x + 0.5 - cx, z + 0.5 - (z_out - 1.5))
                    if d <= 4.6:
                        self.stone(x, 0, z, x, 27, z)
            self.crenellate_ring(cx, z_out - 1.5, 4.6, 28)
        # the passage: floor, walls, barrel vault
        for x in range(GATE_X0, GATE_X1 + 1):
            for z in range(z_in - 1, SIZE - 1):
                self.stone(x, GROUND, z, x, GROUND, z)
                g.fill(x, GROUND + 1, z, x, GROUND + 5, z, "minecraft:air")
        for z in range(z_in - 1, z_out + 3):
            self.stone(GATE_X0 - 1, GROUND, z, GATE_X0 - 1, GROUND + 5, z)
            self.stone(GATE_X1 + 1, GROUND, z, GATE_X1 + 1, GROUND + 5, z)
            self.stone(GATE_X0, GROUND + 6, z, GATE_X1, GROUND + 6, z)
            self.stair(GATE_X0, GROUND + 5, z, "east")
            self.stair(GATE_X1, GROUND + 5, z, "west")
        # portcullis, half dropped, and the murder holes above it
        for x in range(GATE_X0, GATE_X1 + 1):
            for y in range(GROUND + 3, GROUND + 6):
                g.set(x, y, z_out - 1, "minecraft:iron_bars")
            g.set(x, GROUND + 6, z_in + 1, "minecraft:air")
        # arch mouth, outside and in
        for z, facing in ((SIZE - 2, "south"), (z_in - 1, "north")):
            self.stair(GATE_X0, GROUND + 5, z, facing)
            self.stair(GATE_X1, GROUND + 5, z, facing)
        # the bridge over the moat
        for x in range(GATE_X0, GATE_X1 + 1):
            for z in range(SIZE - 1 - MOAT_OUT - 2, SIZE):
                g.set(x, GROUND, z, "minecraft:oak_planks")
                g.fill(x, GROUND + 1, z, x, GROUND + 3, z, "minecraft:air")
        for z in range(SIZE - 1 - MOAT_OUT - 2, SIZE - 1):
            if z % 3 != 0:
                g.set(GATE_X0 - 1, GROUND + 1, z, "minecraft:oak_fence")
                g.set(GATE_X1 + 1, GROUND + 1, z, "minecraft:oak_fence")
            g.set(GATE_X0 - 1, GROUND, z, "minecraft:oak_planks")
            g.set(GATE_X1 + 1, GROUND, z, "minecraft:oak_planks")
        for z in (SIZE - 5, SIZE - 9):                   # piers standing in the water
            for x in (GATE_X0 - 1, GATE_X1 + 1):
                self.stone(x, GROUND - 6, z, x, GROUND - 1, z)
        # lanterns and colours either side of the gate
        for x in (GATE_X0 - 1, GATE_X1 + 1):
            g.set(x, GROUND + 4, SIZE - 2, "minecraft:lantern[hanging=false]")
        for x in (GATE_X0 - 2, GATE_X1 + 2):
            g.set(x, GROUND + 8, z_out + 1, "minecraft:red_wall_banner[facing=south]")
            g.set(x, GROUND + 12, z_out + 1, "minecraft:white_wall_banner[facing=south]")

    def keep(self) -> None:
        g = self.g
        lo, hi = KEEP_LO, KEEP_HI
        self.stone(lo, 0, lo, hi, KEEP_ROOF, hi)
        for level, floor in enumerate(KEEP_FLOORS):
            top = KEEP_FLOORS[level + 1] if level + 1 < len(KEEP_FLOORS) else KEEP_ROOF
            g.fill(lo + 2, floor + 1, lo + 2, hi - 2, top - 1, hi - 2, "minecraft:air")
            material = "minecraft:stone_bricks" if level == 0 else "minecraft:oak_planks"
            g.fill(lo + 2, floor, lo + 2, hi - 2, floor, hi - 2, material)
        # roof, parapet and four corner turrets
        g.fill(lo + 2, KEEP_ROOF, lo + 2, hi - 2, KEEP_ROOF, hi - 2, "minecraft:stone_bricks")
        g.fill(lo + 1, KEEP_ROOF + 1, lo + 1, hi - 1, KEEP_ROOF + 4, hi - 1, "minecraft:air")
        self.crenellate_line(lo, lo, hi, lo, KEEP_ROOF + 2)
        self.crenellate_line(lo, hi, hi, hi, KEEP_ROOF + 2)
        self.crenellate_line(lo, lo, lo, hi, KEEP_ROOF + 2)
        self.crenellate_line(hi, lo, hi, hi, KEEP_ROOF + 2)
        for x in range(lo, hi + 1):
            for z in range(lo, hi + 1):
                if dist_to_edge(x, z, lo, hi) == 0:
                    self.stone(x, KEEP_ROOF + 1, z, x, KEEP_ROOF + 1, z)
        for cx, cz in ((lo, lo), (hi - 3, lo), (lo, hi - 3), (hi - 3, hi - 3)):
            self.stone(cx, KEEP_ROOF, cz, cx + 3, KEEP_ROOF + 5, cz + 3)
            g.set(cx + 1, KEEP_ROOF + 7, cz + 1, "minecraft:red_banner[rotation=0]")
            g.fill(cx + 1, KEEP_ROOF + 1, cz + 1, cx + 2, KEEP_ROOF + 5, cz + 2, "minecraft:air")
            for x in range(cx, cx + 4):
                for z in range(cz, cz + 4):
                    if x in (cx, cx + 3) or z in (cz, cz + 3):
                        if (x + z) % 2 == 0:
                            self.stone(x, KEEP_ROOF + 6, z, x, KEEP_ROOF + 6, z)
        # doorway on the courtyard side, facing the gate
        for x in (31, 32):
            g.fill(x, GROUND + 1, hi - 1, x, GROUND + 3, hi, "minecraft:air")
        g.set(31, GROUND + 1, hi, "minecraft:oak_door[facing=south,half=lower,hinge=left]")
        g.set(31, GROUND + 2, hi, "minecraft:oak_door[facing=south,half=upper,hinge=left]")
        g.set(32, GROUND + 1, hi, "minecraft:oak_door[facing=south,half=lower,hinge=right]")
        g.set(32, GROUND + 2, hi, "minecraft:oak_door[facing=south,half=upper,hinge=right]")
        self.stair(30, GROUND + 3, hi, "east")
        self.stair(33, GROUND + 3, hi, "west")
        # windows: arrow loops low down, glazed lights at the top
        for level, floor in enumerate(KEEP_FLOORS):
            for t in range(lo + 4, hi - 2, 5):
                for y in (floor + 3, floor + 4):
                    for x, z in ((t, lo), (t, hi), (lo, t), (hi, t)):
                        if level == len(KEEP_FLOORS) - 1:
                            g.set(x, y, z, "minecraft:glass_pane")
                        else:
                            g.set(x, y, z, "minecraft:air")
        self.keep_stairs()
        self.keep_rooms()

    def keep_stairs(self) -> None:
        """One straight flight per floor, tucked against the west wall, with the
        floor above opened up over it."""
        g = self.g
        x = KEEP_LO + 3
        for floor, top in zip(KEEP_FLOORS, KEEP_FLOORS[1:] + [KEEP_ROOF]):
            rise = top - floor
            for step in range(rise):
                z = KEEP_LO + 3 + step
                self.stair(x, floor + 1 + step, z, "south")
                self.stair(x + 1, floor + 1 + step, z, "south")
                g.fill(x, floor + 2 + step, z, x + 1, floor + 4 + step, z, "minecraft:air")
            g.fill(x, top, KEEP_LO + 3, x + 1, top, KEEP_LO + 2 + rise, "minecraft:air")

    def keep_rooms(self) -> None:
        g = self.g
        lo, hi = KEEP_LO + 2, KEEP_HI - 2
        floor = KEEP_FLOORS[0]
        # great hall: a long table with benches, a hearth, chandeliers
        for z in range(lo + 4, hi - 2):
            g.set(31, floor + 1, z, "minecraft:oak_fence")
            g.set(32, floor + 1, z, "minecraft:oak_fence")
            g.set(31, floor + 2, z, "minecraft:oak_slab[type=top]")
            g.set(32, floor + 2, z, "minecraft:oak_slab[type=top]")
            if z % 2 == 0:
                self.stair(30, floor + 1, z, "east", material="oak")
                self.stair(33, floor + 1, z, "west", material="oak")
        for x in range(30, 34):
            g.fill(x, floor + 1, lo, x, floor + 4, lo, "minecraft:bricks")
        g.set(31, floor + 1, lo + 1, "minecraft:campfire[lit=true,facing=south]")
        g.set(32, floor + 1, lo + 1, "minecraft:campfire[lit=true,facing=south]")
        for x, z in ((28, lo + 6), (35, lo + 6), (28, hi - 4), (35, hi - 4)):
            g.set(x, KEEP_FLOORS[1] - 1, z, "minecraft:iron_chain[axis=y]")
            g.set(x, KEEP_FLOORS[1] - 2, z, "minecraft:lantern[hanging=true]")
        g.set(29, floor + 1, hi - 1, "minecraft:chest[facing=west]",
              {"id": "minecraft:chest", "LootTable": "minecraft:chests/simple_dungeon"})

        # barracks
        floor = KEEP_FLOORS[1]
        for i, z in enumerate((lo + 2, lo + 5, lo + 8)):
            g.set(hi - 1, floor + 1, z, "minecraft:red_bed[facing=west,part=head]")
            g.set(hi - 2, floor + 1, z, "minecraft:red_bed[facing=west,part=foot]")
            g.set(hi - 1, floor + 1, z + 1, "minecraft:barrel[facing=up]")
        g.set(lo + 3, floor + 1, hi - 1, "minecraft:crafting_table")
        g.set(lo + 4, floor + 1, hi - 1, "minecraft:anvil[facing=north]")
        g.set(lo + 6, floor + 1, hi - 1, "minecraft:chest[facing=north]",
              {"id": "minecraft:chest", "LootTable": "minecraft:chests/stronghold_crossing"})

        # the lord's chamber
        floor = KEEP_FLOORS[2]
        g.set(hi - 1, floor + 1, lo + 2, "minecraft:red_bed[facing=west,part=head]")
        g.set(hi - 2, floor + 1, lo + 2, "minecraft:red_bed[facing=west,part=foot]")
        for z in range(lo + 5, hi - 3):
            g.set(hi - 1, floor + 1, z, "minecraft:bookshelf")
        g.set(lo + 2, floor + 1, hi - 2, "minecraft:lectern[facing=north]")
        for x in range(30, 34):
            g.set(x, floor + 1, lo, "minecraft:red_carpet")
        g.set(lo + 2, floor + 1, lo + 2, "minecraft:cauldron")

        # the armoury, under the roof
        floor = KEEP_FLOORS[3]
        g.set(lo + 2, floor + 1, lo + 2, "minecraft:chest[facing=south]",
              {"id": "minecraft:chest", "LootTable": "minecraft:chests/stronghold_library"})
        for z in (lo + 4, lo + 6):
            g.set(lo, floor + 3, z, "minecraft:red_wall_banner[facing=east]")
            g.set(hi, floor + 3, z, "minecraft:red_wall_banner[facing=west]")

    def bailey(self) -> None:
        g = self.g
        # the road from the gate to the keep door
        for x in range(GATE_X0, GATE_X1 + 1):
            for z in range(KEEP_HI + 1, SIZE - 1 - WALL_IN):
                g.set(x, GROUND, z, "minecraft:stone_bricks" if (x + z) % 5 else
                      "minecraft:polished_andesite")
        for z in range(BAILEY + 2, SIZE - BAILEY - 2):
            for x in (KEEP_LO - 2, KEEP_HI + 2):
                g.set(x, GROUND, z, "minecraft:polished_andesite")
        for x in range(KEEP_LO - 2, KEEP_HI + 3):
            for z in (KEEP_LO - 2, KEEP_HI + 2):
                g.set(x, GROUND, z, "minecraft:polished_andesite")

        # the well
        wx, wz = BAILEY + 3, BAILEY + 3
        for x in range(wx - 1, wx + 2):
            for z in range(wz - 1, wz + 2):
                g.set(x, GROUND, z, "minecraft:cobblestone")
                g.set(x, GROUND + 1, z, "minecraft:cobblestone_wall")
        g.set(wx, GROUND, wz, "minecraft:water[level=0]")
        g.set(wx, GROUND + 1, wz, "minecraft:water[level=0]")
        for x, z in ((wx - 1, wz - 1), (wx + 1, wz - 1), (wx - 1, wz + 1), (wx + 1, wz + 1)):
            g.fill(x, GROUND + 2, z, x, GROUND + 3, z, "minecraft:oak_fence")
        for x in range(wx - 1, wx + 2):
            for z in range(wz - 1, wz + 2):
                g.set(x, GROUND + 4, z, "minecraft:oak_slab[type=bottom]")

        # a smithy in one corner
        sx, sz = SIZE - BAILEY - 6, BAILEY + 2
        for x in range(sx, sx + 5):
            for z in range(sz, sz + 4):
                g.set(x, GROUND, z, "minecraft:cobblestone")
        for x, z in ((sx, sz), (sx + 4, sz), (sx, sz + 3), (sx + 4, sz + 3)):
            g.fill(x, GROUND + 1, z, x, GROUND + 3, z, "minecraft:oak_fence")
        for x in range(sx, sx + 5):
            for z in range(sz, sz + 4):
                g.set(x, GROUND + 4, z, "minecraft:spruce_slab[type=bottom]")
        g.set(sx + 1, GROUND + 1, sz + 1, "minecraft:furnace[facing=south,lit=true]")
        g.set(sx + 2, GROUND + 1, sz + 1, "minecraft:furnace[facing=south,lit=true]")
        g.set(sx + 3, GROUND + 1, sz + 1, "minecraft:anvil[facing=south]")
        g.set(sx + 1, GROUND + 1, sz + 2, "minecraft:barrel[facing=up]")
        g.set(sx + 3, GROUND + 1, sz + 2, "minecraft:chest[facing=south]",
              {"id": "minecraft:chest", "LootTable": "minecraft:chests/village/village_weaponsmith"})

        # stables and a hay yard
        hx, hz = BAILEY + 2, SIZE - BAILEY - 7
        for x in range(hx, hx + 6):
            for z in range(hz, hz + 5):
                g.set(x, GROUND, z, "minecraft:dirt_path")
                if x in (hx, hx + 5) or z in (hz, hz + 4):
                    g.set(x, GROUND + 1, z, "minecraft:oak_fence")
        g.set(hx + 2, GROUND + 1, hz, "minecraft:oak_fence_gate[facing=north]")
        for x in range(hx + 1, hx + 4):
            g.set(x, GROUND + 1, hz + 2, "minecraft:hay_block[axis=y]")
        g.set(hx + 1, GROUND + 2, hz + 2, "minecraft:hay_block[axis=y]")

        # a kitchen garden
        gx, gz = SIZE - BAILEY - 8, SIZE - BAILEY - 7
        for x in range(gx, gx + 6):
            for z in range(gz, gz + 5):
                if z == gz + 2:
                    g.set(x, GROUND, z, "minecraft:water[level=0]")
                else:
                    g.set(x, GROUND, z, "minecraft:farmland[moisture=7]")
                    g.set(x, GROUND + 1, z, "minecraft:wheat[age=7]")

        # stairs from the courtyard up to the wall walk, one each side
        for side in (0, 1):
            x = BAILEY if side == 0 else SIZE - 1 - BAILEY
            for step in range(WALK - GROUND):
                z = 26 + step
                self.stair(x, GROUND + 1 + step, z, "south")
                self.stone(x, GROUND, z, x, GROUND + step, z)
        # torches on the inside face of the wall. A wall torch hangs off the
        # block behind it, which is the one opposite its facing, so it sits one
        # block into the bailey and faces away from the wall
        for t in range(BAILEY + 2, SIZE - BAILEY - 1, 6):
            g.set(t, GROUND + 4, WALL_IN + 1, "minecraft:wall_torch[facing=south]")
            g.set(t, GROUND + 4, SIZE - 2 - WALL_IN, "minecraft:wall_torch[facing=north]")
            g.set(WALL_IN + 1, GROUND + 4, t, "minecraft:wall_torch[facing=east]")
            g.set(SIZE - 2 - WALL_IN, GROUND + 4, t, "minecraft:wall_torch[facing=west]")

    def building(self, x0, z0, w, d, height, wall, roof, base=GROUND) -> None:
        """A small gabled building: walls, a ridged roof of stairs, a door and
        two windows. The bailey needs somewhere for people to live."""
        g = self.g
        x1, z1 = x0 + w - 1, z0 + d - 1
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                g.set(x, base, z, "minecraft:cobblestone")
        for y in range(base + 1, base + height + 1):
            for x in range(x0, x1 + 1):
                for z in range(z0, z1 + 1):
                    on_edge = x in (x0, x1) or z in (z0, z1)
                    if on_edge:
                        corner = x in (x0, x1) and z in (z0, z1)
                        g.set(x, y, z, "minecraft:oak_log[axis=y]" if corner else wall)
                    else:
                        g.set(x, y, z, "minecraft:air")
        # gabled roof, ridge running along z
        span = (w + 1) // 2
        for step in range(span):
            y = base + height + 1 + step
            for z in range(z0 - 1, z1 + 2):
                g.set(x0 + step, y, z, f"minecraft:{roof}_stairs[facing=east]")
                g.set(x1 - step, y, z, f"minecraft:{roof}_stairs[facing=west]")
                if step:
                    for x in range(x0 + step + 1, x1 - step):
                        g.set(x, y, z, "minecraft:air")
                for x in range(x0 + step + 1, x1 - step):
                    g.set(x, y - 1, z, f"minecraft:{roof}_planks")
        # door and windows
        cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
        dx, dz = cx, z1
        g.set(dx, base + 1, dz, "minecraft:oak_door[facing=south,half=lower,hinge=left]")
        g.set(dx, base + 2, dz, "minecraft:oak_door[facing=south,half=upper,hinge=left]")
        for z in (cz - 2, cz + 2):
            g.set(x0, base + 2, z, "minecraft:glass_pane")
            g.set(x1, base + 2, z, "minecraft:glass_pane")
        g.set(cx, base + height, dz, "minecraft:lantern[hanging=true]")

    def chapel(self) -> None:
        g = self.g
        x0, z0, w, d = BAILEY + 1, 24, 7, 11
        self.building(x0, z0, w, d, 5, "minecraft:stone_bricks", "spruce")
        cx = x0 + w // 2
        # altar, candles and a rose window at the far end
        g.set(cx, GROUND + 1, z0 + 1, "minecraft:smooth_stone_slab[type=top]")
        g.set(cx - 1, GROUND + 1, z0 + 1, "minecraft:candle[candles=3,lit=true]")
        g.set(cx + 1, GROUND + 1, z0 + 1, "minecraft:candle[candles=2,lit=true]")
        g.set(cx, GROUND + 3, z0, "minecraft:yellow_stained_glass_pane")
        g.set(cx - 1, GROUND + 3, z0, "minecraft:red_stained_glass_pane")
        g.set(cx + 1, GROUND + 3, z0, "minecraft:blue_stained_glass_pane")
        for z in range(z0 + 3, z0 + 9, 2):        # pews
            self.stair(cx - 1, GROUND + 1, z, "north", material="oak")
            self.stair(cx + 1, GROUND + 1, z, "north", material="oak")
        g.set(cx, GROUND + 1, z0 + 4, "minecraft:lectern[facing=south]")

    def barracks(self) -> None:
        g = self.g
        x0, z0, w, d = SIZE - BAILEY - 8, 24, 7, 11
        self.building(x0, z0, w, d, 5, "minecraft:oak_planks", "spruce")
        for z in range(z0 + 2, z0 + 9, 3):
            g.set(x0 + 1, GROUND + 1, z, "minecraft:red_bed[facing=east,part=head]")
            g.set(x0 + 2, GROUND + 1, z, "minecraft:red_bed[facing=east,part=foot]")
            g.set(x0 + w - 2, GROUND + 1, z, "minecraft:barrel[facing=up]")
        g.set(x0 + w - 2, GROUND + 1, z0 + 1, "minecraft:chest[facing=west]",
              {"id": "minecraft:chest", "LootTable": "minecraft:chests/pillager_outpost"})

    def build(self) -> Grid:
        self.ground_and_moat()
        self.curtain_wall()
        self.corner_towers()
        self.gatehouse()
        self.keep()
        self.bailey()
        self.chapel()
        self.barracks()
        return self.g


def lattice(x: int, z: int, cell: int, salt: int) -> float:
    """Smooth value noise on a lattice, so damage comes in patches rather than
    as even static."""
    x0, z0 = x // cell, z // cell
    tx, tz = (x % cell) / cell, (z % cell) / cell

    def corner(a, b):
        return hashed(a, 0, b, salt) / 1000.0

    def lerp(a, b, t):
        t = t * t * (3 - 2 * t)
        return a + (b - a) * t

    return lerp(lerp(corner(x0, z0), corner(x0 + 1, z0), tx),
                lerp(corner(x0, z0 + 1), corner(x0 + 1, z0 + 1), tx), tz)


class TowerKeep(Castle):
    """A tower house inside a low bailey wall: the small castle, common enough
    that the country feels lived in."""

    SIZE = 32
    HEIGHT = 40
    BASE = 8               # courtyard level
    WALL = 3               # ring distance of the bailey wall
    TOWER_LO, TOWER_HI = 10, 21
    FLOORS = [8, 14, 20, 26]
    ROOF = 31

    def __init__(self, weathered: int = 0):
        self.g = Grid(self.SIZE, self.HEIGHT, self.SIZE)
        self.weathered = weathered

    def near_edge(self, x, z) -> int:
        return dist_to_edge(x, z, 0, self.SIZE - 1)

    def build(self) -> Grid:
        g, base = self.g, self.BASE
        size = self.SIZE
        g.fill(0, 0, 0, size - 1, base - 3, size - 1, "minecraft:stone")
        g.fill(0, base - 2, 0, size - 1, base - 1, size - 1, "minecraft:dirt")
        g.fill(0, base, 0, size - 1, base, size - 1, "minecraft:grass_block[snowy=false]")

        # bailey wall with a walk on top
        lo, hi = self.WALL, size - 1 - self.WALL
        for x in range(size):
            for z in range(size):
                d = self.near_edge(x, z)
                if self.WALL <= d <= self.WALL + 1:
                    self.stone(x, 0, z, x, base + 6, z)
                    g.fill(x, base + 7, z, x, base + 10, z, "minecraft:air")
                elif d > self.WALL + 1:
                    g.set(x, base, z, paving(x, z))
                    g.fill(x, base + 1, z, x, base + 10, z, "minecraft:air")
        self.crenellate_line(lo, lo, hi, lo, base + 7)
        self.crenellate_line(lo, hi, hi, hi, base + 7)
        self.crenellate_line(lo, lo, lo, hi, base + 7)
        self.crenellate_line(hi, lo, hi, hi, base + 7)
        for cx, cz in ((lo, lo), (hi - 1, lo), (lo, hi - 1), (hi - 1, hi - 1)):
            self.stone(cx, 0, cz, cx + 1, base + 9, cz + 1)
            for x in range(cx, cx + 2):
                for z in range(cz, cz + 2):
                    if (x + z) % 2 == 0:
                        self.stone(x, base + 10, z, x, base + 10, z)

        # gateway on the south side
        gate_x = size // 2 - 1
        for x in (gate_x, gate_x + 1):
            g.fill(x, base + 1, hi - 1, x, base + 3, hi + 1, "minecraft:air")
            self.stone(x, base + 4, hi - 1, x, base + 4, hi + 1)
        for z in (hi - 1, hi + 1):
            self.stair(gate_x - 1, base + 4, z, "east")
            self.stair(gate_x + 2, base + 4, z, "west")
        g.set(gate_x, base + 1, hi, "minecraft:oak_door[facing=south,half=lower,hinge=left]")
        g.set(gate_x, base + 2, hi, "minecraft:oak_door[facing=south,half=upper,hinge=left]")
        g.set(gate_x + 1, base + 1, hi, "minecraft:oak_door[facing=south,half=lower,hinge=right]")
        g.set(gate_x + 1, base + 2, hi, "minecraft:oak_door[facing=south,half=upper,hinge=right]")
        for x in (gate_x - 1, gate_x + 2):
            g.set(x, base + 3, hi + 1, "minecraft:lantern[hanging=false]")

        # the tower house
        tl, th = self.TOWER_LO, self.TOWER_HI
        self.stone(tl, 0, tl, th, self.ROOF, th)
        for level, floor in enumerate(self.FLOORS):
            top = self.FLOORS[level + 1] if level + 1 < len(self.FLOORS) else self.ROOF
            g.fill(tl + 1, floor + 1, tl + 1, th - 1, top - 1, th - 1, "minecraft:air")
            g.fill(tl + 1, floor, tl + 1, th - 1, floor, th - 1,
                   "minecraft:stone_bricks" if level == 0 else "minecraft:oak_planks")
        g.fill(tl + 1, self.ROOF, tl + 1, th - 1, self.ROOF, th - 1, "minecraft:stone_bricks")
        g.fill(tl + 1, self.ROOF + 1, tl + 1, th - 1, self.ROOF + 4, th - 1, "minecraft:air")
        self.crenellate_line(tl, tl, th, tl, self.ROOF + 2)
        self.crenellate_line(tl, th, th, th, self.ROOF + 2)
        self.crenellate_line(tl, tl, tl, th, self.ROOF + 2)
        self.crenellate_line(th, tl, th, th, self.ROOF + 2)
        for x in range(tl, th + 1):
            for z in range(tl, th + 1):
                if dist_to_edge(x, z, tl, th) == 0:
                    self.stone(x, self.ROOF + 1, z, x, self.ROOF + 1, z)
        # stair turret in one corner, carrying the ladder out onto the roof
        self.stone(tl, self.ROOF, tl, tl + 3, self.ROOF + 5, tl + 3)
        g.fill(tl + 1, self.ROOF + 1, tl + 1, tl + 2, self.ROOF + 5, tl + 2, "minecraft:air")
        for x in range(tl, tl + 4):
            for z in range(tl, tl + 4):
                if (x in (tl, tl + 3) or z in (tl, tl + 3)) and (x + z) % 2 == 0:
                    self.stone(x, self.ROOF + 6, z, x, self.ROOF + 6, z)
        g.fill(tl + 1, self.ROOF, tl + 1, tl + 2, self.ROOF, tl + 2, "minecraft:air")

        # door, windows, ladders between floors
        door_x = (tl + th) // 2
        g.fill(door_x, base + 1, th, door_x + 1, base + 2, th, "minecraft:air")
        g.set(door_x, base + 1, th, "minecraft:oak_door[facing=south,half=lower,hinge=left]")
        g.set(door_x, base + 2, th, "minecraft:oak_door[facing=south,half=upper,hinge=left]")
        g.set(door_x + 1, base + 1, th, "minecraft:oak_door[facing=south,half=lower,hinge=right]")
        g.set(door_x + 1, base + 2, th, "minecraft:oak_door[facing=south,half=upper,hinge=right]")
        for floor in self.FLOORS:
            glazed = floor > self.FLOORS[1]
            pane = "minecraft:glass_pane" if glazed else "minecraft:air"
            for t in (tl + 3, th - 3):
                for y in (floor + 3, floor + 4):
                    g.set(t, y, tl, pane)
                    g.set(t, y, th, pane)
                    g.set(tl, y, t, pane)
                    g.set(th, y, t, pane)
        for floor, top in zip(self.FLOORS, self.FLOORS[1:] + [self.ROOF]):
            for y in range(floor + 1, top + 1):
                g.set(tl + 2, y, tl + 1, "minecraft:ladder[facing=south]")
            g.set(tl + 2, top, tl + 1, "minecraft:air")
            g.set(tl + 2, top, tl + 2, "minecraft:air")

        # furnishings
        g.set(th - 2, self.FLOORS[0] + 1, tl + 2, "minecraft:campfire[lit=true,facing=south]")
        g.set(th - 2, self.FLOORS[1] + 1, tl + 2, "minecraft:red_bed[facing=west,part=head]")
        g.set(th - 3, self.FLOORS[1] + 1, tl + 2, "minecraft:red_bed[facing=west,part=foot]")
        g.set(tl + 2, self.FLOORS[1] + 1, th - 2, "minecraft:chest[facing=west]",
              {"id": "minecraft:chest", "LootTable": "minecraft:chests/pillager_outpost"})
        g.set(th - 2, self.FLOORS[2] + 1, th - 2, "minecraft:crafting_table")
        g.set(th - 3, self.FLOORS[2] + 1, th - 2, "minecraft:barrel[facing=up]")
        g.set(tl + 2, self.FLOORS[3] + 1, th - 2, "minecraft:chest[facing=west]",
              {"id": "minecraft:chest", "LootTable": "minecraft:chests/simple_dungeon"})

        # courtyard: a well, a woodpile, a patch of wheat
        wx, wz = size - 7, 7
        for x in range(wx - 1, wx + 2):
            for z in range(wz - 1, wz + 2):
                g.set(x, base, z, "minecraft:cobblestone")
                g.set(x, base + 1, z, "minecraft:cobblestone_wall")
        g.set(wx, base, wz, "minecraft:water[level=0]")
        g.set(wx, base + 1, wz, "minecraft:water[level=0]")
        for x in range(6, 10):
            g.set(x, base, size - 8, "minecraft:oak_log[axis=x]")
        g.set(6, base + 1, size - 8, "minecraft:oak_log[axis=x]")
        for x in range(size - 10, size - 5):
            for z in range(size - 9, size - 6):
                g.set(x, base, z, "minecraft:farmland[moisture=7]")
                g.set(x, base + 1, z, "minecraft:wheat[age=7]")
        for t in (7, size - 8):
            g.set(t, base + 4, self.WALL + 1, "minecraft:wall_torch[facing=south]")
        return g


def ruinate(grid: Grid, keep_level: int = GROUND) -> Grid:
    """Knock a castle down: patches of wall gone, moss and vines over what is
    left, rubble underfoot. Damage grows with height, so footings survive and
    battlements do not."""
    top = grid.sy - 1
    masonry_names = ("stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks",
                     "cobblestone", "chiseled_stone_bricks", "oak_planks",
                     "spruce_planks", "stone_brick_stairs", "spruce_stairs", "oak_log",
                     "glass_pane", "oak_stairs", "bookshelf", "iron_bars")
    furniture = ("red_bed", "oak_door", "lantern", "wall_torch", "torch", "campfire",
                 "chest", "barrel", "crafting_table", "hay_block", "wheat", "candle",
                 "lectern", "red_carpet", "oak_fence", "anvil", "furnace")
    rubble = []
    for x in range(grid.sx):
        for z in range(grid.sz):
            patch = lattice(x, z, 13, 5)
            collapse = lattice(x, z, 27, 9)
            for y in range(keep_level + 1, top + 1):
                name = grid.name_at(x, y, z)
                if not name or name == "air":
                    continue
                height = (y - keep_level) / max(1.0, top - keep_level)
                damage = 0.55 * patch + 0.95 * height + 0.18 * collapse
                if name in masonry_names and damage > 0.80:
                    grid.set(x, y, z, "minecraft:air")
                    if hashed(x, y, z, 21) < 90:
                        rubble.append((x, z))
                elif name in furniture and hashed(x, y, z, 33) < 700:
                    grid.set(x, y, z, "minecraft:air")
    for x, z in rubble:
        for y in range(keep_level + 1, keep_level + 4):
            if grid.name_at(x, y, z) in ("", "air"):
                grid.set(x, y, z, "minecraft:cobblestone" if hashed(x, y, z, 41) < 600
                         else "minecraft:mossy_cobblestone")
                break
    # vines down whatever wall is still standing
    for x in range(grid.sx):
        for z in range(grid.sz):
            if lattice(x, z, 9, 17) < 0.62:
                continue
            for y in range(keep_level + 1, top):
                if grid.name_at(x, y, z) not in ("", "air"):
                    continue
                for dx, dz, side in ((1, 0, "west"), (-1, 0, "east"),
                                     (0, 1, "north"), (0, -1, "south")):
                    neighbour = grid.name_at(x + dx, y, z + dz)
                    if neighbour in ("stone_bricks", "mossy_stone_bricks",
                                     "cracked_stone_bricks", "cobblestone"):
                        if hashed(x, y, z, 55) < 240:
                            grid.set(x, y, z, f"minecraft:vine[{side}=true]")
                        break
    # nothing is left hanging in mid air once its wall has gone
    loose = ("red_banner", "red_wall_banner", "torch", "wall_torch", "lantern", "ladder",
             "chest", "barrel", "crafting_table", "anvil", "furnace", "campfire",
             "candle", "lectern", "red_bed", "red_carpet", "oak_door", "cobblestone_wall",
             "hay_block", "wheat", "farmland", "iron_bars", "oak_fence", "bookshelf")
    for x in range(grid.sx):
        for z in range(grid.sz):
            for y in range(top, keep_level, -1):
                if grid.name_at(x, y, z) not in loose:
                    continue
                def unsupported(name):
                    return name in ("", "air") or name in loose
                below = grid.name_at(x, y - 1, z)
                sides = [grid.name_at(x + dx, y, z + dz)
                         for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                # top down, so a ladder loses its wall and then falls the whole way
                if unsupported(below) and all(unsupported(s) for s in sides):
                    grid.set(x, y, z, "minecraft:air")

    # a ruin belongs to the wood again: soil returns, and the trees with it
    for x in range(grid.sx):
        for z in range(grid.sz):
            under = grid.name_at(x, keep_level, z)
            over = grid.name_at(x, keep_level + 1, z)
            if (under in ("cobblestone", "mossy_cobblestone", "andesite", "gravel",
                          "dirt_path", "polished_andesite", "stone_bricks")
                    and over in ("", "air") and lattice(x, z, 7, 71) > 0.42):
                grid.set(x, keep_level, z, "minecraft:grass_block[snowy=false]")

    # the courtyard goes back to grass and scrub
    for x in range(grid.sx):
        for z in range(grid.sz):
            if grid.name_at(x, keep_level, z) == "grass_block" and \
                    grid.name_at(x, keep_level + 1, z) in ("", "air"):
                roll = hashed(x, keep_level, z, 63)
                if roll < 120:
                    grid.set(x, keep_level + 1, z, "minecraft:short_grass")
                elif roll < 150:
                    grid.set(x, keep_level + 1, z, "minecraft:fern")
    return grid


LAND_BIOMES = [f"{NS}:downs", f"{NS}:oakwood", f"{NS}:heath", f"{NS}:crag"]


def emit(writer: PackWriter, name: str, grid: Grid, sink: int) -> None:
    structures.add(writer, f"{NS}:{name}", grid, LAND_BIOMES, sink=sink)
    path = writer.root / "data" / NS / "structure" / f"{name}.nbt"
    print(f"  {name:14s} {grid.sx}x{grid.sy}x{grid.sz}  {grid.filled():6d} blocks  "
          f"{len(grid.palette):3d} states  {path.stat().st_size / 1024:5.0f} KB")


def main() -> int:
    pack = ROOT / "packs" / "castle_country"
    renders = ROOT / "renders"
    writer = PackWriter(pack, "Castle country - crags, downs and oak woods", "26.2")
    writer.mcmeta()
    print("building templates")

    great = Castle().build()
    emit(writer, "great_castle", great, sink=-(GROUND + 1))
    render_iso(great, renders / "castle_iso.png", scale=6, label="great castle")
    render_iso(great, renders / "castle_iso_back.png", scale=6, turn=2,
               label="great castle - from behind")
    render_plan(great, renders / "castle_plan.png",
                levels=[GROUND, GROUND + 3, 18, 25, WALK + 1], scale=4,
                label="great castle - plans")

    ruin = ruinate(Castle(weathered=3).build())
    emit(writer, "ruined_castle", ruin, sink=-(GROUND + 1))
    render_iso(ruin, renders / "castle_ruin_iso.png", scale=6, label="ruined castle")

    keep = TowerKeep().build()
    emit(writer, "tower_keep", keep, sink=-(TowerKeep.BASE + 1))
    render_iso(keep, renders / "tower_keep_iso.png", scale=8, label="tower keep")

    # two sets, so the small ones are common and never land on a great castle
    writer.add("structure_set", f"{NS}:great_castles",
               structures.spread({f"{NS}:great_castle": 3, f"{NS}:ruined_castle": 2},
                                 spacing=24, separation=10, salt=51263611))
    writer.add("structure_set", f"{NS}:tower_keeps",
               structures.spread(f"{NS}:tower_keep", spacing=13, separation=6,
                                 salt=90114377,
                                 exclusion=(f"{NS}:great_castles", 5)))
    print("wrote structure sets: great_castles (24/10), tower_keeps (13/6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
