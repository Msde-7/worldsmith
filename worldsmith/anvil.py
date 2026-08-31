"""Reading a world the game has already generated.

The region files are the only place the game writes down what it actually did,
so this is how worldsmith checks itself: where the structure starts landed, and
which blocks ended up where.
"""
from __future__ import annotations

import gzip
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from .voxel import Grid, NbtReader, block_states

SECTOR = 4096


def region_dir(world: Path) -> Path:
    """Overworld region files, in either of the two layouts the game writes."""
    world = Path(world)
    nested = world / "dimensions" / "minecraft" / "overworld" / "region"
    return nested if nested.is_dir() else world / "region"


def read_region(path: Path) -> dict[tuple[int, int], dict]:
    """{(chunk x, chunk z): chunk NBT} for one .mca file."""
    raw = Path(path).read_bytes()
    if len(raw) < 2 * SECTOR:
        return {}
    chunks: dict[tuple[int, int], dict] = {}
    for index in range(1024):
        offset = struct.unpack(">I", b"\0" + raw[index * 4:index * 4 + 3])[0]
        if offset == 0 or raw[index * 4 + 3] == 0:
            continue
        start = offset * SECTOR
        length = struct.unpack(">I", raw[start:start + 4])[0]
        scheme = raw[start + 4]
        payload = raw[start + 5:start + 4 + length]
        if scheme == 1:
            payload = gzip.decompress(payload)
        elif scheme == 2:
            payload = zlib.decompress(payload)
        elif scheme != 3:
            continue
        nbt = NbtReader(payload).root()
        chunks[(nbt.get("xPos", 0), nbt.get("zPos", 0))] = nbt
    return chunks


def read_world(world: Path):
    """Every chunk in the world, one region file at a time."""
    for path in sorted(region_dir(world).glob("*.mca")):
        yield from read_region(path).items()


def unpack_longs(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Entries of `bits` bits packed into longs without straddling them (1.16+).

    Heightmaps, block states and biomes all use this layout, and a world read is
    thousands of these, so it is done a whole section at a time.
    """
    words = np.asarray(packed).astype(np.uint64)
    shifts = np.arange(64 // bits, dtype=np.uint64) * np.uint64(bits)
    values = (words[:, None] >> shifts[None, :]) & np.uint64((1 << bits) - 1)
    return values.reshape(-1)[:count].astype(np.int64)


def unpack_heightmap(packed: np.ndarray, world_height: int = 384) -> np.ndarray:
    """A chunk's heightmap, [z][x]. Entries are as wide as the world is tall, so
    a dimension shorter or taller than the overworld packs them differently."""
    bits = max(1, math.ceil(math.log2(world_height + 1)))
    return unpack_longs(packed, bits, 256).reshape(16, 16)


def _paletted(section: dict, key: str, count: int, side: int):
    """A section's block or biome names, as (side, side, side) indexed [y][z][x]."""
    holder = section.get(key) or {}
    palette = [str(entry.get("Name") if isinstance(entry, dict) else entry)
               for entry in (holder.get("palette") or [])]
    if not palette:
        return None
    data = holder.get("data")
    if data is None or len(data) == 0:
        index = np.zeros(count, dtype=np.int64)
    else:
        bits = max(4 if key == "block_states" else 1, math.ceil(math.log2(len(palette))))
        index = unpack_longs(np.asarray(data), bits, count)
    return np.array(palette, dtype=object)[index].reshape(side, side, side)


def section_blocks(chunk: dict, lo: int, hi: int) -> dict[int, np.ndarray]:
    """{block y: (16, 16) of block names} over y in [lo, hi], indexed [z][x]."""
    out: dict[int, np.ndarray] = {}
    for section in chunk.get("sections") or []:
        base = int(section.get("Y", 0)) * 16
        if base > hi or base + 15 < lo:
            continue
        names = _paletted(section, "block_states", 4096, 16)
        if names is None:
            continue
        for dy in range(16):
            if lo <= base + dy <= hi:
                out[base + dy] = names[dy]
    return out


def biome_at(chunk: dict, x: int, y: int, z: int) -> str | None:
    """The biome the game stored for this block, at quart resolution."""
    for section in chunk.get("sections") or []:
        base = int(section.get("Y", 0)) * 16
        if not (base <= y < base + 16):
            continue
        names = _paletted(section, "biomes", 64, 4)
        if names is None:
            return None
        return str(names[(y - base) // 4][(z & 15) // 4][(x & 15) // 4])
    return None


def biome_in_world(world: Path, x: int, y: int, z: int) -> str | None:
    """The biome the game stored at this block, read from the one region file
    that holds it rather than by walking the world."""
    path = region_dir(world) / f"r.{x >> 9}.{z >> 9}.mca"
    if not path.is_file():
        return None
    chunk = read_region(path).get((x >> 4, z >> 4))
    return biome_at(chunk, x, y, z) if chunk else None


def structure_starts(world: Path) -> dict[str, list[dict]]:
    """Every structure the game recorded, by id.

    The start itself carries no bounding box any more, so the box is the union
    of its pieces, which is what actually got built.
    """
    found: dict[str, list[dict]] = {}
    for _, chunk in read_world(world):
        starts = (chunk.get("structures") or {}).get("starts") or {}
        for ident, start in starts.items():
            boxes = [[int(v) for v in kid["BB"]] for kid in (start.get("Children") or [])
                     if kid.get("BB") is not None]
            if not boxes:
                continue
            box = ([min(b[i] for b in boxes) for i in range(3)]
                   + [max(b[i] for b in boxes) for i in range(3, 6)])
            found.setdefault(ident, []).append({
                "chunk": (int(start.get("ChunkX", 0)), int(start.get("ChunkZ", 0))),
                "box": box,
                "centre": ((box[0] + box[3]) // 2, (box[2] + box[5]) // 2),
                "rotation": str((start.get("Children") or [{}])[0].get("rotation", "NONE")),
            })
    return found


def read_box(world: Path, x0: int, z0: int, x1: int, z1: int, y0: int, y1: int) -> Grid:
    """The blocks in a box of the world, as a Grid the previewer can draw."""
    grid = Grid(x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
    known, _ = block_states()
    # cave_air and void_air are air the game carved, and drawing them as blocks
    # would fill every cave the build stands over
    air = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")
    for path in sorted(region_dir(world).glob("*.mca")):
        rx, rz = (int(part) for part in path.stem.split(".")[1:3])
        if not (rx * 512 <= x1 and rx * 512 + 511 >= x0
                and rz * 512 <= z1 and rz * 512 + 511 >= z0):
            continue
        for (cx, cz), chunk in read_region(path).items():
            bx, bz = cx * 16, cz * 16
            if bx > x1 or bx + 15 < x0 or bz > z1 or bz + 15 < z0:
                continue
            xs = range(max(0, x0 - bx), min(16, x1 - bx + 1))
            zs = range(max(0, z0 - bz), min(16, z1 - bz + 1))
            for y, plane in section_blocks(chunk, y0, y1).items():
                for dz in zs:
                    for dx in xs:
                        name = str(plane[dz][dx])
                        if name in air or name not in known:
                            continue
                        grid.set(bx + dx - x0, y - y0, bz + dz - z0, name)
    return grid
