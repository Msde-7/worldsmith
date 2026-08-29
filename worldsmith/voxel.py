"""A grid of blocks that writes a Minecraft structure template.

The game builds structures from `.nbt` templates, so a custom build is a box of
block states plus the worldgen JSON in structures.py. Block states are checked
against the vendored block list: a bad state in a template fails silently in
game, and the block that should have been there is simply missing.
"""
from __future__ import annotations

import functools
import gzip
import json
import re
import struct
from pathlib import Path

import numpy as np

from .registry import DEFAULT_VERSION, VANILLA_ROOT

TAG_END, TAG_BYTE, TAG_INT = 0, 1, 3
TAG_DOUBLE, TAG_STRING, TAG_LIST, TAG_COMPOUND = 6, 8, 9, 10

SPEC = re.compile(r"^([a-z0-9_.:/-]+)(?:\[(.*)\])?$")


@functools.lru_cache(maxsize=None)
def block_states(version: str = DEFAULT_VERSION) -> tuple[frozenset[str], dict]:
    """Every block id, and the property values each one accepts."""
    path = VANILLA_ROOT / version / "blocks.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return frozenset(data.get("blocks", ())), data.get("properties", {})


@functools.lru_cache(maxsize=None)
def data_version(version: str = DEFAULT_VERSION) -> int:
    """The DataVersion a template of this version must carry."""
    path = VANILLA_ROOT / version / "version.json"
    if not path.is_file():
        raise FileNotFoundError(f"no vendored version.json for {version}")
    return int(json.loads(path.read_text(encoding="utf-8"))["data_version"])


def parse_block(spec: str, version: str = DEFAULT_VERSION) -> dict:
    """`oak_stairs[facing=north]` to the {Name, Properties} a template holds."""
    match = SPEC.match(spec)
    if not match:
        raise ValueError(f"unreadable block spec {spec!r}")
    name, properties = match.group(1), match.group(2)
    if ":" not in name:
        name = "minecraft:" + name
    known, allowed_by_block = block_states(version)
    if known and name not in known:
        raise ValueError(f"no such block: {name}")
    state = {"Name": name}
    if properties:
        allowed = allowed_by_block.get(name, {})
        parsed = {}
        for pair in properties.split(","):
            key, _, value = pair.partition("=")
            key, value = key.strip(), value.strip()
            if allowed and key not in allowed:
                raise ValueError(f"{name} has no property {key!r} (has {sorted(allowed)})")
            if allowed and value not in allowed[key]:
                raise ValueError(f"{name}[{key}={value}] invalid (want {allowed[key]})")
            parsed[key] = value
        state["Properties"] = parsed
    return state


def format_block(state: dict) -> str:
    """The inverse of parse_block, for reading a template back."""
    properties = state.get("Properties") or {}
    if not properties:
        return str(state["Name"])
    inner = ",".join(f"{k}={v}" for k, v in sorted(properties.items()))
    return f"{state['Name']}[{inner}]"


# --- NBT ---------------------------------------------------------------------

def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack(">H", len(raw)) + raw


def _tag_of(value) -> int:
    if isinstance(value, bool):
        return TAG_BYTE
    if isinstance(value, int):
        return TAG_INT
    if isinstance(value, float):
        return TAG_DOUBLE
    if isinstance(value, str):
        return TAG_STRING
    if isinstance(value, dict):
        return TAG_COMPOUND
    if isinstance(value, (list, tuple)):
        return TAG_LIST
    raise TypeError(f"cannot write {type(value)} as NBT")


def _payload(value) -> bytes:
    tag = _tag_of(value)
    if tag == TAG_BYTE:
        return struct.pack(">b", int(value))
    if tag == TAG_INT:
        return struct.pack(">i", value)
    if tag == TAG_DOUBLE:
        return struct.pack(">d", value)
    if tag == TAG_STRING:
        return _string(value)
    if tag == TAG_LIST:
        if not value:
            return struct.pack(">Bi", TAG_END, 0)
        return (struct.pack(">Bi", _tag_of(value[0]), len(value))
                + b"".join(_payload(item) for item in value))
    out = bytearray()
    for key, item in value.items():
        out += bytes([_tag_of(item)]) + _string(key) + _payload(item)
    return bytes(out) + bytes([TAG_END])


def write_nbt(root: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = bytes([TAG_COMPOUND]) + _string("") + _payload(root)
    # mtime 0: the same build writes the same bytes, so rebuilding a pack does
    # not churn every template in the diff
    path.write_bytes(gzip.compress(raw, mtime=0))
    return path


class NbtReader:
    """Just enough of the format to read a template or a chunk back."""

    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def _take(self, n: int) -> bytes:
        chunk = self.data[self.at:self.at + n]
        self.at += n
        return chunk

    def _number(self, fmt: str, size: int):
        return struct.unpack(fmt, self._take(size))[0]

    def string(self) -> str:
        return self._take(self._number(">H", 2)).decode("utf-8", "replace")

    def payload(self, tag: int):
        if tag == TAG_BYTE:
            return self._number(">b", 1)
        if tag == 2:
            return self._number(">h", 2)
        if tag == TAG_INT:
            return self._number(">i", 4)
        if tag == 4:
            return self._number(">q", 8)
        if tag == 5:
            return self._number(">f", 4)
        if tag == TAG_DOUBLE:
            return self._number(">d", 8)
        if tag == 7:
            return np.frombuffer(self._take(self._number(">i", 4)), dtype=np.int8)
        if tag == TAG_STRING:
            return self.string()
        if tag == TAG_LIST:
            child, count = self._number(">B", 1), self._number(">i", 4)
            return [self.payload(child) for _ in range(count)]
        if tag == TAG_COMPOUND:
            out = {}
            while True:
                child = self._number(">B", 1)
                if child == TAG_END:
                    return out
                name = self.string()      # read first: a subscript assignment
                out[name] = self.payload(child)   # evaluates the value before the key
        if tag == 11:
            return np.frombuffer(self._take(4 * self._number(">i", 4)), dtype=">i4")
        if tag == 12:
            return np.frombuffer(self._take(8 * self._number(">i", 4)), dtype=">i8")
        raise ValueError(f"unknown NBT tag {tag}")

    def root(self):
        tag = self._number(">B", 1)
        if tag != TAG_COMPOUND:
            raise ValueError(f"root tag is {tag}, not a compound")
        self.string()
        return self.payload(TAG_COMPOUND)


def read_nbt(path: Path) -> dict:
    return NbtReader(gzip.decompress(Path(path).read_bytes())).root()


# --- the grid ----------------------------------------------------------------

class Grid:
    """A box of blocks. Index 0 means "leave the world alone"."""

    def __init__(self, size_x: int, size_y: int, size_z: int,
                 version: str = DEFAULT_VERSION):
        self.sx, self.sy, self.sz = size_x, size_y, size_z
        self.version = version
        self.cells = np.zeros((size_x, size_y, size_z), dtype=np.int32)
        self.palette: list[str] = []
        self._index: dict[str, int] = {}
        self.block_entities: dict[tuple[int, int, int], dict] = {}

    @classmethod
    def load(cls, path: Path, version: str = DEFAULT_VERSION) -> "Grid":
        root = read_nbt(path)
        grid = cls(*(int(v) for v in root["size"]), version=version)
        palette = [format_block(state) for state in root["palette"]]
        for block in root["blocks"]:
            x, y, z = (int(v) for v in block["pos"])
            grid.set(x, y, z, palette[int(block["state"])], block.get("nbt"))
        return grid

    def id_of(self, spec: str) -> int:
        if spec not in self._index:
            parse_block(spec, self.version)          # raises on a typo
            self.palette.append(spec)
            self._index[spec] = len(self.palette)    # 1 based; 0 is "untouched"
        return self._index[spec]

    def inside(self, x, y, z) -> bool:
        return 0 <= x < self.sx and 0 <= y < self.sy and 0 <= z < self.sz

    def set(self, x, y, z, spec: str, nbt: dict | None = None) -> None:
        if not self.inside(x, y, z):
            return
        self.cells[x, y, z] = self.id_of(spec)
        key = (int(x), int(y), int(z))
        if nbt is None:
            self.block_entities.pop(key, None)
        else:
            self.block_entities[key] = nbt

    def get(self, x, y, z) -> str | None:
        if not self.inside(x, y, z):
            return None
        index = int(self.cells[x, y, z])
        return self.palette[index - 1] if index else None

    def name_at(self, x, y, z) -> str:
        spec = self.get(x, y, z)
        return "" if spec is None else spec.split("[")[0].split(":")[-1]

    def fill(self, x0, y0, z0, x1, y1, z1, spec: str) -> None:
        value = self.id_of(spec)
        self.cells[max(0, min(x0, x1)):min(self.sx, max(x0, x1) + 1),
                   max(0, min(y0, y1)):min(self.sy, max(y0, y1) + 1),
                   max(0, min(z0, z1)):min(self.sz, max(z0, z1) + 1)] = value

    def counts(self) -> dict[str, int]:
        used = {index: int(n) for index, n in
                zip(*np.unique(self.cells, return_counts=True)) if index}
        return {self.palette[index - 1]: n for index, n in used.items()}

    def filled(self) -> int:
        return int(np.count_nonzero(self.cells))

    def to_nbt(self) -> dict:
        blocks = []
        for x, y, z in zip(*(axis.tolist() for axis in np.nonzero(self.cells))):
            entry = {"state": int(self.cells[x, y, z]) - 1, "pos": [x, y, z]}
            nbt = self.block_entities.get((x, y, z))
            if nbt is not None:
                entry["nbt"] = nbt
            blocks.append(entry)
        return {
            "size": [self.sx, self.sy, self.sz],
            "entities": [],
            "blocks": blocks,
            "palette": [parse_block(spec, self.version) for spec in self.palette],
            "DataVersion": data_version(self.version),
        }

    def save(self, path: Path) -> Path:
        return write_nbt(self.to_nbt(), path)
