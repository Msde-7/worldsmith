"""Datapack loading: vanilla data as the base layer, the user's pack on top.

Ids keep their sub-directory path, so worldgen/noise/nether/temperature.json is
minecraft:nether/temperature and does not collide with minecraft:temperature.
Getting that wrong silently swaps the nether's climate noise into the overworld.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# category -> relative directory under data/<namespace>/
CATEGORIES = {
    "noise": "worldgen/noise",
    "density_function": "worldgen/density_function",
    "noise_settings": "worldgen/noise_settings",
    "biome": "worldgen/biome",
    "multi_noise_biome_source_parameter_list": "worldgen/multi_noise_biome_source_parameter_list",
    "configured_carver": "worldgen/configured_carver",
    "configured_feature": "worldgen/configured_feature",
    "placed_feature": "worldgen/placed_feature",
    "dimension": "dimension",
    "dimension_type": "dimension_type",
}

VANILLA_ROOT = Path(__file__).resolve().parent.parent / "vanilla"
DEFAULT_VERSION = "26.2"


def _read_dir(root: Path, namespace: str, category: str) -> dict[str, dict]:
    base = root / "data" / namespace / CATEGORIES[category]
    out: dict[str, dict] = {}
    if not base.is_dir():
        return out
    for path in base.rglob("*.json"):
        rel = path.relative_to(base).as_posix()[:-len(".json")]
        try:
            out[f"{namespace}:{rel}"] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    return out


class Pack:
    """One datapack directory, or the vendored vanilla data."""

    def __init__(self, root: str | os.PathLike, name: str | None = None):
        self.root = Path(root)
        self.name = name or self.root.name
        self.data: dict[str, dict[str, dict]] = {c: {} for c in CATEGORIES}
        data_dir = self.root / "data"
        self.namespaces = (sorted(p.name for p in data_dir.iterdir() if p.is_dir())
                           if data_dir.is_dir() else [])
        for namespace in self.namespaces:
            for category in CATEGORIES:
                self.data[category].update(_read_dir(self.root, namespace, category))
        mcmeta = self.root / "pack.mcmeta"
        self.mcmeta = json.loads(mcmeta.read_text(encoding="utf-8")) if mcmeta.is_file() else None

    def __repr__(self):
        counts = ", ".join(f"{c}={len(v)}" for c, v in self.data.items() if v)
        return f"<Pack {self.name} [{counts}]>"


class Registries:
    """Vanilla first, then each overlay pack. Later packs win."""

    def __init__(self, packs: list[Pack]):
        self.packs = packs
        self.data: dict[str, dict[str, dict]] = {c: {} for c in CATEGORIES}
        for pack in packs:
            for category, entries in pack.data.items():
                self.data[category].update(entries)

    @classmethod
    def load(cls, pack_paths: list[str | os.PathLike] | None = None, version: str = DEFAULT_VERSION,
             include_vanilla: bool = True) -> "Registries":
        packs: list[Pack] = []
        if include_vanilla:
            vanilla_dir = VANILLA_ROOT / version
            if not vanilla_dir.is_dir():
                available = sorted(p.name for p in VANILLA_ROOT.iterdir()) if VANILLA_ROOT.is_dir() else []
                raise FileNotFoundError(
                    f"no vendored vanilla data for {version} (have: {available or 'none'})")
            packs.append(Pack(vanilla_dir, name=f"vanilla-{version}"))
        for path in pack_paths or []:
            packs.append(Pack(path))
        return cls(packs)

    def get(self, category: str, ident: str):
        if ":" not in ident:
            ident = "minecraft:" + ident
        return self.data[category].get(ident)

    def ids(self, category: str) -> list[str]:
        return sorted(self.data[category])

    def origin(self, category: str, ident: str) -> str | None:
        """Which pack supplied this entry."""
        if ":" not in ident:
            ident = "minecraft:" + ident
        for pack in reversed(self.packs):
            if ident in pack.data[category]:
                return pack.name
        return None
