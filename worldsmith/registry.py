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
    # a custom build is a template under data/<ns>/structure/*.nbt plus these
    # three: what it is, the pool holding its template, how often it appears
    "structure": "worldgen/structure",
    "template_pool": "worldgen/template_pool",
    "structure_set": "worldgen/structure_set",
    # Biome tags are what put villages, temples and monuments in a custom world:
    # a structure generates in the biomes its has_structure/* tag lists. They are
    # loaded so they can be validated. Vanilla merges tags across packs unless an
    # entry sets "replace"; that is not modelled here, so a later pack simply wins.
    "biome_tag": "tags/worldgen/biome",
}

TEMPLATE_DIR = "structure"           # the .nbt builds themselves

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


def _read_templates(root: Path, namespace: str) -> dict[str, Path]:
    base = root / "data" / namespace / TEMPLATE_DIR
    return {f"{namespace}:{path.relative_to(base).as_posix()[:-len('.nbt')]}": path
            for path in base.rglob("*.nbt")} if base.is_dir() else {}


class Pack:
    """One datapack directory, or the vendored vanilla data."""

    def __init__(self, root: str | os.PathLike, name: str | None = None):
        self.root = Path(root)
        self.name = name or self.root.name
        self.data: dict[str, dict[str, dict]] = {c: {} for c in CATEGORIES}
        data_dir = self.root / "data"
        self.namespaces = (sorted(p.name for p in data_dir.iterdir() if p.is_dir())
                           if data_dir.is_dir() else [])
        self.templates: dict[str, Path] = {}
        for namespace in self.namespaces:
            for category in CATEGORIES:
                self.data[category].update(_read_dir(self.root, namespace, category))
            self.templates.update(_read_templates(self.root, namespace))
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
        self.templates: dict[str, Path] = {}
        for pack in packs:
            for category, entries in pack.data.items():
                self.data[category].update(entries)
            self.templates.update(pack.templates)

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

    def biome_set(self, biomes, seen: set[str] | None = None,
                  missing: set[str] | None = None) -> set[str] | None:
        """Expand a biome list the way a structure's `biomes` field is written,
        as ids, or a tag, or a list mixing both. None means a tag is not here to
        expand, in which case nothing downstream should pretend to know. The
        tags that were not here are collected in `missing`, which is the one
        thing a caller can say about it that is worth reading.
        """
        if biomes is None:
            return None
        if isinstance(biomes, str):
            biomes = [biomes]
        seen = set() if seen is None else seen
        out: set[str] = set()
        for entry in biomes:
            required = True
            if isinstance(entry, dict):
                required = entry.get("required", True) is not False
                entry = entry.get("id")
            if not isinstance(entry, str):
                continue
            if not entry.startswith("#"):
                out.add(entry if ":" in entry else "minecraft:" + entry)
                continue
            ident = entry[1:]
            if ident in seen:                      # tags may name each other
                continue
            seen.add(ident)
            tag = self.get("biome_tag", ident)
            if tag is None:
                if not required:
                    continue
                if missing is not None:
                    missing.add(entry)
                return None
            inner = self.biome_set(tag.get("values") or [], seen, missing)
            if inner is None:
                return None
            out |= inner
        return out

    def start_template(self, structure: dict | None):
        """The .nbt a jigsaw structure starts from, reached through the pool it
        names. `start_pool` is a template_pool id, not a template id, and the
        two are usually written the same, which is what hides the difference."""
        pool = self.get("template_pool", (structure or {}).get("start_pool") or "") or {}
        for entry in pool.get("elements") or []:
            location = ((entry or {}).get("element") or {}).get("location")
            if isinstance(location, str):
                return self.templates.get(location if ":" in location
                                          else "minecraft:" + location)
        return None

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
