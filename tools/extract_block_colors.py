"""Average each block's top-face texture into a colour the preview can use.

The renderer needs one RGB value per block. That table used to be typed out by
hand, so it lagged behind the blocks packs actually reach for, and a block that
was missing rendered magenta. This reads the real textures out of the client jar
instead.

    python tools/extract_block_colors.py [--version 26.2]

It downloads the client jar into .runtime/ (about 26 MB, cached) and writes
vanilla/<version>/block_colors.json. Only the averages are kept, a few bytes per
block; no Mojang asset is redistributed.

Blocks whose texture is greyscale and tinted at runtime -- grass, foliage, water
-- come out grey, which is why worldsmith/colors.py keeps a small hand-tuned
table layered on top.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worldsmith.play import RUNTIME, VERSION_MANIFEST, _download  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# For a map seen from above the top face is what matters. cube_all calls it
# "all", cube_bottom_top "top", a log's top is its "end".
FACE_KEYS = ("top", "up", "all", "end", "particle", "side", "texture", "cross", "layer0")


def client_jar(version: str) -> Path:
    jar = RUNTIME / f"client-{version}.jar"
    if jar.is_file():
        return jar
    import urllib.request
    with urllib.request.urlopen(VERSION_MANIFEST) as response:
        manifest = json.load(response)
    entry = next((v for v in manifest["versions"] if v["id"] == version), None)
    if entry is None:
        raise SystemExit(f"Minecraft {version} is not in the version manifest")
    with urllib.request.urlopen(entry["url"]) as response:
        meta = json.load(response)
    return _download(meta["downloads"]["client"]["url"], jar, f"Minecraft {version} client")


class Assets:
    def __init__(self, jar: Path):
        self.zip = zipfile.ZipFile(jar)
        self.names = set(self.zip.namelist())

    def json_at(self, path: str):
        return json.loads(self.zip.read(path)) if path in self.names else None

    def model(self, ident: str):
        return self.json_at(f"assets/minecraft/models/{ident.split(':')[-1]}.json")

    def textures_for(self, ident: str, depth: int = 0) -> dict:
        """Merge a model's textures with its parents'. Children win."""
        model = self.model(ident)
        if model is None or depth > 8:
            return {}
        inherited = self.textures_for(model["parent"], depth + 1) if "parent" in model else {}
        return {**inherited, **(model.get("textures") or {})}

    def image(self, texture: str) -> np.ndarray | None:
        path = f"assets/minecraft/textures/{texture.split(':')[-1]}.png"
        if path not in self.names:
            return None
        with self.zip.open(path) as handle:
            img = Image.open(handle).convert("RGBA")
            data = np.asarray(img, dtype=np.float64)
        # animated textures are a vertical strip of square frames; take the first
        if data.shape[0] > data.shape[1] and data.shape[0] % data.shape[1] == 0:
            data = data[:data.shape[1]]
        return data


def model_of(assets: Assets, block: str) -> str | None:
    """The model a block shows in its default state."""
    states = assets.json_at(f"assets/minecraft/blockstates/{block}.json")
    if states is None:
        return None
    candidates = []
    for variant in (states.get("variants") or {}).values():
        candidates.append(variant)
    for part in states.get("multipart") or []:
        candidates.append(part.get("apply"))
    for entry in candidates:
        if isinstance(entry, list):
            entry = entry[0] if entry else None
        if isinstance(entry, dict) and "model" in entry:
            return entry["model"]
    return None


def average(data: np.ndarray) -> tuple[int, int, int] | None:
    alpha = data[..., 3]
    opaque = alpha > 16
    if not opaque.any():
        return None
    rgb = data[..., :3][opaque]
    return tuple(int(round(v)) for v in rgb.mean(axis=0))


def colour_of(assets: Assets, block: str) -> tuple[int, int, int] | None:
    model = model_of(assets, block)
    if model is None:
        return None
    textures = assets.textures_for(model)
    if not textures:
        return None
    ordered = [k for k in FACE_KEYS if k in textures] + \
              [k for k in textures if k not in FACE_KEYS]
    for key in ordered:
        value = textures[key]
        for _ in range(4):
            # "#all" points at another key, and 26.2 may wrap the id in an
            # object: {"sprite": "minecraft:block/glass", "force_translucent": true}
            if isinstance(value, dict):
                value = value.get("sprite")
            elif isinstance(value, str) and value.startswith("#"):
                value = textures.get(value[1:])
            else:
                break
        if not isinstance(value, str):
            continue
        data = assets.image(value)
        if data is None:
            continue
        rgb = average(data)
        if rgb is not None:
            return rgb
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="26.2")
    args = ap.parse_args()

    blocks = json.loads((ROOT / "vanilla" / args.version / "blocks.json")
                        .read_text(encoding="utf-8"))["blocks"]
    assets = Assets(client_jar(args.version))

    colours, missed = {}, []
    for ident in blocks:
        rgb = colour_of(assets, ident.split(":")[-1])
        if rgb is None:
            missed.append(ident)
        else:
            colours[ident.split(":")[-1]] = list(rgb)

    out = ROOT / "vanilla" / args.version / "block_colors.json"
    out.write_text(json.dumps(colours, indent=0, sort_keys=True), encoding="utf-8")
    print(f"{len(colours)}/{len(blocks)} blocks -> {out.relative_to(ROOT)}")
    if missed:
        print(f"no texture found for {len(missed)}: {', '.join(m.split(':')[-1] for m in missed[:12])}"
              + (" ..." if len(missed) > 12 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
