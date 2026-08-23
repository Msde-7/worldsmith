"""Images. This is the feedback loop: terrain you can look at.

Four views, because they answer different questions:

  map      surface blocks with hillshading. What does it look like?
  height   elevation ramp with contour bands. How tall, how steep?
  biomes   climate assignment. Is anything placed where I meant?
  section  vertical slice. Are there overhangs, is it hollow?
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colors import biome_color, block_color, grass_color, palette_rgb, parse_hex
from .scene import Scene, biome_climate_table
from .terrain import Terrain, cross_section
from .world import World


def _upscale(img: Image.Image, scale: int) -> Image.Image:
    if scale <= 1:
        return img
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def hillshade(height: np.ndarray, step: int, azimuth: float = 315.0, altitude: float = 45.0,
              exaggeration: float = 1.6) -> np.ndarray:
    """Classic Lambert hillshade; returns a multiplier around 1.0."""
    h = height.astype(np.float64) * exaggeration
    dy, dx = np.gradient(h, max(1, step))
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    shade = (np.sin(alt) * np.cos(slope) +
             np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(0.55 + 0.75 * shade, 0.25, 1.65)


def _grass_tint(scene: Scene) -> np.ndarray:
    """Per-pixel grass tint from the biome's temperature and downfall."""
    table = biome_climate_table(scene.world, scene.biomes)
    colors = np.array([grass_color(t, d) for t, d in table], dtype=np.float64)
    return colors[scene.biome_index]


def render_map(scene: Scene, scale: int = 4, shade: bool = True, tint_grass: bool = True,
               water_depth_shading: bool = True) -> Image.Image:
    palette = palette_rgb(scene.palette).astype(np.float64)
    rgb = palette[scene.surface_block]

    if tint_grass:
        grassy = np.array([name.split(":")[-1] in ("grass_block", "moss_block", "mycelium", "podzol")
                           for name in scene.palette], dtype=bool)
        mask = grassy[scene.surface_block]
        if mask.any():
            rgb[mask] = _grass_tint(scene)[mask]

    water = np.array([name.split(":")[-1] in ("water", "flowing_water") for name in scene.palette],
                     dtype=bool)[scene.surface_block]
    if water.any():
        # the sea takes the biome's own water_color, as it does in game
        tints = np.array([parse_hex((scene.world.registries.get("biome", b) or {})
                                    .get("effects", {}).get("water_color"))
                          for b in scene.biomes], dtype=np.float64)
        shallow = tints[scene.biome_index]
        deep = shallow * 0.32 + np.array([8, 16, 40]) * 0.68
        if water_depth_shading:
            depth = scene.terrain.water_depth().astype(np.float64)
            t = np.clip(depth / 40.0, 0.0, 1.0)[..., None]
        else:
            t = 0.0
        rgb[water] = ((1 - t) * shallow + t * deep)[water]

    if shade:
        h = scene.terrain.surface_y.astype(np.float64)
        # flatten the sea so hillshade shows the seabed only faintly
        h = np.maximum(h, scene.world.sea_level - 3)
        rgb = rgb * hillshade(h, scene.terrain.step)[..., None]

    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    return _upscale(img, scale)


def render_height(terrain: Terrain, scale: int = 4, contour_every: int = 16) -> Image.Image:
    h = terrain.surface_y.astype(np.float64)
    valid = terrain.solid_anywhere
    lo = float(h[valid].min()) if valid.any() else 0.0
    hi = float(h[valid].max()) if valid.any() else 1.0
    span = max(1.0, hi - lo)
    t = np.clip((h - lo) / span, 0.0, 1.0)

    # blue -> teal -> green -> tan -> brown -> white
    stops = np.array([
        [20, 40, 110], [30, 110, 170], [60, 150, 120], [120, 180, 90],
        [210, 200, 130], [160, 120, 80], [120, 90, 70], [250, 250, 255],
    ], dtype=np.float64)
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(t, pos, stops[:, i]) for i in range(3)], axis=-1)

    if contour_every > 0:
        band = (np.floor(h / contour_every) % 2 == 0)
        rgb[band] *= 0.92
        edge = np.zeros_like(band)
        edge[1:, :] |= (np.floor(h[1:] / contour_every) != np.floor(h[:-1] / contour_every))
        edge[:, 1:] |= (np.floor(h[:, 1:] / contour_every) != np.floor(h[:, :-1] / contour_every))
        rgb[edge] *= 0.7

    sea = terrain.surface_y < terrain.sea_level
    rgb[sea] = rgb[sea] * 0.55 + np.array([20, 40, 110]) * 0.45
    rgb[~valid] = [10, 10, 14]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    return _upscale(img, scale)


def render_biomes(scene: Scene, scale: int = 4, shade: bool = True) -> Image.Image:
    colors = np.array([biome_color(name, i) for i, name in enumerate(scene.biomes)], dtype=np.float64)
    rgb = colors[scene.biome_index]
    if shade:
        rgb = rgb * hillshade(np.maximum(scene.terrain.surface_y.astype(float),
                                         scene.world.sea_level - 3), scene.terrain.step,
                              exaggeration=1.0)[..., None]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    return _upscale(img, scale)


def render_section(world: World, x0: int, z: int, length: int, scale: int = 2,
                   step: int = 1, y_step: int = 1, sea_level: int | None = None) -> Image.Image:
    xs, ys, solid = cross_section(world, x0, z, length, step=step, y_step=y_step)
    sea = world.sea_level if sea_level is None else sea_level
    h, w = solid.shape
    rgb = np.zeros((h, w, 3), dtype=np.float64)

    stone = np.array(block_color("stone"), dtype=np.float64)
    deep = np.array(block_color("deepslate"), dtype=np.float64)
    water = np.array(block_color("water"), dtype=np.float64)
    sky = np.array([26, 30, 40], dtype=np.float64)
    grid = np.array([46, 52, 66], dtype=np.float64)

    yy = ys[:, None] * np.ones((1, w))
    depth_t = np.clip((0 - yy) / 64.0, 0.0, 1.0)[..., None]
    rock = stone * (1 - depth_t) + deep * depth_t
    rgb[:] = sky
    below_sea = (yy < sea)
    rgb[below_sea & ~solid] = water
    rgb[solid] = rock[solid]

    # surface highlight: solid block with air directly above
    above = np.zeros_like(solid)
    above[:-1, :] = solid[:-1, :] & ~solid[1:, :]
    rgb[above] = np.minimum(255.0, rgb[above] * 1.35 + 26.0)

    for level in range(int(np.ceil(ys[0] / 32.0)) * 32, int(ys[-1]) + 1, 32):
        row = np.searchsorted(ys, level)
        if 0 <= row < h and level != 0:
            rgb[row] = rgb[row] * 0.6 + grid * 0.4
    zero = np.searchsorted(ys, 0)
    if 0 <= zero < h:
        rgb[zero] = rgb[zero] * 0.4 + np.array([90, 100, 120]) * 0.6

    img = Image.fromarray(np.clip(rgb[::-1], 0, 255).astype(np.uint8), "RGB")
    return _upscale(img, scale)


def _font(size: int = 12):
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(images: list[tuple[str, Image.Image]], columns: int = 2,
                  pad: int = 12, title: str | None = None, footer: list[str] | None = None,
                  background=(18, 18, 22)) -> Image.Image:
    """Label and tile several views into one image, so one look answers "what did
    that change do?". A panel wider than one grid cell, usually the
    cross-section, gets a row of its own."""
    font = _font(13)
    title_font = _font(18)
    label_h = 20
    entries = [(e[0], e[1], (len(e) > 2 and e[2])) for e in images]
    normal = [(lbl, img) for lbl, img, w in entries if not w]
    wide = [(lbl, img) for lbl, img, w in entries if w]
    if not normal:                       # everything is wide: fall back to a stack
        normal, wide = wide, []
    cell_w = max(img.width for _, img in normal)
    cell_h = max((img.height for _, img in normal), default=0) + label_h
    rows = (len(normal) + columns - 1) // columns
    top = 34 if title else pad
    foot_h = (len(footer) * 16 + pad) if footer else 0
    width = max(columns * cell_w + (columns + 1) * pad,
                max((img.width for _, img in wide), default=0) + 2 * pad)
    height = top + rows * (cell_h + pad) + pad + foot_h
    height += sum(img.height + label_h + pad for _, img in wide)
    sheet = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad, 9), title, fill=(235, 235, 240), font=title_font)
    for i, (label, img) in enumerate(normal):
        r, c = divmod(i, columns)
        x = pad + c * (cell_w + pad)
        y = top + pad + r * (cell_h + pad)
        draw.text((x, y), label, fill=(190, 195, 205), font=font)
        sheet.paste(img, (x, y + label_h))
    y = top + rows * (cell_h + pad) + pad
    for label, img in wide:
        draw.text((pad, y), label, fill=(190, 195, 205), font=font)
        sheet.paste(img, (pad, y + label_h))
        y += img.height + label_h + pad
    if footer:
        for line in footer:
            draw.text((pad, y), line, fill=(165, 170, 180), font=font)
            y += 16
    return sheet
