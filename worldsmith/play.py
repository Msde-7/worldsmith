"""`worldsmith play`: from a datapack to standing in the world, in one command.

Everything a person would otherwise do by hand:

  * work out which dimension to use, and rewrite it as the overworld so a new
    world simply is that terrain
  * fetch a matching Java runtime and server jar the first time (cached)
  * pick a spawn point worth spawning at, a flat place to stand with the most
    dramatic thing in the world in view, using the renderer rather than guesswork
  * generate the world, pre-build the area around spawn, set the spawn point,
    creative mode, cheats on, clear weather, midday
  * install it into .minecraft/saves under a readable name, and drop the pack
    into .minecraft/datapacks so new worlds can use it too
  * open the launcher and say where to look

Running the server writes an `eula.txt`, which accepts Mojang's EULA; it happens
inside worldsmith's own cache directory.
"""
from __future__ import annotations

import gzip
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pack import export_zip
from .registry import Registries
from .terrain import sample_terrain
from .world import World

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime"
VERSION_MANIFEST = "https://launchermeta.mojang.com/mc/game/version_manifest_v2.json"
ADOPTIUM = ("https://api.adoptium.net/v3/binary/latest/{major}/ga/{os}/{arch}/jre/"
            "hotspot/normal/eclipse")

WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"


def log(message: str = "") -> None:
    print(message, flush=True)


def adoptium_url(major: int) -> str:
    """Adoptium names the platform in the path, so pick the right build."""
    system = "windows" if WINDOWS else "mac" if MACOS else "linux"
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x64"
    return ADOPTIUM.format(major=major, os=system, arch=arch)


def minecraft_dir() -> Path:
    """Where the vanilla launcher keeps saves, datapacks and screenshots."""
    if WINDOWS:
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / ".minecraft"
    if MACOS:
        return Path.home() / "Library" / "Application Support" / "minecraft"
    return Path.home() / ".minecraft"


@dataclass
class Runtime:
    java: Path
    jar: Path
    version: str


def _download(url: str, dest: Path, label: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"    downloading {label} ...")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    log(f"    {dest.name}  {dest.stat().st_size / 1048576:.0f} MB")
    return dest


def ensure_runtime(version: str) -> Runtime:
    """Server jar for `version` plus a Java runtime new enough to run it."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    jar = RUNTIME / f"server-{version}.jar"
    meta_path = RUNTIME / f"version-{version}.json"

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(VERSION_MANIFEST) as response:
            manifest = json.load(response)
        entry = next((v for v in manifest["versions"] if v["id"] == version), None)
        if entry is None:
            raise SystemExit(f"Minecraft {version} is not in the version manifest")
        with urllib.request.urlopen(entry["url"]) as response:
            meta = json.load(response)
        meta_path.write_text(json.dumps({
            "downloads": {"server": meta["downloads"]["server"]},
            "javaVersion": meta.get("javaVersion", {}),
        }), encoding="utf-8")

    if not jar.is_file():
        _download(meta["downloads"]["server"]["url"], jar, f"Minecraft {version} server")

    major = int((meta.get("javaVersion") or {}).get("majorVersion", 21))
    java = _ensure_java(major)
    return Runtime(java=java, jar=jar, version=version)


def _java_binaries(root: Path) -> list[Path]:
    """java under an unpacked runtime. macOS buries it in Contents/Home."""
    return sorted(root.rglob("java.exe" if WINDOWS else "java"))


def _ensure_java(major: int) -> Path:
    target = RUNTIME / f"jre{major}"
    found = _java_binaries(target)
    if found:
        return found[0]
    system = shutil.which("java")
    if system and _java_major(Path(system)) >= major:
        return Path(system)
    # unpack_archive picks the format from the suffix, so name it accordingly
    archive = RUNTIME / (f"jre{major}.zip" if WINDOWS else f"jre{major}.tar.gz")
    if not archive.is_file():
        _download(adoptium_url(major), archive, f"Java {major} runtime")
    if not target.is_dir():
        shutil.unpack_archive(archive, target)
    found = _java_binaries(target)
    if not found:
        raise SystemExit(f"no java binary under {target}")
    if not WINDOWS:
        found[0].chmod(0o755)
    return found[0]


def _java_major(java: Path) -> int:
    try:
        out = subprocess.run([str(java), "-version"], capture_output=True, text=True, timeout=20)
        text = (out.stderr or "") + (out.stdout or "")
        token = text.split('"')[1]
        return int(token.split(".")[0]) if not token.startswith("1.") else int(token.split(".")[1])
    except Exception:
        return 0


@dataclass
class Viewpoint:
    x: int
    z: int
    y: int
    landmark: tuple[int, int, int] | None
    note: str


def pick_viewpoint(world: World, span: int = 1536, step: int = 8) -> Viewpoint:
    """Somewhere worth spawning: a flat, solid place to stand with the biggest
    thing in the neighbourhood in view.

    Scored on the relief within ~160 blocks rather than raw height, so it works
    for spires, canyons and floating islands alike.
    """
    n = span // step
    terrain = sample_terrain(world, -span // 2, -span // 2, n, n, step=step)
    height = terrain.surface_y.astype(np.float64)
    solid = terrain.solid_anywhere
    sea = world.sea_level

    xs, zs = terrain.xs, terrain.zs
    standable = solid & (terrain.surface_y >= sea)
    if not standable.any():                        # a world with no dry land
        standable = solid
    if not standable.any():
        return Viewpoint(0, world.sea_level + 1, 64, None, "this world has no terrain near the origin")

    def window(values, radius_cells, reducer):
        pad = np.pad(values, radius_cells, mode="edge")
        stack = [pad[a:a + values.shape[0], b:b + values.shape[1]]
                 for a in range(2 * radius_cells + 1) for b in range(2 * radius_cells + 1)]
        return reducer(np.stack(stack), axis=0)

    flat_radius = max(1, 12 // step)
    local_hi = window(height, flat_radius, np.max)
    local_lo = window(height, flat_radius, np.min)
    flatness = local_hi - local_lo                       # small = level ground

    drama_radius = max(2, 160 // step)
    drama = window(np.where(solid, height, np.nan), drama_radius,
                   np.nanmax) - window(np.where(solid, height, np.nan), drama_radius, np.nanmin)
    drama = np.nan_to_num(drama)

    centre = np.hypot(*np.meshgrid(xs, zs)[::-1])
    good = standable & (flatness <= 3)
    if not good.any():
        good = standable & (flatness <= 6)
    if not good.any():
        good = standable

    score = np.where(good, drama - centre / 40.0, -1e9)
    index = int(np.argmax(score))
    zi, xi = np.unravel_index(index, height.shape)
    sx, sz = int(xs[xi]), int(zs[zi])

    # the landmark to face: tallest solid column within ~200 blocks
    reach = max(1, 200 // step)
    z0, z1 = max(0, zi - reach), min(height.shape[0], zi + reach + 1)
    x0, x1 = max(0, xi - reach), min(height.shape[1], xi + reach + 1)
    patch = np.where(solid[z0:z1, x0:x1], height[z0:z1, x0:x1], -1e9)
    lz, lx = np.unravel_index(int(np.argmax(patch)), patch.shape)
    landmark = (int(xs[x0 + lx]), int(zs[z0 + lz]), int(patch[lz, lx]))

    exact = sample_terrain(world, sx, sz, 1, 1, step=1)
    stand_y = int(exact.surface_y[0, 0]) + 1
    if not bool(exact.solid_anywhere[0, 0]):
        stand_y = max(sea + 1, int(height[zi, xi]) + 1)

    rise = landmark[2] - (stand_y - 1)
    distance = int(np.hypot(landmark[0] - sx, landmark[1] - sz))
    note = (f"{rise} blocks of relief {distance} blocks away" if rise > 12
            else f"terrain around y {stand_y - 1}")
    return Viewpoint(sx, sz, stand_y, landmark, note)


def viewpoint_at_build(world, source, registries, structure_id: str, seed: int,
                       reach: int = 4096) -> Viewpoint | None:
    """Spawn beside a build rather than at the most dramatic terrain.

    The placement model knows where builds land before the world is generated,
    so this needs no second pass over the finished world.
    """
    from .placement import set_reports

    structure = registries.get("structure", structure_id)
    if structure is None:
        raise SystemExit(f"unknown structure {structure_id}")
    owner = next((ident for ident in registries.ids("structure_set")
                  if any(e.get("structure") == structure_id
                         for e in (registries.get("structure_set", ident) or {}).get("structures") or [])),
                 None)
    if owner is None:
        raise SystemExit(f"no structure set places {structure_id}")
    reports = [r for r in set_reports(registries, world, source, owner, seed,
                                      -reach, -reach, reach, reach)
               if r.build == structure_id]
    kept = [r for r in reports if r.accepted]
    if not kept:
        return None
    best = min(kept, key=lambda r: abs(r.box[0]) + abs(r.box[1]))
    x = (best.box[0] + best.box[2]) // 2
    z = best.box[3] + 4                      # just outside one edge, looking in
    ground = sample_terrain(world, x, z, 1, 1, step=1)
    y = int(ground.surface_y[0, 0]) + 1
    middle = ((best.box[0] + best.box[2]) // 2, (best.box[1] + best.box[3]) // 2)
    return Viewpoint(x, z, y, (middle[0], middle[1], best.surface_y + 8),
                     f"beside {structure_id.split(':')[-1]}")


def as_overworld(pack: Path, dimension_id: str, registries: Registries, work: Path) -> Path:
    """Copy the pack with its dimension also written as minecraft:overworld, so a
    new world simply is this terrain."""
    staged = work / pack.name
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(pack, staged)
    dimension = registries.get("dimension", dimension_id)
    target = staged / "data" / "minecraft" / "dimension" / "overworld.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dimension, indent=2) + "\n", encoding="utf-8")
    return staged


def _patch_byte(raw: bytes, name: bytes, value: int) -> bytes:
    marker = bytes([1]) + struct.pack(">H", len(name)) + name
    index = raw.find(marker)
    if index < 0:
        return raw
    patched = bytearray(raw)
    patched[index + len(marker)] = value
    return bytes(patched)


def _patch_string(raw: bytes, name: bytes, value: str) -> bytes:
    marker = bytes([8]) + struct.pack(">H", len(name)) + name
    index = raw.find(marker)
    if index < 0:
        return raw
    at = index + len(marker)
    old_len = struct.unpack(">H", raw[at:at + 2])[0]
    encoded = value.encode("utf-8")
    return raw[:at] + struct.pack(">H", len(encoded)) + encoded + raw[at + 2 + old_len:]


def finish_level_dat(level_dat: Path, name: str) -> None:
    """Name the world and turn on cheats, so /tp works, without writing a whole
    NBT serialiser: both tags carry their own length."""
    raw = gzip.decompress(level_dat.read_bytes())
    raw = _patch_string(raw, b"LevelName", name)
    raw = _patch_byte(raw, b"allowCommands", 1)
    level_dat.write_bytes(gzip.compress(raw))


def server_properties(seed: int, gamemode: str) -> str:
    """Settings for the throwaway server that builds the world.

    `generate-structures` must be on. The server bakes it into the world at
    creation as generate_features, so a world built with it off has no village,
    temple, monument or stronghold in it and never will, whatever the pack's
    biome tags say. tools/verify_in_game.py deliberately turns it off instead:
    structures reshape the ground, which would wreck a heightmap comparison.
    """
    return "\n".join([
        f"level-seed={seed}", "level-name=world", "online-mode=false", "max-tick-time=-1",
        "sync-chunk-writes=true", f"gamemode={gamemode}", "difficulty=peaceful",
        "generate-structures=true", "spawn-protection=0", "view-distance=10",
        "simulation-distance=4", "allow-nether=false", "",
    ])


def generate_world(runtime: Runtime, work: Path, pack: Path, seed: int,
                   spawn: Viewpoint, radius: int, gamemode: str, pregen_seconds: int) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    (work / "eula.txt").write_text("eula=true\n", encoding="utf-8")
    (work / "server.properties").write_text(server_properties(seed, gamemode), encoding="utf-8")
    datapacks = work / "world" / "datapacks"
    datapacks.mkdir(parents=True, exist_ok=True)
    target = datapacks / pack.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(pack, target)

    proc = subprocess.Popen([str(runtime.java), "-Xmx2G", "-jar", str(runtime.jar), "nogui"],
                            cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    ready = False
    started = time.time()
    for line in proc.stdout:
        lines.append(line.rstrip())
        low = line.lower()
        if "failed to load" in low or "exception" in low:
            log("    " + line.rstrip())
        if 'for help, type "help"' in low:
            ready = True
            break
        if time.time() - started > 600:
            break
    if not ready:
        proc.kill()
        (work / "server.log").write_text("\n".join(lines), encoding="utf-8")
        raise SystemExit("the server never finished starting; see " + str(work / "server.log"))

    threading.Thread(target=lambda: [lines.append(t.rstrip()) for t in proc.stdout],
                     daemon=True).start()

    def send(command: str):
        proc.stdin.write(command + "\n")
        proc.stdin.flush()

    send(f"setworldspawn {spawn.x} {spawn.y} {spawn.z}")
    send("gamerule keepInventory true")
    send("time set day")
    send("weather clear 1000000")

    if pregen_seconds > 0 and radius > 0:
        # forceload brings chunks all the way to `full`, but takes at most 256
        # chunks per command, and that is 256 chunks, so tiles have to be
        # chunk-aligned or a 256-block square straddles 17 chunks and is refused.
        tiles = 0
        x_start = ((spawn.x - radius) // 16) * 16
        z_start = ((spawn.z - radius) // 16) * 16
        for z0 in range(z_start, spawn.z + radius, 256):
            for x0 in range(x_start, spawn.x + radius, 256):
                send(f"forceload add {x0} {z0} {x0 + 255} {z0 + 255}")
                tiles += 1
        log(f"    building up to {tiles * 256} chunks around spawn ({pregen_seconds}s budget)")
        region_dir = work / "world" / "dimensions" / "minecraft" / "overworld" / "region"
        deadline = time.time() + pregen_seconds
        last, stable = -1, 0
        while time.time() < deadline:
            time.sleep(5)
            send("save-all flush")
            time.sleep(3)
            size = sum(f.stat().st_size for f in region_dir.glob("*.mca")) if region_dir.is_dir() else 0
            if size == last and size > 0:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            last = size
        send("forceload remove all")

    send("save-all flush")
    time.sleep(4)
    send("stop")
    try:
        proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        proc.kill()
    (work / "server.log").write_text("\n".join(lines), encoding="utf-8")
    return work / "world"


def install_world(world_dir: Path, name: str) -> Path:
    saves = minecraft_dir() / "saves"
    saves.mkdir(parents=True, exist_ok=True)
    folder = "".join(c for c in name if c.isalnum() or c in " -_").strip() or "worldsmith"
    dest = saves / folder
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(world_dir, dest, ignore=shutil.ignore_patterns("session.lock"))
    finish_level_dat(dest / "level.dat", name)
    return dest


def install_datapack(pack: Path) -> Path | None:
    """Also drop the pack in .minecraft/datapacks so it shows up on the
    world-creation screen for new worlds."""
    folder = minecraft_dir() / "datapacks"
    folder.mkdir(parents=True, exist_ok=True)
    return export_zip(pack, folder / f"{pack.name}.zip")


def launcher_candidates() -> list[Path]:
    if WINDOWS:
        return [
            Path(r"C:\XboxGames\Minecraft Launcher\Content\Minecraft.exe"),
            Path(r"C:\Program Files (x86)\Minecraft Launcher\MinecraftLauncher.exe"),
            (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Minecraft Launcher"
             / "MinecraftLauncher.exe"),
        ]
    if MACOS:
        return [Path("/Applications/Minecraft.app"),
                Path.home() / "Applications" / "Minecraft.app"]
    return [Path("/usr/bin/minecraft-launcher"), Path("/opt/minecraft-launcher/minecraft-launcher"),
            Path("/var/lib/flatpak/exports/bin/com.mojang.Minecraft")]


def launch_minecraft() -> str | None:
    for path in launcher_candidates():
        if MACOS and path.is_dir():
            subprocess.Popen(["open", "-a", str(path)])
            return str(path)
        if path.is_file():
            if WINDOWS:
                subprocess.Popen([str(path)],
                                 creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
            else:
                subprocess.Popen([str(path)], start_new_session=True)
            return str(path)
    # the launcher registers a minecraft:// URL handler on every platform
    opener = ("start" if WINDOWS else "open" if MACOS else "xdg-open")
    try:
        if WINDOWS:
            os.startfile("minecraft://")
        else:
            subprocess.Popen([opener, "minecraft://"], start_new_session=True)
        return "minecraft:// handler"
    except (OSError, FileNotFoundError):
        return None
