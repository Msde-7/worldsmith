"""The RandomState equivalent: seeds every noise and compiles the noise router.

This is where a world seed turns into concrete noise objects. Each noise is
seeded from xoroshiro(seed).forkPositional().fromHashOf(<full noise id>), which
is why renaming a noise changes the terrain even at the same seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .density import DensityCompiler, Node, Const
from .jrandom import JavaRandom, Xoroshiro
from .noise import BlendedNoise, NormalNoise
from .registry import Registries

ROUTER_FIELDS = (
    "barrier", "fluid_level_floodedness", "fluid_level_spread", "lava",
    "temperature", "vegetation", "continents", "erosion", "depth", "ridges",
    "preliminary_surface_level", "final_density", "vein_toggle", "vein_ridged", "vein_gap",
)

# Noise parameters the game hard-codes rather than reading from data.
BUILTIN_NOISE = {
    "minecraft:surface": (-6, [1.0, 1.0, 1.0]),
    "minecraft:surface_secondary": (-6, [1.0, 1.0, 0.0, 1.0]),
    "minecraft:clay_bands_offset": (-8, [1.0]),
    "minecraft:badlands_pillar": (-2, [1.0, 1.0, 1.0, 1.0]),
    "minecraft:badlands_pillar_roof": (-8, [1.0]),
    "minecraft:badlands_surface": (-6, [1.0, 1.0, 1.0]),
    "minecraft:iceberg_pillar": (-6, [1.0, 1.0, 1.0, 1.0]),
    "minecraft:iceberg_pillar_roof": (-3, [1.0]),
    "minecraft:iceberg_surface": (-6, [1.0, 1.0, 1.0]),
}


class MissingEntry(Exception):
    pass


@dataclass
class NoiseSettings:
    min_y: int = -64
    height: int = 384
    size_horizontal: int = 1
    size_vertical: int = 2

    @property
    def cell_width(self) -> int:
        return self.size_horizontal * 4

    @property
    def cell_height(self) -> int:
        return self.size_vertical * 4

    @property
    def max_y(self) -> int:
        return self.min_y + self.height


@dataclass
class World:
    """A seeded, compiled dimension: router + surface rule + block palette."""

    registries: Registries
    settings_id: str
    seed: int
    noise: NoiseSettings
    sea_level: int
    default_block: str
    default_fluid: str
    legacy_random_source: bool
    aquifers_enabled: bool = False
    router: dict[str, Node] = field(default_factory=dict)
    surface_rule: dict | None = None
    compiler: DensityCompiler | None = None
    _noise_cache: dict = field(default_factory=dict)

    # The DensityCompiler environment.
    @property
    def cell_width(self) -> int:
        return self.noise.cell_width

    @property
    def cell_height(self) -> int:
        return self.noise.cell_height

    @property
    def max_y(self) -> int:
        return self.noise.max_y

    def get_density(self, ident: str):
        return self.registries.get("density_function", ident)

    def get_noise(self, ident) -> NormalNoise | None:
        """ident is either an id string or an inline {firstOctave, amplitudes}."""
        if isinstance(ident, dict):
            return NormalNoise(self._random_for("inline"), int(ident.get("firstOctave", 0)),
                               [float(a) for a in ident.get("amplitudes", [])])
        if ":" not in ident:
            ident = "minecraft:" + ident
        if ident in self._noise_cache:
            return self._noise_cache[ident]
        params = self.registries.get("noise", ident)
        if params is None:
            if ident in BUILTIN_NOISE:
                first_octave, amplitudes = BUILTIN_NOISE[ident]
            else:
                raise MissingEntry(f"unknown noise: {ident}")
        else:
            first_octave = int(params.get("firstOctave", 0))
            amplitudes = [float(a) for a in params.get("amplitudes", [])]

        if self.legacy_random_source:
            if ident == "minecraft:temperature":
                noise = NormalNoise(JavaRandom(self.seed), -7, [1.0, 1.0])
                self._noise_cache[ident] = noise
                return noise
            if ident == "minecraft:vegetation":
                noise = NormalNoise(JavaRandom(self.seed + 1), -7, [1.0, 1.0])
                self._noise_cache[ident] = noise
                return noise
            if ident == "minecraft:offset":
                noise = NormalNoise(self._positional().from_hash_of("offset"), 0, [0.0])
                self._noise_cache[ident] = noise
                return noise

        noise = NormalNoise(self._random_for(ident), first_octave, amplitudes)
        self._noise_cache[ident] = noise
        return noise

    def get_blended_noise(self, xz_scale, y_scale, xz_factor, y_factor, smear) -> BlendedNoise:
        key = ("blended", xz_scale, y_scale, xz_factor, y_factor, smear)
        if key not in self._noise_cache:
            random = (JavaRandom(self.seed) if self.legacy_random_source
                      else self._positional().from_hash_of("minecraft:terrain"))
            self._noise_cache[key] = BlendedNoise(random, xz_scale, y_scale, xz_factor, y_factor, smear)
        return self._noise_cache[key]

    def _positional(self):
        base = JavaRandom(self.seed) if self.legacy_random_source else Xoroshiro.create(self.seed)
        return base.fork_positional()

    def _random_for(self, ident: str):
        return self._positional().from_hash_of(ident)

    @classmethod
    def create(cls, registries: Registries, settings_id: str, seed: int) -> "World":
        settings = registries.get("noise_settings", settings_id)
        if settings is None:
            raise MissingEntry(f"unknown noise_settings: {settings_id} "
                               f"(have: {', '.join(registries.ids('noise_settings')[:8])}...)")
        noise_cfg = settings.get("noise") or {}
        noise = NoiseSettings(
            min_y=int(noise_cfg.get("min_y", -64)),
            height=int(noise_cfg.get("height", 384)),
            size_horizontal=int(noise_cfg.get("size_horizontal", 1)),
            size_vertical=int(noise_cfg.get("size_vertical", 2)),
        )
        world = cls(
            registries=registries,
            settings_id=settings_id,
            seed=seed,
            noise=noise,
            sea_level=int(settings.get("sea_level", 63)),
            default_block=(settings.get("default_block") or {}).get("Name", "minecraft:stone"),
            default_fluid=(settings.get("default_fluid") or {}).get("Name", "minecraft:water"),
            legacy_random_source=bool(settings.get("legacy_random_source", False)),
            aquifers_enabled=bool(settings.get("aquifers_enabled", False)),
            surface_rule=settings.get("surface_rule"),
        )
        world.compiler = DensityCompiler(world)
        router_json = settings.get("noise_router") or {}
        for name in ROUTER_FIELDS:
            entry = router_json.get(name)
            world.router[name] = world.compiler.compile(entry) if entry is not None else Const(0.0)
        return world
