"""Static checks for a worldgen datapack.

The failure mode this exists to prevent: Minecraft rejects a malformed worldgen
file silently at world creation. You get a flat void world, or the vanilla
overworld, with one line buried in the log. Everything here is a check that
would otherwise cost a world-creation round trip.

Findings come back as ERROR (the game will reject or ignore this), WARNING
(loads, but almost certainly not what was meant) or INFO.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from .climate import BiomeSource, unreachable_biomes
from .colors import is_missing
from .density import DENSITY_FIELDS, DENSITY_TYPES, REQUIRED_FIELDS
from .registry import Pack, Registries
from .surface import SURFACE_CONDITION_TYPES, SURFACE_RULE_TYPES
from .world import BUILTIN_NOISE, ROUTER_FIELDS

ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"

# 26.2 = pack_format 107. Anything older than 1.18 (format 8) has no
# data-driven worldgen at all.
KNOWN_PACK_FORMATS = {
    107: "26.2", 105: "26.1", 88: "1.21.11", 81: "1.21.9", 71: "1.21.8",
    61: "1.21.5", 57: "1.21.4", 48: "1.21.2", 41: "1.21", 26: "1.20.5",
    18: "1.20.2", 12: "1.19.4", 10: "1.19", 9: "1.18.2", 8: "1.18",
}


@dataclass
class Finding:
    level: str
    where: str
    message: str
    hint: str | None = None

    def format(self) -> str:
        head = f"[{self.level}] {self.where}: {self.message}"
        return head if not self.hint else head + f"\n         hint: {self.hint}"


class Validator:
    def __init__(self, registries: Registries, pack: Pack | None = None, version: str = "26.2"):
        self.registries = registries
        self.pack = pack
        self.version = version
        self.findings: list[Finding] = []
        blocks_file = Path(__file__).resolve().parent.parent / "vanilla" / version / "blocks.json"
        data = json.loads(blocks_file.read_text(encoding="utf-8")) if blocks_file.is_file() else {}
        self.block_ids = set(data.get("blocks", []))
        self.block_properties = data.get("properties", {})

    def add(self, level, where, message, hint=None):
        self.findings.append(Finding(level, where, message, hint))

    def has(self, category: str, ident: str) -> bool:
        return self.registries.get(category, ident) is not None

    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == ERROR]

    def report(self) -> str:
        if not self.findings:
            return "no findings"
        order = {ERROR: 0, WARNING: 1, INFO: 2}
        return "\n".join(f.format() for f in sorted(self.findings, key=lambda f: order[f.level]))

    def validate_pack(self, own_ids_only: bool = True) -> list[Finding]:
        if self.pack is not None:
            self.check_mcmeta()
        scope = self.pack.data if (self.pack is not None and own_ids_only) else self.registries.data
        for ident, obj in scope.get("noise", {}).items():
            self.check_noise_parameters(ident, obj)
        for ident, obj in scope.get("density_function", {}).items():
            self.check_density(f"density_function/{ident}", obj, depth=0)
        for ident, obj in scope.get("noise_settings", {}).items():
            self.check_noise_settings(ident, obj)
        for ident, obj in scope.get("dimension", {}).items():
            self.check_dimension(ident, obj)
        for ident, obj in scope.get("biome", {}).items():
            self.check_biome(ident, obj)
        for ident, obj in scope.get("biome_tag", {}).items():
            self.check_biome_tag(ident, obj)
        return self.findings

    def check_mcmeta(self):
        meta = self.pack.mcmeta
        if meta is None:
            self.add(ERROR, "pack.mcmeta", "missing",
                     'every datapack needs pack.mcmeta with {"pack": {"pack_format": N, "description": "..."}}')
            return
        pack = meta.get("pack") or {}
        fmt = pack.get("pack_format")
        if fmt is None:
            self.add(ERROR, "pack.mcmeta", "no pack_format")
        elif fmt not in KNOWN_PACK_FORMATS:
            expected = [k for k, v in KNOWN_PACK_FORMATS.items() if v == self.version]
            self.add(WARNING, "pack.mcmeta", f"pack_format {fmt} is not a version worldsmith knows",
                     f"{self.version} expects pack_format {expected}")
        elif KNOWN_PACK_FORMATS[fmt] != self.version:
            self.add(WARNING, "pack.mcmeta",
                     f"pack_format {fmt} is {KNOWN_PACK_FORMATS[fmt]} but validating against {self.version}")
        if isinstance(fmt, int) and fmt > 81:
            for key in ("min_format", "max_format"):
                value = pack.get(key)
                if value is None:
                    self.add(ERROR, "pack.mcmeta",
                             f"pack_format {fmt} is above 81, so '{key}' is mandatory",
                             'the game refuses the pack outright: "Pack declares support for version '
                             'newer than 81, but is missing mandatory fields min_format and max_format". '
                             f'Use "{key}": [{fmt}, 0].')
                elif not (isinstance(value, list) and len(value) == 2
                          and all(isinstance(v, int) for v in value)):
                    self.add(ERROR, "pack.mcmeta", f"'{key}' must be a [major, minor] pair of integers")
        if not self.pack.namespaces:
            self.add(ERROR, "data/", "no namespace directories under data/")

    def check_noise_parameters(self, ident, obj):
        where = f"noise/{ident}"
        if not isinstance(obj, dict):
            self.add(ERROR, where, "must be an object")
            return
        if "firstOctave" not in obj:
            self.add(ERROR, where, "missing 'firstOctave' (note the camelCase: it is the one "
                                   "camelCase field in all of worldgen)")
        amps = obj.get("amplitudes")
        if not isinstance(amps, list) or not amps:
            self.add(ERROR, where, "'amplitudes' must be a non-empty list")
            return
        if all(a == 0 for a in amps):
            self.add(WARNING, where, "every amplitude is 0, so this noise is constant 0")
        if amps[0] == 0:
            self.add(INFO, where, "leading amplitude is 0: the first octave is skipped, "
                                  "which changes the effective scale")

    def check_density(self, where, obj, depth: int, seen: tuple = ()):
        if depth > 96:
            self.add(ERROR, where, "density function nested more than 96 deep")
            return
        if isinstance(obj, (int, float)):
            return
        if isinstance(obj, str):
            ident = obj if ":" in obj else "minecraft:" + obj
            if not self.has("density_function", ident):
                self.add(ERROR, where, f"reference to unknown density function '{obj}'",
                         "check spelling and that the file exists under worldgen/density_function/")
            elif ident in seen:
                self.add(ERROR, where, f"reference cycle through '{obj}'")
            return
        if not isinstance(obj, dict):
            self.add(ERROR, where, f"expected a number, id or object, got {type(obj).__name__}")
            return
        raw = obj.get("type")
        if not isinstance(raw, str):
            self.add(ERROR, where, "object has no 'type'")
            return
        t = raw.split(":")[-1]
        if t not in DENSITY_TYPES:
            close = _closest(t, DENSITY_TYPES)
            self.add(ERROR, where, f"unknown density function type '{raw}'",
                     f"did you mean '{close}'?" if close else None)
            return
        allowed = DENSITY_FIELDS.get(t, {})
        for key in obj:
            if key != "type" and key not in allowed:
                close = _closest(key, allowed)
                self.add(ERROR, where, f"'{t}' has no field '{key}'",
                         f"did you mean '{close}'?" if close else f"allowed: {sorted(allowed) or 'none'}")
        for key in REQUIRED_FIELDS.get(t, ()):  # missing required
            if key not in obj:
                self.add(ERROR, where, f"'{t}' is missing required field '{key}'")

        # per-type semantics
        if t == "y_clamped_gradient":
            from_y, to_y = obj.get("from_y"), obj.get("to_y")
            if from_y is not None and from_y == to_y:
                self.add(ERROR, where, "from_y == to_y makes the gradient undefined")
        if t == "clamp":
            lo, hi = obj.get("min"), obj.get("max")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
                self.add(ERROR, where, f"clamp min ({lo}) > max ({hi})")
        if t == "range_choice":
            lo, hi = obj.get("min_inclusive"), obj.get("max_exclusive")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo >= hi:
                self.add(WARNING, where, "min_inclusive >= max_exclusive, so when_in_range never runs")
        if t == "interval_select":
            th, fn = obj.get("thresholds") or [], obj.get("functions") or []
            if len(fn) != len(th) + 1:
                self.add(ERROR, where,
                         f"interval_select needs len(functions) == len(thresholds) + 1 "
                         f"(got {len(fn)} functions, {len(th)} thresholds)")
            if list(th) != sorted(th):
                self.add(ERROR, where, "thresholds must be in ascending order")
        if t in ("noise", "shifted_noise", "shift", "shift_a", "shift_b", "weird_scaled_sampler"):
            key = "argument" if t.startswith("shift") and t != "shifted_noise" else "noise"
            ref = obj.get(key)
            if isinstance(ref, str):
                ident = ref if ":" in ref else "minecraft:" + ref
                if not self.has("noise", ident) and ident not in BUILTIN_NOISE:
                    self.add(ERROR, where, f"reference to unknown noise '{ref}'",
                             "noise parameter files live under worldgen/noise/")
        if t == "spline":
            self.check_spline(where + "/spline", obj.get("spline"), depth)
        if t == "old_blended_noise":
            for key in ("xz_factor", "y_factor"):
                if obj.get(key) == 0:
                    self.add(ERROR, where, f"{key} must not be 0 (division by zero)")

        for key, kind in allowed.items():
            if key not in obj:
                continue
            if kind == "D":
                self.check_density(f"{where}/{key}", obj[key], depth + 1, seen)
            elif kind == "D[]":
                for i, sub in enumerate(obj[key] or []):
                    self.check_density(f"{where}/{key}[{i}]", sub, depth + 1, seen)

    def check_spline(self, where, spline, depth):
        if isinstance(spline, (int, float)):
            return
        if not isinstance(spline, dict):
            self.add(ERROR, where, "spline must be a number or an object")
            return
        if "coordinate" not in spline:
            self.add(ERROR, where, "spline has no 'coordinate'")
        else:
            self.check_density(where + "/coordinate", spline["coordinate"], depth + 1)
        points = spline.get("points")
        if not isinstance(points, list) or not points:
            self.add(ERROR, where, "spline has no points")
            return
        last = None
        for i, point in enumerate(points):
            if not isinstance(point, dict):
                self.add(ERROR, f"{where}/points[{i}]", "point must be an object")
                continue
            loc = point.get("location")
            if not isinstance(loc, (int, float)):
                self.add(ERROR, f"{where}/points[{i}]", "point needs a numeric 'location'")
            else:
                if last is not None and loc <= last:
                    self.add(ERROR, f"{where}/points[{i}]",
                             f"locations must strictly increase (got {loc} after {last})")
                last = loc
            if "value" not in point:
                self.add(ERROR, f"{where}/points[{i}]", "point needs a 'value'")
            elif isinstance(point["value"], dict):
                self.check_spline(f"{where}/points[{i}]/value", point["value"], depth + 1)

    def check_noise_settings(self, ident, obj):
        where = f"noise_settings/{ident}"
        if not isinstance(obj, dict):
            self.add(ERROR, where, "must be an object")
            return
        noise = obj.get("noise")
        if not isinstance(noise, dict):
            self.add(ERROR, where, "missing 'noise' block with min_y/height/size_horizontal/size_vertical")
        else:
            min_y = noise.get("min_y")
            height = noise.get("height")
            if not isinstance(min_y, int) or min_y % 16 != 0:
                self.add(ERROR, where, f"noise.min_y must be a multiple of 16 (got {min_y})")
            if not isinstance(height, int) or height % 16 != 0:
                self.add(ERROR, where, f"noise.height must be a multiple of 16 (got {height})")
            if isinstance(min_y, int) and isinstance(height, int):
                if not (-2048 <= min_y <= 2031):
                    self.add(ERROR, where, f"noise.min_y {min_y} is outside the allowed [-2048, 2031]")
                if not (0 < height <= 4064):
                    self.add(ERROR, where, f"noise.height {height} is outside the allowed (0, 4064]")
                if min_y + height > 2032:
                    self.add(ERROR, where,
                             f"min_y + height = {min_y + height} exceeds the world top of 2032")
            for key in ("size_horizontal", "size_vertical"):
                v = noise.get(key)
                if not isinstance(v, int) or not (1 <= v <= 4):
                    self.add(ERROR, where, f"noise.{key} must be an integer 1..4 (got {v})")
        sea = obj.get("sea_level")
        if isinstance(noise, dict) and isinstance(sea, int) and isinstance(noise.get("min_y"), int):
            top = noise["min_y"] + noise.get("height", 0)
            if sea > top:
                self.add(WARNING, where, f"sea_level {sea} is above the world top ({top})")
            elif sea < noise["min_y"]:
                self.add(INFO, where, f"sea_level {sea} is below the world, so nothing floods "
                                      "(vanilla floating_islands does this on purpose)")
        for key in ("default_block", "default_fluid"):
            self.check_block_state(f"{where}/{key}", obj.get(key))
        router = obj.get("noise_router")
        if not isinstance(router, dict):
            self.add(ERROR, where, "missing 'noise_router'")
        else:
            for key in router:
                if key not in ROUTER_FIELDS:
                    close = _closest(key, ROUTER_FIELDS)
                    self.add(ERROR, f"{where}/noise_router",
                             f"unknown router field '{key}'",
                             f"did you mean '{close}'?" if close else None)
            for key in ROUTER_FIELDS:
                if key not in router:
                    self.add(ERROR, f"{where}/noise_router", f"missing required router field '{key}'",
                             "every field is mandatory; use 0 for the ones you do not need")
            for key, value in router.items():
                if key in ROUTER_FIELDS:
                    self.check_density(f"{where}/noise_router/{key}", value, 0)
        if obj.get("surface_rule") is None:
            self.add(WARNING, where, "no surface_rule: the whole surface will be default_block")
        else:
            self.check_surface_rule(f"{where}/surface_rule", obj["surface_rule"])

    def check_surface_rule(self, where, rule, depth=0):
        if depth > 64:
            self.add(ERROR, where, "surface rule nested more than 64 deep")
            return
        if not isinstance(rule, dict):
            self.add(ERROR, where, "surface rule must be an object")
            return
        raw = rule.get("type")
        if not isinstance(raw, str):
            self.add(ERROR, where, "surface rule has no 'type'")
            return
        t = raw.split(":")[-1]
        if t == "bandlands":
            return
        if t not in SURFACE_RULE_TYPES:
            close = _closest(t, SURFACE_RULE_TYPES | {"bandlands"})
            self.add(ERROR, where, f"unknown surface rule type '{raw}'",
                     f"did you mean '{close}'?" if close else "valid: sequence, condition, block, bandlands")
            return
        if t == "sequence":
            seq = rule.get("sequence")
            if not isinstance(seq, list):
                self.add(ERROR, where, "'sequence' must be a list")
                return
            if not seq:
                self.add(WARNING, where, "empty sequence")
            for i, sub in enumerate(seq):
                self.check_surface_rule(f"{where}/sequence[{i}]", sub, depth + 1)
        elif t == "condition":
            self.check_surface_condition(f"{where}/if_true", rule.get("if_true"))
            self.check_surface_rule(f"{where}/then_run", rule.get("then_run"), depth + 1)
        elif t == "block":
            self.check_block_state(f"{where}/result_state", rule.get("result_state"), required=True)

    def check_surface_condition(self, where, cond, depth=0):
        if not isinstance(cond, dict):
            self.add(ERROR, where, "condition must be an object")
            return
        raw = cond.get("type")
        if not isinstance(raw, str):
            self.add(ERROR, where, "condition has no 'type'")
            return
        t = raw.split(":")[-1]
        if t not in SURFACE_CONDITION_TYPES:
            close = _closest(t, SURFACE_CONDITION_TYPES)
            self.add(ERROR, where, f"unknown surface condition '{raw}'",
                     f"did you mean '{close}'?" if close else None)
            return
        if t == "not":
            self.check_surface_condition(f"{where}/invert", cond.get("invert"), depth + 1)
        if t == "biome":
            biomes = cond.get("biome_is")
            if isinstance(biomes, str):
                biomes = [biomes]
            if not isinstance(biomes, list) or not biomes:
                self.add(ERROR, where, "'biome_is' must be a biome id or a non-empty list of them")
            else:
                for b in biomes:
                    if str(b).startswith("#"):
                        continue                      # biome tag; resolved by the game
                    if not self.has("biome", str(b)):
                        self.add(ERROR, where, f"condition references unknown biome '{b}'")
        if t == "noise_threshold":
            ref = cond.get("noise")
            if isinstance(ref, str):
                ident = ref if ":" in ref else "minecraft:" + ref
                if not self.has("noise", ident) and ident not in BUILTIN_NOISE:
                    self.add(ERROR, where, f"unknown noise '{ref}'")
            lo, hi = cond.get("min_threshold"), cond.get("max_threshold")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
                self.add(ERROR, where, "min_threshold > max_threshold, so this is never true")
        if t == "vertical_gradient" and not cond.get("random_name"):
            self.add(ERROR, where, "vertical_gradient needs a 'random_name'")

    def check_dimension(self, ident, obj):
        where = f"dimension/{ident}"
        if not isinstance(obj, dict):
            self.add(ERROR, where, "must be an object")
            return
        dtype = obj.get("type")
        if dtype is None:
            self.add(ERROR, where, "missing 'type' (a dimension_type id)")
        elif isinstance(dtype, str) and not self.has("dimension_type", dtype) and \
                dtype not in ("minecraft:overworld", "minecraft:the_nether", "minecraft:the_end",
                              "minecraft:overworld_caves"):
            self.add(ERROR, where, f"unknown dimension_type '{dtype}'")
        gen = obj.get("generator")
        if not isinstance(gen, dict):
            self.add(ERROR, where, "missing 'generator'")
            return
        gtype = str(gen.get("type", "")).split(":")[-1]
        if gtype not in ("noise", "flat", "debug"):
            self.add(ERROR, where, f"unknown generator type '{gen.get('type')}'")
            return
        if gtype != "noise":
            return
        settings = gen.get("settings")
        if isinstance(settings, str) and not self.has("noise_settings", settings):
            self.add(ERROR, where, f"generator.settings references unknown noise_settings '{settings}'")
        source = gen.get("biome_source")
        if not isinstance(source, dict):
            self.add(ERROR, where, "generator has no 'biome_source'")
            return
        stype = str(source.get("type", "")).split(":")[-1]
        if stype == "multi_noise":
            entries = source.get("biomes")
            if not entries and "preset" in source:
                # the entries behind a preset are Mojang's, so they are resolved
                # to prove the pack will find biomes, and not audited further
                self.check_preset(where, str(source["preset"]))
            elif not isinstance(entries, list) or not entries:
                self.add(ERROR, where, "multi_noise biome_source needs a non-empty 'biomes' list "
                                       "or a 'preset'")
            else:
                self.check_biome_entries(where, entries)
        elif stype == "fixed":
            if not self.has("biome", str(source.get("biome"))):
                self.add(ERROR, where, f"fixed biome_source references unknown biome '{source.get('biome')}'")
        elif stype not in ("checkerboard", "the_end"):
            self.add(ERROR, where, f"unknown biome_source type '{source.get('type')}'")

    def check_preset(self, where, preset):
        """A preset names a biome table that has to be vendored to be usable."""
        table = self.registries.get("multi_noise_biome_source_parameter_list", preset)
        if table is None:
            self.add(ERROR, where, f"biome_source names unknown preset '{preset}'")
            return
        entries = table.get("biomes")
        if not entries:
            self.add(ERROR, where, f"no biome table is vendored for preset '{preset}', so the "
                                   "preview would have no biomes to place",
                     "run python tools/extract_biome_parameters.py to read it out of the server jar")
            return
        biomes = {str(e.get("biome")) for e in entries if isinstance(e, dict)}
        missing = sorted(b for b in biomes if not self.has("biome", b))
        if missing:
            self.add(ERROR, where, f"preset '{preset}' places {len(missing)} biome(s) this pack "
                                   f"cannot resolve: {', '.join(missing[:4])}")
        else:
            self.add(INFO, where, f"biome_source uses preset '{preset}': "
                                  f"{len(entries)} entries, {len(biomes)} biomes")

    def check_biome_entries(self, where, entries):
        names = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                self.add(ERROR, f"{where}/biomes[{i}]", "entry must be an object")
                continue
            name = entry.get("biome")
            names.append(name)
            if not self.has("biome", str(name)):
                self.add(ERROR, f"{where}/biomes[{i}]", f"unknown biome '{name}'",
                         "either use a vanilla biome id or add worldgen/biome/<name>.json to the pack")
            params = entry.get("parameters")
            if not isinstance(params, dict):
                self.add(ERROR, f"{where}/biomes[{i}]", "entry has no 'parameters'")
                continue
            for key in ("temperature", "humidity", "continentalness", "erosion", "depth", "weirdness"):
                if key not in params:
                    self.add(ERROR, f"{where}/biomes[{i}]", f"parameters missing '{key}'")
                    continue
                v = params[key]
                if isinstance(v, list):
                    if len(v) != 2:
                        self.add(ERROR, f"{where}/biomes[{i}]", f"'{key}' range must have 2 entries")
                    elif v[0] > v[1]:
                        self.add(ERROR, f"{where}/biomes[{i}]", f"'{key}' range is reversed: {v}")
                    elif not all(-2.0 <= x <= 2.0 for x in v):
                        self.add(WARNING, f"{where}/biomes[{i}]",
                                 f"'{key}' range {v} lies outside the usual [-1, 1]")
            if "offset" not in params:
                self.add(ERROR, f"{where}/biomes[{i}]", "parameters missing 'offset' "
                                                        "(use 0.0 unless you are deliberately biasing)")
        try:
            source = BiomeSource.from_json({"type": "minecraft:multi_noise", "biomes": entries})
            dead = unreachable_biomes(source)
            for name in dead:
                self.add(WARNING, where, f"biome '{name}' never wins anywhere in climate space: "
                                         "another entry always fits better, so it will not generate")
        except Exception:
            pass

    def check_biome(self, ident, obj):
        where = f"biome/{ident}"
        if not isinstance(obj, dict):
            self.add(ERROR, where, "must be an object")
            return
        for key in BIOME_REQUIRED:
            if key not in obj:
                self.add(ERROR, where, f"biome is missing required field '{key}'")
        for key in obj:
            if key not in BIOME_FIELDS:
                close = _closest(key, BIOME_FIELDS)
                self.add(ERROR, where, f"biome has no field '{key}'",
                         f"did you mean '{close}'?" if close else f"allowed: {sorted(BIOME_FIELDS)}")
        effects = obj.get("effects")
        if isinstance(effects, dict):
            if "water_color" not in effects:
                self.add(ERROR, where, "effects must contain 'water_color'")
            for key in effects:
                if key not in BIOME_EFFECTS:
                    moved = ATTRIBUTE_FOR_EFFECT.get(key)
                    hint = (f"in {self.version} it lives in attributes as {moved!r}" if moved
                            else f"allowed effects: {sorted(BIOME_EFFECTS)}")
                    self.add(ERROR, where, f"'{key}' is not a biome effect in {self.version}", hint)
        attributes = obj.get("attributes")
        if attributes is not None:
            if not isinstance(attributes, dict):
                self.add(ERROR, where, "'attributes' must be an object")
            else:
                for key in attributes:
                    if key not in BIOME_ATTRIBUTES:
                        self.add(WARNING, where, f"unrecognised biome attribute '{key}'")
        features = obj.get("features")
        if features is not None:
            # one list per generation step, up to 11; trailing empty steps may be omitted
            if not isinstance(features, list) or len(features) > 11:
                self.add(ERROR, where, "'features' must be a list of at most 11 lists, one per "
                                       "generation step (use [] for a bare terrain biome)")
            elif any(not isinstance(step, list) for step in features):
                self.add(ERROR, where, "each entry of 'features' must itself be a list")

    def check_biome_tag(self, ident, obj):
        """A biome tag is how a structure finds its biomes.

        Get one wrong and the game says nothing: the tag is simply empty, and the
        world generates without a single village in it.
        """
        where = f"biome_tag/{ident}"
        if not isinstance(obj, dict):
            self.add(ERROR, where, "must be an object")
            return
        for key in obj:
            if key not in TAG_FIELDS:
                close = _closest(key, TAG_FIELDS)
                self.add(ERROR, where, f"tag has no field '{key}'",
                         f"did you mean '{close}'?" if close else f"allowed: {sorted(TAG_FIELDS)}")
        if "replace" in obj and not isinstance(obj["replace"], bool):
            self.add(ERROR, where, "'replace' must be true or false",
                     "omit it, or use false, to add to the tag rather than override it")
        values = obj.get("values")
        if values is None:
            self.add(ERROR, where, "tag has no 'values'")
            return
        if not isinstance(values, list):
            self.add(ERROR, where, "'values' must be a list")
            return
        if not values and obj.get("replace") is not True:
            # "replace": true with an empty list is the way to switch a vanilla
            # tag off, so only an empty *additive* tag is pointless
            self.add(WARNING, where, "tag is empty and does not replace, so it does nothing")
        seen: set[str] = set()
        for i, entry in enumerate(values):
            self.check_tag_entry(f"{where}/values[{i}]", entry, seen)

    def check_tag_entry(self, where, entry, seen: set):
        required = True
        if isinstance(entry, dict):
            required = entry.get("required", True)
            entry = entry.get("id")
        if not isinstance(entry, str):
            self.add(ERROR, where, "entry must be a biome id, a '#tag' reference, "
                                   'or {"id": "...", "required": false}')
            return
        is_tag = entry.startswith("#")
        ident = _qualify(entry[1:] if is_tag else entry)
        # '#foo' and 'foo' name different things, so the marker stays in the key
        key = ("#" if is_tag else "") + ident
        if key in seen:
            self.add(WARNING, where, f"'{entry}' is listed twice")
        seen.add(key)
        if not required:
            return          # the point of an optional entry is that it may be absent
        if is_tag:
            # vanilla's own tags are not vendored, so only tags this pack could
            # define are checkable
            if self._owns(ident) and not self.has("biome_tag", ident):
                self.add(ERROR, where, f"unknown biome tag '{entry}'",
                         "it is not defined by this pack")
        elif not self.has("biome", ident):
            close = _closest(ident.split(":")[-1],
                             {b.split(":")[-1] for b in self.registries.ids("biome")})
            self.add(ERROR, where, f"unknown biome '{entry}'",
                     f"did you mean '{close}'?" if close else None)

    def _owns(self, ident: str) -> bool:
        """Is this id in a namespace the pack under test defines?"""
        return self.pack is not None and _qualify(ident).split(":")[0] in self.pack.namespaces

    def check_block_state(self, where, state, required: bool = False):
        if state is None:
            if required:
                self.add(ERROR, where, "missing block state")
            return
        if not isinstance(state, dict):
            self.add(ERROR, where, 'block state must be an object like {"Name": "minecraft:stone"}')
            return
        name = state.get("Name")
        if not isinstance(name, str):
            self.add(ERROR, where, "block state has no 'Name' (capital N)")
            return
        ident = name if ":" in name else "minecraft:" + name
        if self.block_ids and ident not in self.block_ids:
            close = _closest(ident.split(":")[-1], {b.split(":")[-1] for b in self.block_ids})
            self.add(ERROR, where, f"unknown block '{name}'",
                     f"did you mean 'minecraft:{close}'?" if close else None)
        elif is_missing(ident):
            self.add(INFO, where, f"'{name}' has no preview colour; it will render magenta")
        props = state.get("Properties")
        if props is not None:
            if not isinstance(props, dict):
                self.add(ERROR, where, "'Properties' must be an object of string->string")
            else:
                known = self.block_properties.get(ident)
                for key, value in props.items():
                    if not isinstance(value, str):
                        self.add(ERROR, where, f"property '{key}' must be a *string* (got {value!r})")
                    if known and key not in known:
                        self.add(ERROR, where, f"block '{name}' has no property '{key}'",
                                 f"it has: {sorted(known)}")


TAG_FIELDS = {"replace", "values"}

# 26.2 biome schema, read off the vendored vanilla files rather than recalled.
BIOME_FIELDS = {"temperature", "downfall", "has_precipitation", "effects", "spawners",
                "spawn_costs", "carvers", "features", "attributes", "temperature_modifier",
                "creature_spawn_probability"}
BIOME_REQUIRED = ("temperature", "downfall", "has_precipitation", "effects",
                  "spawners", "spawn_costs", "carvers", "features")
BIOME_EFFECTS = {"water_color", "foliage_color", "grass_color", "dry_foliage_color",
                 "grass_color_modifier"}
# fields that used to live in `effects` and moved to `attributes` in 26.x
ATTRIBUTE_FOR_EFFECT = {
    "sky_color": "minecraft:visual/sky_color",
    "fog_color": "minecraft:visual/fog_color",
    "water_fog_color": "minecraft:visual/water_fog_color",
    "particle": "minecraft:visual/ambient_particles",
    "ambient_sound": "minecraft:audio/ambient_sounds",
    "mood_sound": "minecraft:audio/ambient_sounds",
    "additions_sound": "minecraft:audio/ambient_sounds",
    "music": "minecraft:audio/background_music",
    "music_volume": "minecraft:audio/music_volume",
}
BIOME_ATTRIBUTES = {
    "minecraft:visual/sky_color", "minecraft:visual/fog_color", "minecraft:visual/water_fog_color",
    "minecraft:visual/water_fog_end_distance", "minecraft:visual/ambient_particles",
    "minecraft:audio/background_music", "minecraft:audio/ambient_sounds", "minecraft:audio/music_volume",
    "minecraft:gameplay/snow_golem_melts", "minecraft:gameplay/increased_fire_burnout",
    "minecraft:gameplay/can_pillager_patrol_spawn",
}


def _qualify(ident: str) -> str:
    """'plains' -> 'minecraft:plains'. Registries.get does this internally; tag
    checking needs the qualified form for de-duplication and for hints."""
    return ident if ":" in ident else "minecraft:" + ident


def _closest(word: str, options) -> str | None:
    matches = difflib.get_close_matches(word, list(options), n=1, cutoff=0.72)
    return matches[0] if matches else None


def validate_path(path, version: str = "26.2") -> tuple[Validator, list[Finding]]:
    pack = Pack(path)
    registries = Registries.load([path], version=version)
    validator = Validator(registries, pack, version)
    return validator, validator.validate_pack()
