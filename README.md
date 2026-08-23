# worldsmith

Write Minecraft terrain in JSON, render it, and look at it before you load the game.

![basalt spires](renders/spires.png)

Minecraft's world generation has been pure data since 1.18: density functions,
splines, surface rules, biome boxes. What is missing is the other half of the loop,
a way to *see* what the JSON does without launching the game and flying around.
worldsmith is that half: a reimplementation of the game's worldgen in Python that
draws the terrain in seconds.

```bash
git clone https://github.com/Msde-7/worldsmith
cd worldsmith
pip install -r requirements.txt

python -m worldsmith.cli new    packs/mine --namespace mine --name mine
python -m worldsmith.cli check  packs/mine
python -m worldsmith.cli render packs/mine --out renders/mine.png
python -m worldsmith.cli play   packs/mine     # into Minecraft, ready to walk into
```

Python 3.10+, numpy and pillow. numba is optional and makes renders about ten
times faster.

## It is the same terrain, and that is measured

`tools/verify_in_game.py` runs a real Minecraft 26.2 server on the pack, reads the
heightmaps back out of the region files and compares them column by column:

```
$ python tools/verify_in_game.py packs/basalt_spires
columns compared : 331776
exact matches    : 331776 (100.000%)
max  |error|     : 0 blocks
```

Same story on vanilla's own terrain: 51,200 columns, 100.000% exact. Turn
aquifers on, which the engine deliberately does not model, and it drops to 99.19%
with a mean error of 0.12 blocks. That gap is the whole of the difference.

Underneath, 170 checks pin the engine to [deepslate](https://github.com/misode/deepslate)
and to real JVM output: all 35 vanilla density functions, all 7 dimension routers,
every noise and every RNG stream, bit for bit.

## The loop

One render, four views: what it looks like, how tall it is, where the biomes
landed, and a cross-section for overhangs and hollows. Change one thing, look
again. The basalt spires pack on its first pass, its third, and its last:

| first attempt | narrower, taller | final |
|---|---|---|
| ![v1](renders/spires_v1.png) | ![v3](renders/spires_v3.png) | ![final](renders/spires.png) |

Blobby islands became spires by raising the spire mask threshold so only the top
few percent of columns qualify, then steepening the height spline. Thirty seconds,
no world reload.

`--size 256 --step 1` renders block by block instead of one pixel per noise cell,
which is the view for close-ups:

![closeup](renders/spires_closeup.png)

## Into the game

`worldsmith play` is the last mile in one command: it rewrites the pack's dimension
as the overworld, fetches a server jar and Java runtime, **uses the renderer to
pick a spawn worth spawning at** (flat ground with the biggest landmark in view),
pre-builds the chunks around it, and installs the save into
`.minecraft/saves` with creative and cheats on.

![spawn area](renders/spawn_area.png)

It only adds that one save. The pack it copies into `.minecraft/datapacks` just
shows up in the Data Packs list on the world-creation screen, so New World still
gives you an ordinary world. Running the server writes `eula=true` inside
`.runtime/`, which accepts Mojang's EULA.

## Example packs

`red_canyons`: a tableland cut by slot gorges. The canyon network is `abs(noise)`
near zero along a winding line; the benches are a staircase spline that turns
smooth noise into flat tops with abrupt risers.

![red canyons](renders/canyons.png)

`sky_islands`: `depth` as a band instead of a ramp, two opposing y gradients
min-ed together, so terrain has a top *and* a bottom and the void takes the rest.

![sky islands](renders/sky_islands.png)

`tools/build_canyons.py` and `tools/build_skyislands.py` are the scripts that wrote
those two, kept because spline points read better as numbers with comments next to
them. `worldsmith new` scaffolds a third, a plain continents-and-oceans world laid
out to be edited (`renders/starter.png`).

## Commands

```bash
worldsmith new     packs/mine --namespace mine --name mine
worldsmith check   packs/mine        # schema, dangling refs, dead biomes, smoke test
worldsmith render  packs/mine --out renders/mine.png
worldsmith play    packs/mine        # into Minecraft
worldsmith export  packs/mine        # a zip for someone else's game
worldsmith reference terrain         # the height formula and what every knob does
worldsmith probe   packs/mine --at 100 64 -200 --density mine:offset mine:factor
worldsmith column  packs/mine --at 100 -200      # one column, top to bottom
```

Run them as `python -m worldsmith.cli <command>` from the repository. `check` is
the one to run before every render: Minecraft rejects malformed worldgen silently
at world creation and hands you a void world, so the validator covers every node
type and field, dangling references, spline ordering, block ids and properties,
the mandatory router fields, and biome boxes that can never win.

## What is in here

| | |
|---|---|
| `worldsmith/jrandom.py` | `java.util.Random` and xoroshiro128++, bit exact, plus `Mth.getSeed` |
| `worldsmith/noise.py` | ImprovedNoise, PerlinNoise, NormalNoise, BlendedNoise, 2D simplex |
| `worldsmith/kernels.py` | numba kernel for the Perlin inner loop; falls back to numpy |
| `worldsmith/density.py` | every density function node, compiled and vectorised |
| `worldsmith/terrain.py` | heightmaps and cross-sections; lattice or per-block sampling |
| `worldsmith/surface.py` | surface rules, including the badlands clay bands |
| `worldsmith/climate.py` | multi-noise biome assignment, and unreachable-biome detection |
| `worldsmith/validate.py` | schema, references, splines, block ids, biome boxes |
| `worldsmith/render.py` | the four views and the contact sheet |
| `worldsmith/play.py` | server runtime, spawn picking, world install |
| `vanilla/26.2/` | the vendored vanilla data it is checked against |
| `tools/` | in-game verification, the deepslate oracle, the two authoring scripts |

Renders are fast because every purely 2D function is evaluated once per column
rather than once per block, the Perlin inner loop is a parallel numba kernel, and
terrain is sampled on the noise-cell lattice whenever the pack's `final_density`
is rooted in an `interpolated` node, which makes that sampling exact. Anything
else, vanilla included, falls back to an exact per-block scan.

## Known limits

* **Aquifers are not modelled**, so with `aquifers_enabled: true` the game can seal
  underground voids with stone the engine does not draw. Custom packs generally
  leave them off; vanilla has them on.
* **Carvers and features are not run.** No caves, trees or ores. The preview is
  bare worldgen terrain, which is what you are editing.
* **Vanilla's biome presets cannot be previewed.** `{"preset": "minecraft:overworld"}`
  resolves in Java code rather than in data, so such a dimension renders with
  terrain only. Custom packs list their biomes and preview fine.

Vanilla itself, rendered by the engine:

![vanilla overworld](renders/vanilla_overworld.png)

## Tests

```bash
python tests/run_all.py                              # 203 checks, no network
python tools/verify_in_game.py packs/basalt_spires   # the real game, opt-in
```

The second downloads a server jar and Java runtime into `.runtime/` on first use
and writes an `eula.txt` there.

## Using it with Claude Code

`.claude/skills/worldsmith/SKILL.md` turns "make me a world of sheer basalt spires
over an acid sea" into the datapack, and, the part that matters, makes the model
render the result and *look at it* before iterating. Copy that directory into
`~/.claude/skills/` to have it everywhere. The example packs were built exactly
that way: describe, generate, render, look, adjust.

## License and credit

MIT, see [LICENSE](LICENSE). One exception: `vanilla/26.2/` is Minecraft's own
worldgen data, extracted by [misode/mcmeta](https://github.com/misode/mcmeta) and
vendored here so the engine can be checked against it. That data belongs to Mojang
and is not covered by this license.

The engine is validated against [deepslate](https://github.com/misode/deepslate)
(MIT) by misode, which also powers
[misode.github.io/worldgen](https://misode.github.io/worldgen). worldsmith is an
independent project and is not affiliated with or endorsed by Mojang or Microsoft.
