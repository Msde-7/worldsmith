# worldsmith

Write Minecraft terrain in JSON, render it, and look at it before you load the game.

![basalt spires](renders/spires.png)

Minecraft's world generation has been pure data since 1.18: density functions,
splines, surface rules, biome boxes. What is missing is the other half of the loop,
a way to *see* what the JSON does without launching the game and flying around.
worldsmith is that half: a reimplementation of the game's worldgen in Python that
draws the terrain in seconds. Then `worldsmith play` builds the world and drops
you in it:

![basalt spires in game](renders/ingame_spires.jpg)

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

Same story on vanilla's own terrain, aquifers and all: 51,200 columns, 99.959%
exact, mean error 0.002 blocks. Aquifers are the hard part there, since the game
walls off underground water with stone that shows up in the heightmap; modelling
them took that sample from 99.19% to 99.96%.

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
![red canyons in game](renders/ingame_canyons.jpg)

`sky_islands`: `depth` as a band instead of a ramp, two opposing y gradients
min-ed together, so terrain has a top *and* a bottom and the void takes the rest.

![sky islands](renders/sky_islands.png)
![sky islands in game](renders/ingame_sky_islands.jpg)

`tools/build_canyons.py` and `tools/build_skyislands.py` are the scripts that wrote
those two, kept because spline points read better as numbers with comments next to
them. `worldsmith new` scaffolds a third, a plain continents-and-oceans world laid
out to be edited (`renders/starter.png`).

## Commands

```bash
worldsmith new     packs/mine --namespace mine --name mine [--caves] [--like minecraft:plains]
worldsmith check   packs/mine        # schema, dangling refs, dead biomes, biome tags, smoke test
worldsmith render  packs/mine --out renders/mine.png
worldsmith play    packs/mine        # into Minecraft
worldsmith export  packs/mine        # a zip for someone else's game
worldsmith reference terrain         # the height formula and what every knob does
worldsmith probe   packs/mine --at 100 64 -200 --density mine:offset mine:factor
worldsmith column  packs/mine --at 100 -200      # one column, top to bottom
```

`--caves` cuts vanilla's cave system into the terrain. Vanilla's caves are density
functions rather than carvers, so these are the one kind of cave the preview draws
exactly. They have to go inside the `interpolated` node the way vanilla does it: a
world that cut them in outside that node matched the game on 87% of columns against
100% for the shape vanilla uses, which is the sort of thing you only find by
generating both, with aquifers off on both sides so only the placement differed.

`--caves` turns aquifers on with them, and writes the four aquifer router fields
that make that mean anything. Caves without aquifers flood: the game has no water
table to consult, so it fills every cavity below sea level with water and everything
under y=-54 with lava, and on a 400 column sample only 5% of the cave volume came
out as air against 91% with aquifers on. Two consequences worth knowing. Such a pack
renders on the exact per-block scan, roughly 5x slower, because a cave layer makes
`final_density` a `min` and the density is then no longer linear in y. And the cave
functions carry vanilla's own altitudes (`noodle` is gated to y -60..321, `entrances`
ramps between y -10 and 30), so a pack that moves `min_y` or `height` keeps its caves
where vanilla put them.

Aquifers are also the one part of the engine that is not exact, so a cave pack
previews a little less precisely than a plain one. Against a real server over
331,776 columns, `--caves` came out 99.441% exact and 99.724% within one block, mean
error 0.017 blocks. The columns it misses are the ones where the game perches a body
of water or drops a barrier lid over a cavity and the engine puts it a few blocks
off. A plain pack has no aquifers and stays exact.

`--like minecraft:plains` borrows a vanilla biome's trees, ores, mobs and carvers.
These are features and carvers, not density functions, so the game places them after
generation and a render never shows them.

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
| `worldsmith/aquifer.py` | underground fluid levels and the stone barriers between them |
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

* **Aquifer boundaries are not perfect.** The fluid levels and the barriers
  between them are modelled, but where the game's floodedness check is a near
  tie the engine can pick the other side, which is the last 0.04% of columns.
  Most custom packs leave `aquifers_enabled` off and are unaffected.
* **Carvers and features are not run**, so the trees, ores and ravines `--like`
  brings along are placed by the game afterwards and never show up in a render.
  Noise caves are the exception, since they live in the density functions.
* **Vanilla's biome presets cannot be previewed.** `{"preset": "minecraft:overworld"}`
  resolves in Java code rather than in data, so such a dimension renders with
  terrain only. Custom packs list their biomes and preview fine.

Vanilla itself, rendered by the engine:

![vanilla overworld](renders/vanilla_overworld.png)

## Tests

```bash
python tests/run_all.py                              # 221 checks, no network
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
