---
name: worldsmith
description: Turn a description of a landscape into a working Minecraft worldgen datapack - density functions, splines, surface rules and biome placement - and iterate on it by rendering the terrain and looking at it. Use whenever someone asks for custom Minecraft terrain, a custom dimension, a custom overworld, "make me a world that...", or wants existing worldgen JSON changed, debugged or explained.
---

# worldsmith

Write Minecraft terrain in JSON, render it, look at it, change it. The renderer
is a bit-exact reimplementation of the game's worldgen, verified column-for-column
against Minecraft itself, so what the image shows is what the game will build.

Run every command from the root of the worldsmith checkout.

## The loop

```
python -m worldsmith.cli new packs/<name> --namespace <ns> --name <name>
# edit the JSON
python -m worldsmith.cli check  packs/<name>
python -m worldsmith.cli render packs/<name> --out renders/<name>.png
# LOOK AT THE IMAGE with the Read tool, change one thing, render again
python -m worldsmith.cli play   packs/<name>          # into Minecraft, ready to walk into
```

**Always read the rendered PNG before saying anything about how the terrain
looks.** That is the entire point of the tool: you can see the terrain. Two or
three render-and-look cycles produce far better terrain than one careful guess.

## Putting a world in Minecraft

`play` is the whole job in one command, and it is what to reach for whenever
someone wants to actually *see* a world rather than a picture of one:

```
python -m worldsmith.cli play packs/<name>
python -m worldsmith.cli play packs/a packs/b packs/c      # several at once
python -m worldsmith.cli play packs/<name> --no-launch     # install, don't open the game
python -m worldsmith.cli play packs/<name> --name "Ash Wastes" --seed 7
```

It validates the pack, rewrites its dimension as `minecraft:overworld` so a new
world simply *is* that terrain, fetches a matching server jar and Java runtime
the first time (cached in `.runtime/`, ~240 MB), **picks the spawn point with the
renderer**, a flat place to stand with the biggest landmark in view, pre-builds
the chunks around it, sets creative + cheats + clear midday, installs the save
into `%APPDATA%/.minecraft/saves` under a readable name, also drops the pack into
`.minecraft/datapacks` for new worlds, and prints the coordinates plus a `/tp` to
the landmark.

Things worth knowing:

* Generating runs a real Minecraft server locally, which writes `eula=true` in
  `.runtime/`. Say so when you run it; it accepts Mojang's EULA.
* The save is stamped with the version that built it (`--mc-version`, default
  26.2). The player has to launch **that version or newer**, or Minecraft refuses
  to open the world. Tell them which one.
* Don't launch the game unless they asked you to; pass `--no-launch` otherwise.
* `--pregen 0` skips pre-building if they just want it installed quickly.
* Existing worlds are **replaced** by name, so pick `--name` deliberately.

## Before writing any JSON

Read the reference. It is short, it is version-accurate for 26.2, and it
contains the numbers you would otherwise guess wrong:

```
python -m worldsmith.cli reference terrain     # the height formula, factor, jaggedness
python -m worldsmith.cli reference density     # every node type and its fields
python -m worldsmith.cli reference surface     # surface rules
python -m worldsmith.cli reference biomes      # climate parameters
python -m worldsmith.cli reference mistakes    # the ones that cost a world reload
```

The single most important fact: the ground sits at

```
surface_y  ~=  min_y + height/3 * (1.5 + offset)
```

so for a normal world (min_y -64, height 384) an `offset` of **-0.5 puts the
surface at sea level** and every 0.1 of offset is about 13 blocks. An offset
spline that forgets the -0.5 builds the world at cloud height.

## Turning words into knobs

| they said | change |
|---|---|
| higher / lower ground | the `offset` spline |
| flatter, plateaus, mesas | raise `factor` (6+) |
| dramatic, mountainous | lower `factor` (1.5-3) |
| spiky, jagged peaks | `jaggedness` > 0, but only near ridges |
| sheer cliffs, vertical walls | high `factor` *and* a sharp step in `offset` |
| overhangs, arches, caves in cliffs | low `factor`, heavier `base_3d_noise` |
| floating islands | make `depth` a band, not a ramp: subtract `abs(y - centre)` |
| wider / narrower features | `xz_scale` on the shaping noise, or `firstOctave` |
| more / less ocean | move the ocean end of the `offset` spline |
| different rock, sand, snow | the `surface_rule` |
| different biomes | the `biome_source` parameter boxes |

`firstOctave` is a power of two: -7 gives ~128-block features, -5 gives ~32-block
features. Less negative = smaller, tighter features.

## Measure instead of guessing

Before choosing a spline threshold, look at what the input noise actually does:

```
python -m worldsmith.cli probe  packs/<name> --at 100 64 -200 --density <ns>:offset <ns>:factor
python -m worldsmith.cli column packs/<name> --at 100 -200
```

and for distributions, sample in Python:

```python
from worldsmith.registry import Registries
from worldsmith.world import World
from worldsmith.density import Ctx, prepare
import numpy as np
w = World.create(Registries.load(["packs/<name>"]), "<ns>:<settings>", 12345)
node = w.compiler.compile_ref("<ns>:spire_field"); prepare(node)
xs = np.random.default_rng(0).integers(-20000, 20000, 40000).astype(float)[None, :]
zs = np.random.default_rng(1).integers(-20000, 20000, 40000).astype(float)[None, :]
v = np.ravel(np.broadcast_to(np.asarray(node.eval(Ctx(xs, np.array([[64.0]]), zs)), float), xs.shape))
print(np.percentile(v, [1, 50, 90, 95, 99]))
```

"Only the top 5% of columns should be spires" then becomes a real threshold
rather than a guess.

## Rendering

```
--size 1024 --step 4      a wide overview (1 pixel per noise cell) - the default
--size 256  --step 1      block-accurate detail; slower, use for close-ups
--views map,height,biomes,section
--center X Z --seed N     check somewhere else, or another seed, before declaring victory
```

The four views answer different questions: `map` what it looks like, `height`
how tall and how steep, `biomes` whether anything landed where you meant,
`section` whether there are overhangs, floating chunks or a hollow world.

## Checking

`check` runs a schema validator (every density function type and field, dangling
references, spline ordering, block ids and their properties, biome boxes that can
never win) and then a smoke test that actually generates terrain and reports
whether anything was built at all. Fix every ERROR before rendering; the game
rejects malformed worldgen silently and hands you a void world.

## Verifying in the real game (optional)

```
python tools/verify_in_game.py packs/<name>
python tools/verify_in_game.py packs/<name> --sample 200    # quicker, one region
```

Runs a real Minecraft server on the pack and compares its stored heightmaps with
the engine's, column by column. It downloads a server jar and Java runtime into
`.runtime/` on first use and accepts Mojang's EULA there, so ask before running
it.

## Shipping it elsewhere

`python -m worldsmith.cli export packs/<name>` writes a zip for someone else's
game. It goes in `<world>/datapacks/`, or in `.minecraft/datapacks` so it can be
enabled from the world-creation screen. Scaffold with `--replace-overworld` to have new worlds use
the terrain as their overworld; note that existing worlds keep their old terrain,
because generator settings are baked in at world creation.
