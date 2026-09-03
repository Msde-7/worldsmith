---
name: worldsmith
description: Turn a description of a landscape or of a thing to build into a working Minecraft datapack - terrain from density functions, splines, surface rules and biome placement, and builds the game places from block templates - then iterate on either by rendering it and looking at the picture. Use whenever someone asks for custom Minecraft terrain, a custom dimension or overworld, or for a castle, tower, ruin, village or any other structure placed in a world, or wants existing worldgen or structure JSON changed, debugged or explained.
---

# worldsmith

Write Minecraft terrain in JSON, render it, look at it, change it. The renderer
is a bit-exact reimplementation of the game's worldgen, verified column-for-column
against Minecraft itself, so what the image shows is what the game will build.

There are two halves. Terrain is the one below, and **builds**, meaning anything
made of blocks that the game places for you, are the section "The other half,
building things". They work on their own or together with terrain.

Run every command from the root of the worldsmith checkout.

## The loop

```
python -m worldsmith.cli new packs/<name> --namespace <ns> --name <name>
#   --caves                  cut vanilla's cave system into the terrain
#   --like minecraft:plains  borrow a vanilla biome's trees, ores, mobs, carvers
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

## The other half, building things

Terrain and builds are separate halves and neither needs the other. A pack can
be terrain only, builds only, which is a build dropped into an ordinary vanilla
world, or both. Read the reference before writing one:

```
python -m worldsmith.cli reference builds
```

Start from something that already works rather than an empty file:

```
python -m worldsmith.cli new packs/keep --namespace keep --name keep --with-build
```

That scaffolds a small hut with every part a build has to get right (a buried
footing, hollow rooms, a door, a light, a loot chest) plus the three JSON files
that place it. Then replace the geometry.

A build is a box of blocks plus three JSON files, and `structures.add` writes
all four. Keep the geometry in a script rather than a one-off, so the build can
be re-run and changed. The example packs do it that way in `examples/`, and
`examples/build_castles.py` is a large one to read if you want to see how far it
goes.

`worldsmith.shapes` has the geometry rather than writing loops for it: `speckle`
(masonry mixed from several blocks so a wall does not read as extruded),
`hollow_box`, `cylinder`, `perimeter`, `ring_cells`, `crenellate`, `gable_roof`
and `stair_flight`.

```python
from worldsmith import structures
from worldsmith.pack import PackWriter
from worldsmith.shapes import crenellate, hollow_box, perimeter, speckle
from worldsmith.voxel import Grid

grid = Grid(32, 24, 32)                       # x, y, z
grid.fill(0, 0, 0, 31, 7, 31, "minecraft:stone")          # buried footing
grid.fill(4, 8, 4, 27, 20, 27, "minecraft:stone_bricks")
grid.fill(5, 9, 5, 26, 19, 26, "minecraft:air")           # air is what hollows it
grid.set(15, 9, 27, "minecraft:oak_door[facing=south,half=lower,hinge=left]")
grid.set(9, 9, 9, "minecraft:chest[facing=north]",
         {"id": "minecraft:chest", "LootTable": "minecraft:chests/simple_dungeon"})

writer = PackWriter("packs/keep", "a keep")
writer.mcmeta()
structures.add(writer, "keep:tower", grid, ["minecraft:plains", "minecraft:forest"],
               sink=-9)                       # floor at y=8, so -(8+1)
# or name them with a tag: structures.add(..., "#minecraft:is_forest", ...)
writer.add("structure_set", "keep:towers",
           structures.spread("keep:tower", spacing=16, separation=7, salt=771223))
```

Then the same loop as terrain, with the same rule: **look at the picture**.

```
python -m worldsmith.cli build  packs/keep --id keep:tower --plan 8,14   # draw it
python -m worldsmith.cli build  packs/keep --id keep:tower --site 0     # on its ground
python -m worldsmith.cli sites  packs/keep            # where it lands, on what ground
python -m worldsmith.cli render packs/keep --builds   # those sites on the terrain
python -m worldsmith.cli check  packs/keep            # the silent mistakes
python -m worldsmith.cli play   packs/keep --spawn-at keep:tower
```

`build --site 0` is the one to look at: the build standing on the terrain of
the site the game will actually choose, drawn without generating a world.

`sites` is the one to read carefully. It lists every site the game will
consider, whether the biome check keeps it, and the ground under the footprint:
height, relief and how much is under water. Relief over about 10 blocks means
the build will sit in a hole or on a pedestal, and the fix is either a deeper
buried footing, a biome list that only names flat biomes, or terrain that has
more flat ground in it.

What will bite, in order of how much time it costs:

* **Air.** `structures.add` uses `legacy_single_pool_element` for you. If you
  write the pool JSON yourself, use it too: the modern element throws air away
  and every room comes out solid.
* **Trees.** Features are placed after structures, so grass inside a build grows
  a wood. Pave anything that should stay clear: gravel, cobblestone, stone and
  `dirt_path` are not soil; `coarse_dirt` and `podzol` are.
* **Height.** `sink` is `-(floor + 1)`. Bury several courses below the floor or
  the build stands on a pillar of air on a slope.
* **Big builds.** `terrain_adaptation` defaults to `beard_box`, which clears the
  whole box. Anything wider than a house needs it.
* **Block states.** They are checked against the real block list as you write
  them, so a typo raises here rather than vanishing in game.
* **Size.** Vanilla ships nothing wider than 48 blocks; 96 is checked here and
  places every block. A 64x52x64 build is a hundred thousand blocks per copy,
  which is what makes a world slow to generate, not the count of builds.
* **The ground around a build is not the ground you rendered.** The game
  reshapes terrain against a structure and worldsmith does not model it.
  Measured around a 64 wide build: nothing matches within 3 blocks of it
  (median 10 blocks different), 84% matches by 8 blocks out, and by 12 blocks
  it is back to normal. `sites` relief is the ground before the build lands,
  which is the number that says whether it will look wrong. Never place
  anything relative to the terrain right beside a build.

A build pack can be added to a world someone already plays, which terrain
cannot: drop it in `<world>/datapacks/` and the builds appear in chunks
generated from then on, not in the ones already explored. Checked on a world
made without the pack: 0 builds at spawn, then 12 in a patch generated 2000
blocks away after the pack went in.

After a world exists, check what the game actually did:

```
python -m worldsmith.cli inspect <world> --pack packs/keep --structure keep:tower --render out.png
```

It lists every placed build, and with `--pack` compares the placed blocks
against the template block for block, and says whether the placement model
called that site right. Anything under about 99% that is not ore in the buried
footing, or trees in something meant to be overgrown, is worth reading.

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
| caves | `--caves`, or min the cave functions in yourself (see below) |
| trees, ores, mobs | `--like <vanilla biome>`; the game places them, the render never shows them |

Caves belong **inside** the `interpolated` node, the way vanilla does it:

```
final_density = min(interpolated(min(<terrain>, minecraft:overworld/caves/entrances)),
                    minecraft:overworld/caves/noodle)
```

Cutting them in outside that node looks equivalent and is not. A world built that
way matched the real game on 87% of columns; the shape above matched on 100%. Both
sides of that test had aquifers off, so only the placement differed.

Caves need aquifers, so `--caves` turns them on and writes the four aquifer
router fields (`barrier`, `fluid_level_floodedness`, `fluid_level_spread`, `lava`)
with vanilla's noises. Skip that and the caves flood: the game fills every cavity
below sea level with water and below y=-54 with lava, which measured 5% dry cave
volume against 91% with aquifers on. Flipping `aquifers_enabled` while leaving
those four fields at 0 is its own bug, and gives flat sheets of water underground.

Two more things that catch people out with `--caves`:

* the pack renders about 5x slower. `final_density` becomes a `min`, so it is no
  longer linear in y and `render`, `check` and `column` all take the exact
  per-block scan instead of the lattice shortcut.
* the cave functions carry vanilla's altitudes (`noodle` gated to y -60..321,
  `entrances` ramping y -10..30). Move `min_y` or `height` and the caves stay put.
* the preview stops being exact, because aquifers are the one part of the engine
  that approximates. A `--caves` pack measured 99.441% exact against a real server
  over 331,776 columns (99.724% within one block); the misses are columns where the
  game perches water or drops a barrier lid a few blocks off where the engine put it.

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
never win, and for builds a missing template, a pool that names nothing, a
separation above its spacing, and biomes the dimension never places) and then
a smoke test that actually generates terrain and reports whether anything was
built at all. Fix every ERROR before rendering; the game
rejects malformed worldgen silently and hands you a void world.

## Verifying in the real game (optional)

```
python tools/verify_in_game.py packs/<name>
python tools/verify_in_game.py packs/<name> --sample 200    # quicker, one region
python tools/verify_placement.py --size 64                  # the build placement model
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
