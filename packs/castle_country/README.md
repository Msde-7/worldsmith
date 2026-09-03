# Castle country

One of the example packs. A custom overworld of rolling downs, oak woods,
lakes and flat-topped crags, with castles standing in it, written to show the
two halves of worldsmith working together. Nothing here is part of worldsmith
itself. The castles are geometry in `examples/build_castles.py`, and anything
else made of blocks would be written the same way.

## What is in here

    data/castle/worldgen/density_function/   the terrain shaping
    data/castle/worldgen/noise/crag.json     the noise the crags are cut from
    data/castle/worldgen/biome/              sea, shore, downs, oakwood, heath, crag
    data/castle/worldgen/noise_settings/     surface rules and the noise router
    data/castle/dimension/                   the dimension itself
    data/castle/structure/*.nbt              the castles, as structure templates
    data/castle/worldgen/structure/          how each one is placed
    data/castle/worldgen/template_pool/      the single-element pools they sit in
    data/castle/worldgen/structure_set/      how often, and how far apart

## The castles

| structure | size | spacing | what it is |
|---|---|---|---|
| `castle:great_castle` | 64x52x64 | one per 24 chunks (with ruins) | moat, bridge, gatehouse with a portcullis, four round corner towers, wall walk, four-storey keep, chapel, barracks, smithy, stables, well, garden |
| `castle:ruined_castle` | 64x52x64 | shares the great castle set, weight 2:3 | the same castle after a few centuries: walls down, vines, rubble, the courtyard gone back to wood |
| `castle:tower_keep` | 32x40x32 | one per 13 chunks | a tower house of four floors inside a low bailey wall, with a well and a wheat patch |

Two details matter and are easy to get wrong:

* the pool elements are `legacy_single_pool_element`. The modern
  `single_pool_element` ignores air in a template, and these castles use air to
  hollow their rooms and to cut the moat.
* `terrain_adaptation` is `beard_box`, the way the ancient city does it, so a
  hillside cannot push up through the bailey.

## Rebuilding it

    python examples/build_castle_country.py   # terrain
    python examples/build_castles.py          # the castles and their worldgen JSON
    python -m worldsmith.cli check packs/castle_country
    python -m worldsmith.cli play  packs/castle_country

## Checking it

    python -m worldsmith.cli sites packs/castle_country
    python -m worldsmith.cli inspect <world> --pack packs/castle_country --structure castle:great_castle --render renders/as_built.png
