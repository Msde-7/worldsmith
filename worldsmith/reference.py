"""The worldgen syntax reference, printed by `worldsmith reference <topic>`.

Numbers marked "measured" come from sampling the vendored vanilla 26.2 data with
this engine rather than from memory.
"""
from __future__ import annotations

TOPICS = {
    "layout": """FILE LAYOUT
===========
A worldgen datapack for 26.2 (pack_format 107):

  pack.mcmeta                                   {"pack":{"pack_format":107,"description":"..."}}
  data/<ns>/worldgen/noise/<id>.json            noise parameters  {firstOctave, amplitudes}
  data/<ns>/worldgen/density_function/<id>.json one density function per file
  data/<ns>/worldgen/noise_settings/<id>.json   the generator: router + surface rule + world height
  data/<ns>/worldgen/biome/<id>.json            custom biomes (or reuse minecraft: ones)
  data/<ns>/dimension/<id>.json                 dimension: dimension_type + generator + biome source

Ids keep their sub-directory: worldgen/density_function/hills/base.json is
"<ns>:hills/base".

TWO WAYS TO SHIP IT
  * New dimension:  data/<ns>/dimension/<name>.json, reachable with
    /execute in <ns>:<name> run tp @s ~ ~ ~
  * Replace the overworld:  write the same generator to
    data/minecraft/dimension/overworld.json. Every new world made with the
    pack enabled uses your terrain. (Existing worlds keep their old terrain.)

Datapacks go in <world>/datapacks/ , or in .minecraft/datapacks (via the
"Data Packs" button on the world-creation screen) so a new world starts with it.
""",
    "router": """THE NOISE ROUTER
================
noise_settings.noise_router has 15 fields. ALL of them must be present or the
file is rejected and the dimension silently falls back. Fields you do not use
are the number 0.

  final_density              THE ONE THAT MATTERS.  > 0 is solid, <= 0 is air
                             (or water below sea_level).
  temperature, vegetation    climate for biome placement (vegetation == humidity)
  continents, erosion, depth, ridges
                             the other four climate parameters (ridges == weirdness)
  preliminary_surface_level  approximate ground height; only used by the
                             `above_preliminary_surface` surface condition
  barrier, fluid_level_floodedness, fluid_level_spread, lava
                             aquifers.  0 when aquifers_enabled is false
  vein_toggle, vein_ridged, vein_gap
                             ore veins.  0 when ore_veins_enabled is false

Everything else in the file:

  noise: {min_y, height, size_horizontal, size_vertical}
      min_y and height must be multiples of 16, min_y + height <= 2032.
      size_horizontal/size_vertical are 1..4 and set the noise cell:
      cell width = size_horizontal * 4, cell height = size_vertical * 4.
      Terrain is only defined at cell corners and interpolated between them,
      so a bigger cell = smoother/cheaper, a smaller cell = sharper/costlier.
      The overworld uses 1 and 2 -> cells 4 wide, 8 tall.
  sea_level            water fills every air block below this y.
  default_block        what solid density becomes (usually minecraft:stone)
  default_fluid        what fills below sea level
  aquifers_enabled     underground water pockets; needs the 4 aquifer fields
  ore_veins_enabled    copper/iron vein blobs; needs the 3 vein fields
  legacy_random_source false for anything modern
  surface_rule         what block sits on top (see `reference surface`)
  spawn_target         [] is fine for a custom dimension
""",
    "density": """DENSITY FUNCTION TYPES (26.2)
=============================
A density function is a number, a reference string ("<ns>:<id>"), or an object
with a "type".

ARITHMETIC
  add / mul / min / max      argument1, argument2
  abs / square / cube        argument
  half_negative              argument      d>0 ? d : d*0.5
  quarter_negative           argument      d>0 ? d : d*0.25
  squeeze                    argument      c=clamp(d,-1,1); c/2 - c^3/24
  invert                     argument      1/d
  clamp                      input, min, max
  constant                   argument      (or just write the bare number)

POSITION
  y_clamped_gradient         from_y, to_y, from_value, to_value
                             linear in y, clamped outside the range. This is
                             how "higher = less solid" gets expressed.

NOISE
  noise                      noise, xz_scale, y_scale
                             samples at (x*xz_scale, y*y_scale, z*xz_scale).
                             y_scale 0 makes it purely 2D.
  shifted_noise              noise, shift_x, shift_y, shift_z, xz_scale, y_scale
                             like `noise` but adds the shift functions to the
                             sample position; vanilla passes shift_x/shift_z to
                             wobble the climate grid so biomes are not square.
  shift_a / shift_b / shift  argument (a noise id)
                             the wobble itself; shift_a varies with x/z,
                             shift_b with z/x. Use minecraft:shift_x and
                             minecraft:shift_z, which are already defined.
  old_blended_noise          xz_scale, y_scale, xz_factor, y_factor,
                             smear_scale_multiplier
                             the classic 3D terrain noise. Vanilla uses
                             0.25 / 0.125 / 80 / 160 / 8. Bigger xz_factor =
                             wider features; bigger y_factor = more vertical
                             stretch (mesas); smaller = blobbier/overhangy.

SELECTION
  range_choice               input, min_inclusive, max_exclusive,
                             when_in_range, when_out_of_range
  interval_select            input, thresholds[n], functions[n+1]
                             functions MUST be exactly one longer than
                             thresholds. Replaces the old weird_scaled_sampler.
  spline                     spline: {coordinate, points:[{location,value,derivative}]}
                             piecewise cubic; `value` may itself be a spline,
                             which is how vanilla nests continents -> erosion ->
                             ridges.  locations must strictly increase.

CACHING / STRUCTURE  (these change results, not just speed)
  flat_cache                 argument. Evaluated once per 4x4 column at y=0.
                             Wrap every purely-2D function in this: it is what
                             makes climate values constant within a 4x4 cell.
  cache_2d                   argument. One value per column
  cache_once                 argument. Memoises repeated use in one position
  cache_all_in_cell          argument
  interpolated               argument. Sampled at cell corners and trilinearly
                             interpolated. Wrap the main terrain in this: it is
                             what makes vanilla terrain smooth rather than
                             blocky, and it is much cheaper.

SPECIAL
  find_top_surface           density, upper_bound, lower_bound, cell_height
                             scans down for the highest y where density > 0
  end_islands                the End's island field
  blend_alpha / blend_offset / blend_density / beardifier
                             old-world blending and structure flattening; in a
                             fresh dimension these are 1 / 0 / passthrough / 0

REMOVED: initial_density_without_jaggedness and terrain_shaper no longer exist.
weird_scaled_sampler is gone too: the game refuses a pack that names it.
""",
    "terrain": """HOW TERRAIN HEIGHT ACTUALLY WORKS
=================================
Vanilla's shape, which every custom pack should start from:

  depth  = y_clamped_gradient(min_y..max_y : 1.5 .. -1.5)  +  offset
  sloped_cheese = 4 * quarter_negative(depth * factor)  +  base_3d_noise
  final_density = interpolated(squeeze(0.64 * (slides + sloped_cheese)))

Read it as: `depth` is a vertical ramp that is positive underground and negative
in the sky. `offset` slides that ramp up or down per column, so offset is the
knob for "how high is the ground here". `factor` multiplies the ramp, so it
controls how fast density changes with height, which is exactly how much the
3D noise can push the surface around.

  ==> THE HEIGHT FORMULA
      surface_y  ~=  min_y + height/3 * (1.5 + offset)
      For the overworld (min_y -64, height 384):  y ~= -64 + 128 * (1.5 + offset)
        offset -0.78 -> y  30      offset -0.44 -> y  72
        offset -0.66 -> y  45      offset -0.30 -> y  90
        offset -0.50 -> y  64      offset  0.00 -> y 128
      That -0.5 is not arbitrary: it is what puts the ground at sea level.
      A custom offset spline that forgets it builds the world in the clouds.

  ==> RELIEF
      factor large  (6)  -> flat: the noise barely moves the surface (oceans, plains)
      factor small  (1.5)-> dramatic: the same noise swings the surface tens of blocks
      Vanilla measured range: 0.6 .. 6.3, median 4.7.

  ==> JAGGEDNESS
      jaggedness * half_negative(noise(minecraft:jagged, xz_scale 1500)) is added
      to depth before multiplying by factor. Vanilla keeps it 0 almost
      everywhere (median 0.0) and raises it to ~0.55 only at peaks, which is
      what makes mountain ridges spiky while plains stay smooth.

  ==> SLIDES
      Near the world ceiling, add a y_clamped_gradient going negative so terrain
      cannot reach the top; near the floor, one going positive so there is a
      solid base. Without the top slide you get a world capped in stone.

  ==> OVERHANGS / FLOATING ISLANDS
      Overhangs happen when the 3D noise beats the depth ramp. Lower `factor`
      and raise the noise weight. For true floating islands, make `depth` a
      band rather than a ramp: subtract abs(y - island_y) so density is
      positive only in a slab, then let noise break it up.

MEASURED VANILLA RANGES (60k random columns, percentiles 1/25/50/75/99):
  continents      -0.78  -0.22   0.01   0.23   0.76      (min -1.38, max 1.35)
  erosion         -0.70  -0.21   0.00   0.21   0.68
  ridges          -0.82  -0.24   0.00   0.24   0.83
  ridges_folded   -0.99  -0.65  -0.27   0.22   0.96      (the "peaks & valleys" fold)
  offset          -0.73  -0.62  -0.50  -0.46   0.01
  factor           1.37   3.95   4.69   5.78   6.30
  jaggedness       0.00   0.00   0.00   0.00   0.00      (max 0.55, only at peaks)
Use these to sanity-check a spline: if your offset spline outputs +0.4 anywhere,
that column's ground is at y ~ 179.
""",
    "surface": """SURFACE RULES
=============
The game walks each column downwards and asks the rule tree what block to place
at every stone position. The first rule that returns a block wins.

RULES
  sequence   {"sequence": [...]}        first match wins
  condition  {"if_true": <cond>, "then_run": <rule>}
  block      {"result_state": {"Name": "minecraft:grass_block",
                               "Properties": {"snowy": "false"}}}
             Property values are STRINGS, always.
  bandlands  the badlands terracotta banding

CONDITIONS
  Every field below is mandatory, `is_3d` aside: the game refuses the whole
  pack over a missing one rather than defaulting it.

  stone_depth   {offset, surface_type: floor|ceiling, add_surface_depth,
                 secondary_depth_range}
                true within `offset+1` blocks of the top (floor) or bottom
                (ceiling) of the stone.  offset 0 == "the surface block itself".
  water         {offset, surface_depth_multiplier, add_stone_depth}
                true when this block is above the local water line + offset.
                offset -1 == "at or above water level" == dry land.
  y_above       {anchor: {absolute|above_bottom|below_top: N},
                 surface_depth_multiplier, add_stone_depth}
  biome         {"biome_is": "<id>"}  or a list of ids
  noise_threshold {noise, min_threshold, max_threshold, is_3d}
  vertical_gradient {random_name, true_at_and_below, false_at_and_above}
                a randomised fade between two heights: bedrock and the
                stone/deepslate transition
  steep         the ground drops 4+ blocks over 2 blocks (cliff faces)
  hole          a dip in the surface-depth noise
  temperature   cold enough to snow
  above_preliminary_surface  keeps surface blocks out of caves
  not           {"invert": <cond>}

TYPICAL ORDER
  1. bedrock at the bottom (vertical_gradient)
  2. stone_depth offset 0  -> the top block: grass if `water offset -1`,
     otherwise gravel/sand
  3. stone_depth with add_surface_depth -> a band of dirt below it
  4. vertical_gradient -> deepslate low down
  5. fall through to default_block

If nothing matches, the block stays default_block. A world of bare stone means
the rule tree never fired, usually because a `biome` condition names a biome the
dimension does not actually place.
""",
    "biomes": """BIOME PLACEMENT
===============
A multi_noise biome source assigns the biome whose 6D box is nearest to the
climate at that point. Boxes need not tile the space and gaps are fine, but
overlapping boxes mean the loser never appears anywhere.

  "biome_source": {
    "type": "minecraft:multi_noise",
    "biomes": [
      {"biome": "<ns>:ashen_flats",
       "parameters": {
         "temperature": [0.2, 1.0], "humidity": [-1.0, -0.35],
         "continentalness": [-0.11, 1.0], "erosion": [0.05, 1.0],
         "depth": [0.0, 0.0], "weirdness": [-1.0, 1.0], "offset": 0.0}}
    ]}

All seven keys are required. A single number means a point, a [min,max] pair a
range. `offset` is a constant distance penalty: raise it to make an entry a
fallback that only wins when nothing else is close.

WHAT THE PARAMETERS MEAN
  continentalness  where the land is. Vanilla's bands:
      mushroom  -1.2 .. -1.05     deep ocean -1.05 .. -0.455
      ocean    -0.455 .. -0.19    coast      -0.19 .. -0.11
      near     -0.11 .. 0.03      mid         0.03 .. 0.3
      far        0.3 .. 1.0
  erosion          how flattened.  -1 = mountainous, +1 = flat plains.
  temperature      5 bands: -1..-0.45, -0.45..-0.15, -0.15..0.2, 0.2..0.55, 0.55..1
  humidity         5 bands: -1..-0.35, -0.35..-0.1, -0.1..0.1, 0.1..0.3, 0.3..1
  weirdness        the raw ridge noise. Vanilla splits it into a negative and a
                   positive half so each biome type appears twice; the folded
                   value (peaks and valleys) is what actually shapes terrain.
  depth            0 at the surface, ~1 deep underground. Surface biomes use
                   [0,0] or [0,1]; cave biomes use [0.2,1] or similar.

CHECKING IT
  `worldsmith check` samples the climate cube and reports any entry that never
  wins: the "my biome doesn't spawn" bug, caught before you load the world.
  A custom biome file needs at least temperature, downfall, effects and
  (for 26.2) effects.water_color.
""",
    "builds": """PUTTING A BUILD IN A WORLD
=========================
A build is a box of blocks the game copies in: a template .nbt plus three JSON
files. worldsmith.structures.add writes all four from a voxel.Grid, and
worldsmith.shapes has the geometry a build is made of: speckle for masonry that
does not read as extruded, hollow_box, cylinder, perimeter and ring_cells,
crenellate, gable_roof and stair_flight.

  data/<ns>/structure/<name>.nbt        the blocks
  worldgen/structure/<name>.json        what it is, and where it may go
  worldgen/template_pool/<name>.json    the pool holding the template
  worldgen/structure_set/<name>s.json   how often, and how far apart

  ==> THE ELEMENT TYPE
      Use `legacy_single_pool_element`. The modern `single_pool_element`
      ignores air in a template, so every room a build hollows out comes back
      solid and every trench it digs comes back filled. This is silent.

  ==> HOW HIGH IT LANDS
      With project_start_to_heightmap set, the template's y=0 lands at
      (heightmap at the site) + start_height. So a build whose floor is at
      template y=N wants start_height {"absolute": -(N+1)} to sit flush.
      Bury the courses below the floor: on a slope they are what stops the
      build standing on a pillar of air.

  ==> TERRAIN ADAPTATION
      none          the ground is left exactly as it is
      beard_thin    a skirt of dirt under the pieces; vanilla's villages
      beard_box     the whole bounding box is cleared of terrain and supported;
                    the ancient city uses it, and anything wider than a house
                    wants it or a hillside will stand inside the walls
      bury / encapsulate   for builds meant to be underground

  ==> WHERE IT LANDS
      random_spread picks one chunk in every spacing-by-spacing region of
      chunks, from seed, region and salt. separation must be below spacing.
      spread_type linear or triangular, the second clustering toward the middle
      of a region. An exclusion_zone keeps one set away from another's sites.
      A set holding several builds draws one per site by weight.
      `worldsmith sites <pack>` lists all of that without generating anything.

  ==> THE ANCHOR, WHICH IS NOT THE MIDDLE
      A jigsaw build is anchored at the chunk's MINIMUM corner and turned about
      that corner, so the box grows away from it in a direction that depends on
      a random rotation. The game then reads the ground height and the biome at
      the MIDDLE of that box. For a 64 wide build those two points are 32
      blocks apart, which is enough to be a different biome and a different
      answer. `worldsmith render <pack> --builds` draws the real box.

  ==> BIOMES
      The biomes list is the whole placement rule the game gives you. Either a
      list of ids or a tag ("#minecraft:is_forest"), and worldsmith expands
      vanilla's tags, so `sites` filters on a tag the way the game does.
      Name biomes your own dimension actually places, and remember that a pack
      that controls its own terrain controls this: a biome that only exists on
      flat ground is a rule that says "build on flat ground".

  ==> WHAT WILL BITE
      * a block state that does not parse is dropped, leaving a hole
      * features are placed AFTER structures, so a grass courtyard grows a
        wood. Pave anything you do not want trees in: gravel, dirt_path,
        cobblestone and stone are not soil, coarse_dirt and podzol are
      * chests need {"id": "minecraft:chest", "LootTable": "<table>"}
      * size: vanilla ships nothing above 48 blocks across. 96 works, checked
        against a server with every one of its 9216 floor blocks in place
      * check the size, `worldsmith build <pack>`, before wondering why a world
        takes a while: a 64x52x64 build is a hundred thousand blocks per copy

  ==> THE GROUND AROUND IT IS NOT THE GROUND YOU RENDERED
      terrain_adaptation reshapes the terrain against the build, by ten blocks
      or more, and worldsmith's terrain does not model that. So `sites` relief
      is the ground before the build lands, which is the right number for "will
      this look wrong", and anything that needs the ground after it lands has to
      read the finished world:

        worldsmith inspect <world> --pack <pack> --structure <id>

      which also says whether the model called that site right.

  ==> THE LOOP
      worldsmith new    <pack> --with-build             start from one that works
      worldsmith build  <pack> --id <id> --plan 10,17   look at it
      worldsmith build  <pack> --id <id> --site 0       look at it on its ground
      worldsmith sites  <pack>                          where it lands, on what
      worldsmith render <pack> --builds                 see that on the terrain
      worldsmith check  <pack>                          the silent mistakes
      worldsmith play   <pack>                          walk into it
""",
    "mistakes": """MISTAKES THAT COST A WORLD RELOAD
=================================
* Missing a router field. All 15 are mandatory; the file is rejected wholesale.
* "firstOctave" is camelCase. Everything else in worldgen is snake_case.
* Forgetting the -0.5 in the offset spline: the ground ends up near y 128 with
  the ocean far below it. See `reference terrain`.
* No top slide: terrain grows until it hits the world ceiling and the sky is
  solid stone.
* interval_select with len(functions) == len(thresholds). It must be n+1.
* Spline point locations out of order, or duplicated. They must strictly
  increase.
* Block property values written as numbers or booleans. They are strings:
  {"level": "0"}, not {"level": 0}.
* Two biome entries with identical parameters: one of them never generates.
* Using a `biome` surface condition for a biome the dimension never places, so
  the surface stays bare stone.
* aquifers_enabled true with the aquifer router fields left at 0: you get flat
  sheets of water underground.
* A structure that never generates. Structures find their biomes through
  data/minecraft/tags/worldgen/biome/has_structure/<name>.json, so a custom
  world has none until those tags list its biomes. A typo in one is silent:
  the tag is simply empty and no village is ever placed.
* min_y/height not multiples of 16.
* A build placed with single_pool_element instead of legacy_single_pool_element:
  the air in its template is thrown away, so its rooms are solid. See
  `reference builds`.
* Overwriting data/minecraft/dimension/overworld.json and expecting an EXISTING
  world to change. Terrain settings are baked in at world creation.
""",
    "workflow": """THE LOOP
========
  worldsmith new packs/<name> --namespace <ns> --name <name>
  ...edit the JSON...
  worldsmith check packs/<name>          # schema + dangling refs + smoke test
  worldsmith render packs/<name> --out renders/<name>.png
  ...look at the image, change one thing, render again...
  worldsmith export packs/<name>         # zip for the game

Useful while debugging:
  worldsmith column packs/<name> --at 100 -200
      prints one column's density from the world ceiling down, so you can see
      exactly where it crosses zero.
  worldsmith probe packs/<name> --at 100 64 -200 --density <ns>:offset <ns>:factor
      prints every router value and any named density function at one point.
  worldsmith render ... --step 1 --size 256
      block-accurate detail instead of one pixel per noise cell.
  worldsmith render ... --seed 7 --center 5000 -3000
      different seed / somewhere else, to check it is not a one-spot fluke.
""",
}


def reference_text(topic_name: str = "index") -> str:
    if topic_name in ("index", "", None):
        lines = ["worldsmith reference topics:", ""]
        for name in TOPICS:
            first = TOPICS[name].splitlines()[0]
            lines.append(f"  {name:10s} {first}")
        lines.append("")
        lines.append("  worldsmith reference all    prints everything")
        return "\n".join(lines)
    if topic_name == "all":
        return "\n\n\n".join(TOPICS[name] for name in TOPICS)
    if topic_name not in TOPICS:
        return f"unknown topic '{topic_name}'. try: {', '.join(TOPICS)}"
    return TOPICS[topic_name]
