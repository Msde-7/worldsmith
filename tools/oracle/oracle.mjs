// Generates golden values from deepslate (the library behind misode.github.io/worldgen)
// so the worldsmith Python engine can be verified against a known-good implementation.
import fs from 'node:fs'
import path from 'node:path'
import {
	XoroshiroRandom, LegacyRandom, NormalNoise, BlendedNoise, NoiseParameters,
	Identifier, WorldgenRegistries, NoiseGeneratorSettings, RandomState,
	DensityFunction, NoiseChunkGenerator, FixedBiomeSource, BlockState,
} from 'deepslate'

const ROOT = path.resolve(process.argv[2] ?? '../../vanilla/26.2')
const OUT = path.resolve(process.argv[3] ?? '../../tests/golden')
const SEED = BigInt(process.argv[4] ?? '12345')

function readAll(dir, cb, prefix = '') {
	if (!fs.existsSync(dir)) return
	for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
		if (entry.isDirectory()) readAll(path.join(dir, entry.name), cb, prefix + entry.name + '/')
		else if (entry.name.endsWith('.json')) {
			cb(prefix + entry.name.slice(0, -5), JSON.parse(fs.readFileSync(path.join(dir, entry.name), 'utf8')))
		}
	}
}

const wg = path.join(ROOT, 'data/minecraft/worldgen')
readAll(path.join(wg, 'noise'), (id, json) =>
	WorldgenRegistries.NOISE.register(Identifier.create(id), NoiseParameters.fromJson(json)))
// density functions may reference each other; register lazily so order doesn't matter
readAll(path.join(wg, 'density_function'), (id, json) =>
	WorldgenRegistries.DENSITY_FUNCTION.register(Identifier.create(id), () => DensityFunction.fromJson(json)))
readAll(path.join(wg, 'noise_settings'), (id, json) =>
	WorldgenRegistries.NOISE_SETTINGS.register(Identifier.create(id), () => NoiseGeneratorSettings.fromJson(json)))

const out = { seed: SEED.toString(), version: '26.2' }

// ---------- 1. RNG streams ----------
{
	const rng = {}
	for (const seed of ['0', '12345', '-4242424242', '9223372036854775807']) {
		const r = XoroshiroRandom.create(BigInt(seed))
		const longs = [], doubles = [], ints = []
		for (let i = 0; i < 6; i++) longs.push(r.nextLong().toString())
		for (let i = 0; i < 6; i++) doubles.push(r.nextDouble())
		for (let i = 0; i < 6; i++) ints.push(r.nextInt(256 - i))
		rng['xoroshiro:' + seed] = { longs, doubles, ints }

		const l = new LegacyRandom(BigInt(seed))
		const llongs = [], ldoubles = [], lints = []
		for (let i = 0; i < 6; i++) llongs.push(l.nextLong().toString())
		for (let i = 0; i < 6; i++) ldoubles.push(l.nextDouble())
		for (let i = 0; i < 6; i++) lints.push(l.nextInt(256 - i))
		rng['legacy:' + seed] = { longs: llongs, doubles: ldoubles, ints: lints }
	}
	// positional
	const pos = XoroshiroRandom.create(SEED).forkPositional()
	const hashed = {}
	for (const name of ['minecraft:continentalness', 'minecraft:terrain', 'octave_-9', 'surface', 'minecraft:offset']) {
		const r = pos.fromHashOf(name)
		hashed[name] = [r.nextDouble(), r.nextDouble(), r.nextInt(256)]
	}
	const at = {}
	for (const [x, y, z] of [[0, 0, 0], [1, 2, 3], [-500, 60, 1200], [123456, -40, -654321], [1000000, 0, -1000000]]) {
		const r = pos.at(x, y, z)
		at[`${x},${y},${z}`] = [r.nextDouble(), r.nextFloat()]
	}
	out.rng = { streams: rng, fromHashOf: hashed, at }
}

// ---------- 2. Noise ----------
{
	const noises = {}
	const pos = XoroshiroRandom.create(SEED).forkPositional()
	const coords = [[0, 0, 0], [1.5, 2.5, -3.5], [100, 64, 200], [-3000.25, -12.5, 7777.75], [123456.5, 300, -98765.5]]
	for (const id of ['minecraft:continentalness', 'minecraft:erosion', 'minecraft:ridge', 'minecraft:offset', 'minecraft:temperature', 'minecraft:vegetation', 'minecraft:jagged', 'minecraft:aquifer_barrier']) {
		const params = WorldgenRegistries.NOISE.getOrThrow(Identifier.parse(id))
		const n = new NormalNoise(pos.fromHashOf(id), params)
		noises[id] = { maxValue: n.maxValue, samples: coords.map(([x, y, z]) => n.sample(x, y, z)) }
	}
	const bn = new BlendedNoise(pos.fromHashOf('minecraft:terrain'), 0.25, 0.125, 80, 160, 8)
	noises['blended:terrain'] = { maxValue: bn.maxValue, samples: coords.map(([x, y, z]) => bn.sample(x, y, z)) }
	out.noise = { coords, values: noises }
}

// ---------- 3. Density functions (vanilla overworld router) ----------
function routerGolden(settingsId) {
	const settings = WorldgenRegistries.NOISE_SETTINGS.getOrThrow(Identifier.parse(settingsId))
	const rs = new RandomState(settings, SEED)
	const coords = []
	for (const x of [0, 37, -412, 5000]) for (const z of [0, -91, 730]) for (const y of [-32, 0, 63, 100, 180]) coords.push([x, y, z])
	const res = {}
	for (const key of ['continents', 'erosion', 'ridges', 'depth', 'temperature', 'vegetation', 'finalDensity', 'preliminarySurfaceLevel', 'barrier', 'veinToggle', 'veinRidged', 'veinGap', 'fluidLevelFloodedness', 'fluidLevelSpread', 'lava']) {
		const fn = rs.router[key]
		if (!fn) continue
		res[key] = coords.map(([x, y, z]) => fn.compute(DensityFunction.context(x, y, z)))
	}
	return { coords, values: res }
}
out.router = { 'minecraft:overworld': routerGolden('minecraft:overworld') }
for (const id of ['minecraft:amplified', 'minecraft:nether', 'minecraft:end', 'minecraft:caves', 'minecraft:floating_islands', 'minecraft:large_biomes']) {
	try { out.router[id] = routerGolden(id) } catch (e) { out.router[id] = { error: String(e) } }
}

// ---------- 4. Heightmaps (end-to-end) ----------
{
	const settings = WorldgenRegistries.NOISE_SETTINGS.getOrThrow(Identifier.parse('minecraft:overworld'))
	const rs = new RandomState(settings, SEED)
	const biomeSource = new FixedBiomeSource(Identifier.create('plains'))
	const gen = new NoiseChunkGenerator(biomeSource, settings)
	const cols = []
	for (let i = 0; i < 24; i++) {
		const x = (i % 6) * 37 - 100
		const z = Math.floor(i / 6) * 53 - 80
		cols.push([x, z, gen.getBaseHeight(x, z, 'WORLD_SURFACE_WG', rs)])
	}
	out.heightmap = { settings: 'minecraft:overworld', columns: cols }
}

fs.mkdirSync(OUT, { recursive: true })
fs.writeFileSync(path.join(OUT, 'deepslate_golden.json'), JSON.stringify(out, null, 1))
console.log('wrote', path.join(OUT, 'deepslate_golden.json'))
console.log('router keys:', Object.keys(out.router['minecraft:overworld']?.values ?? {}).join(', '))
