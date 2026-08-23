// Per-density-function golden values, so a mismatch can be pinned to one node.
import fs from 'node:fs'
import path from 'node:path'
import {
	NoiseParameters, Identifier, WorldgenRegistries, NoiseGeneratorSettings,
	RandomState, DensityFunction,
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
const densityFunctionIds = []
readAll(path.join(wg, 'density_function'), (id, json) => {
	densityFunctionIds.push('minecraft:' + id)
	WorldgenRegistries.DENSITY_FUNCTION.register(Identifier.create(id), () => DensityFunction.fromJson(json))
})
readAll(path.join(wg, 'noise_settings'), (id, json) =>
	WorldgenRegistries.NOISE_SETTINGS.register(Identifier.create(id), () => NoiseGeneratorSettings.fromJson(json)))

const settings = WorldgenRegistries.NOISE_SETTINGS.getOrThrow(Identifier.parse('minecraft:overworld'))
const randomState = new RandomState(settings, SEED)
const visitor = randomState.createVisitor(settings.noise, settings.legacyRandomSource)

const coords = []
for (const x of [0, 37, -412, 5000]) {
	for (const z of [0, -91, 730]) {
		for (const y of [-32, 0, 63, 100, 180]) coords.push([x, y, z])
	}
}

const out = { coords, values: {} }
for (const id of densityFunctionIds) {
	try {
		const fn = WorldgenRegistries.DENSITY_FUNCTION.getOrThrow(Identifier.parse(id)).mapAll(visitor)
		out.values[id] = coords.map(([x, y, z]) => fn.compute(DensityFunction.context(x, y, z)))
	} catch (e) {
		out.values[id] = { error: String(e) }
	}
}

fs.mkdirSync(OUT, { recursive: true })
fs.writeFileSync(path.join(OUT, 'deepslate_density_functions.json'), JSON.stringify(out, null, 1))
console.log('wrote', densityFunctionIds.length, 'density functions')
