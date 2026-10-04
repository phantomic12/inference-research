// Regenerates site/src/data/ from data/ + docs/ before every Astro build, so the
// site can never ship a stale snapshot. Runs the repo's own build script; it does
// not touch data/, schemas/ or docs/00-index.md.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const script = resolve(here, '../../tools/build_site_data.py');
const payload = resolve(here, '../src/data/meta.json');

if (!existsSync(script)) {
  console.error(`[site] ${script} not found — run this from the repo checkout`);
  process.exit(1);
}

const candidates = ['python', 'python3', 'py'];
let ran = false;
let failure = null;
for (const py of candidates) {
  const res = spawnSync(py, [script], { stdio: 'inherit', cwd: resolve(here, '../..') });
  if (res.error) continue;
  if (res.status === 0) { ran = true; break; }
  failure = `${py} exited ${res.status}`;
}
if (!ran) {
  console.error(`[site] could not run tools/build_site_data.py (${failure}).`);
  console.error('[site] Install Python 3.10+ or run `npm run data` manually, then build.');
  process.exit(1);
}
if (!existsSync(payload)) {
  console.error('[site] build_site_data.py ran but produced no site/src/data/meta.json');
  process.exit(1);
}