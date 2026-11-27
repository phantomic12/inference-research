// Regenerates site/src/data/ from data/ + docs/ before every Astro build, so the
// site can never ship a stale snapshot. Runs the repo's own build script; it does
// not touch data/, schemas/ or docs/00-index.md.
import { spawnSync } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const script = resolve(root, 'tools/build_site_data.py');
const payload = resolve(here, '../src/data/meta.json');
const candidates = ['python', 'python3', 'py'];

// Run a repo tool with the first Python binary that works. A missing script or
// a nonzero exit is a HARD failure: the old silent skip let five gitignored
// ranking docs disappear from a clean checkout, and their pages 404'd.
function runTool(rel, { outputTo, expect = [] } = {}) {
  const abs = resolve(root, rel);
  if (!existsSync(abs)) {
    console.error(`[site] ${rel} not found — run this from the repo checkout`);
    process.exit(1);
  }
  let failure = null;
  for (const py of candidates) {
    const res = spawnSync(py, [rel], {
      cwd: root,
      encoding: 'utf8',
      stdio: outputTo ? ['ignore', 'pipe', 'inherit'] : 'inherit',
    });
    if (res.error) continue;
    if (res.status === 0) {
      if (outputTo) writeFileSync(resolve(root, outputTo), res.stdout);
      for (const expected of expect) {
        if (!existsSync(resolve(root, expected))) {
          console.error(`[site] ${rel} ran but produced no ${expected}`);
          process.exit(1);
        }
      }
      return;
    }
    failure = `${py} ${rel} exited ${res.status}`;
  }
  console.error(`[site] could not run ${rel} (${failure ?? `none of ${candidates.join(', ')} is available`}).`);
  console.error('[site] Install Python 3.10+ or run `npm run data` manually, then build.');
  process.exit(1);
}

// The ranking/synthesis docs are gitignored as "regenerated in CI", so they do
// not exist on a clean checkout. validate.yml and the deploy workflow generate
// them; a local `npm run build` must do the same. This runs BEFORE
// build_site_data.py because its load_docs() reads docs/ into docs.json, which
// is what drives the Astro /docs/[slug] routes. Keep this list in sync with
// .github/workflows/validate.yml and .github/workflows/deploy-site.yml.
runTool('tools/index.py', { expect: ['docs/00-index.md'] });
runTool('tools/rank_models.py', { outputTo: 'docs/29-local-model-ranking.md' });
runTool('tools/intelligence_density.py', { expect: ['docs/30-intelligence-density.md'] });
runTool('tools/mine_derivatives.py', { expect: ['docs/32-finetunes-and-quants.md'] });
runTool('tools/decision_ranking.py', { outputTo: 'docs/33-decision-models.md' });
runTool('tools/agentic_ranking.py', { expect: ['docs/31-agentic-ranking.md'] });
runTool('tools/quant_retention.py', { expect: ['docs/34-quant-retention.md'] });

if (!existsSync(script)) {
  console.error(`[site] ${script} not found — run this from the repo checkout`);
  process.exit(1);
}

let ran = false;
let failure = null;
for (const py of candidates) {
  const res = spawnSync(py, [script], { stdio: 'inherit', cwd: root });
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

// Also build joined-view data (cost-per-token, compatibility, roofline)
runTool('tools/build_joined_views.py');
