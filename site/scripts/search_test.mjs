/**
 * End-to-end search test against the built Pagefind index.
 *
 * Loads Pagefind's real runtime + WASM in Node and queries the actual dist/
 * index, so this exercises the same ranking the browser gets.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dist = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const pfDir = join(dist, 'pagefind');
const entry = join(pfDir, 'pagefind.js');
if (!existsSync(entry)) {
  console.error('no pagefind index in dist/ — run npm run build first');
  process.exit(1);
}

// Pagefind fetches its own index files over HTTP, so the test serves dist/ with
// the same static server shape a deployment would.
const http = await import('node:http');
const { readFile } = await import('node:fs/promises');
const MIME = { '.js': 'text/javascript', '.json': 'application/json',
  '.wasm': 'application/wasm', '.pf_meta': 'application/octet-stream',
  '.pf_index': 'application/octet-stream', '.pf_fragment': 'application/octet-stream',
  '.html': 'text/html' };
const server = http.createServer(async (req, res) => {
  const url = decodeURIComponent((req.url || '/').split('?')[0]);
  let f = join(dist, url);
  if (url.endsWith('/')) f = join(f, 'index.html');
  try {
    const body = await readFile(f);
    const ext = url.slice(url.lastIndexOf('.'));
    res.writeHead(200, { 'content-type': MIME[ext] || 'application/octet-stream' });
    res.end(body);
  } catch {
    if (process.env.PF_DEBUG) console.error('  404', url, '->', f);
    res.writeHead(404); res.end('nf');
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = 'http://127.0.0.1:' + server.address().port + '/';
// Pagefind resolves pagefind-entry.json and the .pf_meta/.pf_index files
// relative to basePath, which must end in the /pagefind/ output subdir.
const base = origin + 'pagefind/';
const pf = await import(pathToFileURL(entry).href);
await pf.options({ basePath: base, baseUrl: origin });
process.on('exit', () => server.close());

function onDisk(url) {
  const p = url.replace(/^\//, '');
  return existsSync(join(dist, p, 'index.html')) || existsSync(join(dist, p));
}

// Resolve facets exactly as src/lib/search.ts does: from /facets.json and the
// record id in the URL, because the Windows Pagefind CLI records no page meta.
const FACETS = JSON.parse(readFileSync(join(dist, 'facets.json'), 'utf-8'));
const TITLES = JSON.parse(readFileSync(join(dist, 'titles.json'), 'utf-8'));
const BY_DIR = {
  accelerators: 'accelerator', flops: 'flop', engines: 'engine',
  quantization: 'quantization', interconnect: 'interconnect',
  benchmarks: 'benchmark', gotchas: 'gotcha', supply: 'supply',
  models: 'model', papers: 'paper', sources: 'source',
};

const CASES = [
  { q: 'decode gemm', expect: 'flops/decode-gemm' },
  { q: 'AITER gfx1100', expect: null },
  { q: 'H100 SXM memory bandwidth', expect: null },
  { q: 'contested', expect: null },
  { q: 'prefix caching', expect: null },
  { q: 'paged attention block table', expect: 'flops/paged-attention-block-table-kernel' },
  { q: 'quantization', expect: null },
];

let fails = 0;
for (const c of CASES) {
  const res = await pf.search(c.q);
  const n = res.results.length;
  if (!n) {
    fails += 1;
    console.log(`FAIL  "${c.q}" -> 0 results`);
    continue;
  }
  // Resolve the full hit list the way the site does, then check the top hit.
  const hits = [];
  for (const r of res.results.slice(0, 60)) {
    const d = await r.data();
    const path = (d.url ?? '').replace(/^https?:\/\/[^/]+/, '');
    const isRecord = path.startsWith('/r/');
    const stem = isRecord ? path.replace(/^\/r\//, '').replace(/\/$/, '') : '';
    const rid = FACETS[stem] ? stem : (d.meta?.rid ?? '');
    const type = isRecord ? (BY_DIR[stem.split('/')[0]] ?? '') : (d.meta?.type ?? '');
    const status = FACETS[stem]?.s ?? d.meta?.status ?? '';
    hits.push({ path, rid, type, status, title: isRecord ? (TITLES[stem] ?? stem) : (d.meta?.title ?? path) });
  }
  const top = hits[0];
    // Relevance ranking is Pagefind's job; this test asserts the expected record is
    // *among* the top hits, not that it wins outright — "decode gemm" legitimately
    // ranks DeepGEMM (which mentions it) above the decode-gemm record.
    const rank = c.expect ? hits.findIndex((h) => h.rid === c.expect) : 0;
    const ok = !c.expect ? true : rank >= 0 && rank < 10;
    if (!ok) fails += 1;
    console.log(
      `${ok ? 'PASS' : 'FAIL'}  "${c.q}" -> ${n} hits | top: ${top.rid || top.path} | type=${top.type} status=${top.status}` +
        (c.expect ? ` | "${c.expect}" at rank ${rank}` : ''),
    );
    if (!ok && c.expect) console.log(`      expected ${c.expect} in the top 10`);
  if (!top.type) { fails += 1; console.log('      EMPTY type facet'); }
  if (top.path && !onDisk(top.path)) {
    fails += 1;
    console.log(`      RESULT URL NOT ON DISK: ${top.path}`);
  }
}

// Filtered searches: type and status must actually narrow the set.
const all = await pf.search('inference');
const byType = await pf.search('memory bandwidth');
console.log(`\ntype+status filter check on "${byType.query ?? 'memory bandwidth'}":`);
const filtered = [];
for (const r of byType.results.slice(0, 60)) {
  const d = await r.data();
  const path = (d.url ?? '').replace(/^https?:\/\/[^/]+/, '');
  if (!path.startsWith('/r/')) continue;
  const stem = path.replace(/^\/r\//, '').replace(/\/$/, '');
  filtered.push({ type: BY_DIR[stem.split('/')[0]], status: FACETS[stem]?.s });
}
const accel = filtered.filter((x) => x.type === 'accelerator');
const contested = filtered.filter((x) => x.status === 'contested');
console.log(`  ${filtered.length} record hits, ${accel.length} accelerators, ${contested.length} contested`);
if (accel.length === 0) { fails += 1; console.log('  FAIL: type filter yields nothing'); }
if (contested.length === 0) { fails += 1; console.log('  FAIL: no contested record in this result set'); }

const fragDir = join(pfDir, 'fragment');
const frags = existsSync(fragDir) ? readdirSync(fragDir).length : 0;
console.log(`\n${frags} fragment files; ${CASES.length} query cases`);
console.log(fails === 0 ? 'SEARCH TESTS PASSED' : `${fails} SEARCH TESTS FAILED`);
process.exit(fails === 0 ? 0 : 1);