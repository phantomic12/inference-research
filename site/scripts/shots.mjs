/**
 * Screenshot the built site in headless Chrome over HTTP.
 *
 * Serving matters: the pages reference root-absolute asset paths (/_astro/…),
 * which a file:// load cannot resolve — the stylesheet is silently dropped and
 * the page screenshots unstyled. This serves dist/ exactly as a static host
 * would, so the shots show the real rendering.
 *
 *   node scripts/shots.mjs
 */
import { spawn } from 'node:child_process';
import http from 'node:http';
import { mkdirSync, existsSync, readFileSync, statSync } from 'node:fs';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const outDir = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '.site-shots');
mkdirSync(outDir, { recursive: true });

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
if (!existsSync(CHROME)) {
  console.error('chrome not found at', CHROME);
  process.exit(1);
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.wasm': 'application/wasm',
  '.pf_meta': 'application/octet-stream', '.pf_index': 'application/octet-stream',
  '.pf_fragment': 'application/octet-stream', '.pagefind': 'application/octet-stream',
};

function handler(req, res) {
  let url;
  try {
    url = decodeURIComponent((req.url || '/').split('?')[0]);
  } catch {
    url = '/';
  }
  let f = join(site, url);
  if (url.endsWith('/')) f = join(f, 'index.html');
  if (!existsSync(f) || statSync(f).isDirectory()) {
    const nf = join(site, '404.html');
    res.writeHead(404, { 'content-type': MIME['.html'] });
    res.end(existsSync(nf) ? readFileSync(nf) : '404');
    return;
  }
  res.writeHead(200, { 'content-type': MIME[extname(f)] || 'application/octet-stream' });
  res.end(readFileSync(f));
}
const server = http.createServer(handler);
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = 'http://127.0.0.1:' + server.address().port;
console.log('serving dist/ at', base);

const PAGES = [
  ['home', '/', 1500, 2500],
  ['accelerators', '/accelerator/', 1500, 2100],
  ['gotchas', '/gotcha/', 1500, 2100],
  ['record-h100', '/r/accelerators/nvidia-h100-sxm/', 1500, 2300],
  ['record-contested', '/r/accelerators/nvidia-b300/', 1500, 1900],
  ['record-draft', '/r/supply/ai-datacentre-power-capacity-run-cost/', 1500, 1700],
  ['record-gotcha', '/r/gotchas/aiter-gate-skips-rdna3-gfx1100/', 1500, 1700],
  ['record-benchmark', '/r/benchmarks/h100-8gpu-node-loaded-76-percent-of-tdp-measured/', 1500, 1700],
  ['record-paper', '/r/papers/alpaserve/', 1500, 1500],
  ['docs-01', '/docs/01-hardware-selection/', 1500, 2100],
  ['docs-06', '/docs/06-glossary/', 1500, 1600],
  ['search', '/search/?q=prefix+caching', 1500, 1400],
  ['compare', '/compare/?type=accelerator&picks=accelerators/nvidia-h100-sxm,accelerators/nvidia-b200,accelerators/amd-instinct-mi300x', 1600, 1700],
  ['graph', '/graph/', 1500, 1500],
  ['notfound', '/no-such-page/', 1200, 800],
];

const profile = join(outDir, '.chrome-profile');
mkdirSync(profile, { recursive: true });

let fails = 0;
for (const [name, path, w, h] of PAGES) {
  const out = join(outDir, `${name}.png`);
  const res = spawn(CHROME, [
    '--headless', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
    '--no-default-browser-check', `--user-data-dir=${profile}`,
    `--window-size=${w},${h}`, '--virtual-time-budget=30000',
    `--screenshot=${out}`, base + path,
  ], { stdio: 'ignore' });
  const code = await new Promise((r) => res.on('exit', r));
  const bytes = existsSync(out) ? statSync(out).size : 0;
  if (bytes > 1000) {
    console.log(`PASS  ${name.padEnd(18)} ${String(bytes).padStart(8)} bytes  ${path}`);
  } else {
    fails += 1;
    console.log(`FAIL  ${name.padEnd(18)} ${bytes} bytes (exit ${code})  ${path}`);
  }
}

server.close();
console.log(`\nshots in ${outDir}`);
console.log(fails === 0 ? 'ALL PAGES RENDERED' : `${fails} PAGE(S) FAILED`);
process.exit(fails === 0 ? 0 : 1);