/**
 * Browser-level proof that search renders, driving real Chrome over CDP.
 *
 * Chrome's --screenshot and --dump-dom both snapshot at load, which is before
 * a WASM fetch resolves, so neither can see the search results. This attaches
 * to Chrome's DevTools protocol, waits in real wall-clock time, then reads the
 * DOM. Same engine a reader uses.
 */
import { spawn } from 'node:child_process';
import http from 'node:http';
import { existsSync, readFileSync, mkdirSync } from 'node:fs';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const out = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '.site-shots');
mkdirSync(out, { recursive: true });
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
};

const server = http.createServer((req, res) => {
  let url;
  try { url = decodeURIComponent((req.url || '/').split('?')[0]); } catch { url = '/'; }
  let f = join(site, url);
  if (url.endsWith('/')) f = join(f, 'index.html');
  if (!existsSync(f)) { res.writeHead(404); res.end('nf'); return; }
  res.writeHead(200, { 'content-type': MIME[extname(f)] || 'application/octet-stream' });
  res.end(readFileSync(f));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = 'http://127.0.0.1:' + server.address().port;
console.log('serving', site, 'at', base);

const port = 9300 + Math.floor(Math.random() * 500);
const profile = join(out, '.chrome-cdp');
mkdirSync(profile, { recursive: true });
const chrome = spawn(CHROME, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank',
], { stdio: 'ignore' });

async function cdpWs() {
  for (let i = 0; i < 80; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (r.ok) return (await r.json()).webSocketDebuggerUrl;
    } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('devtools never came up');
}

const ws = new WebSocket(await cdpWs());
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

let msgId = 0;
const pending = new Map();
const events = [];
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  else if (msg.method) events.push(msg);
};
function send(method, params = {}, sessionId) {
  const id = ++msgId;
  return new Promise((res) => {
    pending.set(id, res);
    ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
}

const { result: t } = await send('Target.createTarget', { url: 'about:blank' });
const { result: s } = await send('Target.attachToTarget', { targetId: t.targetId, flatten: true });
const session = s.sessionId;
await send('Page.enable', {}, session);
await send('Runtime.enable', {}, session);
await send('Log.enable', {}, session);

const CASES = [
  ['prefix caching', '/search/?q=prefix+caching', {}],
  ['decode gemm', '/search/?q=decode+gemm', {}],
  ['gfx1100 (type=gotcha)', '/search/?q=gfx1100&type=gotcha', { type: 'gotcha' }],
  ['contested only', '/search/?q=memory+bandwidth&status=contested', { status: 'contested' }],
  ['masthead box', '/accelerator/', { masthead: 'mi300' }],
];

let fails = 0;
for (const [label, path, expect] of CASES) {
  events.length = 0;
  await send('Page.navigate', { url: base + path }, session);
  for (let i = 0; i < 100; i++) {
    if (events.some((e) => e.method === 'Page.loadEventFired')) break;
    await new Promise((r) => setTimeout(r, 100));
  }
  // Real wall-clock wait for the WASM index to load and the query to resolve.
  await new Promise((r) => setTimeout(r, 8000));

  const evalJs = async (expression) => {
    const msg = await send(
      'Runtime.evaluate',
      { expression, returnByValue: true, awaitPromise: true },
      session,
    );
    if (msg.error || !msg.result || msg.result.exceptionDetails) return undefined;
    return msg.result.result?.value;
  };
  let names = [];
  let badges = [];
  let typeOk = true;

  if (expect.masthead) {
    await evalJs(`(() => { const i = document.getElementById('q'); i.value = ${JSON.stringify(expect.masthead)};
      i.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);
    await new Promise((r) => setTimeout(r, 8000));
    const html = (await evalJs('document.getElementById("sr").outerHTML')) || '';
    names = [...html.matchAll(/class="sr-name">([^<]*)</g)].map((m) => m[1]);
    badges = [...html.matchAll(/class="status status-(\w+)"/g)].map((m) => m[1]);
  } else {
    const html = (await evalJs('document.querySelector("main").outerHTML')) || '';
    names = [...html.matchAll(/class="row-name"[^>]*>([^<]+)</g)].map((m) => m[1]);
    badges = [...html.matchAll(/class="status status-(\w+)"/g)].map((m) => m[1]);
    if (expect.type) {
      const subs = (await evalJs(
        '[...document.querySelectorAll("main .row-sub")].map(e => e.textContent).join("|")')) || '';
      typeOk = subs.split('|').filter(Boolean).every((x) => x.includes(expect.type));
    }
  }

  const errs = events.filter((e) =>
    e.method === 'Runtime.exceptionThrown' ||
    (e.method === 'Log.entryAdded' && e.params?.entry?.level === 'error'));
  const ok = names.length > 0 && errs.length === 0 && typeOk;
  if (!ok) fails += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}`);
  console.log(`      ${names.length} results; badges: ${[...new Set(badges)].join(', ') || 'none'}`);
  for (const n of names.slice(0, 3)) console.log(`        - ${n.replace(/<[^>]*>/g, '').slice(0, 76)}`);
  if (!typeOk) console.log(`      FAIL type filter leaked`);
  if (errs.length) {
    console.log('      console errors: ' + errs.slice(0, 2)
      .map((e) => e.params?.entry?.text || e.params?.exceptionDetails?.text).join(' | '));
  }
}

server.close();
chrome.kill();
console.log(fails === 0 ? '\nSEARCH RENDERS IN A REAL BROWSER' : `\n${fails} BROWSER CHECK(S) FAILED`);
process.exit(fails === 0 ? 0 : 1);