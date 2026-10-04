/**
 * Client-side search over the built site, backed by Pagefind.
 *
 * Pagefind indexes the *rendered* HTML after `astro build`, so search covers the
 * notes prose, every field and the doc pages, and it ships as static JSON+WASM —
 * no server, and it works offline from the filesystem.
 *
 * The /search/ page and the masthead box share this module so both filter and
 * rank identically.
 *
 * Filtering note: Pagefind's own filter index is built by a CLI flag that the
 * Windows binary in this toolchain does not expose, so type/status filtering is
 * applied here over the returned result set instead. Every record page carries
 * `data-pagefind-meta` for `type`, `status` and `rid`, so those values are
 * available on every hit with no extra index.
 */

export interface SearchHit {
  url: string;
  title: string;
  type: string;
  rid: string;
  status: string;
  summary: string;
}

interface PagefindResult {
  data(): Promise<{
    url: string;
    excerpt: string;
    meta: Record<string, string>;
  }>;
}

interface PagefindModule {
  search(term: string, opts?: Record<string, unknown>): Promise<{ results: PagefindResult[] }>;
  options(opts: { baseUrl?: string; excerptLength?: number }): Promise<void>;
}

const TYPE_ORDER = [
  'accelerator', 'flop', 'engine', 'quantization', 'interconnect', 'benchmark',
  'gotcha', 'supply', 'model', 'paper', 'source', 'doc', 'typeindex', 'page',
];

let cached: Promise<PagefindModule | null> | null = null;

export function loadPagefind(): Promise<PagefindModule | null> {
  if (cached) return cached;
  // Pagefind's runtime only exists in dist/ after `pagefind --site dist`, so it
  // is not resolvable at build time. Building the specifier at runtime keeps it
  // a genuine browser-side dynamic import of a static file, with the bundler
  // never asked to resolve it.
  const spec = '/pagefind/pagefind.js';
  cached = import(/* @vite-ignore */ spec)
    .then(async (m: any) => {
      // basePath must point at the /pagefind/ output subdir: Pagefind resolves
      // pagefind-entry.json and the .pf_meta/.pf_index files relative to it.
      // baseUrl is what result URLs are reported relative to.
      await m.options({ basePath: '/pagefind/', baseUrl: '/', excerptLength: 40 });
      return m as PagefindModule;
    })
    .catch(() => null);
  return cached;
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
}

export function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Wrap query-term matches in <mark>, on already-escaped text. */
export function highlight(escaped: string, terms: string[]): string {
  let out = escaped;
  for (const t of terms) {
    if (t.length < 2) continue;
    out = out.replace(new RegExp(`(${escapeRe(t)})`, 'gi'), '<mark>$1</mark>');
  }
  return out;
}

export function queryTerms(q: string): string[] {
  return q.toLowerCase().split(/\s+/).filter((t) => t.length >= 2);
}

export function hitUrl(url: string): string {
  return url.endsWith('/') ? url : `${url}/`;
}

/**
 * A record page is one under /r/; a type index under /<type>/. A filtered search
 * should return records, not the index pages that merely list them, unless the
 * reader filtered by type.
 */
function isIndexPage(url: string, type: string): boolean {
  if (url.includes('/docs/') || url.includes('/compare/') || url.includes('/graph/')) return false;
  if (url.startsWith('/r/')) return false;
  return url.replace(/^\//, '').split('/')[0] !== type;
}

export async function runSearch(
  q: string,
  filters: { type?: string; status?: string } = {},
): Promise<SearchHit[] | null> {
  const pf = await loadPagefind();
  if (!pf) return null;
  const res = await pf.search(q);

  const wantType = filters.type ?? '';
  const wantStatus = filters.status ?? '';
  // --- facets ------------------------------------------------------------
  // Pagefind's CLI on Windows (pagefind_extended) exposes no flag to build its
  // filter index, and it only records meta declared at index time. So the two
  // facets /search/ filters on are resolved from a small static map shipped
  // alongside the site: for a record, the type is the directory in its URL and
  // the status is in the map. Cheap, exact, and independent of the index format.
  const FACETS: Record<string, { t: string; s: string }> = await fetch('/facets.json')
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}));
  const TITLES: Record<string, string> = await fetch('/titles.json')
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}));

  const BY_DIR: Record<string, string> = {
    accelerators: 'accelerator', flops: 'flop', engines: 'engine',
    quantization: 'quantization', interconnect: 'interconnect',
    benchmarks: 'benchmark', gotchas: 'gotcha', supply: 'supply',
    models: 'model', papers: 'paper', sources: 'source',
  };

  const facetsOf = (h: SearchHit): { t: string; s: string } => {
    const known = FACETS[h.rid];
    if (known) return known;
    if (!h.url.startsWith('/r/')) return { t: h.type, s: h.status };
    const dir = h.url.replace(/^\/r\//, '').split('/')[0];
    return { t: BY_DIR[dir] ?? h.type, s: h.status };
  };

  const raw: SearchHit[] = [];
  // Over-fetch, then narrow client-side: a filtered view needs enough candidates
  // to still fill a page of results.
  const limit = wantType || wantStatus ? 250 : 60;
  for (const r of res.results.slice(0, limit)) {
    const d = await r.data();
    const hit: SearchHit = {
      url: hitUrl(d.url),
      // The record page's <title> is the record name, but a doc or index page's
      // is its own; prefer the explicit titles map when the URL is a record.
      title: '',
      type: d.meta.type ?? '',
      rid: d.meta.rid ?? '',
      status: d.meta.status ?? '',
      summary: d.excerpt ?? '',
    };
    const path = hit.url.replace(/^\/r\//, '');
    const isRecord = hit.url.startsWith('/r/');
    if (isRecord) {
      const stem = path.replace(/\/$/, '');
      hit.rid = FACETS[stem] ? stem : hit.rid;
      hit.title = TITLES[hit.rid] ?? hit.rid;
    }
    if (!hit.title) hit.title = d.meta.title ?? path;
    const f = facetsOf(hit);
    hit.type = f.t || hit.type;
    hit.status = f.s || hit.status;
    if (wantType && hit.type !== wantType) continue;
    if (wantStatus && hit.status !== wantStatus) continue;
    raw.push(hit);
  }

  // Records first, then docs, then index pages: a reader searching for a fact
  // wants the record, not the page that lists records.
  raw.sort((a, b) => {
    const rank = (h: SearchHit) => {
      if (h.url.startsWith('/r/')) return 0;
      if (h.type === 'doc') return 1;
      if (h.url.includes('/docs/')) return 1;
      return 2;
    };
    const ra = rank(a);
    const rb = rank(b);
    if (ra !== rb) return ra - rb;
    const ta = TYPE_ORDER.indexOf(a.type);
    const tb = TYPE_ORDER.indexOf(b.type);
    return (ta < 0 ? 99 : ta) - (tb < 0 ? 99 : tb);
  });

  const filtered = raw.filter((h) => !isIndexPage(h.url, wantType));
  return (filtered.length ? filtered : raw).slice(0, 60);
}

export function statusBadge(status: string): string {
  if (!status) return '';
  return `<span class="status status-${escapeHtml(status)}">${escapeHtml(status)}</span>`;
}