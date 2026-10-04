/**
 * Compare view.
 *
 * Loads one type's matrix from /c/<type>-slim.json on demand, so the page itself
 * stays small and a reader comparing two accelerators downloads the accelerator
 * matrix, not the whole corpus. Selection lives in the URL, so a comparison can
 * be pasted into a message.
 */

export interface CompareData {
  labels: Record<string, string>;
  rows: { r: string; n: string; s: string; c: Record<string, string> }[];
}

const cache = new Map<string, CompareData>();

export async function loadCompare(type: string): Promise<CompareData | null> {
  if (!type) return null;
  if (cache.has(type)) return cache.get(type)!;
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}c/${encodeURIComponent(type)}-slim.json`);
    if (!res.ok) return null;
    const data = (await res.json()) as CompareData;
    cache.set(type, data);
    return data;
  } catch {
    return null;
  }
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
}

function recUrl(rid: string): string {
  return `${import.meta.env.BASE_URL}r/${rid}/`;
}

/** Order fields so identity-ish keys come first and prose sinks to the bottom. */
function fieldOrder(labels: Record<string, string>): string[] {
  const rank = (k: string) => {
    const head = k.split('.')[0];
    if (['name', 'vendor', 'architecture', 'scheme', 'kind', 'class', 'family'].includes(head)) return 0;
    if (['severity', 'status'].includes(head)) return 1;
    if (['flops', 'memory', 'vram', 'bandwidth', 'tdp', 'bits', 'params', 'kv'].some((s) => head.includes(s))) return 2;
    if (head === 'notes') return 9;
    return 5;
  };
  return Object.keys(labels).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

export async function renderCompare(host: HTMLElement): Promise<void> {
  const params = new URLSearchParams(location.search);
  let type = params.get('type') ?? 'accelerator';
  const picks = (params.get('picks') ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

  host.innerHTML = `
    <div class="compare-picker">
      <div class="control">
        <label for="c-type">type</label>
        <select id="c-type">
          ${(await typeList()).map((t) => `<option value="${esc(t)}"${t === type ? ' selected' : ''}>${esc(t)}</option>`).join('')}
        </select>
      </div>
      <div class="control" style="flex:1 1 22rem">
        <label for="c-add">add records (up to 4)</label>
        <select id="c-add" multiple size="6"></select>
      </div>
      <span class="control-count" id="c-count"></span>
    </div>
    <div id="c-out"><p class="hint">Loading ${esc(type)}…</p></div>`;

  const typeSel = host.querySelector('#c-type') as HTMLSelectElement;
  const addSel = host.querySelector('#c-add') as HTMLSelectElement;
  const out = host.querySelector('#c-out') as HTMLElement;
  const count = host.querySelector('#c-count') as HTMLElement;

  let chosen = [...picks];
  let data: CompareData | null = null;

  async function refreshOpts() {
    if (!data) return;
    const q = (addSel.value.match(/^q:/)?.[1] ?? '').toLowerCase();
    addSel.innerHTML = data.rows
      .filter((r) => !chosen.includes(r.r))
      .filter((r) => !q || r.n.toLowerCase().includes(q) || r.r.toLowerCase().includes(q))
      .slice(0, 300)
      .map((r) => `<option value="${esc(r.r)}">${esc(r.n)}</option>`)
      .join('');
  }

  function writeUrl() {
    const p = new URLSearchParams();
    p.set('type', type);
    if (chosen.length) p.set('picks', chosen.join(','));
    history.replaceState(null, '', `${import.meta.env.BASE_URL}compare/?${p.toString()}`);
  }

  async function load() {
    out.innerHTML = `<p class="hint">Loading ${esc(type)}…</p>`;
    data = await loadCompare(type);
    if (!data) {
      out.innerHTML = `<p class="empty-state">No compare data for ${esc(type)}.</p>`;
      return;
    }
    // Preset picks may name records of another type; drop what does not belong.
    chosen = chosen.filter((c) => data!.rows.some((r) => r.r === c));
    await refreshOpts();
    draw();
    writeUrl();
  }

  function draw() {
    if (!data) return;
    const rows = chosen
      .map((c) => data!.rows.find((r) => r.r === c))
      .filter((r): r is NonNullable<typeof r> => Boolean(r));
    count.textContent = `${rows.length} of 4 selected`;
    if (!rows.length) {
      out.innerHTML = `<p class="empty-state">Select records above, or use a preset above.</p>`;
      return;
    }

    const keys = fieldOrder(data.labels);
    const shown = keys.filter((k) => rows.some((r) => r.c[k]));

    const head =
      `<tr><th class="rownum"></th><th class="fieldname">field</th>` +
      rows
        .map(
          (r) =>
            `<th><a href="${recUrl(r.r)}">${esc(r.n)}</a>` +
            `<div style="margin-top:.25rem"><span class="status status-${esc(r.s)}">${esc(r.s)}</span></div></th>`,
        )
        .join('') +
      `</tr>`;

    const body = shown
      .map((k) => {
        const vals = rows.map((r) => r.c[k] ?? '');
        // Shade a cell when its value differs from every other selected record,
        // so a disagreement is visible without reading the numbers.
        const distinct = new Set(vals.map((v) => v.trim())).size;
        return (
          `<tr data-field="${esc(k)}">` +
          `<td class="rownum"></td>` +
          `<td class="fieldname" title="${esc(k)}">${esc(data!.labels[k] ?? k)}</td>` +
          vals
            .map(
              (v) =>
                `<td class="${distinct > 1 ? 'differ' : ''}">${
                  v === '' ? '<span class="hint">—</span>' : esc(v)
                }</td>`,
            )
            .join('') +
          `</tr>`
        );
      })
      .join('');

    const disagreements = shown.filter((k) => {
      const vals = rows.map((r) => r.c[k] ?? '').filter((v) => v.trim());
      return new Set(vals.map((v) => v.trim())).size > 1;
    }).length;

    out.innerHTML =
      `<div class="compare-grid"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>` +
      `<p class="hint" style="margin-top:.6rem">` +
      `${shown.length} fields shown · ${disagreements} differ across these records` +
      ` · values are verbatim from the records, including vendor claims</p>`;

    // Row numbering, injected client-side so the markup stays readable.
    out.querySelectorAll('tbody tr').forEach((tr, i) => {
      const n = tr.querySelector('.rownum');
      if (n) n.textContent = String(i + 1);
    });
  }

  typeSel.addEventListener('change', () => {
    type = typeSel.value;
    chosen = [];
    load();
  });
  addSel.addEventListener('change', () => {
    for (const o of Array.from(addSel.selectedOptions)) {
      if (chosen.length >= 4) break;
      if (!chosen.includes(o.value)) chosen.push(o.value);
    }
    addSel.selectedIndex = -1;
    draw();
    writeUrl();
  });

  await load();
}

async function typeList(): Promise<string[]> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}type-list.json`);
    if (res.ok) return (await res.json()) as string[];
  } catch {
    /* fall through */
  }
  return ['accelerator', 'benchmark', 'compiler', 'engine', 'flop', 'gotcha', 'interconnect', 'metric_exposure', 'model', 'paper', 'quantization', 'source', 'supply'];
}