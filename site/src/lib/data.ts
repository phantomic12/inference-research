/**
 * Shared record helpers. Everything the site renders comes from the payload
 * written by tools/build_site_data.py — no record is ever hand-written here.
 */
import meta from '../data/meta.json';
import docs from '../data/docs.json';

export type Status = 'verified' | 'draft' | 'contested' | 'deprecated';

export interface Row {
  rid: string;
  name: string;
  status: Status | string;
  confidence: number | null;
  updated: string | null;
  head: string;
  tail: string[];
  facets: Record<string, string>;
  nbackrefs: number;
  nrefs: number;
}

export interface Field {
  key: string;
  label: string;
  value: unknown;
  prose: boolean;
  ref: boolean;
  url: boolean;
}

export interface RefGroup {
  field: string;
  label: string;
  targets: string[];
}

export interface Backref {
  from: string;
  via: string;
}

export interface Record_ {
  rid: string;
  type: string;
  name: string;
  status: Status | string;
  confidence: number | null;
  updated: string | null;
  head: string;
  tail: string[];
  facets: Record<string, string>;
  fields: Field[];
  notes: string;
  refs: RefGroup[];
  backrefs: Backref[];
}

export interface DocEntry {
  slug: string;
  title: string;
  order: number;
  chars: number;
  markdown: string;
}

export interface TypeInfo {
  type: string;
  title: string;
  dir: string;
  count: number;
  primary: boolean;
  facets: string[];
}

export const META = meta as {
  total: number;
  counts: Record<string, number>;
  types: TypeInfo[];
  status: Record<string, number>;
  status_order: Status[];
  mean_confidence: number | null;
  dangling: { from: string; field: string; value: string }[];
  docs: { slug: string; title: string; order: number }[];
};

export const DOCS = docs as DocEntry[];
export const DOC_INDEX = META.docs;

/** Page URL for a qualified record id such as `accelerators/nvidia-h100-sxm`. */
export function recUrl(rid: string): string {
  return `/r/${rid}/`;
}

export function typeUrl(type: string): string {
  return `/${type}/`;
}

export function docUrl(slug: string): string {
  return `/docs/${slug}/`;
}

export function docBySlug(slug: string): DocEntry | undefined {
  return DOCS.find((d) => d.slug === slug);
}

export const PRIMARY_TYPES = META.types.filter((t) => t.primary);
export const SOURCE_TYPE = META.types.find((t) => t.type === 'source')!;

export function typeInfo(type: string): TypeInfo | undefined {
  return META.types.find((t) => t.type === type);
}

export function typeTitle(type: string): string {
  return typeInfo(type)?.title ?? type;
}

/** Singular display for a record type, e.g. `benchmark` -> `Benchmark`. */
export function singular(type: string): string {
  const t = typeTitle(type);
  return t.replace(/s$/, '');
}

/**
 * Split a qualified id into its directory and stem. Used by routing and by the
 * cross-reference renderers.
 */
export function splitRid(rid: string): { dir: string; stem: string } {
  const i = rid.indexOf('/');
  return i === -1 ? { dir: '', stem: rid } : { dir: rid.slice(0, i), stem: rid.slice(i + 1) };
}

/** Confidence as a visible 0.00-1.00 reading plus a coarse band label. */
export function confBand(c: number | null): { label: string; cls: string } {
  if (c === null || c === undefined) return { label: '—', cls: 'conf-none' };
  if (c >= 0.85) return { label: `high · ${c.toFixed(2)}`, cls: 'conf-high' };
  if (c >= 0.65) return { label: `medium · ${c.toFixed(2)}`, cls: 'conf-mid' };
  return { label: `low · ${c.toFixed(2)}`, cls: 'conf-low' };
}

export const STATUS_MEANING: Record<string, string> = {
  verified: 'Checked against a primary source.',
  contested: 'Two sources disagree and the repo deliberately did not resolve it. Do not treat a single number here as settled.',
  draft: 'Not yet verified. Provisional.',
  deprecated: 'Superseded or no longer accurate.',
};

/** Sort helper: contested first, then draft, then verified, then deprecated. */
export function statusRank(s: string | null | undefined): number {
  const i = META.status_order.indexOf((s ?? '') as Status);
  return i === -1 ? META.status_order.length : i;
}