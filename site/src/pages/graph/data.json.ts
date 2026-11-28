// Build-time JSON endpoint for the /graph/ page payload.
//
// The page used to inline 3.4 MB of escaped JSON in a <script> tag. This route
// emits the same filtered graph as a separate, cacheable dist/graph/data.json.
// It is generated from the build-time graph.json rather than committed under
// site/public/, so it can never go stale against the record corpus.
import type { APIRoute } from 'astro';
import graph from '../../data/graph.json';

interface Node { i: string; t: string; n: string; s: string }
interface Edge { 0: string; 1: string; 2: string }

const { nodes: allNodes, edges: allEdges } = graph as { nodes: Node[]; edges: Edge[] };
const nodes = allNodes.filter((n) => n.t !== 'source');
const edges = allEdges
  .filter((e) => !e[0].startsWith('sources/') && !e[1].startsWith('sources/'));

export const GET: APIRoute = () =>
  new Response(JSON.stringify({ nodes, edges }), {
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });
