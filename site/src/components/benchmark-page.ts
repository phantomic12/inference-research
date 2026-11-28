/**
 * Rows per page for the benchmark type index.
 *
 * Kept in one place because both the first page (the generic [type] route) and
 * the continuation pages (/benchmark/2/ …) must agree on the slice size.
 */
export const BENCHMARK_PAGE_SIZE = 250;