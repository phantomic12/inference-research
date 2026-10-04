#!/usr/bin/env python3
"""Shared record-type registry: schema name -> directory name, list of types.

Every tool imports DIRS/TYPES from here so a type is defined in exactly one place.

REF_FIELDS is the same idea for cross-references: the fields whose values are
BARE record slugs rather than prose or configuration names. It lives here, and
not in either validator, because two tools resolve references — validate.py
(gates CI) and build_site_data.py (builds the site) — and two definitions of
"which fields hold ids" is how a dangling reference gets reported by one tool
and silently dropped by the other. That is exactly the defect this table was
extracted to end.
"""
from __future__ import annotations

# schema/type name  ->  data/ directory name
DIRS = {
    "accelerator": "accelerators",
    "flop": "flops",
    "engine": "engines",
    "quantization": "quantization",
    "interconnect": "interconnect",
    "benchmark": "benchmarks",
    "gotcha": "gotchas",
    "supply": "supply",
    "model": "models",
    "paper": "papers",
    "source": "sources",
    "compiler": "compilers",
    "metric_exposure": "metric-exposures",
}

TYPES = list(DIRS)

# Fields whose values are bare record slugs, and the type each should point at
# ("any" = no type constraint, the field may cross types). `sources` is
# deliberately absent: it is handled separately because a dangling source has
# always been a hard error, not budgeted debt.
#
# A field NOT listed here is not a reference, even when its value looks like a
# slug: benchmark.model is a model NAME, gotcha.concepts is a free-text scope
# marker, accelerator.form_factors is a config enum, accelerator.memory_type is
# a memory technology. Counting those as references would drown the signal.
REF_FIELDS: dict[str, dict[str, str]] = {
    "accelerator": {"interconnect": "interconnect"},
    "benchmark": {
        "engine_id": "engine",
        "accelerator_ids": "accelerator",
        "interconnect_ids": "interconnect",
        "format_id": "quantization",
    },
    "flop": {"affected_by_hardware": "accelerator"},
    "gotcha": {"affects": "any"},
    "paper": {"hardware_relevance": "accelerator"},
    "quantization": {"native_support": "accelerator", "emulated_support": "accelerator"},
    "supply": {"accelerator_ids": "accelerator"},
    "metric_exposure": {"engine_id": "engine"},
}

# Fields that hold a single reference and may be null (so an absent value is
# not a dangling reference).
NULLABLE_REF_FIELDS = frozenset({"format_id"})

# (type, field) pairs where an unresolved reference is ALWAYS fatal, not
# budgeted debt. benchmark.engine_id is required by the schema and names the
# thing that produced the number; a benchmark whose engine cannot be resolved
# cannot be interpreted at all, so this has always been an error and stays one.
HARD_REF_FIELDS = frozenset({("benchmark", "engine_id")})

TITLES = {
    "accelerator": "Accelerators",
    "flop": "FLOP classes",
    "engine": "Inference engines",
    "quantization": "Quantization formats",
    "interconnect": "Interconnect",
    "benchmark": "Benchmarks",
    "gotcha": "Gotchas",
    "supply": "Where to buy / source hardware",
    "model": "Models",
    "paper": "Papers",
    "source": "Sources",
    "compiler": "Compilers & kernel DSLs",
    "metric_exposure": "Metric exposures",
}