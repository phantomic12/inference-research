#!/usr/bin/env python3
"""Shared record-type registry: schema name -> directory name, list of types.

Every tool imports DIRS/TYPES from here so a type is defined in exactly one place.
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
    "source": "sources",
}

TYPES = list(DIRS)

TITLES = {
    "accelerator": "Accelerators",
    "flop": "FLOP classes",
    "engine": "Inference engines",
    "quantization": "Quantization formats",
    "interconnect": "Interconnect",
    "benchmark": "Benchmarks",
    "gotcha": "Gotchas",
    "source": "Sources",
}