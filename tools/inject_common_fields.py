#!/usr/bin/env python3
"""Inject the common fields into every schema that lacks them.

Run once after adding a record type; safe to re-run (idempotent). Common fields
are defined here so they cannot drift between schemas.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

COMMON_PROPERTIES = OrderedDict([
    ("status", {"enum": ["verified", "draft", "contested", "deprecated"]}),
    ("confidence", {"type": "number"}),
    ("updated", {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}),
    ("sources", {"type": "array", "items": {"type": "string"},
                 "description": "source ids; each must exist in data/sources/"}),
])

# `source` records ARE the sources, so they carry no `sources` field of their own.
NO_SOURCES_FIELD = {"source"}


def main() -> None:
    changed = []
    for t in TYPES:
        p = ROOT / "schemas" / f"{t}.schema.json"
        schema = json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
        props = schema["properties"]
        required = schema["required"]

        for key, spec in COMMON_PROPERTIES.items():
            if key in props:
                continue
            if key == "sources" and t in NO_SOURCES_FIELD:
                continue
            props[key] = OrderedDict(spec)
            if key not in required:
                required.append(key)

        # keep id, type, name first, then common, then type-specific
        head = [k for k in ("id", "type", "name") if k in props]
        tail = [k for k in props if k not in head]
        ordered = OrderedDict((k, props[k]) for k in head + tail)
        schema["properties"] = ordered
        schema["required"] = [k for k in head + tail if k in required]

        new = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
        if new != p.read_text(encoding="utf-8"):
            p.write_text(new, encoding="utf-8")
            changed.append(t)

    print(f"updated: {', '.join(changed) if changed else '(nothing)'}")


if __name__ == "__main__":
    main()