#!/usr/bin/env python3
"""Self-tests for the ingestion tooling. Run: python tools/test_tools.py

Each test builds records in a throwaway temp dir, so it never touches data/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent


def run_tool(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run the COPY of the tool inside cwd.

    The tools derive their repo root from __file__, so invoking the originals
    would read and write the real data/ directory. Always run the temp copy.
    """
    return subprocess.run(
        [sys.executable, str(cwd / "tools" / script), *args],
        cwd=cwd, capture_output=True, text=True,
    )


class Harness(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ir-test-"))
        for sub in ("tools", "schemas", "docs"):
            shutil.copytree(ROOT / sub, self.tmp / sub)
        for d in ("accelerators", "flops", "engines", "quantization",
                  "interconnect", "benchmarks", "gotchas", "sources"):
            (self.tmp / "data" / d).mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel: str, obj: dict) -> None:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def read(self, rel: str) -> dict:
        return json.loads((self.tmp / rel).read_text(encoding="utf-8"))

    def validate(self, *extra: str) -> subprocess.CompletedProcess:
        return run_tool("validate.py", *extra, cwd=self.tmp)

    def good_source(self, slug: str = "acme-spec") -> dict:
        return {
            "id": slug, "type": "source", "name": f"{slug} sheet",
            "status": "verified", "confidence": 0.8, "updated": "2026-10-03",
            "url": "https://example.com/spec", "publisher": "ACME",
            "kind": "spec-sheet", "published": None,
            "accessed": "2026-10-03", "archived_url": None, "notes": "",
        }

    def good_accelerator(self, slug: str = "acme-a100") -> dict:
        return {
            "id": slug, "type": "accelerator", "name": "ACME A100",
            "status": "verified", "confidence": 0.9, "updated": "2026-10-03",
            "sources": ["acme-spec"], "vendor": "asic-other",
            "architecture": "test-arch", "release_year": 2025, "process_nm": 12,
            "form_factors": ["pcie"], "vram_gb": 80.0, "memory_type": "hbm3e",
            "memory_bus_bit": 5120, "memory_bandwidth_gbps": 3000.0,
            "memory_bandwidth_basis": "HBM3e 6Gbps x 5120-bit",
            "flops": [{"precision": "fp8", "tflops": 4000.0, "dense": True,
                       "vendor_claim": True, "source_id": "acme-spec"}],
            "tdp_w": 700.0, "interconnect": ["nvlink"], "unified_memory": False,
            "consumer": False, "notes": "",
        }


class TestValidate(Harness):
    def test_empty_repo_is_valid(self):
        r = self.validate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("0 records validated", r.stdout)

    def test_good_records_pass(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/accelerators/acme-a100.json", self.good_accelerator())
        r = self.validate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("2 records validated", r.stdout)

    def test_unknown_field_is_error(self):
        rec = self.good_accelerator()
        rec["mystery_field"] = 1
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/accelerators/acme-a100.json", rec)
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("unknown field 'mystery_field'", r.stdout)

    def test_dangling_source_is_error(self):
        rec = self.good_accelerator()
        rec["sources"] = ["does-not-exist"]
        self.write("data/accelerators/acme-a100.json", rec)
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("cites unknown id 'does-not-exist'", r.stdout)

    def test_id_filename_mismatch_is_error(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        rec = self.good_accelerator()
        rec["id"] = "something-else"
        self.write("data/accelerators/acme-a100.json", rec)
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("!= filename", r.stdout)

    def test_bad_date_is_error(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        rec = self.good_accelerator()
        rec["updated"] = "03/10/2026"
        self.write("data/accelerators/acme-a100.json", rec)
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("must be YYYY-MM-DD", r.stdout)

    def test_bad_enum_is_error(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        rec = self.good_accelerator()
        rec["vendor"] = "nintendo"
        self.write("data/accelerators/acme-a100.json", rec)
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("not in allowed enum", r.stdout)

    def test_wrong_type_is_error(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        rec = self.good_accelerator()
        rec["vram_gb"] = "eighty gigabytes"
        self.write("data/accelerators/acme-a100.json", rec)
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("expected type", r.stdout)

    def test_invalid_json_reported_not_raised(self):
        (self.tmp / "data/sources/broken.json").write_text("{not json", encoding="utf-8")
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("invalid JSON", r.stdout)

    def test_benchmark_engine_must_resolve(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/benchmarks/b1.json", {
            "id": "b1", "type": "benchmark", "name": "run",
            "status": "draft", "confidence": 0.5, "updated": "2026-10-03",
            "sources": ["acme-spec"], "engine_id": "ghost-engine",
            "accelerator_ids": [], "interconnect_ids": [], "model": "m",
            "format_id": None, "metric": "decode_tok_s", "value": 1.0,
            "unit": "tok/s", "methodology": "bs1", "measured_by": "self",
            "reproducible": False, "notes": "",
        })
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("is not a record id", r.stdout)

    def test_dangling_hardware_ref_is_warning_only(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/accelerators/acme-a100.json", self.good_accelerator())
        # the fixture's `interconnect: ["nvlink"]` has no interconnect record yet
        r = self.validate()
        self.assertEqual(r.returncode, 0)
        self.assertIn("WARN", r.stdout)
        self.assertIn("not a record id (ok if planned)", r.stdout)


class TestNewRecord(Harness):
    def test_skeleton_validates(self):
        r = run_tool("new_record.py", "source", "--name", "ACME Spec Sheet",
                     "--url", "https://example.com/s", "--kind", "spec-sheet",
                     cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = self.read("data/sources/acme-spec-sheet.json")
        self.assertEqual(rec["id"], "acme-spec-sheet")
        self.assertEqual(rec["type"], "source")
        self.assertEqual(rec["kind"], "spec-sheet")
        v = self.validate()
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_slugify_and_source_link(self):
        run_tool("new_record.py", "source", "--name", "Spec", "--url", "https://x",
                 "--kind", "spec-sheet", cwd=self.tmp)
        r = run_tool("new_record.py", "accelerator", "--name", "ACME B200 (SXM)",
                     "--source", "spec", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = self.read("data/accelerators/acme-b200-sxm.json")
        self.assertEqual(rec["sources"], ["spec"])
        self.assertEqual(rec["vendor"], None)

    def test_refuses_to_clobber(self):
        args = ("source", "--name", "Spec", "--url", "https://x", "--kind", "blog")
        self.assertEqual(run_tool("new_record.py", *args, cwd=self.tmp).returncode, 0)
        second = run_tool("new_record.py", *args, cwd=self.tmp)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("refusing to clobber", second.stderr + second.stdout)

    def test_force_overwrites(self):
        args = ("source", "--name", "Spec", "--url", "https://x", "--kind", "blog")
        run_tool("new_record.py", *args, cwd=self.tmp)
        r = run_tool("new_record.py", *args, "--force", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class TestQuery(Harness):
    def seed(self) -> None:
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/accelerators/acme-a100.json", self.good_accelerator())

    def test_list_and_get(self):
        self.seed()
        r = run_tool("query.py", "list", "accelerator", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("acme-a100", r.stdout)
        g = run_tool("query.py", "get", "accelerators/acme-a100", cwd=self.tmp)
        self.assertEqual(json.loads(g.stdout)["name"], "ACME A100")

    def test_get_unknown_suggests(self):
        self.seed()
        g = run_tool("query.py", "get", "accelerators/nonexistent-thing", cwd=self.tmp)
        self.assertNotEqual(g.returncode, 0)
        self.assertIn("no such record", g.stdout + g.stderr)

    def test_where_exact_match(self):
        self.seed()
        r = run_tool("query.py", "where", "vendor=asic-other", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("accelerators/acme-a100", r.stdout)
        none = run_tool("query.py", "where", "vendor=acme", cwd=self.tmp)
        self.assertIn("no records where", none.stdout)

    def test_search_ranks_and_prints(self):
        self.seed()
        r = run_tool("query.py", "search", "hbm3e", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("acme-a100", r.stdout)
        empty = run_tool("query.py", "search", "zzzznotfound", cwd=self.tmp)
        self.assertIn("no matches", empty.stdout)

    def test_refs_both_directions(self):
        self.seed()
        r = run_tool("query.py", "refs", "accelerators/acme-a100", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("acme-spec", r.stdout)
        self.assertIn("[ok]", r.stdout)

    def test_stats_and_gaps(self):
        self.seed()
        s = run_tool("query.py", "stats", cwd=self.tmp)
        self.assertEqual(s.returncode, 0, s.stdout + s.stderr)
        self.assertIn("2 records", s.stdout)
        g = run_tool("query.py", "gaps", cwd=self.tmp)
        self.assertEqual(g.returncode, 0, g.stdout + g.stderr)
        self.assertIn("data/flops/", g.stdout)


class TestIndex(Harness):
    def test_index_writes_all_sections(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/accelerators/acme-a100.json", self.good_accelerator())
        r = run_tool("index.py", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        text = (self.tmp / "docs" / "00-index.md").read_text(encoding="utf-8")
        self.assertIn("## Accelerators (1)", text)
        self.assertIn("acme-a100", text)
        self.assertIn("GENERATED by tools/index.py", text)
        self.assertIn("## Gaps", text)
        self.assertIn("data/flops/", text)

    def test_index_survives_broken_record(self):
        (self.tmp / "data/sources/broken.json").write_text("{oops", encoding="utf-8")
        r = run_tool("index.py", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("## Sources (0)", (self.tmp / "docs" / "00-index.md").read_text(encoding="utf-8"))


class TestRegistry(Harness):
    def test_dirs_exist_for_every_type(self):
        r = run_tool("validate.py", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_schema_for_every_type(self):
        sys.path.insert(0, str(TOOLS))
        from registry import TYPES  # noqa: PLC0415
        for t in TYPES:
            self.assertTrue((ROOT / "schemas" / f"{t}.schema.json").exists(), t)

    def test_schema_type_const_matches_dir_mapping(self):
        sys.path.insert(0, str(TOOLS))
        from registry import DIRS, TYPES
        for t in TYPES:
            schema = json.loads((ROOT / "schemas" / f"{t}.schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["properties"]["type"]["const"], t)
            self.assertTrue((ROOT / "data" / DIRS[t]).is_dir(), DIRS[t])


if __name__ == "__main__":
    unittest.main(verbosity=2)