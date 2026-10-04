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
        sys.path.insert(0, str(TOOLS))
        from registry import DIRS
        for d in DIRS.values():
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

    def good_interconnect(self, slug: str = "acme-nvlink") -> dict:
        return {
            "id": slug, "type": "interconnect", "name": "ACME NVLink",
            "status": "verified", "confidence": 0.9, "updated": "2026-10-03",
            "sources": ["acme-spec"], "kind": "nvlink", "version": "4",
            "bandwidth_gbps": 50.0, "bandwidth_basis": "per link, unidirectional",
            "link_count": 18, "topology": "point-to-point", "scale_up": True,
            "scale_out": False, "switching": None, "notes": "",
        }

    def good_flop(self, slug: str = "acme-decode-gemm") -> dict:
        return {
            "id": slug, "type": "flop", "name": "ACME Decode GEMM",
            "status": "verified", "confidence": 0.9, "updated": "2026-10-03",
            "sources": ["acme-spec"], "class": "decode_gemm",
            "arithmetic_intensity": "low at batch 1", "bound_by": "memory",
            "scales_with": ["batch"], "affected_by_hardware": [],
            "workarounds": [], "notes": "",
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


class TestIndexDeterminism(Harness):
    def test_index_has_no_date_stamp(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        r = run_tool("index.py", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        header = (self.tmp / "docs" / "00-index.md").read_text(
            encoding="utf-8").splitlines()[2]
        self.assertIn("GENERATED by tools/index.py", header)
        # A date in the header makes every regeneration a diff, so CI's
        # staleness check fails on any day other than the commit date.
        self.assertNotRegex(header, r"\d{4}-\d{2}-\d{2}")

    def test_two_runs_produce_identical_bytes(self):
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/accelerators/acme-a100.json", self.good_accelerator())
        run_tool("index.py", cwd=self.tmp)
        first = (self.tmp / "docs" / "00-index.md").read_bytes()
        run_tool("index.py", cwd=self.tmp)
        self.assertEqual(first, (self.tmp / "docs" / "00-index.md").read_bytes())


class TestIntakeOverlay(unittest.TestCase):
    """The overlay is the tree intake validates AND indexes. It must be exactly
    HEAD + the eligible set -- never the live working tree, which in this repo
    also holds other agents' in-flight skeletons."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ir-overlay-"))
        sys.path.insert(0, str(TOOLS))
        import intake
        self.intake = intake
        self._real_root = intake.ROOT
        # Point intake at the temp repo for the duration of the test.
        intake.ROOT = self.tmp

        for sub in ("tools", "schemas", "docs"):
            shutil.copytree(ROOT / sub, self.tmp / sub)
        from registry import DIRS
        for d in DIRS.values():
            (self.tmp / "data" / d).mkdir(parents=True, exist_ok=True)
        (self.tmp / "docs" / "00-index.md").write_text("# Index\n", encoding="utf-8")
        for cmd in ("git init -q", "git add -A", "git -c user.email=t@t -c user.name=t "
                                        "commit -qm base"):
            subprocess.run(cmd.split(), cwd=self.tmp, check=True,
                           capture_output=True)

    def tearDown(self) -> None:
        self.intake.ROOT = self._real_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel: str, obj: dict) -> None:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def rec(self, slug: str) -> dict:
        """A complete accelerator record, via the shared Harness fixtures."""
        return Harness.__new__(Harness).good_accelerator(slug)

    def build(self, eligible_paths: list[str]) -> Path:
        dest = Path(tempfile.mkdtemp(prefix="ir-overlay-dest-"))
        items = [{"path": self.tmp / p} for p in eligible_paths]
        try:
            return self.intake.build_overlay(dest, items)
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise

    def test_overlay_is_head_plus_eligible_only(self):
        self.write("data/accelerators/committed.json", self.rec("committed"))
        subprocess.run(["git", "add", "-A"], cwd=self.tmp, check=True,
                       capture_output=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-qm", "rec"], cwd=self.tmp, check=True,
                       capture_output=True)
        # eligible new record
        self.write("data/accelerators/eligible.json", self.rec("eligible"))
        # in-flight skeleton: present in working tree, never eligible
        self.write("data/accelerators/inflight.json",
                   {"id": "inflight", "type": "accelerator", "name": "wip",
                    "vendor": None, "architecture": None, "vram_gb": None})

        overlay = self.build(["data/accelerators/eligible.json"])
        try:
            names = sorted(p.name for p in (overlay / "data" / "accelerators").glob("*.json"))
            self.assertEqual(names, ["committed.json", "eligible.json"])
            self.assertFalse((overlay / "data" / "accelerators" / "inflight.json").exists())
            # tools + schemas are copied so index.py/validate.py can run there
            self.assertTrue((overlay / "tools" / "index.py").exists())
            self.assertTrue((overlay / "schemas" / "accelerator.schema.json").exists())
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def test_overlay_uses_head_blob_not_working_tree_edit(self):
        """A crawler editing an already-committed record must not leak into the
        overlay: the commit will carry the old content for that file."""
        self.write("data/accelerators/committed.json", self.rec("committed"))
        subprocess.run(["git", "add", "-A"], cwd=self.tmp, check=True,
                       capture_output=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-qm", "rec"], cwd=self.tmp, check=True,
                       capture_output=True)
        # The same path, edited in the working tree by "another agent". It is
        # simply not in the eligible set, so the overlay must carry HEAD's copy.
        dirty = self.rec("committed")
        dirty["vram_gb"] = 999.0
        self.write("data/accelerators/committed.json", dirty)

        overlay = self.build([])
        try:
            got = json.loads((overlay / "data" / "accelerators" /
                              "committed.json").read_text(encoding="utf-8"))
            self.assertEqual(got["vendor"], "asic-other")
            self.assertEqual(got["vram_gb"], 80.0)
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def test_index_from_overlay_is_deterministic(self):
        self.write("data/accelerators/eligible.json", self.rec("eligible"))
        overlay = self.build(["data/accelerators/eligible.json"])
        try:
            first = self.intake.generate_index(overlay)
            second = self.intake.generate_index(overlay)
            self.assertEqual(first, second)
            self.assertIn("eligible", first)
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def test_inflight_record_absent_from_generated_index(self):
        self.write("data/accelerators/eligible.json", self.rec("eligible"))
        self.write("data/accelerators/inflight-wip.json",
                   {"id": "inflight-wip", "type": "accelerator", "name": "Half Written Chip",
                    "vendor": "someone"})
        overlay = self.build(["data/accelerators/eligible.json"])
        try:
            text = self.intake.generate_index(overlay)
            self.assertIn("eligible", text)
            self.assertNotIn("inflight-wip", text)
            self.assertNotIn("Half Written Chip", text)
        finally:
            shutil.rmtree(overlay, ignore_errors=True)


class TestCostPerToken(Harness):
    """tools/cost_per_token.py must refuse to print a number it cannot show is
    in a record. The whole point of the join is that the price-basis and
    GPU-count steps are auditable, so the tests attack exactly those."""

    def cost_tool(self, *args: str) -> subprocess.CompletedProcess:
        return run_tool("cost_per_token.py", *args, cwd=self.tmp)

    def seed_joinable_pair(self) -> None:
        """One priced supply record and one throughput benchmark that share an
        accelerator id. Both price_basis and unit carry the literal substrings
        the tool's evidence check requires."""
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/accelerators/acme-a100.json", self.good_accelerator())
        self.write("data/supply/acme-cloud.json", {
            "id": "acme-cloud", "type": "supply", "name": "ACME cloud",
            "status": "verified", "confidence": 0.9, "updated": "2026-10-03",
            "sources": ["acme-spec"], "kind": "cloud", "vendor": "ACME",
            "accelerator_ids": ["acme-a100"], "region": "us-test-1",
            "channels": ["on-prem-cloud"], "price_usd": 2.0,
            "price_basis": "per GPU-hour. acme.8xlarge = 8x A100 at $16.00/instance-hr / 8 = $2.00",
            "availability": "in-stock", "lead_time_weeks": None,
            "export_controlled": None, "notes": "",
        })
        self.write("data/engines/trtllm.json", {
            "id": "trtllm", "type": "engine", "name": "TensorRT-LLM",
            "status": "verified", "confidence": 0.9, "updated": "2026-10-03",
            "sources": ["acme-spec"],
        })
        self.write("data/benchmarks/acme-tps.json", {
            "id": "acme-tps", "type": "benchmark", "name": "ACME run",
            "status": "verified", "confidence": 0.9, "updated": "2026-10-03",
            "sources": ["acme-spec"], "engine_id": "trtllm",
            "accelerator_ids": ["acme-a100"], "interconnect_ids": [],
            "model": "acme-model-7b", "format_id": None,
            "metric": "tps_aggregate", "value": 8000.0,
            "unit": "output tokens/s (aggregate, 8 GPUs)",
            "methodology": "Hardware: 8x NVIDIA A100 SXM 80GB on ONE node.",
            "measured_by": "vendor", "reproducible": False, "notes": "",
        })

    def test_real_repo_join_is_fully_sourced(self):
        """Against the real data/: not one quoted price basis or GPU count may
        fail. This is the regression that stops a 8x price-basis error from
        reaching a doc."""
        r = run_tool("cost_per_token.py", cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("UNSOURCED INPUT", r.stdout)

    def test_real_repo_table_is_all_aggregate_or_labelled(self):
        r = run_tool("cost_per_token.py", "--format", "json", cwd=ROOT)
        rows = json.loads(r.stdout)
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn(row["concurrency"], ("aggregate", "single-user"))
            self.assertTrue(row["supply_ref"].startswith("supply/"))
            self.assertTrue(row["bench_ref"].startswith("benchmarks/"))
            self.assertGreater(row["usd_per_mtok"], 0)

    def test_contradicting_record_stops_the_table(self):
        """If a real record's text no longer contains the figure the join
        quotes, the tool must stop rather than print a stale number. Built by
        copying a genuine priced record into the temp repo and corrupting the
        price it quotes."""
        for sub in ("supply", "benchmarks", "sources"):
            shutil.copytree(ROOT / "data" / sub, self.tmp / "data" / sub,
                            dirs_exist_ok=True)
        p = self.tmp / "data" / "supply" / "aws-ec2-p5-h100.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["price_basis"] = "per H100 GPU-hour, price redacted in this fixture."
        p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        r = self.cost_tool()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("UNSOURCED INPUT", r.stdout)
        self.assertIn("aws-ec2-p5-h100", r.stdout)
        # ...and the override still refuses to be silent about it
        forced = self.cost_tool("--force")
        self.assertIn("UNSOURCED INPUT", forced.stdout)

    def test_gaps_never_invents_a_price(self):
        r = self.cost_tool("gaps")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("no record", r.stdout)

    def test_evidence_check_flags_missing_substring(self):
        sys.path.insert(0, str(TOOLS))
        import cost_per_token
        self.assertTrue(hasattr(cost_per_token, "check_evidence"))
        # The real repo must be clean; assert the check is actually running by
        # corrupting one in-memory quote.
        self.assertEqual(cost_per_token.check_evidence(), [])
        saved = cost_per_token.PRICE_QUOTES["aws-ec2-p5-h100"][0]
        cost_per_token.PRICE_QUOTES["aws-ec2-p5-h100"][0] = dict(
            saved, basis="a figure that appears in no record at all")
        try:
            self.assertNotEqual(cost_per_token.check_evidence(), [])
        finally:
            cost_per_token.PRICE_QUOTES["aws-ec2-p5-h100"][0] = saved
        self.assertEqual(cost_per_token.check_evidence(), [])


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


class TestDirectoryPlacement(Harness):
    """Records must live in the directory matching their declared type.

    This is a regression test for a real bug: agents wrote records into
    data/interconnects/ (plural) and data/flop/ (singular) and validate.py
    did not notice. The canonical directories are defined in registry.py
    and must be enforced by validate.py.
    """

    def test_interconnect_in_interconnects_dir_is_error(self):
        """A record with type='interconnect' in data/interconnects/ must fail."""
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/interconnects/acme-nvlink.json", self.good_interconnect())
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("non-canonical directory", r.stdout)
        self.assertIn("interconnects", r.stdout)

    def test_flop_in_flop_dir_is_error(self):
        """A record with type='flop' in data/flop/ (singular) must fail."""
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/flop/acme-decode-gemm.json", self.good_flop())
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("non-canonical directory", r.stdout)
        self.assertIn("flop", r.stdout)

    def test_correctly_placed_interconnect_passes(self):
        """A record with type='interconnect' in data/interconnect/ must pass."""
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/interconnect/acme-nvlink.json", self.good_interconnect())
        r = self.validate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_correctly_placed_flop_passes(self):
        """A record with type='flop' in data/flops/ must pass."""
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/flops/acme-decode-gemm.json", self.good_flop())
        r = self.validate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_multiple_stray_dirs_all_reported(self):
        """Multiple stray directories must each be reported."""
        self.write("data/sources/acme-spec.json", self.good_source())
        self.write("data/interconnects/acme-nvlink.json", self.good_interconnect())
        self.write("data/flop/acme-decode-gemm.json", self.good_flop())
        r = self.validate()
        self.assertEqual(r.returncode, 1)
        self.assertIn("interconnects", r.stdout)
        self.assertIn("flop", r.stdout)

    def test_stray_dir_without_json_is_not_error(self):
        """An empty stray directory (no .json files) is not an error."""
        self.write("data/sources/acme-spec.json", self.good_source())
        (self.tmp / "data" / "interconnects").mkdir(parents=True)
        r = self.validate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class TestCanonicalDirectories(Harness):
    """The canonical directory list must be explicit and complete.

    If someone adds a new type or a stray directory appears, these tests
    fail loudly rather than silently accommodating the change.
    """

    CANONICAL_DIRS = [
        "accelerators", "benchmarks", "engines", "flops", "gotchas",
        "interconnect", "models", "papers", "quantization", "sources", "supply",
    ]

    def test_canonical_dir_list_is_exact(self):
        """The canonical directory list must match exactly."""
        sys.path.insert(0, str(TOOLS))
        from registry import DIRS
        self.assertEqual(sorted(DIRS.values()), self.CANONICAL_DIRS)

    def test_no_stray_directories_in_repo(self):
        """No stray directories exist in the real data/ directory."""
        data_dir = ROOT / "data"
        if not data_dir.exists():
            self.skipTest("data/ directory does not exist")
        actual = {d.name for d in data_dir.iterdir() if d.is_dir()}
        expected = set(self.CANONICAL_DIRS)
        strays = actual - expected
        self.assertEqual(strays, [], f"stray directories found: {strays}")

    def test_no_flop_singular_dir(self):
        """data/flop/ (singular) must not exist."""
        self.assertFalse((ROOT / "data" / "flop").exists())

    def test_no_interconnects_plural_dir(self):
        """data/interconnects/ (plural) must not exist."""
        self.assertFalse((ROOT / "data" / "interconnects").exists())


class TestIdFilenameMatch(Harness):
    """Every record's id must match its filename stem.

    This is a cheap invariant that catches copy-paste and bulk-rename errors.
    """

    def test_id_matches_filename_stem(self):
        """All records in the real repo have id == filename stem."""
        sys.path.insert(0, str(TOOLS))
        from registry import DIRS
        data_dir = ROOT / "data"
        if not data_dir.exists():
            self.skipTest("data/ directory does not exist")
        mismatches = []
        for type_name, dir_name in DIRS.items():
            type_dir = data_dir / dir_name
            if not type_dir.exists():
                continue
            for f in sorted(type_dir.glob("*.json")):
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue  # invalid JSON is caught by other tests
                if isinstance(rec, dict) and rec.get("id") != f.stem:
                    mismatches.append(f"{dir_name}/{f.stem}: id={rec.get('id')!r}")
        self.assertEqual(mismatches, [], f"id/filename mismatches: {mismatches}")


if __name__ == "__main__":
    unittest.main(verbosity=2)