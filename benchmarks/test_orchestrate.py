import argparse
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
from pathlib import Path

import fixture_digest
import cell_group
import orchestrate
from runners import bench_common
from test.test_cell_group import CellGroupTest  # imported so unittest.main includes shared tests

HERE = Path(__file__).resolve().parent


def result(framework, backend, variant, losses, precision="f32"):
    return {
        "workload": "mlp_wide",
        "framework": framework,
        "backend": backend,
        "variant": variant,
        "precision": precision,
        "losses": losses,
    }


def cell(framework, backend, variant, losses, precision="f32", p50=1.0):
    """A result with the fields `report` reads, not just the parity ones."""
    r = result(framework, backend, variant, losses, precision=precision)
    r.update(
        {
            "step_ms": {"p10": p50, "p50": p50, "p90": p50},
            "queued_step_ms": p50,
            "compile_s": 1.5,
        }
    )
    return r


def _reject(token):
    raise ValueError(f"not JSON: {token}")


def strict_loads(text):
    """`json.loads` without its NaN/Infinity extension -- JSON as every other reader has it."""
    return json.loads(text, parse_constant=_reject)


class ParityCheckTest(unittest.TestCase):
    def test_precision_tolerances_match_measured_headroom(self):
        self.assertEqual(orchestrate.PARITY_TOL_PRECISION, {"bf16": 4e-3, "f16": 2e-3})

    def test_flat_candidate_fails_even_when_within_bf16_tolerance(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        flat = result(
            "ocannl", "metal", "default", [2.3026, 2.3026, 2.3026], precision="bf16"
        )

        orchestrate.parity_check([ref, flat])

        self.assertLess(flat["parity_max_rel"], orchestrate.PARITY_TOL_PRECISION["bf16"])
        self.assertFalse(flat["parity_loss_moved"])
        self.assertEqual(flat["parity"], "FAIL")

    def test_moving_candidate_within_tolerance_passes(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        moving = result(
            "ocannl", "metal", "tuned", [2.3025, 2.3012, 2.3004], precision="bf16"
        )

        orchestrate.parity_check([ref, moving])

        self.assertTrue(moving["parity_loss_moved"])
        self.assertEqual(moving["parity"], "PASS")


class DivergenceTest(unittest.TestCase):
    """gh-ocannl-676: a cell whose training diverged ran, and its trajectory is the finding.

    The three runners now all emit `null` for a number JSON cannot express, and the OCANNL one
    used to emit OCaml's `nan`, whose line `json.loads` refuses -- so the cell was reported as a
    broken runner and the loss vector that shows the divergence was thrown away. Here the verdict
    is DIVERGED: a parity failure that names its cause.
    """

    def test_null_loss_is_a_diverged_cell_not_a_missing_one(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = result("ocannl", "metal", "default", [2.3026, None, None], precision="f16")

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)
        self.assertEqual(ref["parity"], "REF")

    def test_nan_from_a_python_runner_reads_the_same(self):
        # Python's json.loads accepts its own `NaN` extension, so an older result file can still
        # deliver one; it is the same fact as a null.
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = result("tinygrad", "metal", "jit", [2.3026, float("nan"), float("inf")])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)

    def test_a_diverged_cell_is_not_reported_as_stationary(self):
        # Its finite prefix moved; and with fewer than two finite steps there is nothing to say
        # about movement, which is reported as divergence rather than as a flat trajectory.
        self.assertTrue(orchestrate.loss_moved([2.3026, 1.5, None]))
        self.assertFalse(orchestrate.loss_moved([2.3026, None, None]))

    def test_the_finite_prefix_is_still_compared(self):
        ref = result("pytorch", "cpu", "eager", [2.0, 2.0, 2.0])
        blown = result("ocannl", "cc", "default", [2.0, None, None])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity_max_rel"], 0.0)

    def test_a_finite_value_after_the_divergence_is_not_evidence(self):
        # A loss that comes back finite after a NaN is whatever the arithmetic settled on, not
        # drift: comparing it would put a number on the DIVERGED row that its own contract calls
        # meaningless, and reading it as movement would say the trajectory moved when what it did
        # was blow up.
        ref = result("pytorch", "cpu", "eager", [2.0, 2.0, 2.0])
        blown = result("ocannl", "cc", "default", [2.0, None, 9.0])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity_max_rel"], 0.0)
        self.assertFalse(blown["parity_loss_moved"])
        self.assertEqual(orchestrate.finite_prefix([2.0, None, 9.0]), [2.0])

    def test_the_reference_prefix_bounds_the_comparison_too(self):
        ref = result("pytorch", "cpu", "eager", [2.0, float("nan"), 2.0])
        other = result("ocannl", "cc", "default", [2.0, 5.0, 5.0])

        orchestrate.parity_check([ref, other])

        self.assertEqual(other["parity_max_rel"], 0.0)

    def test_a_diverged_reference_is_no_reference(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, None, None])
        other = result("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])

        orchestrate.parity_check([ref, other])

        self.assertEqual(ref["parity"], "DIVERGED")
        self.assertEqual(other["parity"], "NO-REF")

    def test_report_names_the_divergence_and_writes_parseable_json(self):
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = cell(
            "ocannl", "metal", "default", [2.3026, None, None], precision="f16", p50=None
        )
        orchestrate.parity_check([ref, blown])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, blown], out)
            text = (out / "report.md").read_text()
            rows = [
                strict_loads(line) for line in (out / "results.jsonl").read_text().splitlines()
            ]

        self.assertIn("DIVERGED", text)
        self.assertIn("loss non-finite from step 1", text)
        # A time the runner never measured prints as n/a rather than crashing the report.
        self.assertIn("n/a", text)
        self.assertEqual([r["parity"] for r in rows], ["REF", "DIVERGED"])

    def test_emit_writes_null_for_a_non_finite_number(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bench_common.emit({"losses": [1.0, float("nan")], "step_ms": {"p50": float("inf")}})

        parsed = strict_loads(buf.getvalue().strip())
        self.assertEqual(parsed, {"losses": [1.0, None], "step_ms": {"p50": None}})


class OcannlResultLineTest(unittest.TestCase):
    """The OCANNL runner's own result line, as this orchestrator reads it (gh-ocannl-676).

    The golden of test/operations/bench_result_line holds the line built from fabricated values --
    a diverged trajectory, times that were never measured, a diagnostic full of control characters
    -- and this is the parser that has to accept it. Pinning it in OCaml alone would leave the
    claim "this parses" resting on a second implementation of JSON.
    """

    GOLDEN = HERE.parent / "test/operations/bench_result_line.expected"

    def lines(self):
        return [
            line
            for line in self.GOLDEN.read_text().splitlines()
            if line.startswith("{")  # what run_cell picks out of a cell's output
        ]

    def test_every_emitted_line_parses_strictly(self):
        lines = self.lines()
        self.assertGreaterEqual(len(lines), 2, self.GOLDEN)
        for line in lines:
            with self.subTest(line=line[:60]):
                self.assertIn("losses", strict_loads(line))

    def test_the_diverged_line_is_read_as_a_diverged_cell(self):
        parsed = [strict_loads(line) for line in self.lines()]
        blown = next(r for r in parsed if None in r["losses"])
        ref = result("pytorch", "cpu", "eager", [2.5, 1.75, 1.25])
        ref["workload"] = blown["workload"]

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)


class CellIdentityTest(unittest.TestCase):
    """gh-ocannl-539: scheduling variant and storage precision are independent axes."""

    def test_f32_cells_keep_their_bare_variant_name(self):
        self.assertEqual(orchestrate.cell_name("tuned", "f32"), "tuned")
        self.assertEqual(orchestrate.cell_name("default", "f32"), "default")

    def test_reduced_precision_cells_are_named_by_the_product(self):
        self.assertEqual(orchestrate.cell_name("tuned", "bf16"), "tuned/bf16")
        self.assertEqual(orchestrate.cell_name("default", "f16"), "default/f16")

    def test_skip_entries_are_precision_wildcards(self):
        # A None precision means "at every precision": a scheduling pathology must not be let back
        # in through the bf16/f16 columns. Tested against an injected entry rather than a live one,
        # so emptying SKIP_CELLS does not silently drop coverage of the wildcard itself.
        entry = ("some_workload", "metal", "tuned", None)
        orchestrate.SKIP_CELLS.add(entry)
        try:
            self.assertTrue(orchestrate.cell_skipped("some_workload", "metal", "tuned", "f32"))
            self.assertTrue(orchestrate.cell_skipped("some_workload", "metal", "tuned", "bf16"))
            self.assertFalse(orchestrate.cell_skipped("some_workload", "metal", "default", "f32"))
            self.assertFalse(orchestrate.cell_skipped("some_workload", "cc", "tuned", "bf16"))
        finally:
            orchestrate.SKIP_CELLS.discard(entry)

    def test_cifar_conv_metal_tuned_is_no_longer_skipped(self):
        # gh-ocannl-538: the post-tune re-init hang the entry was added for did not reproduce —
        # the cell completed in ~4 min with no hang, and the sweep reproduced its p50 to 0.005%.
        for precision in ("f32", "bf16"):
            self.assertFalse(
                orchestrate.cell_skipped("cifar_conv", "metal", "tuned", precision), precision
            )

    def test_hip_tuned_cells_are_no_longer_skipped(self):
        # gh-ocannl-532's unparallelized-baseline dispatch is fixed in the tuner and confirmed on
        # the gfx1151 machine that produced the symptom; all three cells complete.
        for workload in ("mlp_wide", "cifar_conv", "cifar_stride"):
            for precision in ("f32", "bf16"):
                self.assertFalse(
                    orchestrate.cell_skipped(workload, "hip", "tuned", precision),
                    (workload, precision),
                )

    def test_every_skip_entry_is_a_four_tuple(self):
        for entry in orchestrate.SKIP_CELLS:
            self.assertEqual(len(entry), 4, entry)

    def test_tuned_f32_and_tuned_bf16_are_distinct_cells(self):
        rows = [
            result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1]),
            result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1], precision="bf16"),
        ]
        names = {orchestrate.cell_name(r["variant"], r["precision"]) for r in rows}

        self.assertEqual(names, {"tuned", "tuned/bf16"})


class ReportGroupingTest(unittest.TestCase):
    def test_rows_group_by_precision_f32_first(self):
        self.assertEqual(orchestrate.precision_rank("f32"), 0)
        self.assertLess(
            orchestrate.precision_rank("bf16"), orchestrate.precision_rank("f16")
        )
        # A gate leg sorts directly after the storage precision it varies, not last: it is a
        # variant of how f16's optimizer step is gated, and reads next to plain f16.
        self.assertGreater(
            orchestrate.precision_rank("f16-static"), orchestrate.precision_rank("f16")
        )
        self.assertLess(
            orchestrate.precision_rank("f16-gated8"),
            orchestrate.precision_rank("unknown-precision"),
        )

    def test_gate_legs_are_parity_gated_at_their_base_precision(self):
        self.assertEqual(
            orchestrate.parity_tol("f16-gated8"), orchestrate.PARITY_TOL_PRECISION["f16"]
        )
        self.assertEqual(
            orchestrate.parity_tol("f16-static"), orchestrate.PARITY_TOL_PRECISION["f16"]
        )


class PrecisionLegTest(unittest.TestCase):
    """gh-ocannl-551: the gate-cost legs are orchestrated cells, and an inexpressible cell is
    reported rather than silently absent."""

    def test_gate_legs_dispatch_their_own_flags(self):
        self.assertEqual(
            orchestrate.precision_env("f16-static"),
            {"BENCH_PRECISION": "f16", "BENCH_STATIC_SCALE": "1"},
        )
        self.assertEqual(
            orchestrate.precision_env("f16-gated16"),
            {"BENCH_PRECISION": "f16", "BENCH_GATE_INTERVAL": "16"},
        )
        self.assertEqual(orchestrate.precision_env("bf16"), {"BENCH_PRECISION": "bf16"})

    def test_cell_env_carries_the_leg_and_clears_the_others(self):
        # The gate flags are cleared per cell and then set by the leg — the two collided as
        # duplicate dict() keywords, which crashed the run only once a gate cell was dispatched.
        stray = {"BENCH_STATIC_SCALE": "1", "BENCH_GATE_INTERVAL": "7"}
        plain = orchestrate.cell_env(stray, "fx.safetensors", "default", "bf16")
        self.assertEqual(plain["BENCH_PRECISION"], "bf16")
        self.assertEqual(plain["BENCH_STATIC_SCALE"], "0")
        self.assertEqual(plain["BENCH_GATE_INTERVAL"], "0")
        gated = orchestrate.cell_env(stray, "fx.safetensors", "tuned", "f16-gated16")
        self.assertEqual(gated["BENCH_PRECISION"], "f16")
        self.assertEqual(gated["BENCH_GATE_INTERVAL"], "16")
        self.assertEqual(gated["BENCH_STATIC_SCALE"], "0")
        self.assertEqual(gated["BENCH_TUNE"], "1")
        static = orchestrate.cell_env(stray, "fx.safetensors", "default", "f16-static")
        self.assertEqual(static["BENCH_STATIC_SCALE"], "1")
        self.assertEqual(static["BENCH_GATE_INTERVAL"], "0")

    def test_precision_spec_rejects_nonsense(self):
        for spec in ("f16-gated0", "f16-gated", "f8", "f16-dynamic"):
            with self.assertRaises(Exception, msg=spec):
                orchestrate.precision_spec(spec)

    def test_gate_legs_need_a_training_workload(self):
        # The forward-only gpt2_mini has no optimizer, hence no loss scale to gate.
        self.assertIsNotNone(
            orchestrate.precision_unavailable("gpt", "infer", "f16-static")
        )
        self.assertIsNotNone(
            orchestrate.precision_unavailable("gpt", "infer", "f16-gated8")
        )
        # ... but gpt2_mini_train does, and plain reduced precisions work in both modes.
        self.assertIsNone(
            orchestrate.precision_unavailable("gpt", "train", "f16-static")
        )
        self.assertIsNone(orchestrate.precision_unavailable("gpt", "infer", "bf16"))
        self.assertIsNone(orchestrate.precision_unavailable("mlp", "train", "f16-gated8"))

    def test_conv_workloads_report_the_missing_runner_support(self):
        reason = orchestrate.precision_unavailable("conv", "train", "bf16")
        self.assertIsNotNone(reason)
        self.assertIn("conv", reason)


class ProvenanceTest(unittest.TestCase):
    """gh-ocannl-644: a tuned cell's result says which pass timed it."""

    def tuned(self, searched, searches=0, replays=0, no_searches=0):
        r = result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1])
        if searched is not None:
            r["searched"] = searched
            r["tune"] = {
                "shipped": "A", "searches": searches, "replays": replays,
                "no_searches": no_searches, "arms": []
            }
        return r

    def test_a_replay_is_the_compliant_case(self):
        replay = self.tuned(False, replays=2)

        self.assertEqual(orchestrate.provenance_check([replay]), [])
        self.assertEqual(replay["provenance"], "REPLAY")

    def test_search_pass_timings_are_a_violation(self):
        # The failure this exists to catch: both passes emit the same framework/backend/variant/
        # precision, so before `searched` a report could quote pass-1 numbers indefinitely.
        searching = self.tuned(True, searches=2)

        self.assertEqual(orchestrate.provenance_check([searching]), [searching])
        self.assertEqual(searching["provenance"], "SEARCH-PASS")

    def test_one_classifier_reads_searched_for_every_consumer(self):
        # The round-1 and round-2 findings were the same shape — a three-valued fact read through
        # a boolean, wrong in a different place each time. There is now one reading of it, and the
        # gate, the `pass` column and the carried-over compile_s label all go through it.
        tune = lambda searches, replays: {  # noqa: E731
            "shipped": "A", "searches": searches, "replays": replays, "arms": []
        }
        searched = {"searched": True, "tune": tune(2, 0)}
        replay = {"searched": False, "tune": tune(0, 2)}
        no_search = {"searched": False, "tune": tune(0, 0)}

        self.assertEqual(orchestrate.search_provenance(searched), "SEARCHED")
        self.assertEqual(orchestrate.search_provenance(replay), "REPLAY")
        self.assertEqual(orchestrate.search_provenance(no_search), "NO-SEARCH")
        self.assertEqual(orchestrate.search_provenance({"searched": None}), "UNKNOWN")
        # No tune object: `searched: false` alone cannot separate a replay from a cell that
        # searched nothing, and guessing either way has already been a bug.
        self.assertEqual(orchestrate.search_provenance({"searched": False}), "UNKNOWN")

    def test_the_runner_states_the_no_search_case_instead_of_it_being_derived(self):
        # gh-ocannl-677: the OCaml call's outcome is one state, and the runner now names it per
        # arm and counts it (`no_searches`) rather than leaving the reader to recover "neither
        # searched nor replayed" from two zeroed counters in a JSON artifact.
        stated = lambda replays, no_searches: {  # noqa: E731
            "searched": False,
            "tune": {"shipped": "A", "searches": 0, "replays": replays,
                     "no_searches": no_searches, "arms": []},
        }

        self.assertEqual(orchestrate.search_provenance(stated(0, 2)), "NO-SEARCH")
        self.assertEqual(orchestrate.search_provenance(stated(2, 0)), "REPLAY")
        # A mixed cell — one arm replayed, the other had nothing to replay — still carries a tuned
        # artifact, so it is a replay.
        self.assertEqual(orchestrate.search_provenance(stated(1, 1)), "REPLAY")

    def test_only_a_real_replay_labels_the_carried_over_compile_cost_cached(self):
        # A search pass under autotune_search=false hands over the cost of compiling the untuned
        # default; calling that "(cached)" would claim a schedule cache it never touched.
        self.assertEqual(orchestrate.COMPILE_S_NOTE.get("REPLAY"), " (cached)")
        self.assertEqual(orchestrate.COMPILE_S_NOTE.get("NO-SEARCH"), " (no search)")
        self.assertIsNone(orchestrate.COMPILE_S_NOTE.get("SEARCHED"))
        self.assertIsNone(orchestrate.COMPILE_S_NOTE.get("UNKNOWN"))

    def test_a_disabled_search_is_neither_a_violation_nor_a_replay(self):
        # gh-ocannl-559's reproducible profile turns the search off: the cell ships the untuned
        # default, having neither searched nor replayed. Gating it would fail BOTH passes of every
        # tuned cell, and calling it a replay would credit the row with a tuned artifact it does
        # not have.
        cell = self.tuned(False, searches=0, replays=0, no_searches=2)

        self.assertEqual(orchestrate.provenance_check([cell]), [])
        self.assertEqual(cell["provenance"], "NO-SEARCH")

        # And the same verdict from an artifact written before `no_searches` existed.
        legacy = self.tuned(False, searches=0, replays=0)
        del legacy["tune"]["no_searches"]

        self.assertEqual(orchestrate.provenance_check([legacy]), [])
        self.assertEqual(legacy["provenance"], "NO-SEARCH")

    def test_a_runner_without_the_field_is_unknown_not_a_violation(self):
        old = self.tuned(None)

        self.assertEqual(orchestrate.provenance_check([old]), [])
        self.assertEqual(old["provenance"], "UNKNOWN")

    def test_cells_that_neither_search_nor_compile_are_not_annotated(self):
        # They have no process distinction to be on the wrong side of; a `pass` entry for them
        # would be a claim about a protocol they do not run.
        default = result("ocannl", "hip", "default", [2.3, 2.2, 2.1])
        default["searched"] = False
        eager = result("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])
        jit = result("tinygrad", "CPU", "jit", [2.3, 2.2, 2.1])

        self.assertEqual(orchestrate.provenance_check([default, eager, jit]), [])
        for r in (default, eager, jit):
            self.assertNotIn("provenance", r)

    def test_a_single_pass_framework_states_its_search_without_being_gated(self):
        # gh-ocannl-675: tinygrad's beam search and torch.compile run in the timing process by
        # protocol. Nothing yet says they pay for it, so the report states it and the gate keeps
        # its hands off — the alternative, saying nothing, reads as though only OCANNL searches.
        beam = result("tinygrad", "AMD", "beam", [2.3, 2.2, 2.1])
        beam["searched"] = True
        compiled = result("pytorch", "cuda", "compiled", [2.3, 2.2, 2.1])
        compiled["searched"] = False

        self.assertEqual(orchestrate.provenance_check([beam, compiled]), [])
        self.assertEqual(beam["provenance"], "SAME-PROCESS")
        self.assertEqual(compiled["provenance"], "CACHED")

    def test_a_single_pass_cell_that_cannot_tell_is_unknown(self):
        beam = result("tinygrad", "AMD", "beam", [2.3, 2.2, 2.1])
        beam["searched"] = None

        self.assertEqual(orchestrate.provenance_check([beam]), [])
        self.assertEqual(beam["provenance"], "UNKNOWN")

    def test_every_verdict_renders(self):
        for verdict in (
            "REPLAY",
            "SEARCH-PASS",
            "NO-SEARCH",
            "SAME-PROCESS",
            "CACHED",
            "UNKNOWN",
        ):
            self.assertIn(verdict, orchestrate.PROVENANCE_MARK)

    def test_the_gated_and_stated_cell_sets_do_not_overlap(self):
        self.assertFalse(orchestrate.TWO_PASS_CELLS & orchestrate.SAME_PROCESS_CELLS)


class TensorizationTest(unittest.TestCase):
    """gh-ocannl-626: a "tensorized" timing must not be able to be a scalar-fallback timing."""

    def cell(self, arms, shipped="A", shipped_mma=...):
        r = result("ocannl", "metal", "tuned", [2.3, 2.2, 2.1])
        r["searched"] = False
        r["step_ms"] = {"p10": 1.0, "p50": 1.1, "p90": 1.2}
        r["queued_step_ms"] = 0.1
        r["compile_s"] = 3.0
        r["parity"] = "REF"
        r["parity_loss_moved"] = True
        r["diverged_at"] = None
        r["tune"] = {
            "shipped": shipped, "searches": 0, "replays": 1, "no_searches": 0, "arms": arms
        }
        if shipped_mma is not ...:
            r["tune"]["shipped_mma"] = shipped_mma
        return r

    def mma(self, tensorization, statements=0, scalar_fallbacks=0):
        return {
            "tensorization": tensorization, "statements": statements,
            "scalar_fallbacks": scalar_fallbacks,
        }


    def arm(self, name, tensorized, tensorization, statements=0, fallbacks=0):
        return {
            "arm": name, "state": "cache-replay", "searched": False, "cache_hit": True,
            "best_ms": 1.0, "best_label": "L", "tensorized": tensorized,
            "tensorization": tensorization, "mma_statements": statements,
            "mma_scalar_fallbacks": fallbacks, "mma_seeded": 0, "mma_timed": 0,
            "mma_best_ms": 1.0, "terminal_failure": None,
        }

    def test_a_declined_tensorize_is_not_reported_as_a_tensorized_timing(self):
        # The defect: the schedule asked for tensor cores, every Tile_mma rendered the lane-0
        # scalar loop, and the row still carried a tensorized variant name.
        cell = self.cell([self.arm("A", True, "scalar-fallback", statements=3, fallbacks=3)])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_a_tensorize_that_emitted_no_statement_is_its_own_verdict(self):
        # Distinct from the fallback: nothing declined, because nothing was emitted to decline.
        cell = self.cell([self.arm("A", True, "not-requested")])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-EMITTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_an_honestly_tensorized_cell_is_not_flagged(self):
        cell = self.cell([self.arm("A", True, "tensorized", statements=4)])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "TENSORIZED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_an_artifact_that_never_asked_is_not_a_mismatch(self):
        cell = self.cell([self.arm("A", False, "not-requested")])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-REQUESTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_a_missing_census_never_reads_as_tensorized(self):
        # The negative control. A runner predating the field, or an arm with no crowned candidate,
        # carries a null label — which must answer UNKNOWN, never the passing reading.
        older = self.cell([{"arm": "A", "best_ms": 1.0}])
        no_winner = self.cell([self.arm("A", False, None)])

        self.assertEqual(orchestrate.tensorization_verdict(older), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_verdict(no_winner), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_check([older, no_winner]), [])
        # And a cell that tuned nothing at all has no arm to consult, so it gets no column entry
        # rather than a fabricated one.
        self.assertIsNone(orchestrate.tensorization_verdict(result("torch", "cuda", "eager", [1.0])))

    def test_the_shipped_routines_census_outranks_the_arm_reports(self):
        # A gh-555 flip refinement that beats the A/B winner ships under `shipped: "flip"` and is
        # not an arm at all; on the timing_ctx path the tuner can also fall back to the untuned
        # default after the arm was crowned. Either way the arm describes a discarded schedule, so
        # the routine that was actually timed is what the column must report.
        flip = self.cell(
            [self.arm("A", True, "tensorized", statements=4)],
            shipped="flip",
            shipped_mma=self.mma("scalar-fallback", statements=3, scalar_fallbacks=3),
        )

        self.assertIsNone(orchestrate.shipped_arm(flip))
        self.assertEqual(orchestrate.tensorization_verdict(flip), "SCALAR-FALLBACK")
        self.assertEqual(orchestrate.tensorization_check([flip]), [flip])

    def test_a_fallback_to_the_untuned_default_is_not_reported_as_the_crowned_arm(self):
        # The arm was crowned in the scratch context and its replay was rejected in the production
        # one, so the timed routine has no mma at all while the arm still says tensorized.
        cell = self.cell(
            [self.arm("A", True, "tensorized", statements=4)],
            shipped_mma=self.mma("not-requested"),
        )

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-EMITTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_a_harness_that_recorded_no_shipped_census_reads_unknown(self):
        # Present-but-empty is a runner that reported arms and forgot the routine: not a finding,
        # and not the passing reading either.
        cell = self.cell([self.arm("A", True, "tensorized", statements=4)], shipped_mma=None)

        self.assertEqual(orchestrate.tensorization_verdict(cell), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_an_artifact_predating_shipped_mma_still_reads_its_arm(self):
        # The key absent (not null) is an older result line; the arm is right whenever the crowned
        # candidate WAS the shipped artifact, which is the common case.
        cell = self.cell([self.arm("A", True, "scalar-fallback", statements=2, fallbacks=2)])

        self.assertNotIn("shipped_mma", cell["tune"])
        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")

    def test_the_verdict_describes_the_arm_that_shipped_not_the_fastest_one(self):
        # tune_placements searches several arms and keeps one artifact; reading any other arm
        # describes a schedule that was discarded (gh-ocannl-638).
        cell = self.cell(
            [
                self.arm("A", True, "tensorized", statements=4),
                self.arm("B", True, "scalar-fallback", statements=2, fallbacks=2),
            ],
            shipped="B",
        )

        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")

    def test_every_verdict_renders_in_the_table(self):
        # The same completeness check the provenance column carries: a verdict with no mark would
        # print as an em dash, which is the "never asked" reading.
        for verdict in ("TENSORIZED", "SCALAR-FALLBACK", "NOT-EMITTED", "NOT-REQUESTED", "UNKNOWN"):
            self.assertIn(verdict, orchestrate.TENSORIZATION_MARK)
        for verdict in orchestrate.TENSORIZATION_MISMATCH:
            self.assertIn(verdict, orchestrate.TENSORIZATION_MARK)
            self.assertIn("**", orchestrate.TENSORIZATION_MARK[verdict])

    def test_the_mismatch_is_visible_in_the_rendered_table(self):
        cells = [
            self.cell([self.arm("A", True, "scalar-fallback", statements=3, fallbacks=3)]),
            self.cell([self.arm("A", True, "tensorized", statements=4)]),
        ]
        orchestrate.tensorization_check(cells)
        out = Path(tempfile.mkdtemp())
        orchestrate.report(cells, out)
        table = (out / "report.md").read_text()

        self.assertIn(" mma |", table)
        self.assertIn("**SCALAR FALLBACK**", table)
        self.assertIn("tensorized", table)


class RunnerProvenanceProbeTest(unittest.TestCase):
    """gh-ocannl-644: what the Python runners report, and what they refuse to claim."""

    def test_a_cell_that_runs_no_search_says_so(self):
        self.assertIs(bench_common.tinygrad_searched(None, beam=0), False)
        self.assertIs(bench_common.torch_searched(object(), compiled=False), False)

    def test_beam_counts_distinguish_a_search_from_a_kernel_cache_replay(self):
        self.assertIs(
            bench_common.tinygrad_searched({"call": 4, "hit": 3, "put": 1}, beam=4), True
        )
        self.assertIs(
            bench_common.tinygrad_searched({"call": 3, "hit": 3, "put": 0}, beam=4), False
        )

    def test_a_probe_that_saw_nothing_says_unknown_rather_than_no(self):
        # The internals moved under the probe. A wrong False is exactly the silent claim the
        # field exists to prevent, so this must not read as "replayed a cache".
        self.assertIsNone(bench_common.tinygrad_searched(None, beam=4))
        self.assertIsNone(
            bench_common.tinygrad_searched({"call": 0, "hit": 0, "put": 0}, beam=4)
        )

    def test_torch_reads_the_fx_graph_cache_counters(self):
        class Torch:
            def __init__(self, **counters):
                self._dynamo = type(
                    "d", (), {"utils": type("u", (), {"counters": {"inductor": counters}})}
                )

        self.assertIs(bench_common.torch_searched(Torch(fxgraph_cache_miss=2), True), True)
        self.assertIs(
            bench_common.torch_searched(Torch(fxgraph_cache_hit=2), compiled=True), False
        )
        self.assertIsNone(bench_common.torch_searched(Torch(), compiled=True))
        self.assertIsNone(bench_common.torch_searched(object(), compiled=True))

    def test_a_beam_search_that_writes_no_cache_entry_still_counts_as_a_search(self):
        # CACHELEVEL=0 / IGNORE_BEAM_CACHE: the search runs and touches no cache. Counting cache
        # traffic alone cannot see it, which is why the probe counts calls too.
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 0, "put": 0}, beam=4), True)
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 1, "put": 0}, beam=4), True)
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 4, "put": 0}, beam=4), False)

    def test_a_bypassed_torch_graph_is_codegen_not_a_replay(self):
        # A run with more than one graph is mixed: one served from the cache, one the cache
        # refused, is still a graph compiled in the timing process.
        class Torch:
            def __init__(self, **counters):
                self._dynamo = type(
                    "d", (), {"utils": type("u", (), {"counters": {"inductor": counters}})}
                )

        mixed = Torch(fxgraph_cache_hit=3, fxgraph_cache_bypass=1)
        self.assertIs(bench_common.torch_searched(mixed, compiled=True), True)
        self.assertIs(
            bench_common.torch_searched(Torch(fxgraph_cache_bypass=2), compiled=True), True
        )

    def test_the_beam_instrument_counts_only_beam_entries_and_passes_values_through(self):
        stored = {("beam_search", "k1"): "winner"}
        calls = []

        class Search:
            @staticmethod
            def diskcache_get(table, key):
                calls.append(("get", table, key))
                return stored.get((table, key))

            @staticmethod
            def diskcache_put(table, key, val):
                stored[(table, key)] = val
                return val

        module = types.ModuleType("tinygrad.engine.search")
        module.diskcache_get, module.diskcache_put = Search.diskcache_get, Search.diskcache_put
        packages = {
            "tinygrad": types.ModuleType("tinygrad"),
            "tinygrad.engine": types.ModuleType("tinygrad.engine"),
            "tinygrad.engine.search": module,
        }
        packages["tinygrad.engine"].search = module
        with unittest.mock.patch.dict(sys.modules, packages):
            counts = bench_common.instrument_tinygrad_beam()
            self.assertEqual(module.diskcache_get("beam_search", "k1"), "winner")
            self.assertIsNone(module.diskcache_get("beam_search", "k2"))
            module.diskcache_put("beam_search", "k2", "found")
            module.diskcache_get("compile", "unrelated")
            module.diskcache_put("compile", "unrelated", "x")

        self.assertEqual(counts, {"call": 0, "hit": 1, "put": 1})
        self.assertIn(("get", "compile", "unrelated"), calls)

    def test_an_uninstrumentable_tinygrad_is_not_an_error(self):
        module = types.ModuleType("tinygrad.engine.search")  # no diskcache helpers
        packages = {
            "tinygrad": types.ModuleType("tinygrad"),
            "tinygrad.engine": types.ModuleType("tinygrad.engine"),
            "tinygrad.engine.search": module,
        }
        with unittest.mock.patch.dict(sys.modules, packages):
            self.assertIsNone(bench_common.instrument_tinygrad_beam())


class FixtureDigestTest(unittest.TestCase):
    """gh-ocannl-645: what a report's numbers were measured on is recorded, not assumed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, name, content=b"weights"):
        path = self.dir / name
        path.write_bytes(content)
        return path

    #: The origin `gen_fixtures.main` records under when none is given, pinned so the
    #: generator tests do not depend on what this box happens to be called.
    this_box = "test-box"

    def gen_fixtures_module(self):
        """`gen_fixtures`, imported without the bench venv, with `build` never reached.

        It imports numpy and safetensors at module scope, but the property the tests below pin --
        the ORDER of `main()`'s steps -- patches `build` out, so nothing here draws a random
        number or writes a safetensors file. Stubbing the two imports is what keeps the check
        running under the bare `python3` the dune rule invokes: skipping there would mean the
        one place it runs unattended never evaluates it.
        """
        stub = types.ModuleType("safetensors.numpy")
        stub.save_file = None
        stubs = {
            "numpy": types.ModuleType("numpy"),
            "safetensors": types.ModuleType("safetensors"),
            "safetensors.numpy": stub,
        }
        with unittest.mock.patch.dict(sys.modules, stubs):
            sys.modules.pop("gen_fixtures", None)
            import gen_fixtures
        patch = unittest.mock.patch.object(
            fixture_digest.platform, "node", return_value=self.this_box
        )
        patch.start()
        self.addCleanup(patch.stop)
        return gen_fixtures

    def check(self, *args, **kwargs):
        """check_fixture_digests with its per-fixture log kept out of the test output."""
        with contextlib.redirect_stdout(io.StringIO()):
            return orchestrate.check_fixture_digests(*args, **kwargs)

    def test_recording_round_trips(self):
        fx = self.fixture("mlp_small.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        changes = fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(
            [(name, origin, was) for name, origin, was, _ in changes],
            [("mlp_small.safetensors", "rog-nv", None)],
        )
        entries = fixture_digest.read_digests(digests)
        self.assertEqual(
            entries[fx.name],
            [fixture_digest.Entry(fixture_digest.sha256_file(fx), fx.stat().st_size, "rog-nv")],
        )
        self.assertEqual(fixture_digest.status(fx, entries)[0], "MATCH")

    def test_a_match_names_the_box_whose_bytes_they_are(self):
        # gh-ocannl-759: "which bytes" was never the whole question. The reports compare boxes,
        # so a number has to carry whose workload it is on, not only that it is on a pinned one.
        fx = self.fixture("gpt2_mini.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")

        verdict, _, _, origins = fixture_digest.status(fx, fixture_digest.read_digests(digests))

        self.assertEqual((verdict, origins), ("MATCH", "minix"))

    def test_regenerating_different_bytes_is_announced_as_a_change(self):
        fx = self.fixture("gpt2_mini.safetensors", b"generated at spec revision A")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        (was,) = fixture_digest.read_digests(digests)[fx.name]

        fx.write_bytes(b"generated at spec revision B")
        changes = fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(len(changes), 1)
        name, origin, previous, now = changes[0]
        self.assertEqual((name, origin, previous), (fx.name, "rog-nv", was))
        self.assertEqual([now], fixture_digest.read_digests(digests)[fx.name])

    def test_regenerating_one_fixture_keeps_the_others_recorded(self):
        # gen_fixtures.py takes a spec list; regenerating one workload must not drop the
        # identities of the fixtures already on disk, or the pin silently narrows to one.
        a, b = self.fixture("a.safetensors", b"aaa"), self.fixture("b.safetensors", b"bbb")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [a, b], "rog-nv")

        fixture_digest.record(digests, [a], "rog-nv")

        self.assertEqual(set(fixture_digest.read_digests(digests)), {a.name, b.name})

    def test_two_boxes_bytes_are_both_recorded_and_both_match(self):
        # The state gh-ocannl-759 found and this format exists for: mlp_small and gpt2_mini hash
        # differently on minix and rog-nv at identical sizes (different venvs, different numpy
        # streams). Neither box's published numbers may be evicted to make the other's fit.
        fx = self.fixture("mlp_small.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")

        fx.write_bytes(b"rog-nv bytes")
        fixture_digest.record(digests, [fx], "rog-nv")

        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries[fx.name]], ["minix", "rog-nv"])
        self.assertEqual(fixture_digest.status(fx, entries)[3], "rog-nv")
        fx.write_bytes(b"minix bytes")
        self.assertEqual(fixture_digest.status(fx, entries)[3], "minix")

    def test_one_box_regenerating_does_not_unpin_the_other(self):
        # The trap the issue was filed about: with one entry per name, whichever box regenerated
        # first pinned its numpy stream and the other box's untouched fixtures then read as a
        # CHANGED workload -- a false alarm that would have been "fixed" by re-recording, i.e. by
        # evicting the first box in turn, forever.
        fx = self.fixture("gpt2_mini.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")
        minix_entries = fixture_digest.read_digests(digests)

        fx.write_bytes(b"rog-nv regenerates its own copy")
        fixture_digest.record(digests, [fx], "rog-nv")

        entries = fixture_digest.read_digests(digests)
        self.assertEqual(entries[fx.name][0], minix_entries[fx.name][0])
        fx.write_bytes(b"minix bytes")
        self.assertEqual(fixture_digest.status(fx, entries)[0], "MATCH")

    def test_agreeing_boxes_are_reported_as_agreeing(self):
        # The happy outcome of a coordinated regeneration, and it should be visible: one set of
        # bytes recorded by both boxes reads as both, not as an arbitrary one of them.
        fx = self.fixture("lenet.safetensors", b"the coordinated bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        fixture_digest.record(digests, [fx], "minix")

        verdict, _, _, origins = fixture_digest.status(fx, fixture_digest.read_digests(digests))

        self.assertEqual((verdict, origins), ("MATCH", "minix,rog-nv"))

    def test_a_differently_generated_fixture_does_not_match(self):
        # The hazard: the difference is applied uniformly to every cell, so the cross-cell parity
        # gate certifies it. Only the digest contradicts the report's workload name.
        fx = self.fixture("lenet.safetensors", b"the bytes the report was measured on")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        entries = fixture_digest.read_digests(digests)

        fx.write_bytes(b"the bytes someone regenerated later")

        verdict, _, _, origins = fixture_digest.status(fx, entries)
        self.assertEqual((verdict, origins), ("MISMATCH", None))

    def test_bytes_no_recorded_box_has_still_mismatch(self):
        # Multi-origin widens what MATCHes to "some recorded box's bytes" -- it must not widen it
        # to "recorded under this name", or the gate becomes a filename check.
        fx = self.fixture("mlp_wide.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")
        fx.write_bytes(b"rog-nv bytes")
        fixture_digest.record(digests, [fx], "rog-nv")
        entries = fixture_digest.read_digests(digests)

        fx.write_bytes(b"a third box nobody recorded")

        self.assertEqual(fixture_digest.status(fx, entries)[0], "MISMATCH")

    def test_an_unrecorded_fixture_is_not_silently_accepted(self):
        fx = self.fixture("mystery.safetensors")

        self.assertEqual(fixture_digest.status(fx, {})[0], "UNRECORDED")

    def test_a_malformed_line_is_an_error_not_a_dropped_pin(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  lenet.safetensors\n")

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_an_unattributed_legacy_line_is_refused_not_guessed(self):
        # A three-field line is the pre-759 format. Adopting it under a guessed origin would
        # record a coin flip as a fact, and the boxes' bytes really do differ.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  17  lenet.safetensors\n")

        with self.assertRaises(ValueError) as refused:
            fixture_digest.read_digests(digests)

        self.assertIn("--record", str(refused.exception))

    def test_one_origin_cannot_be_recorded_twice_for_one_fixture(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.HEADER
            + "aa  1  lenet.safetensors  rog-nv\nbb  2  lenet.safetensors  rog-nv\n"
        )

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_two_files_cannot_be_recorded_under_one_fixture_name(self):
        # `read_digests` refuses one (name, origin) twice; recording must not manufacture the same
        # ambiguity from the other side. Two boxes' copies of one workload passed to one command
        # (`box-a/mlp_small.safetensors`, `box-b/mlp_small.safetensors`) had the later replace the
        # earlier, and the replacement was ANNOUNCED as a changed workload -- a regeneration that
        # never happened, with the first box's bytes gone from the file.
        (self.dir / "box-a").mkdir()
        (self.dir / "box-b").mkdir()
        a = self.fixture("box-a/mlp_small.safetensors", b"box-a bytes")
        b = self.fixture("box-b/mlp_small.safetensors", b"box-b bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [a, b], "rog-nv")

        self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_repeating_one_fixture_path_is_not_an_ambiguity(self):
        # The control: the guard refuses two files claiming one name, not one file named twice --
        # which says nothing contradictory and records exactly what it always did.
        fx = self.fixture("mlp_small.safetensors", b"one box's bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        fixture_digest.record(digests, [fx, fx], "rog-nv")

        self.assertEqual(
            [(e.origin, e.sha256) for e in fixture_digest.read_digests(digests)[fx.name]],
            [("rog-nv", fixture_digest.sha256_file(fx))],
        )

    def test_an_origin_with_whitespace_is_refused(self):
        # The format is whitespace-split, so an origin containing a space would write a line that
        # reads back as a different (or malformed) entry.
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "rog nv")

        self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_an_origin_unsafe_for_a_log_filename_is_refused(self):
        # The same identifiers key sweep log paths. A slash creates an unintended subdirectory;
        # backslash and colon are not portable to the Windows measurement boxes.
        fx = self.fixture("lenet.safetensors")
        for origin in ("rog/nv", r"rog\nv", "rog:nv", ".hidden"):
            digests = self.dir / f"{origin.encode().hex()}.txt"
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    fixture_digest.record(digests, [fx], origin)
                self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_a_fixture_name_with_whitespace_is_refused(self):
        # The name-side twin of test_an_origin_with_whitespace_is_refused: `write_digests` emits
        # names unescaped into the whitespace-split format, so `a b.safetensors` would write a
        # line every later read refuses -- and recording rewrites the whole file, so one such
        # recording breaks the checked-in record. Refused before the file is touched.
        fx = self.fixture("a b.safetensors", b"whatever bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "rog-nv")

        self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_an_origin_read_from_the_file_is_held_to_the_same_identity_rules(self):
        # check_origin guards every writer CLI, but a checked-in or hand-merged row bypasses
        # them: a recorded `minix,rocm` parses into a fixture_origin field byte-identical to two
        # agreeing boxes named minix and rocm. Refused at the one insertion point, line named.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.HEADER + "deadbeef  17  lenet.safetensors  minix,rocm\n"
        )

        with self.assertRaises(ValueError) as refused:
            fixture_digest.read_digests(digests)

        self.assertIn("minix,rocm", str(refused.exception))
        self.assertIn(str(digests), str(refused.exception))

    def test_check_mode_refuses_an_origin_it_would_ignore(self):
        # --check reports against EVERY recorded origin; accepting --origin and doing nothing
        # with it would let `--check --origin "$BOX"` read as having verified that box's bytes.
        fx = self.fixture("lenet.safetensors")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as refused:
                fixture_digest._main(["--check", str(fx), "--origin", "rog-nv"])

        self.assertEqual(refused.exception.code, 2)

    def test_an_origin_that_reads_as_two_origins_is_refused(self):
        # Agreeing boxes are serialized into ONE comma-joined `fixture_origin` field, carried by
        # every result row and report section. A box named `minix,rocm` would publish a field
        # byte-identical to the one two boxes named `minix` and `rocm` produce by agreeing, and no
        # consumer of a published row could tell one box from two. The control is that exact
        # string, arrived at honestly.
        fx = self.fixture("lenet.safetensors", b"same bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "minix,rocm")

        self.assertFalse(digests.exists(), "refused before the file is touched")
        agreed = self.dir / "agreed.txt"
        fixture_digest.record(agreed, [fx], "minix")
        fixture_digest.record(agreed, [fx], "rocm")
        self.assertEqual(
            fixture_digest.status(fx, fixture_digest.read_digests(agreed))[3], "minix,rocm"
        )

    def test_the_checked_in_file_parses_and_names_live_workloads(self):
        # A digest for a workload spec that no longer exists is stale: it pins bytes no current
        # run can produce, and reads as coverage.
        entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
        specs = {p.stem for p in (HERE / "workloads").glob("*.json")}
        self.assertTrue(specs, "no workload specs found next to the test")
        for name in entries:
            self.assertTrue(name.endswith(".safetensors"), name)
            self.assertIn(name[: -len(".safetensors")], specs, name)

    def test_the_checked_in_file_attributes_every_entry(self):
        # The whole point of gh-ocannl-759: no entry may be anonymous, because the published
        # numbers are on more than one box's bytes and the reports have to say which.
        entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
        self.assertTrue(entries, "the checked-in digest file records nothing (gh-ocannl-759)")
        for name, recorded in entries.items():
            for e in recorded:
                self.assertTrue(e.origin and e.origin.strip(), name)

    def test_the_checked_in_header_declares_the_measurement_boxes(self):
        digests = HERE / "fixtures" / fixture_digest.DIGEST_FILE

        self.assertEqual(
            fixture_digest.declared_measurement_boxes(digests),
            ["m4-max", "minix", "rog-nv"],
        )

        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = fixture_digest._main(
                ["--list-declared-measurement-boxes", "--digests", str(digests)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), "m4-max\nminix\nrog-nv\n")

    def test_listing_boxes_does_not_infer_a_matrix_for_a_legacy_file(self):
        # A pre-gh-ocannl-850 file can still be swept for backend coverage, but its row origins
        # are not a declaration that every measuring host is present. Environment aggregation
        # must therefore receive no matrix rather than quietly treating the observed rows as one.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.HEADER + "deadbeef  17  lenet.safetensors  rog-nv\n"
        )

        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = fixture_digest._main(
                ["--list-declared-measurement-boxes", "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), "")

    def test_the_unrecorded_metal_box_is_reported_for_checked_in_fixtures(self):
        digests = HERE / "fixtures" / fixture_digest.DIGEST_FILE
        entries = fixture_digest.read_digests(digests)

        for name, recorded in entries.items():
            self.assertTrue(recorded, name)
            self.assertIn(
                "m4-max",
                fixture_digest.divergent_origins(digests, [name], recorded[0].origin),
                name,
            )

    def test_a_duplicate_box_declaration_is_refused(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.header(["minix", "rog-nv"])
            + f"{fixture_digest.MEASUREMENT_BOXES_FIELD} third-box\n"
        )

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_an_entry_for_an_undeclared_box_is_refused(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.header(["minix"])
            + "deadbeef  17  lenet.safetensors  rog-nv\n"
        )

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_a_regenerating_box_is_told_which_others_it_has_left_behind(self):
        # What gen_fixtures.py prints after regenerating: the other boxes' entries survive, so
        # nothing fails, and that silence is exactly what would let the boxes drift apart again.
        fx = self.fixture("mlp_small.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")
        fx.write_bytes(b"rog-nv regenerates")
        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), ["minix"])
        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "minix"), ["rog-nv"])

    def test_a_box_that_is_the_only_one_recorded_leaves_nobody_behind(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), [])

    def test_a_declared_box_with_no_fixture_entry_is_reported_as_left_behind(self):
        # gh-ocannl-850's missing fact: without an independent box declaration, an absent minix
        # row is indistinguishable from minix never measuring this workload and stays silent.
        fx = self.fixture("lenet.safetensors", b"rog-nv regenerates")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["minix", "rog-nv"]))

        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.measurement_boxes(digests), ["minix", "rog-nv"])
        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), ["minix"])

    def test_a_report_names_a_declared_box_with_no_fixture_entry(self):
        fx = self.fixture("mlp_wide.safetensors", b"rog-nv bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["minix", "rog-nv"]))
        fixture_digest.record(digests, [fx], "rog-nv")
        row = cell("ocannl", "cuda", "default", [2.0, 1.9])
        row.update(
            fixture=fx.name,
            fixture_sha256=fixture_digest.sha256_file(fx),
            fixture_origin="rog-nv",
            parity="PASS",
            parity_loss_moved=True,
        )
        out = self.dir / "report"

        orchestrate.report([row], out, digests_path=digests)
        text = (out / "report.md").read_text()

        self.assertIn("measurement boxes declared by `fixtures/DIGESTS.txt`: minix, rog-nv", text)
        self.assertIn("declared measurement box(es) with no entry", text)
        self.assertIn("minix", text)

    def test_raw_result_rows_preserve_the_box_set_and_missing_entries(self):
        # The JSONL outlives its DIGESTS.txt checkout (and partial.jsonl can outlive the run), so
        # the fleet and missing-entry reading belong in the row, not only in report prose.
        fx = self.fixture("mlp_wide.safetensors", b"rog-nv bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["m4-max", "minix", "rog-nv"]))
        fixture_digest.record(digests, [fx], "rog-nv")
        entries = fixture_digest.read_digests(digests)
        boxes = fixture_digest.measurement_boxes(digests)
        stamp = orchestrate.fixture_result_stamp(
            fx, fixture_digest.sha256_file(fx), "rog-nv", entries, boxes
        )
        # This is the stamp main applies before collect writes partial.jsonl. Asserting it before
        # report() matters: report enriches legacy in-memory rows for final JSONL compatibility,
        # but it cannot repair a partial checkpoint left by an interrupted run.
        self.assertEqual(stamp["measurement_boxes"], ["m4-max", "minix", "rog-nv"])
        self.assertEqual(stamp["fixture_missing_origins"], ["m4-max", "minix"])
        row = cell("ocannl", "cuda", "default", [2.0, 1.9])
        row.update(stamp)
        row.update(parity="PASS", parity_loss_moved=True)
        out = self.dir / "report-with-raw-provenance"

        orchestrate.report([row], out, digests_path=digests)
        raw = json.loads((out / "results.jsonl").read_text())

        self.assertEqual(raw["measurement_boxes"], ["m4-max", "minix", "rog-nv"])
        self.assertEqual(raw["fixture_missing_origins"], ["m4-max", "minix"])

    def test_a_coordinated_regeneration_is_not_reported_as_divergence(self):
        # The success case of the cross-box event: both boxes land the SAME bytes. Naming the
        # other box as "still on a different workload" would tell the operator to redo the thing
        # they just succeeded at. Divergence is judged on bytes, never on the name differing.
        fx = self.fixture("gpt2_mini.safetensors", b"the coordinated bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")

        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), [])

    def test_divergence_is_reported_per_fixture_not_per_box(self):
        # Boxes can agree on one workload and differ on another; the warning must fire on the
        # second without being suppressed by the first.
        agree = self.fixture("lenet.safetensors", b"same everywhere")
        differ = self.fixture("mlp_wide.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [agree, differ], "minix")
        differ.write_bytes(b"rog-nv bytes")
        fixture_digest.record(digests, [agree, differ], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [agree.name], "rog-nv"), [])
        self.assertEqual(
            fixture_digest.divergent_origins(digests, [agree.name, differ.name], "rog-nv"),
            ["minix"],
        )

    def test_an_explicitly_empty_origin_is_refused_not_defaulted(self):
        # How automation fails: `--origin "$BOX"` with $BOX unset. Silently substituting this
        # hostname would attribute the fixture to the wrong box and persist it -- the exact error
        # the file exists to prevent.
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "")

    def test_the_advertised_legacy_migration_actually_runs(self):
        # The refusal names a command; that command has to be able to reach a valid file. Without
        # --adopt-legacy, record() would raise from its own read before rewriting anything.
        fx = self.fixture("lenet.safetensors", b"the legacy bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        sha, size = fixture_digest.sha256_file(fx), fx.stat().st_size
        digests.write_text(fixture_digest.HEADER + f"{sha}  {size}  {fx.name}\n")
        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

        with contextlib.redirect_stdout(io.StringIO()):
            code = fixture_digest._main(
                ["--record", str(fx), "--origin", "minix", "--adopt-legacy", "minix",
                 "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries[fx.name]], ["minix"])
        self.assertEqual(fixture_digest.status(fx, entries)[0], "MATCH")

    def test_adopting_legacy_lines_attributes_them_without_changing_bytes(self):
        # The migration states whose the bytes were; it must not restate WHAT they were. A legacy
        # line for a fixture not being recorded now keeps its digest exactly.
        other = "cifar_conv.safetensors"
        fx = self.fixture("lenet.safetensors", b"recorded now")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + f"deadbeef  17  {other}\n")

        fixture_digest.record(digests, [fx], "minix", legacy_origin="minix")

        entries = fixture_digest.read_digests(digests)
        self.assertEqual(entries[other], [fixture_digest.Entry("deadbeef", 17, "minix")])

    def test_a_migration_does_not_split_one_box_across_two_origin_names(self):
        # The migration writes two facts -- whose the old rows were, and whose the bytes on disk
        # are -- and resolved them from two independent sources: `--adopt-legacy rog-nv` on a host
        # that calls itself `rog-nv-wsl` attributed the rows to rog-nv and recorded the fixtures
        # as rog-nv-wsl. One box then wears two origin names, which reads downstream as two boxes
        # agreeing -- or, for a fixture whose legacy entry was just adopted, as the box having
        # diverged from itself, with `divergent_origins` telling it to coordinate with itself.
        fx = self.fixture("mlp_small.safetensors", b"this box's bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + f"deadbeef  17  {fx.name}\n")
        before = digests.read_text()

        with unittest.mock.patch.object(
            fixture_digest.platform, "node", return_value="rog-nv-wsl"
        ):
            with self.assertRaises(ValueError) as refused:
                fixture_digest._main(
                    ["--record", str(fx), "--adopt-legacy", "rog-nv", "--digests", str(digests)]
                )

        self.assertIn("--origin", str(refused.exception))
        self.assertEqual(digests.read_text(), before, "refused before the file is rewritten")

    def test_the_migration_still_defaults_where_the_two_names_agree(self):
        # First control: on the box the rows are being adopted as, the advertised command still
        # needs no --origin. The refusal above must not have made the ordinary migration harder.
        fx = self.fixture("mlp_small.safetensors", b"this box's bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + f"deadbeef  17  {fx.name}\n")

        with unittest.mock.patch.object(fixture_digest.platform, "node", return_value="rog-nv"):
            with contextlib.redirect_stdout(io.StringIO()):
                code = fixture_digest._main(
                    ["--record", str(fx), "--adopt-legacy", "rog-nv", "--digests", str(digests)]
                )

        self.assertEqual(code, 0)
        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries[fx.name]], ["rog-nv"])

    def test_one_box_may_still_migrate_another_boxs_legacy_rows(self):
        # Second control: two origins in one file ARE right when both were stated -- rog-nv
        # attributing rows it knows were minix's while recording its own bytes as its own. The
        # rule is about an origin nobody typed, not about a file naming two boxes.
        fx = self.fixture("lenet.safetensors", b"rog-nv bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  17  cifar_conv.safetensors\n")

        with contextlib.redirect_stdout(io.StringIO()):
            code = fixture_digest._main(
                ["--record", str(fx), "--origin", "rog-nv", "--adopt-legacy", "minix",
                 "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries["cifar_conv.safetensors"]], ["minix"])
        self.assertEqual([e.origin for e in entries[fx.name]], ["rog-nv"])

    def test_the_legacy_refusal_names_a_command_that_runs_from_the_repo_root(self):
        # The canonical sweep command runs from the repository root, where a bare
        # `fixture_digest.py` names nothing -- the operator cannot paste the remediation.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  17  lenet.safetensors\n")

        with self.assertRaises(ValueError) as refused:
            fixture_digest.read_digests(digests)

        self.assertIn("--adopt-legacy", str(refused.exception))
        self.assertIn(fixture_digest.cli_command(), str(refused.exception))

    def test_the_advertised_path_resolves_against_the_callers_cwd(self):
        # From the repository root it must spell benchmarks/fixture_digest.py, not
        # fixture_digest.py; from an unrelated cwd it falls back to an absolute path.
        root = HERE.parent
        with unittest.mock.patch.object(Path, "cwd", staticmethod(lambda: root)):
            self.assertEqual(fixture_digest.cli_command(), "benchmarks/fixture_digest.py")
        with unittest.mock.patch.object(Path, "cwd", staticmethod(lambda: Path(self.dir))):
            self.assertTrue(Path(fixture_digest.cli_command()).is_absolute())

    def test_the_checked_in_file_keeps_every_box_that_published_numbers(self):
        # gh-ocannl-759 recorded minix's and rog-nv's bytes for the two fixtures both boxes have
        # published numbers on. Neither may be evicted -- by a regeneration, or by a re-record
        # that forgets the other box -- or a standing report is retroactively on a workload
        # nothing pins. (Their digests are deliberately NOT pinned here: a coordinated
        # regeneration is allowed to change them, it is only allowed to change them for BOTH.)
        entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
        for name in ("mlp_small.safetensors", "gpt2_mini.safetensors"):
            self.assertEqual({e.origin for e in entries[name]}, {"minix", "rog-nv"}, name)

    def test_the_sweep_refuses_bytes_nothing_records(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.write_digests(digests, {})

        with self.assertRaises(SystemExit) as refused:
            self.check([fx], digests_path=digests)

        self.assertIn("--no-fixture-digest-check", str(refused.exception))

    def test_the_sweep_runs_and_stamps_a_recorded_fixture(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")

        ids = self.check([fx], digests_path=digests)

        self.assertEqual(ids, {fx: (fixture_digest.sha256_file(fx), "rog-nv")})

    def test_the_opt_out_measures_them_and_still_reports_the_digest(self):
        # A deliberate regeneration is a legitimate reason to run unpinned; it must not also cost
        # the run its record of what it ran on.
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        fx.write_bytes(b"regenerated")

        ids = self.check([fx], digests_path=digests, allow_unpinned=True)

        self.assertEqual(ids, {fx: (fixture_digest.sha256_file(fx), None)})

    def test_a_report_section_says_whose_bytes_its_numbers_are_on(self):
        # The published half of gh-ocannl-759: a reader comparing this report against the other
        # box's has to be able to see, without leaving the report, that the two are on different
        # bytes -- which for mlp_small and gpt2_mini they are.
        rows = [
            dict(
                cell("ocannl", "cuda", "default", [2.3026, 2.3010]),
                fixture="gpt2_mini.safetensors",
                fixture_sha256="043c1ea8",
                fixture_origin="rog-nv",
            )
        ]
        orchestrate.parity_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            text = (out / "report.md").read_text()

        self.assertIn("sha256 `043c1ea8`, rog-nv's bytes", text)

    def test_the_record_cli_pins_existing_bytes_without_regenerating(self):
        # The gh-ocannl-759 command: fixtures that predate the digest file get pinned as they are.
        fx = self.fixture("lenet.safetensors", b"bytes the published numbers are on")
        before = fx.read_bytes()
        digests = self.dir / fixture_digest.DIGEST_FILE

        with contextlib.redirect_stdout(io.StringIO()):
            code = fixture_digest._main(
                ["--record", str(fx), "--origin", "rog-nv", "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        self.assertEqual(fx.read_bytes(), before, "recording must not touch the fixture")
        self.assertEqual(fixture_digest.read_digests(digests)[fx.name][0].origin, "rog-nv")

    def test_the_record_cli_defaults_to_this_host(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with contextlib.redirect_stdout(io.StringIO()):
            fixture_digest._main(["--record", str(fx), "--digests", str(digests)])

        self.assertEqual(
            fixture_digest.read_digests(digests)[fx.name][0].origin, fixture_digest.this_origin()
        )

    def test_the_check_cli_passes_a_recorded_fixture_and_fails_a_perturbed_one(self):
        fx = self.fixture("lenet.safetensors", b"recorded bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")

        with contextlib.redirect_stdout(io.StringIO()) as good:
            passed = fixture_digest._main(["--check", str(fx), "--digests", str(digests)])
        fx.write_bytes(b"perturbed bytes")
        with contextlib.redirect_stdout(io.StringIO()) as bad:
            failed = fixture_digest._main(["--check", str(fx), "--digests", str(digests)])

        self.assertEqual((passed, failed), (0, 1))
        self.assertIn("MATCH — rog-nv's bytes", good.getvalue())
        self.assertIn("MISMATCH", bad.getvalue())

    def test_a_host_that_cannot_name_itself_is_refused_not_given_a_placeholder(self):
        # A literal `unknown-host` is not an origin, it is every nameless box sharing one name:
        # the second such box to record different bytes for a fixture would replace the first's
        # entry under it, which is the whichever-box-records-last provenance loss keying by
        # origin exists to prevent -- only now wearing a name that reads like an answer.
        with unittest.mock.patch.object(fixture_digest.platform, "node", return_value=""):
            self.assertIsNone(fixture_digest.this_origin())
            with self.assertRaises(ValueError) as refused:
                fixture_digest.resolve_origin(None)

        self.assertIn("--origin", str(refused.exception))

    def test_a_named_host_still_supplies_the_default(self):
        # The negative control for the refusal above: absence still defaults where there is a
        # name to default to, which is the whole convenience of the recording CLI.
        with unittest.mock.patch.object(fixture_digest.platform, "node", return_value="rog-nv"):
            self.assertEqual(fixture_digest.resolve_origin(None), "rog-nv")

    def test_the_digest_file_is_validated_before_any_fixture_is_overwritten(self):
        # Generating OVERWRITES the fixture bytes, so anything record() would refuse -- a legacy
        # three-field line, a duplicate origin, a malformed row -- has to be found while the
        # previous bytes still exist. Refusing afterwards loses the bytes the published numbers
        # were measured on AND leaves the new ones unrecorded, which is worse than either alone.
        gen_fixtures = self.gen_fixtures_module()
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        stale = fixtures / "lenet.safetensors"
        stale.write_bytes(b"the bytes the published numbers are on")
        before = stale.read_bytes()
        (fixtures / fixture_digest.DIGEST_FILE).write_text(
            fixture_digest.HEADER + "deadbeef  17  lenet.safetensors\n"
        )
        spec = self.dir / "workloads" / "lenet.json"
        spec.parent.mkdir()
        spec.write_text("{}")
        built = []

        with unittest.mock.patch.object(gen_fixtures, "build", lambda s, d: built.append(s)):
            with self.assertRaises(ValueError):
                gen_fixtures.main(["--origin", "rog-nv"], here=self.dir)

        # And the spec really was one this run would have built, so [] is a refusal and not an
        # empty work list: the same call with a readable digest file builds it (below).
        self.assertEqual(built, [])
        self.assertEqual(stale.read_bytes(), before)

    def test_a_spec_name_the_format_cannot_carry_is_refused_before_building(self):
        # The name-side twin of the digest-file validation above: build() writes
        # fixtures/<spec name>.safetensors, and record() would refuse a whitespace name only
        # AFTER the previous bytes are overwritten. The name check runs first, while the bytes
        # the published numbers rest on still exist.
        gen_fixtures = self.gen_fixtures_module()
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        stale = fixtures / "a b.safetensors"
        stale.write_bytes(b"the bytes the published numbers are on")
        (fixtures / fixture_digest.DIGEST_FILE).write_text(fixture_digest.HEADER)
        spec = self.dir / "workloads" / "spaced.json"
        spec.parent.mkdir()
        spec.write_text('{"name": "a b"}')
        built = []

        with unittest.mock.patch.object(gen_fixtures, "build", lambda s, d: built.append(s)):
            with self.assertRaises(ValueError):
                gen_fixtures.main(["--origin", "rog-nv"], here=self.dir)

        self.assertEqual(built, [], "refused before any build")
        self.assertEqual(stale.read_bytes(), b"the bytes the published numbers are on")

    def test_a_valid_digest_file_lets_the_generator_build(self):
        # The negative control: the guard above must refuse the unreadable file, not every file.
        gen_fixtures = self.gen_fixtures_module()
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        digests = fixtures / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER)

        spec = self.dir / "workloads" / "lenet.json"
        spec.parent.mkdir()
        spec.write_text("{}")
        built = []

        def build(spec_path, out_dir):
            built.append(spec_path)
            path = out_dir / "lenet.safetensors"
            path.write_bytes(b"regenerated")
            return path

        with unittest.mock.patch.object(gen_fixtures, "build", build):
            with contextlib.redirect_stdout(io.StringIO()):
                gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        self.assertEqual(built, [spec])

        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries["lenet.safetensors"]], [self.this_box])

    def test_generated_fixtures_are_named_after_their_spec(self):
        # What the recorded-name check above relies on: gen_fixtures.py writes
        # fixtures/<spec name>.safetensors, and every spec's `name` is its file stem.
        for spec in (HERE / "workloads").glob("*.json"):
            self.assertEqual(json.loads(spec.read_text())["name"], spec.stem, spec.name)


class CellTimeoutTest(unittest.TestCase):
    """gh-ocannl-760: a cell that stops making progress costs the cell, not the sweep."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_cell(self, *args, **kwargs):
        """run_cell with the cell's own output kept out of the test's."""
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result, note = orchestrate.run_cell(*args, **kwargs)
        return result, note, out.getvalue()

    def python(self, source, *argv):
        return [sys.executable, "-c", source, *map(str, argv)]

    def assert_tinygrad_cache_in_scratch(self, cachedb):
        scratch = self.dir.resolve()
        resolved = cachedb.resolve()
        try:
            resolved.relative_to(scratch)
        except ValueError:
            self.fail(
                f"tinygrad cache probe resolved outside the test scratch directory: "
                f"{resolved} (scratch: {scratch})"
            )

    def alive(self, pid):
        """Whether `pid` is still a running process — a zombie is not one.

        The distinction is load-bearing, not pedantry: the grandchild these tests kill is an
        ORPHAN by then (its parent, the cell, was killed too), so whether it disappears or lingers
        as an unreaped zombie is decided by whoever inherits it. Under a normal init it is reaped
        at once; in a container whose PID 1 does not reap, it stays a zombie indefinitely — and
        `kill(pid, 0)` succeeds for a zombie, so a bare signal-0 probe would report the process
        the kill did remove as alive and fail the test in exactly the environments CI runs in.

        The liveness question itself goes through `cell_group.process_is_alive`, which is where the
        answer stops being `os.kill(pid, 0)`: on Windows that call terminates its subject and
        raises `WinError 87` for a pid that already exited. Only the zombie refinement is left
        here, because only POSIX has zombies.
        """
        if not cell_group.process_is_alive(pid):
            return False
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists():
            return True  # not Linux, or no procfs: the signal-0 answer is all there is
        try:
            # `comm` can contain spaces and parentheses; the state letter is the field after the
            # LAST ')'.
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False  # it went away between the probe and the read

    def wait_gone(self, pid, deadline_s=15.0):
        end = time.monotonic() + deadline_s
        while time.monotonic() < end and self.alive(pid):
            time.sleep(0.05)
        return not self.alive(pid)

    def test_a_held_termination_is_delivered_during_exceptional_unwinding(self):
        # Code after a contextmanager's finally is skipped when the body raises.  The old
        # deferral put delivery there, so a SIGTERM held while spawn/cleanup also raised stayed
        # pending until a later child -- or forever when this was the last one.
        cancellation = orchestrate._cancellation
        self.assertEqual(cancellation.depth, 0)
        self.assertIsNone(cancellation.held_signal)
        try:
            with self.assertRaises(SystemExit) as raised:
                with cancellation.deferring():
                    cancellation.held_signal = signal.SIGTERM
                    raise RuntimeError("the concurrent child failure")
            self.assertIn("terminated by signal", str(raised.exception))
            self.assertIn("subprocess it was running", str(raised.exception))
            self.assertIsInstance(raised.exception.__context__, RuntimeError)
        finally:
            cancellation.depth = 0
            cancellation.held_signal = None

    def test_a_held_termination_does_not_replace_an_orchestrator_cleanup_failure(self):
        cancellation = orchestrate._cancellation
        self.assertEqual(cancellation.depth, 0)
        self.assertIsNone(cancellation.held_signal)
        try:
            with self.assertRaises(cell_group.CleanupFailed) as raised:
                with cancellation.deferring():
                    cancellation.held_signal = signal.SIGTERM
                    raise cell_group.CleanupFailed("SURVIVORS still hold the device")
            self.assertIn("SURVIVORS", str(raised.exception))
            self.assertNotIn("killed first", str(raised.exception))
        finally:
            cancellation.depth = 0
            cancellation.held_signal = None

    def test_the_tinygrad_cache_probe_collects_helpers_it_spawned(self):
        # This probe was the final raw subprocess.run site in the matrix sweep.  Importing a
        # framework can itself create helpers, so a direct-child timeout is not enough.
        package = self.dir / "tinygrad"
        package.mkdir()
        (package / "__init__.py").write_text("")
        pidfile = self.dir / "probe-helper.pid"
        cachedb = self.dir / "cache.db"
        (package / "helpers.py").write_text(
            "import os, subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "open(os.environ['PROBE_HELPER_PID'], 'w').write(str(kid.pid))\n"
            "CACHEDB = os.environ['CACHEDB']\n"
        )
        env = os.environ.copy()
        env.update(
            {
                "CACHEDB": str(cachedb),
                "PYTHONPATH": str(self.dir),
                "PROBE_HELPER_PID": str(pidfile),
            }
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "VENV_PY", Path(sys.executable)))
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            found = orchestrate.tinygrad_cachedb(env)

        self.assert_tinygrad_cache_in_scratch(found)
        self.assertEqual(found, cachedb)
        self.assertTrue(pidfile.exists(), "the probe fixture did not spawn its helper")
        helper = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(helper), f"pid {helper} outlived the cache probe")

    def test_the_tinygrad_cache_probe_refuses_an_ambient_cache(self):
        # Negative control for the containment assertion above: the old fallback selected the
        # user's cache after an import timeout. Reproduce the same resolved-path shape even with
        # the explicit cache setting present, and prove the test refuses before inspecting it.
        configured = self.dir / "cache.db"
        ambient = self.dir.parent / "ambient-tinygrad-cache" / "cache.db"
        probe = subprocess.CompletedProcess([], 0, f"{ambient}\n", "")

        with unittest.mock.patch.object(orchestrate, "run_supporting", return_value=probe):
            found = orchestrate.tinygrad_cachedb({"CACHEDB": str(configured)})
            with self.assertRaisesRegex(
                self.failureException, "resolved outside the test scratch directory"
            ):
                self.assert_tinygrad_cache_in_scratch(found)

    def test_a_finished_cell_is_unaffected_by_the_cap(self):
        # The cap must be invisible to every cell that runs: same result line, no note.
        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )

        result, note, _ = self.run_cell("finished", cell, timeout=60)

        self.assertIsNone(note)
        self.assertEqual(result["workload"], "w")

    def test_the_cap_can_be_turned_off(self):
        # `--cell-timeout 0` is the escape hatch for a box whose legitimate cells outrun any cap
        # worth setting; it must mean "no cap", not "a cap of zero seconds".
        cell = self.python(
            "import json, time; time.sleep(0.2); print(json.dumps("
            "{'workload': 'uncapped', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )

        result, note, _ = self.run_cell("uncapped", cell, timeout=0)

        self.assertIsNone(note)
        self.assertEqual(result["workload"], "uncapped")

    def test_a_cell_over_the_cap_is_a_runner_failure_naming_the_cap(self):
        result, note, log = self.run_cell(
            "wedged", self.python("import time; time.sleep(300)"), timeout=1.0
        )

        self.assertIsNone(result)
        self.assertIn("TIMED OUT after 1s", note)
        self.assertIn("--cell-timeout", note)
        self.assertIn("wedged", log)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_the_kill_takes_the_whole_process_group(self):
        # The failure mode this exists for: tinygrad's beam search wedges with a pool of workers
        # alive, and those workers hold the cell's stdout pipe open. Kill the direct child alone
        # and the sweep then blocks reading a pipe nobody will ever close -- the hang moves, it
        # does not go away. So the assertion is both halves: the grandchild dies, and the call
        # returns promptly rather than waiting on the inherited pipe.
        pidfile = self.dir / "grandchild.pid"
        cell = self.python(
            "import subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "time.sleep(300)\n",
            pidfile,
        )

        t0 = time.monotonic()
        result, note, _ = self.run_cell("wedged with workers", cell, timeout=2.0)
        elapsed = time.monotonic() - t0

        self.assertIsNone(result)
        self.assertIn("process group", note)
        self.assertLess(elapsed, 2.0 + orchestrate.CELL_KILL_GRACE_S + 10)
        grandchild = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(grandchild), f"pid {grandchild} survived the cap")

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_the_escalation_is_decided_by_the_group_and_not_by_the_pipe(self):
        # The survivor case the pipe cannot see: a descendant that ignores SIGTERM and does NOT
        # hold the cell's stdout. The cell itself dies on SIGTERM, the pipe closes, and a kill path
        # that escalates only when its read blocks would return here reporting a killed group --
        # while the survivor keeps the GPU and every later cell of the sweep is measured against
        # it. So the SIGKILL has to be owed to the GROUP still having members (gh-ocannl-760
        # review).
        pidfile = self.dir / "stubborn.pid"
        cell = self.python(
            "import signal, subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c',\n"
            "  'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            " time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "sys.stdout.flush()\n"
            "time.sleep(300)\n",
            pidfile,
        )

        # The grace is shortened because this test spends ALL of it: the leader dies on the first
        # SIGTERM, so nothing can end the first pass early and the loop runs to its deadline
        # before the SIGKILL that is the point of the test. At the production 10 s that is ten
        # seconds of sleeping on every run of this suite on both CI platforms, and none of it is
        # under test here -- what is under test is which pass the escalation is owed to, and that
        # is the same at any grace length.
        with unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.5):
            result, note, _ = self.run_cell("stubborn descendant", cell, timeout=2.0)

        self.assertIsNone(result)
        stubborn = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(stubborn), f"pid {stubborn} ignored SIGTERM and survived")
        self.assertNotIn("SURVIVED SIGKILL", note)

    def test_a_member_outliving_sigkill_is_said_so_in_the_failure(self):
        # Nothing survives SIGKILL except a process stuck in the kernel (an uninterruptible driver
        # ioctl), which is not synthesizable -- so the group probe is stood in for. What is pinned
        # is the consequence: the sweep goes on, and the failure says every later cell in the run
        # was measured against whatever is still holding the device.
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            _, note, _ = self.run_cell(
                "stuck in the driver", self.python("import time; time.sleep(300)"), timeout=0.5
            )

        self.assertIn("SURVIVED SIGKILL", note)
        self.assertIn("measured against it", note)

    @unittest.skipUnless(os.name == "posix", "SIGALRM and process groups are POSIX here")
    def test_an_interrupted_cell_gets_the_same_cache_treatment_as_a_capped_one(self):
        # Ctrl-C on a wedged beam cell is the likeliest way anyone meets this bug by hand, and the
        # search it interrupts leaves exactly the partial cache the cap's kill path quarantines.
        called = []

        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)

        with self.assertRaises(KeyboardInterrupt):
            self.run_cell(
                "interrupted",
                self.python("import time; time.sleep(300)"),
                on_incomplete=lambda killed: called.append(("killed", killed))
                or "quarantined the cache",
            )

        # ... and told that this cell was KILLED, which is what decides whether the cache may
        # be moved aside at all.
        self.assertEqual(called, [("killed", True)])

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_sigterm_to_the_sweep_takes_the_running_cell_with_it(self):
        # A job cancellation or a scheduler's time limit is a SIGTERM to the sweep, and Python's
        # default action for it exits without unwinding -- so without a handler the cell and its
        # whole worker pool are orphaned, holding the GPU, with nobody to reap them. The cell is
        # in its own session precisely so the sweep's signals do NOT reach it (gh-ocannl-760
        # review).
        pidfile = self.dir / "orphan.pid"
        driver = self.python(
            "import sys, orchestrate\n"
            "orchestrate.install_termination_handler()\n"
            "cell = [sys.executable, '-c',\n"
            "  'import subprocess, sys, time\\n"
            "kid = subprocess.Popen([sys.executable, \\'-c\\', \\'import time; time.sleep(300)\\'])\\n"
            "open(sys.argv[1], \\'w\\').write(str(kid.pid))\\n"
            "time.sleep(300)\\n', sys.argv[1]]\n"
            "orchestrate.run_cell('cell under a cancelled sweep', cell)\n",
            pidfile,
        )
        sweep = subprocess.Popen(
            driver,
            cwd=str(Path(orchestrate.__file__).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(lambda: sweep.poll() is None and sweep.kill())
        end = time.monotonic() + 30
        while time.monotonic() < end and not pidfile.exists():
            time.sleep(0.05)
        self.assertTrue(pidfile.exists(), "the cell never started")
        grandchild = int(pidfile.read_text())

        sweep.terminate()
        sweep.wait(timeout=30)

        self.assertTrue(
            self.wait_gone(grandchild), f"pid {grandchild} outlived the cancelled sweep"
        )

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "the zombie distinction needs procfs")
    def test_a_group_of_nothing_but_zombies_is_not_alive(self):
        # What the escalation is asking is "does anything still hold the device", and a zombie
        # holds nothing. But `killpg(pgid, 0)` succeeds for a group whose every member is one, and
        # after the kill that is the normal state of the descendants -- orphans, reaped or not at
        # the whim of whoever inherits them. Under a PID 1 that does not reap (a container) a bare
        # signal-0 probe would announce a SIGKILL survivor that does not exist, and sit through
        # both grace periods to do it (gh-ocannl-760 review).
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, start_new_session=True
        )
        self.addCleanup(proc.wait)
        # Bounded, not `while alive`: against a probe that cannot see the distinction this must
        # FAIL rather than hang. Deliberately unreaped throughout -- the process becomes a zombie
        # in its own group (pgid == pid), and signal 0 still finds it.
        end = time.monotonic() + 10
        alive = True
        while time.monotonic() < end and alive:
            alive = (
                cell_group._observe_posix_group(proc.pid, allow_zombie_gone=True)
                is not cell_group.GONE
            )
            if alive:
                time.sleep(0.02)
        os.kill(proc.pid, 0)

        self.assertFalse(alive, "a group of nothing but zombies read as alive")

    def test_two_cells_killed_in_the_same_second_do_not_overwrite_each_other(self):
        # The stamp is second-resolution, so a low cap or a GPU column that wedges at once puts
        # two kills inside one second -- and `os.replace` onto the same name would destroy the
        # first cell's quarantined database, which is the evidence the rename exists to keep.
        db = self.dir / "cache.db"
        db.write_bytes(b"first wedged search")
        first = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})
        db.write_bytes(b"second wedged search")

        second = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})

        kept = sorted(p for p in self.dir.glob("cache.db.wedged-*"))
        self.assertEqual(len(kept), 2, kept)
        self.assertEqual(
            sorted(p.read_bytes() for p in kept),
            [b"first wedged search", b"second wedged search"],
        )
        self.assertNotEqual(first, second)

    def test_the_fleet_default_disables_the_spawn_pool(self):
        # The deadlock exists in tinygrad 0.13.0 and 0.14.0, on both GPU boxes. Zero is the only
        # value that removes that pool; an ambient positive value must not override the pin.
        ambient = {"PATH": "/usr/bin", "PARALLEL": "32"}

        self.assertEqual(orchestrate.DEFAULT_BEAM_PARALLEL, 0)
        self.assertEqual(
            orchestrate.beam_cell_env(ambient, orchestrate.DEFAULT_BEAM_PARALLEL)["PARALLEL"],
            "0",
        )
        self.assertEqual(orchestrate.beam_cell_env(ambient, 4)["PARALLEL"], "4")
        # Zero is a value, not an absence: it is what disables tinygrad's pool outright.
        self.assertEqual(orchestrate.beam_cell_env(ambient, 0)["PARALLEL"], "0")

    def test_an_explicit_upstream_default_drops_ambient_parallel(self):
        # None remains the programmatic spelling of tinygrad's own default. It must not inherit
        # a shell experiment under a result whose absent beam_parallel field says "upstream".
        ambient = {"PATH": "/usr/bin", "PARALLEL": "0"}

        self.assertNotIn("PARALLEL", orchestrate.beam_cell_env(ambient, None))

    @unittest.skipUnless(os.name == "posix", "the deferral is about POSIX signal delivery")
    def test_a_cancellation_inside_the_spawn_window_still_kills_the_cell(self):
        # Between `_execute_child` starting the cell and `Popen` returning it, no name refers to
        # the new process -- so an exception raised there unwinds past a cleanup with nothing to
        # clean, and the isolated runner is orphaned on the GPU with the sweep gone. The signal is
        # delivered here at exactly that point, by a Popen wrapper.
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_int = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGTERM, previous_term)
        self.addCleanup(signal.signal, signal.SIGINT, previous_int)
        orchestrate.install_termination_handler()
        cell = self.python("import time; time.sleep(300)")
        real_popen = cell_group.subprocess.Popen
        spawned = []

        def popen_then_cancel(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            # The child's pid is taken HERE rather than from a file it writes: at this point it
            # may not have run a line of its own yet, and a cancellation that works kills it
            # before it does.
            spawned.append(proc.pid)
            os.kill(os.getpid(), signal.SIGTERM)  # lands inside the window
            return proc

        started = time.monotonic()
        with unittest.mock.patch.object(
            cell_group.subprocess, "Popen", side_effect=popen_then_cancel
        ):
            with self.assertRaises(SystemExit):
                self.run_cell("cancelled mid-spawn", cell, timeout=60)
        elapsed = time.monotonic() - started

        self.assertEqual(len(spawned), 1)
        self.assertTrue(self.wait_gone(spawned[0]), "the cell outlived the cancellation")
        # And it was killed on the way out of the spawn window, not left running until the cell's
        # own cap noticed: a held signal is delivered at the first point where raising is safe.
        self.assertLess(elapsed, orchestrate.CELL_KILL_GRACE_S + 5)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_finished_cell_that_left_workers_behind_has_them_collected(self):
        # `communicate` returned because the LEADER exited and the pipe closed, which says nothing
        # about a worker that redirected its own output -- and that worker still holds the GPU, so
        # every later cell of the sweep would be measured against it (gh-ocannl-760 review).
        pidfile = self.dir / "leftover.pid"
        cell = self.python(
            "import subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "sys.exit(1)\n",
            pidfile,
        )

        result, note, _ = self.run_cell("leaky cell", cell, timeout=60)

        self.assertIsNone(result)
        self.assertIn("exit 1", note)
        self.assertIn("process group behind", note)
        leftover = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(leftover), f"pid {leftover} outlived its cell")

    def test_a_survivor_makes_a_successful_cell_a_failure(self):
        # The cell ran, printed a result line and exited zero -- and something it spawned outlived
        # SIGKILL and still holds the device. That makes this row's own timing suspect and every
        # later row of the run too, so recording it as a success (with a warning on a console
        # nobody keeps) publishes exactly what the failure section exists to stop. The survivor is
        # stood in for: nothing outlives SIGKILL but a process stuck in the kernel.
        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            result, note, _ = self.run_cell("successful with a survivor", cell, timeout=60)

        self.assertIsNone(result)
        self.assertIn("SURVIVED SIGKILL", note)
        self.assertIn("every later cell", note)

    @unittest.skipUnless(os.name == "posix", "the deferral is about POSIX signal delivery")
    def test_a_cancellation_during_the_kill_does_not_abandon_the_group(self):
        # A signal arriving while `kill_cell_group` is in its grace loop raises out of the `except
        # TimeoutExpired` clause that is doing the killing -- and a sibling `except BaseException`
        # does not catch what another `except` clause raises, so the escalation stops halfway and
        # the group outlives the sweep (gh-ocannl-760 review). The cell here holds the grace open
        # by ignoring SIGTERM, and the sweep is cancelled while it does.
        pidfile = self.dir / "mid_kill.pid"
        termfile = self.dir / "mid_kill.termed"
        # The cell RECORDS the SIGTERM instead of `SIG_IGN`-ing it -- it still does not die of one,
        # which is all the test needs of it, and the marker is the moment the grace loop opened.
        # Sleeping a fixed 4 s to land inside a 10 s window was the old way to hit that window,
        # and it cost the suite the wait on every run; waiting for the marker hits it exactly, so
        # the grace can be shortened (below) without the cancellation racing the SIGKILL.
        driver = self.python(
            "import sys, orchestrate\n"
            "orchestrate.install_termination_handler()\n"
            # The grace this cancellation lands in has to be long enough for the parent to see the
            # marker and signal, not long enough to be realistic: what is under test is that the
            # escalation FINISHES across a signal, and the deferral that makes it do so is the
            # same at any grace length.
            "orchestrate.CELL_KILL_GRACE_S = 2.0\n"
            "cell = [sys.executable, '-c',\n"
            "  'import os, signal, sys, time\\n"
            "signal.signal(signal.SIGTERM, lambda *_: open(sys.argv[2], \\'w\\').close())\\n"
            "open(sys.argv[1], \\'w\\').write(str(os.getpid()))\\n"
            "sys.stdout.flush()\\n"
            "time.sleep(300)\\n', sys.argv[1], sys.argv[2]]\n"
            "orchestrate.run_cell('cell cancelled mid-kill', cell, timeout=2)\n",
            pidfile,
            termfile,
        )
        sweep = subprocess.Popen(
            driver,
            cwd=str(Path(orchestrate.__file__).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(lambda: sweep.poll() is None and sweep.kill())
        end = time.monotonic() + 30
        while time.monotonic() < end and not pidfile.exists():
            time.sleep(0.05)
        self.assertTrue(pidfile.exists(), "the cell never started")
        stubborn = int(pidfile.read_text())
        # Through the cap (2 s) and into the grace, where the kill is waiting on a cell that will
        # not answer SIGTERM: the marker is written by the cell's handler, from inside that wait.
        end = time.monotonic() + 30
        while time.monotonic() < end and not termfile.exists():
            time.sleep(0.02)
        self.assertTrue(termfile.exists(), "the kill never reached its grace loop")
        self.assertTrue(self.alive(stubborn), "the cell was gone before the kill was interrupted")

        sweep.terminate()
        sweep.wait(timeout=40)

        self.assertTrue(
            self.wait_gone(stubborn, deadline_s=30),
            f"pid {stubborn} outlived a sweep cancelled mid-kill",
        )

    def test_a_search_that_failed_partway_names_its_cache_too(self):
        # A search that exits nonzero partway leaves the same partial cache a killed one does --
        # some arms committed, the rest never run -- and the next attempt over it reports a
        # provenance nobody wrote. So the cell is asked on this path too, with killed=False: the
        # risk is the same, what may be DONE about it is not (the cache is shared with every later
        # cell of the sweep, and the failure may have been a pre-search one).
        seen = []

        result, note, _ = self.run_cell(
            "failed search",
            self.python("import sys; sys.exit(3)"),
            timeout=60,
            on_incomplete=lambda killed: seen.append(killed) or "CACHE AT RISK: partial search",
        )

        self.assertIsNone(result)
        self.assertEqual(seen, [False])
        self.assertIn("exit 3", note)
        self.assertIn("CACHE AT RISK", note)

    def test_the_failed_path_leaves_the_shared_cache_alone(self):
        # ... and "asked" must not mean "moved": a cell that exited on its own may never have
        # searched at all, while the cache is shared with every later cell of the sweep.
        db = self.dir / "cache.db"
        db.write_bytes(b"the sweep's warm kernels")

        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)}, killed=False)

        self.assertTrue(db.exists())
        self.assertEqual(db.read_bytes(), b"the sweep's warm kernels")
        self.assertIn("CACHE AT RISK", note)
        self.assertIn("exited rather than being killed", note)

    def test_the_report_names_the_pool_a_beam_row_searched_with(self):
        # A beam row's compile cost IS a search cost, and the pool changes it threefold -- so
        # `beam` alone gives the default, `PARALLEL=0` and an explicit N one identity in the table
        # people read numbers out of. The default keeps the bare name (every report so far
        # recorded it that way, and rows that are the same should read the same).
        out = Path(tempfile.mkdtemp())
        rows = [
            cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1]),
            cell("tinygrad", "HIP", "beam", [2.3, 2.2, 2.1]),
            cell("tinygrad", "HIP", "beam", [2.3, 2.2, 2.1]),
        ]
        rows[1]["beam_parallel"] = 0
        rows[2]["beam_parallel"] = None
        orchestrate.parity_check(rows)

        with contextlib.redirect_stdout(io.StringIO()):
            orchestrate.report(rows, out)

        text = (out / "report.md").read_text()
        self.assertIn("| beam P=0 |", text)
        self.assertIn("| beam |", text)

    def test_a_cell_whose_leftovers_were_killed_reports_a_killed_search(self):
        # The leftover sweep interrupted a member -- possibly mid-write -- so the cache is in the
        # state the kill path quarantines, whatever the leader's own exit said.
        seen = []
        pidfile = self.dir / "leftover_killed.pid"
        cell = self.python(
            "import subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "sys.exit(1)\n",
            pidfile,
        )

        _, note, _ = self.run_cell(
            "failed leaving workers",
            cell,
            timeout=60,
            on_incomplete=lambda killed: seen.append(killed) or "cache handled",
        )

        self.assertEqual(seen, [True], "a cell whose members we killed reported killed=False")
        self.assertIn("process group behind", note)
        self.assertTrue(self.wait_gone(int(pidfile.read_text())))

    def test_a_survivor_failure_still_handles_the_cache(self):
        # The survivor branch returns early, and part of the search was forcibly interrupted --
        # which is what `stuck` means -- so the cache is in the state the cap's path quarantines.
        # Returning without asking would leave it for the next beam cell to read.
        seen = []
        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            _, note, _ = self.run_cell(
                "successful over a survivor",
                cell,
                timeout=60,
                on_incomplete=lambda killed: seen.append(killed) or "cache quarantined",
            )

        self.assertEqual(seen, [True])
        self.assertIn("SURVIVED SIGKILL", note)
        self.assertIn("cache quarantined", note)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_supporting_build_is_killed_with_its_own_children(self):
        # The sweep's own subprocesses -- the build, the device probes -- get the same discipline
        # as a cell: `subprocess.run` kills its direct child on an exception but knows nothing of
        # that child's children, and `dune build` forks compilers (gh-ocannl-760 review).
        pidfile = self.dir / "build_worker.pid"
        build = self.python(
            "import subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "time.sleep(300)\n",
            pidfile,
        )

        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 1.5)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)

        with self.assertRaises(KeyboardInterrupt):
            orchestrate.run_supporting(build, capture_output=True)

        worker = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(worker), f"pid {worker} outlived the cancelled build")

    def test_a_cap_that_cannot_mean_anything_is_refused_before_the_first_cell(self):
        # A negative cap expires every `communicate` at once -- killing every cell of the sweep
        # and quarantining their caches on the way -- and nan/inf raise from inside it. Both are
        # worth refusing at the argument rather than cell by cell.
        for bad in ("-1", "nan", "inf", "-0.5"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                orchestrate.cell_timeout_arg(bad)
        self.assertEqual(orchestrate.cell_timeout_arg("0"), 0.0)  # 0 is "no cap", not "cap of 0"
        self.assertEqual(orchestrate.cell_timeout_arg("1800"), 1800.0)

    def test_a_negative_pool_size_is_refused_too(self):
        # `-1` is not "unset" and not "default": tinygrad builds a pool for any truthy PARALLEL
        # and hands the value to `multiprocessing.Pool`, which refuses a negative count -- so
        # every beam cell of the sweep would die inside the runner.
        for bad in ("-1", "-8"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                orchestrate.beam_parallel_arg(bad)
        self.assertEqual(orchestrate.beam_parallel_arg("0"), 0)  # 0 is no pool at all
        self.assertEqual(orchestrate.beam_parallel_arg("4"), 4)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_survivor_holding_the_pipe_does_not_swallow_the_cells_output(self):
        # A member that outlives SIGKILL still owns the captured stdout, so every `communicate`
        # in the kill loop times out and the output would be lost -- exactly where the partial log
        # is the only evidence about a cell nobody will run again. It lives on the exception.
        # Nothing outlives SIGKILL on demand, so the survivor is stood in for by a grandchild
        # that ESCAPES the group (its own session) while still holding the stdout it inherited --
        # which reproduces the property that matters here: every `communicate` in the kill loop
        # times out, so the output can only come off the exception.
        pidfile = self.dir / "pipe_holder.pid"
        cell = self.python(
            "import subprocess, sys, time\n"
            "print('the evidence', flush=True)\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  start_new_session=True)\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "time.sleep(300)\n",
            pidfile,
        )

        def clear_the_holder():
            if pidfile.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)

        self.addCleanup(clear_the_holder)
        with unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 1.0):
            _, note, log = self.run_cell("wedged behind a held pipe", cell, timeout=1.5)

        self.assertIn("TIMED OUT", note)
        self.assertIn("the evidence", log)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_kill_that_could_not_reap_still_lets_go_of_the_pipes(self):
        # The one way out of the kill loop without reaping is a member that outlived SIGKILL
        # holding the write end, so every `communicate` timed out and `Popen` was left owning an
        # open read end for the rest of the sweep -- a descriptor per unreapable cell, leaked on
        # the path least able to spare them (it announced itself as a ResourceWarning under
        # `dune build @benchmarks/runtest`). The survivor is stood in for by a grandchild that
        # ESCAPES the group into its own session while still holding the stdout it inherited,
        # which is what makes every reap in the loop time out.
        pidfile = self.dir / "unreapable.pid"
        holder = cell_group.spawn(
            self.python(
                "import subprocess, sys, time\n"
                "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
                "  start_new_session=True)\n"
                "open(sys.argv[1], 'w').write(str(kid.pid))\n"
                "time.sleep(300)\n",
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # The kill loop cannot reap this leader (that is the point), so the test reaps it.
        self.addCleanup(holder.wait)

        def clear_the_holder():
            if pidfile.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)

        self.addCleanup(clear_the_holder)
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        with unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 1.0):
            orchestrate.kill_cell_group(holder)

        self.assertTrue(holder.stdout.closed, "the kill left the cell's read end open")
        # And the leader is reaped: what the loop could not finish is the READ, blocked on a pipe
        # the survivor holds, not the wait -- the leader took the SIGKILL.
        self.assertIsNotNone(holder.returncode, "the kill left the cell's leader unreaped")

    @unittest.skipUnless(os.name == "posix", "the cancellation is delivered by a signal here")
    def test_a_cancelled_supporting_command_preserves_its_cleanup_failure(self):
        # The other exit from `_run_supporting`, and the same survivor its ordinary path stops
        # the sweep over. Here the sweep is already leaving, so there is nothing to stop -- but a
        # cancellation that reports a clean exit over a process still holding the device is how
        # the NEXT run gets measured against it (gh-ocannl-760 review). The survivor is stood in
        # for as elsewhere: nothing outlives SIGKILL but a process stuck in the kernel.
        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            signal.setitimer(signal.ITIMER_REAL, 1.0)
            with self.assertRaises(cell_group.CleanupFailed) as raised:
                orchestrate.run_supporting(self.python("import time; time.sleep(300)"))

        self.assertIn("SURVIVED SIGKILL", str(raised.exception))

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_successful_cell_whose_leftovers_were_killed_still_handles_the_cache(self):
        # The row stands -- the cell ran and printed its result -- but a member of its group was
        # killed on the way out, possibly mid-write, and the cache it shares with every later beam
        # cell is in the state a kill leaves. It is the cache that is at issue, not the row.
        seen = []
        pidfile = self.dir / "successful_leftover.pid"
        cell = self.python(
            "import json, subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "print(json.dumps({'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))\n",
            pidfile,
        )

        result, note, _ = self.run_cell(
            "successful with leftovers",
            cell,
            timeout=60,
            on_incomplete=lambda killed: seen.append(killed) or "cache quarantined",
        )

        self.assertIsNotNone(result)  # the row is good; the cache is what needed handling
        self.assertIsNone(note)
        self.assertEqual(seen, [True])
        self.assertTrue(self.wait_gone(int(pidfile.read_text())))

    def test_a_quarantine_that_cannot_finish_puts_back_what_it_moved(self):
        # A half-moved family is worse than an unmoved one: a database separated from its WAL
        # opens without the writes it holds, and the cache left behind gains a stale sidecar the
        # next tinygrad process reads against a database it never belonged to.
        db = self.dir / "cache.db"
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
            path.write_bytes(b"live cache " + path.name.encode())
        real_replace = orchestrate.os.replace
        calls = []

        def replace_then_fail(src, dst):
            calls.append(src)
            if len(calls) == 2:  # the -wal move, after the database has already moved
                raise OSError(13, "Permission denied")
            return real_replace(src, dst)

        with unittest.mock.patch.object(orchestrate.os, "replace", replace_then_fail):
            note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})

        self.assertIn("CACHE AT RISK", note)
        self.assertIn("could not quarantine", note)
        # Everything is back where the next cell expects it, and nothing is left half-quarantined.
        self.assertEqual(db.read_bytes(), b"live cache cache.db")
        self.assertEqual(Path(f"{db}-wal").read_bytes(), b"live cache cache.db-wal")
        self.assertEqual(sorted(p.name for p in self.dir.glob("*.wedged-*")), [])

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_supporting_command_leaves_no_descendants_behind(self):
        # `communicate` returned because the LEADER exited, which says nothing about a build's
        # compiler worker or a probe's framework helper -- and those hold the GPU while the sweep
        # measures against them (gh-ocannl-760 review).
        pidfile = self.dir / "supporting_leftover.pid"
        build = self.python(
            "import subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "open(sys.argv[1], 'w').write(str(kid.pid))\n"
            "sys.exit(0)\n",
            pidfile,
        )

        done = orchestrate.run_supporting(build, capture_output=True)

        self.assertEqual(done.returncode, 0)
        worker = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(worker), f"pid {worker} outlived its supporting command")

    def test_a_supporting_survivor_stops_the_sweep_before_it_dispatches(self):
        # Where a cell's survivor fails that cell, a survivor from the BUILD or a device probe is
        # shared by every row the sweep is about to produce -- there is no subset of the results
        # to disbelieve, and a cap cannot bound damage that is already in all of them. So this one
        # stops the sweep. (The survivor is stood in for, as elsewhere: nothing outlives SIGKILL
        # except a process stuck in the kernel.)
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            with self.assertRaises(SystemExit) as raised:
                orchestrate.run_supporting(self.python("pass"), capture_output=True)

        self.assertIn("SURVIVED SIGKILL", str(raised.exception))
        self.assertIn("nothing is dispatched", str(raised.exception))

    def test_a_fractional_cap_is_reported_as_the_cap_it_was(self):
        # `{:.0f}` rounded a sub-second cap to `TIMED OUT after 0s`, one clause before the text
        # saying that zero disables the cap.
        _, note, _ = self.run_cell(
            "briefly capped", self.python("import time; time.sleep(300)"), timeout=0.25
        )

        self.assertIn("TIMED OUT after 0.25s", note)

    def test_the_leftover_probe_itself_runs_deferred(self):
        # A cancellation landing on the probe -- or between it reading the group as alive and the
        # kill starting -- would leave exactly the worker the probe just found, in a session
        # nothing else will reach (gh-ocannl-760 review). So the property is that the probe and
        # its cleanup are inside ONE deferral window, which is what this reads: the depth seen by
        # the probe itself.
        depth_at_probe = []

        def probe(_proc):
            depth_at_probe.append(orchestrate._cancellation.depth)
            return cell_group.GONE

        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )
        with unittest.mock.patch.object(orchestrate, "_group_observation", probe):
            result, note, _ = self.run_cell("probed", cell, timeout=60)

        self.assertIsNone(note)
        self.assertIsNotNone(result)
        self.assertTrue(depth_at_probe, "the ordinary path never probed for leftovers")
        self.assertTrue(
            all(depth > 0 for depth in depth_at_probe),
            f"the leftover probe ran with cancellations undeferred: {depth_at_probe}",
        )

    @unittest.skipUnless(os.name == "posix", "the interrupt path needs POSIX signals")
    def test_an_interrupted_cell_preserves_its_cleanup_failure(self):
        # The interrupt branch exits rather than records, so its print is the operator's only
        # chance to hear that something still holds the device -- while the cancellation's own
        # message says the cell was killed, and the retry they are about to start would be
        # measured against the survivor. The survivor is stood in for, as elsewhere.
        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)

        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            with self.assertRaises(cell_group.CleanupFailed) as raised:
                orchestrate.run_cell(
                    "interrupted over a survivor", self.python("import time; time.sleep(300)")
                )

        self.assertIn("SURVIVED SIGKILL", str(raised.exception))

    def test_the_cache_is_quarantined_even_if_the_cell_log_cannot_be_written(self):
        # The kill is what tore the cache, so undoing it must not sit behind fallible code: the
        # optional cell log writes to an operator-supplied directory, which can be unwritable or
        # full, and losing the sweep to that would leave the partial cache.db in place AND lose
        # the failure record.
        called = []
        with unittest.mock.patch.object(
            orchestrate, "CELL_LOG_DIR", Path("/dev/null/not-a-directory")
        ):
            result, note, _ = self.run_cell(
                "wedged with a broken log dir",
                self.python("import time; time.sleep(300)"),
                timeout=1.0,
                on_incomplete=lambda killed: called.append(("killed", killed))
                or "quarantined the cache",
            )

        self.assertIsNone(result)
        # ... and told that this cell was KILLED, which is what decides whether the cache may
        # be moved aside at all.
        self.assertEqual(called, [("killed", True)])
        self.assertIn("quarantined the cache", note)

    def test_a_killed_beam_cell_quarantines_the_cache_it_was_writing(self):
        # The contaminated-cache consequence recorded on the issue's HIP leg: the search writes
        # its winners into one sqlite file as it goes, so a kill leaves a partial cache that the
        # next run neither replays nor searches past -- while `searched` reports one of the two.
        db = self.dir / "cache.db"
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
            path.write_bytes(b"half a beam search")

        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})

        self.assertFalse(db.exists())
        moved = sorted(p.name for p in self.dir.glob("cache.db.wedged-*"))
        self.assertEqual(len(moved), 3, moved)
        # The sidecars move UNDER the quarantined database's name, because that is the only place
        # sqlite looks for them: `cache.db-wal.wedged-<stamp>` beside `cache.db.wedged-<stamp>` is
        # a database that opens without the writes the killed search never checkpointed -- the
        # evidence the move exists to keep, dropped silently (gh-ocannl-760 review).
        base = next(name for name in moved if not name.endswith(("-wal", "-shm")))
        self.assertEqual(moved, sorted([base, f"{base}-wal", f"{base}-shm"]))
        self.assertIn("quarantined", note)
        self.assertIn("searched", note)

    def test_declining_the_quarantine_still_names_the_risk(self):
        db = self.dir / "cache.db"
        db.write_bytes(b"half a beam search")

        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)}, enabled=False)

        self.assertTrue(db.exists())
        self.assertIn("CACHE AT RISK", note)
        self.assertIn("--no-cache-quarantine", note)

    def test_a_cache_that_was_never_written_is_not_invented(self):
        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(self.dir / "absent.db")})

        self.assertIn("nothing to quarantine", note)

    def test_the_ocannl_note_describes_that_cache_rather_than_tinygrads(self):
        # The two caches differ where it matters: OCANNL commits entries atomically, so the
        # honest statement is about the retry's provenance, not about a torn file.
        note = orchestrate.ocannl_cache_note()

        self.assertIn("autotune_cache", note)
        # And what it must NOT say is that the retry replays: a kill lands mid-search, so the
        # retry replays the finished arms and searches the rest, and `search_provenance` reads
        # `searched` -- true whenever any arm searched. The mixed pass reports SEARCHED, and a
        # note promising REPLAY would misdescribe exactly the case it exists for.
        self.assertIn("SEARCHED", note)
        self.assertEqual(orchestrate.search_provenance({"searched": True}), "SEARCHED")

    def test_a_failed_cell_is_named_in_the_report(self):
        # A failure is invisible in every table above it -- an unrun cell and a wedged one read
        # identically once the report outlives the run log.
        out = Path(tempfile.mkdtemp())
        failures = [("mlp_small tinygrad/CUDA/beam", "TIMED OUT after 1800s; quarantined ...")]
        cells = [cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])]
        orchestrate.parity_check(cells)

        orchestrate.report(cells, out, (), failures)

        text = (out / "report.md").read_text()
        self.assertIn("Runner failures", text)
        self.assertIn("mlp_small tinygrad/CUDA/beam", text)
        self.assertIn("TIMED OUT", text)

    def test_a_run_with_no_failures_says_nothing_about_them(self):
        out = Path(tempfile.mkdtemp())
        cells = [cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])]
        orchestrate.parity_check(cells)

        orchestrate.report(cells, out)

        self.assertNotIn("Runner failures", (out / "report.md").read_text())


class RegimeTest(unittest.TestCase):
    """gh-ocannl-719: the numerics regime is a third axis over the cells, with its own envelope."""

    def test_the_approximate_envelope_is_looser_than_every_exact_one(self):
        # The literal is the pin. Looser than PARITY_TOL and every precision envelope, so an
        # approximate cell of any precision is gated at it; tighter than the loss-movement floor.
        self.assertEqual(orchestrate.PARITY_TOL_APPROX, 1e-2)
        for precision in ("f32", "bf16", "f16", "f16-gated8"):
            self.assertEqual(orchestrate.parity_tol(precision, "approximate"), 1e-2)
            self.assertLess(orchestrate.parity_tol(precision), 1e-2)
        self.assertGreater(orchestrate.PARITY_TOL_APPROX, orchestrate.LOSS_MOVE_MIN_REL)
        # The exact envelopes are untouched by the regime's existence.
        self.assertEqual(orchestrate.parity_tol("f32"), orchestrate.PARITY_TOL)
        self.assertEqual(orchestrate.parity_tol("f32", "exact"), orchestrate.PARITY_TOL)

    def test_the_reference_is_the_exact_torch_cell_even_when_an_approximate_one_ran(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        counterpart = result("pytorch", "cpu", "eager", [2.3126, 2.3110, 2.3100])
        counterpart["regime"] = "approximate"
        candidate = result("ocannl", "cuda", "tuned", [2.3030, 2.3014, 2.3004])
        candidate["regime"] = "approximate"

        orchestrate.parity_check([counterpart, ref, candidate])

        self.assertEqual(ref["parity"], "REF")
        # The approximate torch cell is a counterpart gated like any other row, not a reference:
        # its 4e-3 drift passes the approximate envelope and not the exact one -- and that second
        # verdict is reported rather than assumed.
        self.assertEqual(counterpart["parity"], "PASS")
        self.assertFalse(counterpart["parity_exact_envelope"])
        self.assertEqual(candidate["parity"], "PASS")
        self.assertTrue(candidate["parity_exact_envelope"])

    def test_an_approximate_row_beyond_its_envelope_fails(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        drifted = result("ocannl", "cuda", "tuned", [2.3300, 2.3280, 2.3260])
        drifted["regime"] = "approximate"

        orchestrate.parity_check([ref, drifted])

        self.assertEqual(drifted["parity"], "FAIL")
        self.assertFalse(drifted["parity_exact_envelope"])

    def test_exact_rows_carry_no_exact_envelope_verdict(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        exact = result("ocannl", "cc", "default", [2.3026, 2.3010, 2.3001])

        orchestrate.parity_check([ref, exact])

        self.assertEqual(exact["parity"], "PASS")
        self.assertNotIn("parity_exact_envelope", exact)

    def test_a_sweep_with_no_arguments_runs_the_exact_regime_only(self):
        args = orchestrate.build_arg_parser().parse_args([])

        self.assertEqual(args.profile, ["exact"])

    def test_a_bare_profile_flag_is_refused_rather_than_an_empty_matrix(self):
        # `--profile` with no value would otherwise parse to [] and skip every OCANNL cell,
        # publishing an empty report with a clean exit (Codex P2 on PR #661).
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                orchestrate.build_arg_parser().parse_args(["--profile"])

    def test_the_exact_envelope_verdict_is_about_drift_alone(self):
        # A stationary approximate row within the exact envelope's drift: FAIL for not moving,
        # and "within exact envelope" -- the two facts are reported separately, so the report
        # never attributes a stationary loss to excessive drift (Codex P2 on PR #661).
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        flat = result("ocannl", "cc", "default", [2.3026, 2.3026, 2.3026])
        flat["regime"] = "approximate"

        orchestrate.parity_check([ref, flat])

        self.assertEqual(flat["parity"], "FAIL")
        self.assertFalse(flat["parity_loss_moved"])
        self.assertTrue(flat["parity_exact_envelope"])

    def test_a_mismatched_row_is_shouted_in_the_report_not_published_under_its_label(self):
        # The exact cell inherited OCANNL_PROFILE=approximate: the sweep fails on it, but
        # report.md is written before the failure exits, so the row must say so where its
        # number is read (Codex P1 on PR #661).
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        leaked = cell("ocannl", "cc", "default", [2.3026, 2.3011, 2.3001])
        leaked["profile"] = "approximate"
        orchestrate.parity_check([ref, leaked])
        self.assertEqual(orchestrate.regime_check([ref, leaked]), [leaked])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, leaked], out)
            text = (out / "report.md").read_text()
            rows = [
                strict_loads(line) for line in (out / "results.jsonl").read_text().splitlines()
            ]

        self.assertIn("| regime |", text)
        row = [line for line in text.splitlines() if line.startswith("| ocannl")][0]
        self.assertIn("**REGIME MISMATCH** (dispatched exact; ran approximate)", row)
        self.assertNotIn("| exact |", row)
        self.assertEqual(rows[1]["regime_mismatch"], "approximate")

    def test_both_regimes_can_share_one_sweep_and_an_unknown_one_is_refused(self):
        args = orchestrate.build_arg_parser().parse_args(["--profile", "exact", "approximate"])

        self.assertEqual(args.profile, ["exact", "approximate"])
        # `reproducible` is an OCANNL profile but not a benchmark regime: it has no envelope.
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                orchestrate.build_arg_parser().parse_args(["--profile", "reproducible"])

    def test_the_regime_reaches_each_runner_as_its_own_flag(self):
        # OCANNL: the profile IS the regime, and the exact regime passes no profile at all.
        self.assertEqual(orchestrate.ocannl_regime_args("exact"), [])
        self.assertEqual(
            orchestrate.ocannl_regime_args("approximate"), ["--ocannl_profile=approximate"]
        )
        self.assertEqual(orchestrate.torch_regime_args("exact"), [])
        self.assertEqual(orchestrate.torch_regime_args("approximate"), ["--regime", "approximate"])
        # Exact labels are unchanged; a non-exact cell says so in its label.
        self.assertEqual(orchestrate.regime_label("exact"), "")
        self.assertEqual(orchestrate.regime_label("approximate"), " [approximate]")
        # tinygrad's one cell stands in the approximate regime once the sweep has one.
        self.assertEqual(orchestrate.tinygrad_regime(["exact"]), "exact")
        self.assertEqual(orchestrate.tinygrad_regime(["exact", "approximate"]), "approximate")

    def test_a_row_whose_runner_ran_the_other_regime_is_a_gate_failure(self):
        # An ambient OCANNL_PROFILE=approximate reaching the exact cells, or the reverse: the
        # parity gate cannot see it (an exact trajectory passes the approximate envelope too).
        mislabelled = result("ocannl", "cc", "default", [1.0])
        mislabelled["profile"] = "approximate"
        honest = result("ocannl", "cc", "default", [1.0])
        honest.update(regime="approximate", profile="approximate")
        # A reproducible or performance profile is still the exact regime.
        reproducible = result("ocannl", "cc", "tuned", [1.0])
        reproducible["profile"] = "reproducible"
        torch_wrong = result("pytorch", "cuda", "eager", [1.0])
        torch_wrong.update(regime="approximate", runner_regime="exact")
        # A runner predating the field, and tinygrad (whose regime is the sweep's decision), are
        # not checked.
        older = result("ocannl", "cc", "default", [1.0])
        older["regime"] = "approximate"
        tiny = result("tinygrad", "CPU", "jit", [1.0])
        tiny.update(regime="approximate", regime_settings="tinygrad defaults")

        mismatched = orchestrate.regime_check(
            [mislabelled, honest, reproducible, torch_wrong, older, tiny]
        )

        self.assertEqual(mismatched, [mislabelled, torch_wrong])
        self.assertEqual(mislabelled["regime_mismatch"], "approximate")
        self.assertEqual(torch_wrong["regime_mismatch"], "exact")

    def test_the_regime_is_the_resolution_of_the_profiles_keys_not_its_name(self):
        # Codex P1 on PR #661: OCANNL_TF32_MATMULS=true in the sweep's environment reaches an
        # exact cell with no profile picked, so the profile name says `exact` while the
        # arithmetic is approximate. The runner reports where each approximate-payload key
        # resolved from; an exact row owns only defaults, an approximate row only the profile.
        def knobs(**sources):
            return {
                key: ({"source": "default"} if src is None else {"value": val, "source": src})
                for key, (val, src) in sources.items()
            }

        clean = result("ocannl", "cc", "default", [1.0])
        clean.update(profile=None, regime_knobs=knobs(tf32_matmuls=(None, None)))
        leaked = result("ocannl", "cuda", "default", [1.0])
        leaked.update(
            profile=None,
            regime_knobs=knobs(
                tf32_matmuls=("true", "environment"), cc_backend_fast_math=(None, None)
            ),
        )
        honest = result("ocannl", "cuda", "tuned", [1.0])
        honest.update(
            regime="approximate",
            profile="approximate",
            regime_knobs=knobs(tf32_matmuls=("true", "profile 'approximate' via the commandline")),
        )
        overridden = result("ocannl", "cuda", "tuned", [1.0])
        overridden.update(
            regime="approximate",
            profile="approximate",
            regime_knobs=knobs(
                tf32_matmuls=("false", "commandline"),
                cc_backend_fast_math=("true", "profile 'approximate' via the commandline"),
            ),
        )
        # Another profile pins these keys to exact VALUES, but from the profile: not an
        # exact-regime sweep, and the row says which key came from where.
        reproducible = result("ocannl", "cc", "tuned", [1.0])
        reproducible.update(
            profile="reproducible",
            regime_knobs=knobs(tf32_matmuls=("false", "profile 'reproducible' via the environment")),
        )
        older = result("ocannl", "cc", "default", [1.0])
        older["profile"] = None

        mismatched = orchestrate.regime_check(
            [clean, leaked, honest, overridden, reproducible, older]
        )

        self.assertEqual(mismatched, [leaked, overridden, reproducible])
        self.assertEqual(leaked["regime_mismatch"], "exact but tf32_matmuls=true (environment)")
        self.assertEqual(
            overridden["regime_mismatch"], "approximate but tf32_matmuls=false (commandline)"
        )
        self.assertIn("profile 'reproducible'", reproducible["regime_mismatch"])
        self.assertIsNone(orchestrate.regime_knob_leaks(older))

    def test_the_report_names_the_regime_and_the_exact_envelope_verdict(self):
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        exact = cell("ocannl", "cuda", "tuned", [2.3026, 2.3011, 2.3001], p50=2.0)
        approx = cell("ocannl", "cuda", "tuned", [2.3126, 2.3110, 2.3100], p50=1.0)
        approx["regime"] = "approximate"
        orchestrate.parity_check([ref, exact, approx])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, exact, approx], out)
            text = (out / "report.md").read_text()

        self.assertIn("| regime |", text)
        self.assertIn("the approximate regime 0.01", text)
        table = [line for line in text.splitlines() if line.startswith("| ocannl")]
        # The exact row comes first within the precision though the approximate one is faster:
        # an approximate number is read against the exact one it relaxes.
        self.assertIn("| exact |", table[0])
        self.assertIn("| approximate |", table[1])
        self.assertIn("PASS (4.3e-03, beyond exact envelope)", table[1])
        self.assertNotIn("exact envelope", table[0])
        # The exact rows name their regime too once the column exists.
        self.assertIn("| exact |", [line for line in text.splitlines() if line.startswith("| pytorch")][0])

    def test_an_exact_only_report_has_no_regime_column(self):
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        exact = cell("ocannl", "cc", "default", [2.3026, 2.3011, 2.3001])
        orchestrate.parity_check([ref, exact])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, exact], out)
            text = (out / "report.md").read_text()

        self.assertNotIn("| regime |", text)
        self.assertNotIn("exact envelope", text)
        self.assertIn("| PASS (", text)


class AmbientEnvTest(unittest.TestCase):
    """gh-ocannl-720: what the shell was already carrying is recorded, not guessed at.

    The regime gate (gh-ocannl-719) cross-checks the keys the `approximate` payload owns. Every
    other configuration key reaches the OCANNL cells through `cell_env`'s pass-through of
    `os.environ`, so the artifact has to say what that environment was -- for the keys nobody has
    enumerated as much as for the ones anybody has.
    """

    def test_the_sweep_records_every_ocannl_variable_it_inherited(self):
        # Values as well as names: `OCANNL_CC_VECTOR_BYTES=16` and `=64` are different
        # measurements, and a bare list of names cannot tell the two apart.
        ambient = orchestrate.ambient_ocannl_env(
            {
                "OCANNL_NARROW_COMPUTE_F32": "true",
                "PATH": "/usr/bin",
                "OCANNL_CC_VECTOR_BYTES": "64",
                "HOME": "/home/nobody",
            }
        )

        self.assertEqual(
            ambient,
            {"OCANNL_CC_VECTOR_BYTES": "64", "OCANNL_NARROW_COMPUTE_F32": "true"},
        )
        # Sorted, so two sweeps that inherited the same environment stamp the same mapping
        # whatever order the shell happened to export it in.
        self.assertEqual(
            list(ambient), ["OCANNL_CC_VECTOR_BYTES", "OCANNL_NARROW_COMPUTE_F32"]
        )

    def test_a_key_the_regime_gate_does_not_own_is_recorded_like_any_other(self):
        # The point of recording over enumerating: the rule is the `OCANNL_` prefix, not a list.
        # `regime_check` cross-checks the approximate payload's keys, and a key outside it --
        # `cc_backend_arch_flags` changes what the C backend compiles to, and the key added after
        # this test was written is by definition on nobody's list -- lands in the row all the same.
        self.assertEqual(
            orchestrate.ambient_ocannl_env({"OCANNL_CC_BACKEND_ARCH_FLAGS": "-mcpu=native"}),
            {"OCANNL_CC_BACKEND_ARCH_FLAGS": "-mcpu=native"},
        )

    def test_the_legitimate_variables_are_recorded_rather_than_stripped(self):
        # The sweep dispatches OCANNL_BACKEND itself and benchmarks/README.md documents running
        # under OCANNL_AUTOTUNE_LOG / OCANNL_AUTOTUNE_CACHE_DIR. None of them is refused or
        # filtered out -- a record that quietly omits what it considers normal is not a record.
        env = {
            "OCANNL_BACKEND": "metal",
            "OCANNL_AUTOTUNE_LOG": "1",
            "OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune",
        }

        self.assertEqual(orchestrate.ambient_ocannl_env(env), env)

    def test_only_the_ocannl_rows_carry_the_environment(self):
        ambient = {"OCANNL_NARROW_COMPUTE_F32": "true"}
        ours = result("ocannl", "cc", "default", [2.3026, 2.3010])
        theirs = result("pytorch", "cpu", "eager", [2.3026, 2.3010])

        orchestrate.stamp_ambient_env(ours, ambient)
        orchestrate.stamp_ambient_env(theirs, ambient)

        self.assertEqual(ours["ambient_ocannl_env"], ambient)
        # The torch runner reads none of these; a stamp on its row would invite reading its
        # number as though they had reached it.
        self.assertNotIn("ambient_ocannl_env", theirs)

    def test_each_row_records_its_own_copy(self):
        # One mapping shared by every row of a sweep is a row that changes after it was written.
        ambient = {"OCANNL_BACKEND": "cc"}
        row = orchestrate.stamp_ambient_env(
            result("ocannl", "cc", "default", [2.3026, 2.3010]), ambient
        )

        ambient["OCANNL_BACKEND"] = "metal"

        self.assertEqual(row["ambient_ocannl_env"], {"OCANNL_BACKEND": "cc"})

    def test_the_report_names_the_environment_its_numbers_were_taken_in(self):
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]),
                {"OCANNL_CC_VECTOR_BYTES": "64", "OCANNL_NARROW_COMPUTE_F32": "true"},
            ),
            cell("pytorch", "cpu", "eager", [2.3026, 2.3010]),
        ]
        orchestrate.parity_check(rows)

        text = self.report(rows)

        lines = text.splitlines()
        [index] = [
            i for i, line in enumerate(lines) if line.startswith("ambient OCANNL_* environment:")
        ]
        self.assertEqual(
            lines[index],
            'ambient OCANNL_* environment: {"OCANNL_CC_VECTOR_BYTES": "64", '
            '"OCANNL_NARROW_COMPUTE_F32": "true"}',
        )
        # Beside the measurement boxes, in the header: the configuration a number was taken under
        # is read with the box it was taken on, not hunted for in results.jsonl.
        before = [line for line in lines[:index] if line.strip()]
        self.assertTrue(before[-1].startswith("measurement boxes declared by"))

    def test_a_clean_shell_is_recorded_as_such(self):
        rows = [
            orchestrate.stamp_ambient_env(cell("ocannl", "cc", "default", [2.3026, 2.3010]), {})
        ]
        orchestrate.parity_check(rows)

        self.assertIn("ambient OCANNL_* environment: none\n", self.report(rows))

    def test_a_value_no_text_artifact_could_hold_is_kept_faithful_and_escaped_to_render(self):
        # An environment value is bytes: on POSIX `os.environ` hands a non-UTF-8 byte over as a
        # lone surrogate. The row keeps it verbatim -- `json.dumps` spells it `\udcff` under
        # `ensure_ascii`, so results.jsonl carries it losslessly -- while report.md's UTF-8 write
        # cannot hold it, which is why the rendering escapes and the record does not.
        ambient = orchestrate.ambient_ocannl_env(
            {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"}
        )

        self.assertEqual(ambient, {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"})
        self.assertIn("/tmp/tune-\\udcff", orchestrate.ambient_env_line([], ambient))

    def test_both_artifacts_of_a_sweep_under_such_a_value_are_written(self):
        # End to end, because the failure this prevents lands at the END of a sweep, after every
        # cell has been paid for: report.md's UTF-8 write is what cannot hold a lone surrogate.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]),
                orchestrate.ambient_ocannl_env(
                    {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"}
                ),
            )
        ]
        orchestrate.parity_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            text = (out / "report.md").read_text()
            [row] = [
                strict_loads(line)
                for line in (out / "results.jsonl").read_text().splitlines()
                if line
            ]

        self.assertIn("/tmp/tune-\\udcff", text)
        # The row is the record and keeps the value verbatim; only the rendering escapes.
        self.assertEqual(
            row["ambient_ocannl_env"], {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"}
        )

    def test_a_byte_and_the_characters_that_spell_it_stay_distinguishable(self):
        # Any escaping that leaves ordinary values untouched maps the 0xff byte and the four
        # literal characters `\xff` onto the same text, so two different cache directories would
        # dedupe into one environment and the record would say something false about both.
        byte = orchestrate.ambient_env_line(
            [], orchestrate.ambient_ocannl_env({"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/\udcff"})
        )
        spelling = orchestrate.ambient_env_line(
            [], orchestrate.ambient_ocannl_env({"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/\\xff"})
        )

        self.assertNotEqual(byte, spelling)

    def test_a_value_cannot_forge_a_second_header_line(self):
        # Environment values may contain newlines, and an interpolated one would open what reads
        # as a second `ambient OCANNL_* environment:` line -- the header contradicting itself
        # about the configuration its numbers were taken under.
        line = orchestrate.ambient_env_line(
            [],
            orchestrate.ambient_ocannl_env(
                {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/a\nambient OCANNL_* environment: none"}
            ),
        )

        self.assertEqual(len(line.splitlines()), 1)
        self.assertIn("\\n", line)

    def test_the_report_says_the_environment_even_when_every_cell_failed(self):
        # The environment is a fact about the SWEEP, not about a row: an ambient setting the
        # cells cannot run under is exactly what makes every OCANNL cell fail, and a report that
        # then said `not recorded` would withhold the record at the moment it explains the run.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(
                    [],
                    out,
                    failures=[("mlp_wide ocannl/cc/default", "no result line")],
                    ambient={"OCANNL_NARROW_COMPUTE_F32": "true"},
                )
            text = (out / "report.md").read_text()

        self.assertIn(
            'ambient OCANNL_* environment: {"OCANNL_NARROW_COMPUTE_F32": "true"}\n', text
        )

    def test_rows_that_disagree_about_the_environment_are_refused(self):
        # Two header lines could only say that two environments exist somewhere in the tables,
        # never which row was measured under which -- swapping the two stamps would leave the
        # report byte-identical. Refused like a mixture of measurement-box declarations, with
        # rendering them separately as the remedy.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            orchestrate.stamp_ambient_env(
                cell("ocannl", "metal", "default", [2.3026, 2.3010]),
                {"OCANNL_BACKEND": "metal"},
            ),
        ]

        with self.assertRaises(ValueError) as refused:
            orchestrate.recorded_ambient_env(rows)

        self.assertIn("separate reports", str(refused.exception))

    def test_a_recorded_environment_beside_an_unrecorded_one_is_refused_too(self):
        # `not recorded` is a reading like any other: pairing it with a recorded environment
        # leaves the same ambiguity, and lets the recorded one stand at the head of measurements
        # whose own environment is unrecoverable.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            cell("ocannl", "metal", "default", [2.3026, 2.3010]),
        ]

        with self.assertRaises(ValueError) as refused:
            orchestrate.recorded_ambient_env(rows)

        self.assertIn("not recorded", str(refused.exception))

    def test_the_sweeps_own_value_does_not_excuse_the_rows_from_agreeing(self):
        # The refusal is about what the artifacts SAY, so it cannot be gated on how the header
        # got its value: a supplied environment printed over rows stamped with another would
        # misattribute the measurements results.jsonl records correctly.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            orchestrate.stamp_ambient_env(
                cell("ocannl", "metal", "default", [2.3026, 2.3010]),
                {"OCANNL_BACKEND": "metal"},
            ),
        ]

        with self.assertRaises(ValueError):
            orchestrate.ambient_env_line(rows, {"OCANNL_BACKEND": "cc"})

    def test_a_supplied_environment_its_rows_contradict_is_refused(self):
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            )
        ]

        with self.assertRaises(ValueError) as refused:
            orchestrate.ambient_env_line(rows, {"OCANNL_BACKEND": "metal"})

        self.assertIn("disagrees with what its rows recorded", str(refused.exception))
        # The agreeing case is the sweep's own, and is not disturbed by the check.
        self.assertIn(
            '{"OCANNL_BACKEND": "cc"}',
            orchestrate.ambient_env_line(rows, {"OCANNL_BACKEND": "cc"}),
        )

    def test_the_unstamped_frameworks_are_not_a_lost_record(self):
        # Only OCANNL rows are in that agreement: torch and tinygrad rows carry no stamp by
        # design, and counting them would refuse every ordinary mixed-framework sweep.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            cell("pytorch", "cpu", "eager", [2.3026, 2.3010]),
        ]
        orchestrate.parity_check(rows)

        text = self.report(rows)

        self.assertIn(
            'ambient OCANNL_* environment: {"OCANNL_BACKEND": "cc"}\n', text
        )
        self.assertNotIn("not recorded", text)

    def test_rows_predating_the_stamp_do_not_claim_a_clean_shell(self):
        # Nothing at report time can recover what the shell held while the numbers were taken, so
        # an unstamped row's silence stays silence rather than becoming evidence of `none`.
        rows = [cell("ocannl", "cc", "default", [2.3026, 2.3010])]
        orchestrate.parity_check(rows)

        text = self.report(rows)

        self.assertIn("ambient OCANNL_* environment: not recorded\n", text)
        self.assertNotIn("environment: none", text)

    def test_the_recorded_environment_survives_into_the_raw_rows(self):
        # The report line is a summary; the per-row stamp is what a later script reads, and it has
        # to be JSON like everything else results.jsonl carries.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]),
                {"OCANNL_NARROW_COMPUTE_F32": "true"},
            )
        ]
        orchestrate.parity_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            [row] = [
                strict_loads(line)
                for line in (out / "results.jsonl").read_text().splitlines()
                if line
            ]

        self.assertEqual(row["ambient_ocannl_env"], {"OCANNL_NARROW_COMPUTE_F32": "true"})

    def report(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            return (out / "report.md").read_text()


class CommandLineTest(unittest.TestCase):
    """What the flags parse to — the wiring between a module constant and the sweep.

    `CellTimeoutTest` pins the values themselves: that `DEFAULT_BEAM_PARALLEL` is zero, that
    `beam_parallel_arg` refuses a negative pool, that `beam_cell_env` turns the number into
    `PARALLEL`, that `cell_timeout_arg` refuses a cap that cannot mean anything. None of that
    reaches the sweep unless the flag is declared with that default, and a `default=` edit is
    invisible to every one of those tests.
    """

    def test_a_sweep_with_no_arguments_runs_the_beam_cells_without_a_pool(self):
        # gh-ocannl-843: the pool deadlocks tinygrad 0.13/0.14 on both GPU boxes, so no pool is
        # what an operator gets by not asking -- on a newly provisioned box as much as here.
        args = orchestrate.build_arg_parser().parse_args([])

        self.assertEqual(args.beam_parallel, 0)

    def test_an_explicit_pool_size_reaches_the_beam_cells(self):
        # The opt-in half: the pin is a default, not a ceiling, and an operator who asks for a
        # parallel search gets the size they asked for rather than the pinned zero.
        args = orchestrate.build_arg_parser().parse_args(["--beam-parallel", "4"])

        self.assertEqual(args.beam_parallel, 4)

    def test_a_sweep_with_no_arguments_caps_each_cell_at_half_an_hour(self):
        # gh-ocannl-760: the cap is what makes an unattended sweep lose a wedged cell rather than
        # the sweep, and it only does that from a bare `orchestrate.py`. The literal is the pin --
        # comparing against DEFAULT_CELL_TIMEOUT_S would still pass if the constant and the flag
        # were edited together, which is the drift this claim exists to catch.
        args = orchestrate.build_arg_parser().parse_args([])

        self.assertEqual(args.cell_timeout, 1800)

    def test_an_explicit_cap_reaches_the_cells(self):
        # The opt-in half: an operator who knows a workload outlives the pinned default gets the
        # cap they asked for, through the same `cell_timeout_arg` that refuses a meaningless one.
        args = orchestrate.build_arg_parser().parse_args(["--cell-timeout", "900"])

        self.assertEqual(args.cell_timeout, 900.0)


if __name__ == "__main__":
    unittest.main()
