"""gh-ocannl-1613: benchmark consumers driven on real Train.tune_placements substitution lines.

tune_ship_substituted.exe runs Train.tune_placements under Resource_fault_injection, wires the
reports and ship callbacks through Bench_harness and Bench_json.result_line, and writes four real
runner stdout captures in the current working directory:
  tss_both_refused.out   -- both arms' replays refused; arm B ships substituted
  tss_fallback_ok.out    -- arm A's replay refused; fallback arm B ships measured
  tss_forced_refused.out -- forced arm A's replay refused; arm A ships substituted
  tss_control.out        -- no refusal; arm A ships measured

This script feeds those real result lines to orchestrate.py (substitution_check, substitution_gate
and report), gh728_protocol.py and gh834_session.py, completing the tuner -> harness -> result line
-> benchmark consumer path end to end.
"""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
bench = root / "benchmarks"
sys.path.insert(0, str(bench))
import bench_output  # noqa: E402
import orchestrate  # noqa: E402


def claim(text, condition, detail=""):
    if not condition:
        raise SystemExit("FAIL: " + text + (": " + detail if detail else ""))
    print(text + ": true")


def load_row(name):
    text = Path(name).read_text(encoding="utf-8")
    return bench_output.only_result_record(text)


def run_script(script, *args, isolated=False):
    cmd = [sys.executable] + (["-I"] if isolated else []) + [str(bench / script), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


both_row = load_row("tss_both_refused.out")
fallback_row = load_row("tss_fallback_ok.out")
forced_row = load_row("tss_forced_refused.out")
control_row = load_row("tss_control.out")

both_crown = both_row["tune"]["substituted_from"]
forced_crown = forced_row["tune"]["substituted_from"]

# --- 1. orchestrate.py: substitution_check, substitution_gate and report table. ---
flagged = orchestrate.substitution_check([both_row, fallback_row, forced_row, control_row])
claim("orchestrate flags exactly the two real substituted rows",
      flagged == [both_row, forced_row], repr([r.get("tune", {}).get("shipped") for r in flagged]))
gate = orchestrate.substitution_gate(flagged)
claim("orchestrate substitution_gate emits SUBSTITUTION GATE naming both discarded crowns",
      gate.startswith("SUBSTITUTION GATE: 2 tuned cell(s)")
      and f"(discarded {both_crown})" in gate
      and f"(discarded {forced_crown})" in gate,
      gate)
with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    orchestrate.parity_check([both_row, control_row])
    with contextlib.redirect_stdout(io.StringIO()):
        orchestrate.report([both_row, control_row], out_dir)
    table = (out_dir / "report.md").read_text(encoding="utf-8")
claim("orchestrate report marks only the substituted row SHIPPED UNTUNED",
      table.count("tuned" + orchestrate.SUBSTITUTED_MARK + " |") == 1
      and "`substituted_from`" in table,
      table)

# --- 2. gh728_protocol.py: fails the real substituted lines, passes fallback and control. ---
for label, fname, crown in (
    ("both refused", "tss_both_refused.out", both_crown),
    ("forced refused", "tss_forced_refused.out", forced_crown),
):
    proc = run_script("gh728_protocol.py", fname, "search")
    claim(f"gh728_protocol refuses {label} with ship_substituted=True and discarded crown",
          proc.returncode == 1
          and "ship_substituted=True" in proc.stdout
          and f"substituted_from={crown!r}" in proc.stdout
          and "-- NOT A SEARCH PASS" in proc.stdout,
          f"rc={proc.returncode} out={proc.stdout!r} err={proc.stderr!r}")

for label, fname in (("fallback ok", "tss_fallback_ok.out"), ("control", "tss_control.out")):
    proc = run_script("gh728_protocol.py", fname, "search")
    claim(f"gh728_protocol accepts {label} with ship_substituted=False",
          proc.returncode == 0
          and "ship_substituted=False" in proc.stdout
          and "NOT A" not in proc.stdout,
          f"rc={proc.returncode} out={proc.stdout!r} err={proc.stderr!r}")

# --- 3. gh834_session.py: fails the real substituted lines, passes fallback and control. ---
for label, fname, crown in (
    ("both refused", "tss_both_refused.out", both_crown),
    ("forced refused", "tss_forced_refused.out", forced_crown),
):
    proc = run_script("gh834_session.py", fname, isolated=True)
    claim(f"gh834_session refuses {label} naming its discarded crown",
          proc.returncode == 1
          and (f"INCOMPLETE SESSION: the shipping compile substituted the untuned default "
               f"(discarded {crown})") in proc.stdout,
          f"rc={proc.returncode} out={proc.stdout!r} err={proc.stderr!r}")

for label, fname in (("fallback ok", "tss_fallback_ok.out"), ("control", "tss_control.out")):
    proc = run_script("gh834_session.py", fname, isolated=True)
    claim(f"gh834_session accepts {label} with no diagnostics",
          proc.returncode == 0 and proc.stdout.strip() == "",
          f"rc={proc.returncode} out={proc.stdout!r} err={proc.stderr!r}")
