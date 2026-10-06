"""Controls for the shipping driver's training-table validation and report rendering."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

spec = importlib.util.spec_from_file_location("summary", sys.argv[1])
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

def check(condition, detail):
    # Python -O must not erase a control's verdict while leaving its passing golden line.
    if not condition:
        raise SystemExit("FAIL: " + detail)


with tempfile.TemporaryDirectory() as scratch:
    out = Path(scratch)
    name = "hip-gpt2_mini_train-keep-trainseg.out"
    table = out / name
    status = out / (name[:-4] + ".exit")
    step = {"step_ms": {"p10": 1, "p50": 2, "p90": 3}, "losses": [7]}
    (out / "hip-gpt2_mini_train-keep-r1.out").write_text(json.dumps(step) + "\n")

    def render():
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            code = report.summary(str(out), "keep", "keep")
        return code, text.getvalue()

    def refused(reason):
        code, text = render()
        check(code == 1 and "REFUSED:" in text and reason in text, text)
        check("### Training segments:" not in text, text)
        check("sentinel" not in text, text)

    table.write_text("mode: infer backend: hip\nsentinel-forward-table\n")
    status.write_text("0\n")
    refused("is not a training diagnostic")
    print("forward-only output is refused rather than labeled as training: true")

    table.write_text("mode: train backend: hip\nsentinel-partial-table\n")
    for exit_code in ("124", "1", "137"):
        status.write_text(exit_code + "\n")
        refused("did not complete successfully")
    status.unlink()
    refused("did not complete successfully")
    print("failed, capped and unrecorded outputs are refused rather than published: true")

    status.write_text("0\n")
    table.write_text("mode: train backend: hip\nsentinel-training-table\n")
    code, text = render()
    check(code == 0 and "### Training segments:" in text and "sentinel-training-table" in text, text)
    check("| hip | gpt2_mini_train | keep |" in text, text)
    print("successful training output appears beside its step-time row: true")

    # The cell names follow the treatments the driver was given, whatever they are called.
    (out / "hip-gpt2_mini_train-d1serialcut-r1.out").write_text(json.dumps(step) + "\n")
    (out / "hip-gpt2_mini_train-d1serialoff-r1.out").write_text(json.dumps(step) + "\n")
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = report.summary(str(out), "d1serialcut d1serialoff", "serialoff")
    text = text.getvalue()
    check(code == 0 and "| hip | gpt2_mini_train | d1serialcut |" in text, text)
    check("| 1.000x |" in text and "MISSING REFERENCE" not in text, text)
    check("| keep |" not in text, text)
    print("a treatment's cells are summarized by its name, and only the given treatments': true")

# A prep or dry run (no step stage) owes no step-time matrix; a step stage's empty matrix fails.
with tempfile.TemporaryDirectory() as scratch:
    out = Path(scratch)

    def render(step_stage):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            code = report.summary(str(out), "keep", "keep", step_stage=step_stage)
        return code, text.getvalue()

    code, text = render(step_stage=True)
    check(code == 1 and "NO MEASUREMENT" in text, text)
    print("a step stage whose cells produced no result line fails: true")

    code, text = render(step_stage=False)
    check(code == 0 and "No step stage requested" in text, text)
    check("NO MEASUREMENT" not in text and "| backend |" not in text, text)
    name = "hip-gpt2_mini_train-keep-trainseg.out"
    (out / name).write_text("mode: train backend: hip\nsentinel-prep-table\n")
    (out / (name[:-4] + ".exit")).write_text("0\n")
    code, text = render(step_stage=False)
    check(code == 0 and "sentinel-prep-table" in text and "NO MEASUREMENT" not in text, text)
    print("a run without a step stage passes on its training tables alone: true")

    (out / (name[:-4] + ".exit")).write_text("124\n")
    code, text = render(step_stage=False)
    check(code == 1 and "REFUSED:" in text and "sentinel-prep-table" not in text, text)
    print("a run without a step stage still fails on a refused training table: true")
