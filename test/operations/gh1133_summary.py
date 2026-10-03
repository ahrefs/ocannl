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
        assert code == 1 and "REFUSED:" in text and reason in text, text
        assert "### Training segments:" not in text, text
        assert "sentinel" not in text, text

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
    assert code == 0 and "### Training segments:" in text and "sentinel-training-table" in text, text
    assert "| hip | gpt2_mini_train | keep |" in text, text
    print("successful training output appears beside its step-time row: true")
