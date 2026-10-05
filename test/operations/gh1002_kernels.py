"""Host-only controls for the shipping KERNEL regex and complete-table reader."""
import importlib.util
from pathlib import Path
import sys
import tempfile

owner = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(owner.parent))
spec = importlib.util.spec_from_file_location("gh1002_cells", owner)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def check(condition, detail):
    # A failing control must remain fatal under python -O.
    if not condition:
        raise SystemExit("FAIL: " + detail)


def row(i, total, mma="none", vol=""):
    return (f"bench: kernel {i}/{total} 0.125 ms grid=[2;3] block=[4;8] "
            f"mma:{mma} {vol}w: layer_q layer_k\n")


with tempfile.TemporaryDirectory() as scratch:
    base = Path(scratch) / "cell"
    err = Path(str(base) + ".err")

    def parse(text):
        err.write_text(text)
        return report.kernels(base)

    expected = [{"i": 0, "ms": 0.125, "grid": "2;3", "block": "4;8",
                 "w": ["layer_q", "layer_k"]}]
    check(parse(row(0, 1)) == expected, "legacy row fields")
    print("legacy rows preserve timing, launch dimensions and writes: true")

    text = row(0, 1, "tile_mma 16x8x16 fp16", "vol:yes ")
    match = report.KERNEL.match(text.rstrip("\n"))
    check(match is not None and match.group(6) == "tile_mma 16x8x16 fp16",
          "multiword mma descriptor must stop before vol")
    check(parse(text) == expected, "vol metadata must not contaminate writes")
    print("multiword mma text with vol preserves the write labels: true")

    complete = "unrelated diagnostic\n" + row(0, 3) + row(1, 3) + row(2, 3)
    parsed = parse(complete)
    check(parsed is not None and [k["i"] for k in parsed] == [0, 1, 2],
          "complete zero-based indices")
    print("a complete zero-based table is accepted: true")

    invalid = {
        "missing middle": row(0, 3) + row(2, 3),
        "missing last": row(0, 3) + row(1, 3),
        "missing first": row(1, 3) + row(2, 3),
        "duplicate": row(0, 2) + row(0, 2),
        "out of order": row(1, 2) + row(0, 2),
        "inconsistent total": row(0, 2) + row(1, 3),
        "empty": "unrelated diagnostic\n",
    }
    for name, text in invalid.items():
        check(parse(text) is None, name + " table must be refused")
    print("missing, duplicate, reordered and inconsistent rows are refused: true")
    err.unlink()
    check(report.kernels(base) is None, "missing stderr file must be refused")
    print("a missing kernel-table file is refused: true")
