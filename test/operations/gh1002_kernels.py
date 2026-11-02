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
import bench_output  # noqa: E402  (the kernel row's one reader, gh-ocannl-1767)


def check(condition, detail):
    # A failing control must remain fatal under python -O.
    if not condition:
        raise SystemExit("FAIL: " + detail)


def row(i, total, mma="none", vol="", io="", spread=""):
    return (f"bench: kernel {i}/{total} 0.125 ms {spread}grid=[2;3] block=[4;8] "
            f"{io}mma:{mma} {vol}w: layer_q layer_k\n")


with tempfile.TemporaryDirectory() as scratch:
    base = Path(scratch) / "cell"
    err = Path(str(base) + ".err")

    def parse(text):
        err.write_text(text)
        return report.kernels(base)

    expected = [{"i": 0, "ms": 0.125, "lo": None, "hi": None, "grid": "2;3", "block": "4;8",
                 "w": ["layer_q", "layer_k"]}]
    check(parse(row(0, 1)) == expected, "legacy row fields")
    print("legacy rows preserve timing, launch dimensions and writes: true")

    text = row(0, 1, "tile_mma 16x8x16 fp16", "vol:yes ")
    match = bench_output.KERNEL.match(text.rstrip("\n"))
    check(match is not None and match["mma"] == "tile_mma 16x8x16 fp16",
          "multiword mma descriptor must stop before vol")
    check(parse(text) == expected, "vol metadata must not contaminate writes")
    print("multiword mma text with vol preserves the write labels: true")

    # gh-ocannl-1444: the cost model's bytes read/written sit between the geometry and mma:.
    text = row(0, 1, "tile_mma 16x8x16 fp16", "vol:1 accumulation(s), pinned ", "io:4096/1024~ ")
    match = bench_output.KERNEL.match(text.rstrip("\n"))
    check(match is not None and (match["read"], match["written"], match["io_flags"], match["mma"])
          == ("4096", "1024", "~", "tile_mma 16x8x16 fp16"), "io field: %s" % (match and match.groupdict()))
    check(parse(text) == expected, "the io field must not contaminate timing, geometry or writes")
    legacy = bench_output.KERNEL.match(row(0, 1).rstrip("\n"))
    check(legacy is not None and legacy["read"] is None and legacy["writes"] == "layer_q layer_k",
          "a row without io must still parse")
    print("an io field parses into bytes read/written and leaves the other fields as they were: true")

    # gh-ocannl-1694: the median's range and admission count (Bench_retime.bounds_fields) sit
    # between the time and the geometry; below min_readings the bounds print as nan.
    spread = "ms_lo=0.062500 ms_hi=0.250000 ms_retimed=6/6 "
    text = row(0, 1, "tile_mma 16x8x16 fp16", "vol:0 ", "io:4096/1024 ", spread)
    match = bench_output.KERNEL.match(text.rstrip("\n"))
    check(match is not None and (match["lo"], match["hi"], match["admitted"], match["calls"])
          == ("0.062500", "0.250000", "6", "6"), "range fields: %s" % (match and match.groupdict()))
    check(parse(text) == [dict(expected[0], lo=0.0625, hi=0.25)],
          "the range must not contaminate timing, geometry or writes")
    unstated = parse(row(0, 1, spread="ms_lo=nan ms_hi=nan ms_retimed=2/2 "))
    check(unstated is not None and unstated[0]["ms"] == 0.125
          and unstated[0]["lo"] != unstated[0]["lo"], "an unstated range parses as NaN")
    print("a range parses beside the median, nan where it is unstated, and leaves the rest as it was: true")
    untimed = row(0, 2) + ("bench: kernel 1/2 grid=[2;3] block=[4;8] UNTIMED (none of its 6 "
                           "timing calls was admitted) w: layer_v\n")
    check(parse(untimed) is None, "a table with an UNTIMED row must be refused")
    print("a table with an untimed kernel is incomplete: true")

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
