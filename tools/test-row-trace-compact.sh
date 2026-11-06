#!/usr/bin/env bash
# Hermetic fixtures for tools/row-trace-compact.py: what the digest keeps of a flushing row trace
# (a call once per stage, a template at its mint, one element per axis, the trace line of each),
# in the form stdout gets and the one a log file gets, a trace that ends inside a call, the records
# it cannot read, a trace with nothing in it, and usage errors. Each claim has a negative control,
# a copy of the tool with that guard removed.
#
# The records are spelled as real traces of test/operations/batch_norm_closure spell them, wrapping
# and scope edges included. Needs python3; a host without it skips every leg.
# Usage: tools/test-row-trace-compact.sh [--keep] [--only REGEX]
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch test-row-trace-compact
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/row-trace-compact.py

# P is the prefix of value lines: `debug ` on stdout, none in a log file; scope edges always have it.
P='debug '
# ineq STAGE RES OPND: one call, RES and OPND a row's fields before its `prov`.
ineq() {
  printf '  debug solve_row_ineq begin "tensor/row.ml":4164:31: 12.4ms / 12337791ns\n'
  printf '   %sstage = %s\n' "$P" "$1"
  printf '   %sres = (%s\n (prov (((sh_id 81) (kind Batch)))))\n' "$P" "$2"
  printf '   %sopnd = (%s\n (prov (((sh_id 80) (kind Batch)))))\n' "$P" "$3"
  printf '   debug res_dims_l begin "tensor/row.ml":4172:6: 12.4ms / 12352666ns\n'
  printf '    %sres_dims_l = 0\n' "$P"
  printf '   12.4ms / 12361916ns - debug res_dims_l end\n'
  printf '  12.4ms / 12380958ns - debug solve_row_ineq end\n'
}
templ() { # V
  printf '   debug templ_v begin "tensor/row.ml":4411:10: 12.7ms / 12696833ns\n'
  printf '    %stempl_v = (Row_var %s)\n' "$P" "$1"
  printf '   12.8ms / 12705791ns - debug templ_v end\n'
}
V83='(beg_dims ()) (dims ()) (bcast (Row_var (Row_var 83)))'
BC='(beg_dims ()) (dims ()) (bcast Broadcastable)'
ONE='(Dim ((d 1) (basis default) (proj_id ())))'
TOP='(Dim ((d 1) (basis bcast_if_1) (proj_id ())))'
AFFINE='(Affine (stride 2) (over (Var ((id 85) (name (oh)))))
       (conv
        (((dilation 1) (kernel (Var ((id 86) (name (kh)))))
          (use_padding true))))
       (stride_offset 1))'
SYM='(Sym ((sym s3) (sym_basis default) (sym_proj_id ())))'
out=$TMP/out err=$TMP/err

fixture() {
  echo 'BEGIN DEBUG SESSION debug at elapsed 7.26ms / 7254750ns'
  echo 'debug solve_inequalities begin "tensor/row.ml":5354:35: 7.3ms / 7298875ns'
  echo " ${P}stage = Stage1"
  ineq Stage1 "$V83" "$BC"
  ineq Stage1 "$V83" "$BC"
  ineq Stage2 "$V83" "$BC"
  templ 87
  templ 87
  ineq Stage1 "(beg_dims ()) (dims ($ONE)) (bcast Broadcastable)" "(beg_dims ()) (dims ($TOP)) (bcast Broadcastable)"
  ineq Stage1 "(beg_dims ($AFFINE)) (dims ((Concat ($ONE (Var ((id 84) (name ()))))) $SYM)) (bcast (Row_var (Row_var 83)))" "$BC"
  ineq Stage1 "(beg_dims (${AFFINE/use_padding true/use_padding false})) (dims ((Concat ($ONE (Var ((id 84) (name ()))))) $SYM)) (bcast (Row_var (Row_var 83)))" "$BC"
}
fixture >"$TMP/trace"
P=
fixture >"$TMP/trace-in-file"
P='debug '
cat >"$TMP/want" <<'WANT'
4 Stage1 ..83.. >= <bc>
24 Stage2 ..83.. >= <bc>
    35 templ 87
40 Stage1 <bc>,1 >= <bc>,bcast_if_1=1
50 Stage1 conv_pad(2*v85+1+v86),..83..,(1^v84),(Sym ((sym s3) (sym_basis default) (sym_proj_id ()))) >= <bc>
64 Stage1 conv(2*v85+1+v86),..83..,(1^v84),(Sym ((sym s3) (sym_basis default) (sym_proj_id ()))) >= <bc>
WANT
# The tool's verdict on a trace, into $out and $err, its status in rc.
digest() { # TOOL TRACE
  rc=0
  python3 "$1" "$2" >"$out" 2>"$err" || rc=$?
}
check_digest() { # TOOL [TRACE]
  digest "$1" "${2:-$TMP/trace}"
  [ "$rc" = 0 ] && cmp -s "$out" "$TMP/want" &&
    grep -qF 'row-trace-compact: 6 solve_row_ineq, 5 distinct, 1 template mints, 0 unparsed' "$err"
}
check_in_file() { check_digest "$1" "$TMP/trace-in-file"; } # TOOL
# A crash inside the call: the arguments are logged, no scope edge follows.
ineq Stage6 "$V83" "$BC" | sed -n 1,6p >"$TMP/crashed"
check_crash() { # TOOL
  digest "$1" "$TMP/crashed"
  [ "$rc" = 0 ] && [ "$(cat "$out")" = '1 Stage6 ..83.. >= <bc>' ]
}
{
  ineq Stage1 "$V83" "$BC"
  ineq Stage2 '(beg_dims ()) (dims (' "$BC"
  ineq Stage3 "$V83" "$BC"
} >"$TMP/damaged"
check_damaged() { # TOOL
  digest "$1" "$TMP/damaged"
  [ "$rc" = 1 ] && [ "$(cat "$out")" = $'1 Stage1 ..83.. >= <bc>\n11 Stage2 <unparsed>\n21 Stage3 ..83.. >= <bc>' ] &&
    grep -qF '1 unparsed' "$err"
}
echo 'a test that logged nothing' >"$TMP/empty"
check_empty() { # TOOL
  digest "$1" "$TMP/empty"
  [ "$rc" = 1 ] && [ ! -s "$out" ] && grep -qF 'OCANNL_LOG_LEVEL_ROW=9' "$err"
}
# faulted NAME OLD NEW: a copy of the tool with OLD replaced, which must exist and still compile.
faulted() {
  python3 - "$SRC" "$TMP/$1.py" "$2" "$3" <<'PYTHON'
import sys
src, dst, old, new = sys.argv[1:]
with open(src, encoding="utf-8") as f:
    text = f.read()
if old not in text:
    sys.exit(f"{old!r} is not in {src}")
text = text.replace(old, new)
compile(text, dst, "exec")
with open(dst, "w", encoding="utf-8") as f:
    f.write(text)
PYTHON
}
# control LABEL CHECK NAME OLD NEW: CHECK refuses the fault, on its output, not on a crash.
control() {
  if ! faulted "$3" "$4" "$5"; then
    report 1 "negative control: $1" 'the fault did not apply'
  elif "$2" "$TMP/$3.py"; then
    report 1 "negative control: $1" 'the oracle accepted the fault'
  elif grep -q Traceback "$err"; then
    report 1 "negative control: $1" "the fault crashed the tool: $(cat "$err")"
  else report 0 "negative control: $1"; fi
}

if harness_leg 'each call prints once per stage, each template once, one element per axis #digest'; then
  if check_digest "$SRC"; then report 0 'the digest of the fixture'
  else report 1 'the digest of the fixture' "exit $rc; $(cat "$out" "$err")"; fi
  control 'a later stage of the same call is kept' check_digest no-stage \
    'key = (stage, res, opnd)' 'key = (res, opnd)'
  control 'a template cache hit is no mint' check_digest every-sighting \
    'if value not in minted:' 'if True:'
  control 'the basis of a solved dim is kept' check_digest no-basis \
    'return solved["d"] if basis == "default" else' 'return solved["d"] if True else'
  control 'an Affine dim is one axis' check_digest no-affine 'if kind == "Affine":' 'if kind == "None":'
  control 'padded and unpadded windows stay apart' check_digest no-padding \
    'form = "conv_pad" if conv["use_padding"] == "true" else "conv"' 'form = "conv"'
fi

if harness_leg 'a log file, with bare value lines, digests the same #digest'; then
  if check_in_file "$SRC"; then report 0 'the digest of the fixture in a log file'
  else report 1 'the digest of the fixture in a log file' "exit $rc; $(cat "$out" "$err")"; fi
  control 'value lines need no prefix' check_in_file prefix-required '(?:debug )?' 'debug '
fi

if harness_leg 'every printed line number is the trace line of its record #digest'; then
  bad=
  digest "$SRC" "$TMP/trace"
  while read -r n word _; do
    case $word in templ) pattern='templ_v = ' ;; *) pattern='debug solve_row_ineq begin' ;; esac
    sed -n "${n}p" "$TMP/trace" | grep -qF "$pattern" || bad="$bad $n"
  done <"$out"
  if [ -s "$out" ] && [ -z "$bad" ]; then report 0 'line numbers resolve to their records'
  else report 1 'line numbers resolve to their records' "off:$bad; $(cat "$out" "$err")"; fi
fi

if harness_leg 'a trace ending inside a call keeps that call #partial'; then
  if check_crash "$SRC"; then report 0 'the last call of a crashed trace'
  else report 1 'the last call of a crashed trace' "exit $rc; $(cat "$out" "$err")"; fi
  control 'the end of the trace closes an open call' check_crash no-eof \
    '    if arguments is not None:
        yield begun, "ineq", arguments
' ''
fi

if harness_leg 'a record it cannot read is shown and fails the run #partial'; then
  if check_damaged "$SRC"; then report 0 'a damaged record, between two good ones'
  else report 1 'a damaged record, between two good ones' "exit $rc; $(cat "$out" "$err")"; fi
  control 'a damaged record fails the run' check_damaged status-zero \
    'return 1 if unparsed or calls == 0 else 0' 'return 0'
fi

if harness_leg 'a trace with no solve_row_ineq names the gate and fails #partial'; then
  if check_empty "$SRC"; then report 0 'an empty digest is no verdict'
  else report 1 'an empty digest is no verdict' "exit $rc; $(cat "$out" "$err")"; fi
  control 'an empty digest fails the run' check_empty empty-green \
    'return 1 if unparsed or calls == 0 else 0' 'return 1 if unparsed else 0'
fi

if harness_leg 'a reader that has left costs no traceback #usage'; then
  # Its read end closed before the tool starts: which write meets that is then no race. A short
  # digest stays in the buffer to the last flush, a long one overflows it in mid-run.
  ineq Stage1 "$V83" "$BC" >"$TMP/short"
  for i in $(seq 1 1000); do
    ineq Stage1 "(beg_dims ()) (dims ()) (bcast (Row_var (Row_var $i)))" "$BC"
  done >"$TMP/long"
  check_reader_gone() { # TOOL TRACE
    python3 - "$1" "$2" "$err" <<'PYTHON'
import os, subprocess, sys
tool, trace, err = sys.argv[1:]
read_end, write_end = os.pipe()
os.close(read_end)
with open(err, "w") as stderr:
    subprocess.run([sys.executable, tool, trace], stdout=write_end, stderr=stderr)
PYTHON
    [ ! -s "$err" ]
  }
  check_gone_short() { check_reader_gone "$1" "$TMP/short"; } # TOOL
  check_gone_long() { check_reader_gone "$1" "$TMP/long"; } # TOOL
  if check_gone_short "$SRC"; then report 0 'quiet when the last flush meets a closed pipe'
  else report 1 'quiet when the last flush meets a closed pipe' "$(cat "$err")"; fi
  if check_gone_long "$SRC"; then report 0 'quiet when a write in mid-run meets a closed pipe'
  else report 1 'quiet when a write in mid-run meets a closed pipe' "$(cat "$err")"; fi
  gone_control() { # LABEL CHECK NAME OLD NEW: as control, where the fault's only symptom is noise
    if ! faulted "$3" "$4" "$5"; then report 1 "negative control: $1" 'the fault did not apply'
    elif "$2" "$TMP/$3.py"; then report 1 "negative control: $1" 'the oracle accepted the fault'
    else report 0 "negative control: $1"; fi
  }
  gone_control 'the last flush is the tool'"'"'s own' check_gone_short no-flush \
    '    sys.stdout.flush()
' ''
  gone_control 'the closed pipe is handled' check_gone_long loud 'except BrokenPipeError:' \
    'except ZeroDivisionError:'
fi

if harness_leg 'usage errors exit 2 #usage'; then
  if harness_rejected 2 'cannot read' "$out" python3 "$SRC" "$TMP/no-such-trace"; then
    report 0 'a trace that cannot be read exits 2'
  else report 1 'a trace that cannot be read exits 2' "$(cat "$out")"; fi
  if harness_rejected 2 'required: TRACE' "$out" python3 "$SRC"; then report 0 'no trace exits 2'
  else report 1 'no trace exits 2' "$(cat "$out")"; fi
fi
finish
