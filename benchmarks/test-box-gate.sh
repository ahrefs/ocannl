#!/usr/bin/env bash
# ocannl-harness: standalone
# Controls for the box gate (benchmarks/box-gate.sh, benchmarks/box_gate.py): a benchmark driver
# that loads fixtures runs benchmarks/check-box.sh on its box and on the fixtures it loads before
# it starts, refuses to start on a GAP with the check's own output, and starts on READY. Each
# driver runs against a stand-in check-box.sh answering GAP, then READY. The GAP run exits 1 with
# the check's GAP verdict and a `refusing to start` line and reaches nothing past the gate; the
# READY run reaches the driver's first action past it; both hand the check exactly the box and the
# fixtures the driver's cells load. The gate's own surface -- no --box, --skip-box-check, a check
# that judged nothing, malformed options, a run with no measuring step -- is exercised on the shell
# side here (the Python side in benchmarks/test_orchestrate.py, beside orchestrate.py's own legs
# and the inventory that every fixture-loading driver calls the gate), and each oracle has a
# mutated twin it must reject.
# Usage: benchmarks/test-box-gate.sh [--keep|--only REGEX|--help]
#
# The drivers run from copies in a scratch git checkout whose benchmarks/check-box.sh is the
# stand-in: it appends its arguments to TMP/calls and answers what TMP/verdict says. A stub dune
# and opam on PATH, and a stub bench_gpt.exe in the scratch _build, record into TMP/started that a
# driver went on past the gate, and fail, so no driver gets further than its first action.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3 git
harness_scratch test-box-gate

mkdir -p "$TMP/tree/benchmarks/fixtures" "$TMP/bin" "$TMP/runs"
tree=$(cd "$TMP/tree" && pwd -P) || exit 2
fx=$tree/benchmarks/fixtures
mkdir -p "$tree/benchmarks/runners"
cp "$here"/*.sh "$here"/*.py "$here/ocannl_config" "$tree/benchmarks/" || exit 2
cp "$here"/runners/*.py "$tree/benchmarks/runners/" || exit 2
for f in mlp_wide mlp_small lenet gpt2_mini gpt2_mini_s512 gpt2_mini_s1024 gpt2_mini_train \
  gpt2_mini_train_s512 gpt2_mini_train_s1024 gpt2_mini_train_b1_s128 gpt2_mini_train_b1_s256 \
  gpt2_mini_train_b1_s512; do
  : >"$fx/$f.safetensors"
done
cat >"$tree/benchmarks/check-box.sh" <<EOF
#!/usr/bin/env bash
# The gates' name reading (--canonical-box) is the real script's, and test-check-box.sh's to test.
[ "\$1" != --canonical-box ] || { printf '%s\n' "\$2"; exit 0; }
printf '%s\n' "\$*" >>"$TMP/calls"
printf '%s\n' "\${BENCH_VENV_PY-unset}" >>"$TMP/venv"
case \$(cat "$TMP/verdict") in
  ready) echo 'check-box: venv: ok -- stand-in'; echo 'check-box: stand-in: READY (1 items)'; exit 0 ;;
  gap) echo 'check-box: fixture stand-in: GAP -- absent from the stand-in store'
    echo '  repair: copy the stand-in in'; echo 'check-box: stand-in: GAP (1 of 2 items)'; exit 1 ;;
  *) echo 'check-box: the stand-in judged nothing' >&2; exit 2 ;;
esac
EOF
for tool in dune opam; do
  printf '#!/bin/sh\necho "%s $*" >>"%s/started"\nexit 1\n' "$tool" "$TMP" >"$TMP/bin/$tool"
done
runner=$tree/_build/default/benchmarks/runners/ocannl/bench_gpt.exe
mkdir -p "$(dirname "$runner")"
printf '#!/bin/sh\necho "bench_gpt BENCH_FIXTURE=$BENCH_FIXTURE" >>"%s/started"\nexit 1\n' "$TMP" >"$runner"
chmod +x "$TMP/bin/"* "$runner"
git -C "$tree" init -q && git -C "$tree" add -A benchmarks &&
  git -C "$tree" -c user.name=harness -c user.email=harness@localhost commit -qm scratch || exit 2
export PATH="$TMP/bin:$PATH"
gate=$tree/benchmarks/box-gate.sh
cp "$gate" "$TMP/box-gate.sh.orig" || exit 2

# [drive RUN VERDICT CMD...]: CMD with the stand-in answering VERDICT, every `@OUT@` in it
# replaced by runs/RUN/out (which nothing makes before the driver does); output in
# runs/RUN/stdout, status in $got, the check's arguments in runs/RUN/calls, what went past the
# gate in runs/RUN/started.
drive() {
  local run=$1 d=$TMP/runs/$1 a cmd=()
  printf '%s\n' "$2" >"$TMP/verdict"
  shift 2
  for a in "$@"; do cmd+=("${a//@OUT@/$d/out}"); done
  rm -f "$TMP/calls" "$TMP/started" "$TMP/venv"
  mkdir -p "$d"
  got=0
  "${cmd[@]}" >"$d/stdout" 2>&1 </dev/null || got=$?
  mv "$TMP/calls" "$d/calls" 2>/dev/null || : >"$d/calls"
  mv "$TMP/started" "$d/started" 2>/dev/null || : >"$d/started"
  mv "$TMP/venv" "$d/venv" 2>/dev/null || : >"$d/venv"
}
# [refused RUN WANT DRIVER]: exit 1, the check's GAP verdict and DRIVER's refusal on the output,
# the check called once with WANT, and nothing past the gate: no first action, no OUT.
refused() {
  local d=$TMP/runs/$1
  [ "$got" = 1 ] && grep -q '^check-box: stand-in: GAP (1 of 2 items)$' "$d/stdout" &&
    grep -q "^$3: refusing to start: " "$d/stdout" && [ "$(cat "$d/calls")" = "$2" ] &&
    [ ! -s "$d/started" ] && [ ! -e "$d/out" ]
}
# [started RUN WANT MARK]: no refusal, the check called once with WANT (nothing when WANT is
# empty), and the driver reached its first action past the gate: MARK, a pattern on
# runs/RUN/started, or `out` for a driver whose first action is making its OUT.
started() {
  local d=$TMP/runs/$1
  ! grep -q 'refusing to start' "$d/stdout" && [ "$(cat "$d/calls")" = "$2" ] &&
    if [ "$3" = out ]; then [ -d "$d/out" ]; else grep -q -- "$3" "$d/started"; fi
}
# [pair LABEL RUN WANT DRIVER MARK CMD...]: CMD under GAP must be refused, under READY must start.
pair() {
  local label=$1 run=$2 want=$3 driver=$4 mark=$5
  shift 5
  drive "$run-gap" gap "$@"
  if refused "$run-gap" "$want" "$driver"; then report 0 "$label refuses to start on a GAP"
  else report 1 "$label refuses to start on a GAP" "rc=$got; see $TMP/runs/$run-gap (--keep)"; fi
  drive "$run-ready" ready "$@"
  if started "$run-ready" "$want" "$mark"; then report 0 "$label starts on READY"
  else report 1 "$label starts on READY" "rc=$got; see $TMP/runs/$run-ready (--keep)"; fi
}
want() { # BOX DIR NAME...: the arguments the check must be handed
  local box=$1 dir=$2 line n
  shift 2
  line="--box $box --fixture-dir $dir"
  for n in "$@"; do line="$line --fixture $n"; done
  printf '%s' "$line"
}
# [verdict LABEL OK RUN]: report LABEL from the status of the check OK.
verdict() { if [ "$2" = 0 ]; then report 0 "$1"; else report 1 "$1" "rc=$got; see $TMP/runs/$3 (--keep)"; fi; }
gh514_want=$(want m4-max "$fx" mlp_wide gpt2_mini)
gh514() { bash "$tree/benchmarks/gh514_cells.sh" "$@" cc f32 "$tree"; }

if harness_leg 'gh514_cells.sh#shell'; then
  pair gh514_cells.sh gh514 "$gh514_want" gh514_cells 'dune build' gh514 --box m4-max
fi
if harness_leg 'gh1133_cells.sh#shell'; then
  all="gpt2_mini gpt2_mini_s512 gpt2_mini_s1024 gpt2_mini_train gpt2_mini_train_s512 gpt2_mini_train_s1024"
  # shellcheck disable=SC2086 # a list of names
  pair gh1133_cells.sh gh1133 "$(want m4-max "$fx" $all)" gh1133 'dune build' \
    bash "$tree/benchmarks/gh1133_cells.sh" --box m4-max @OUT@ "$TMP" 60 build step
  # Each measuring step adds what it loads, once: seg the inference fixtures, trainseg the rest.
  drive gh1133-segs gap bash "$tree/benchmarks/gh1133_cells.sh" --box m4-max @OUT@ "$TMP" 60 trainseg seg step
  # shellcheck disable=SC2086 # a list of names
  refused gh1133-segs "$(want m4-max "$fx" gpt2_mini_train gpt2_mini_train_s512 gpt2_mini_train_s1024 \
    gpt2_mini gpt2_mini_s512 gpt2_mini_s1024)" gh1133
  verdict 'gh1133_cells.sh: trainseg, seg and step check every fixture once, in step order' $? gh1133-segs
  drive gh1133-seg gap bash "$tree/benchmarks/gh1133_cells.sh" --box m4-max @OUT@ "$TMP" 60 seg
  refused gh1133-seg "$(want m4-max "$fx" gpt2_mini gpt2_mini_s512 gpt2_mini_s1024)" gh1133
  verdict 'gh1133_cells.sh: seg checks only the fixtures it loads' $? gh1133-seg
fi
if harness_leg 'gh1003_fold_cells.sh#shell'; then
  pair gh1003_fold_cells.sh gh1003 "$(want m4-max "$fx" gpt2_mini gpt2_mini_s512 gpt2_mini_s1024 \
    gpt2_mini_train_s512 gpt2_mini_train_s1024)" gh1003 'dune build' \
    bash "$tree/benchmarks/gh1003_fold_cells.sh" --box m4-max @OUT@ 60 build metal train
  # The scrub of BENCH_* keeps the cells hermetic; the check still sees the box's interpreter
  # override, set or not.
  drive gh1003-venv ready env BENCH_VENV_PY=/shared/venv/bin/python \
    bash "$tree/benchmarks/gh1003_fold_cells.sh" --box m4-max @OUT@ 60 dry
  [ "$(cat "$TMP/runs/gh1003-venv/venv")" = /shared/venv/bin/python ]
  verdict 'gh1003_fold_cells.sh: the check sees BENCH_VENV_PY through the BENCH_* scrub' $? gh1003-venv
  drive gh1003-novenv ready env -u BENCH_VENV_PY bash "$tree/benchmarks/gh1003_fold_cells.sh" --box m4-max @OUT@ 60 dry
  [ "$(cat "$TMP/runs/gh1003-novenv/venv")" = unset ]
  verdict 'gh1003_fold_cells.sh: an unset BENCH_VENV_PY stays unset for the check' $? gh1003-novenv
  drive gh1003-cc gap bash "$tree/benchmarks/gh1003_fold_cells.sh" --box m4-max @OUT@ 60 cc dry
  refused gh1003-cc "$(want m4-max "$fx" gpt2_mini gpt2_mini_s512)" gh1003
  verdict 'gh1003_fold_cells.sh: cc and dry check only the fixtures they load' $? gh1003-cc
fi
if harness_leg 'gh728_cells.sh#shell'; then
  pair gh728_cells.sh gh728 "$(want rog-nv "$fx" gpt2_mini)" gh728_cells 'dune build' \
    bash "$tree/benchmarks/gh728_cells.sh" --box rog-nv cc @OUT@ "$fx/gpt2_mini.safetensors" 2 2
  # An absent fixture reaches the check, whose GAP names the repair, rather than a usage error.
  drive gh728-absent gap bash "$tree/benchmarks/gh728_cells.sh" --box rog-nv cc @OUT@ "$fx/gpt2_mini_b1.safetensors" 2 2
  refused gh728-absent "$(want rog-nv "$fx" gpt2_mini_b1)" gh728_cells
  verdict 'gh728_cells.sh: an absent fixture is the check'"'"'s GAP' $? gh728-absent
fi
if harness_leg 'gh834_cells.sh#shell'; then
  pair gh834_cells.sh gh834 "$(want tuf "$fx" gpt2_mini)" gh834 'dune build' \
    bash "$tree/benchmarks/gh834_cells.sh" --box tuf cc @OUT@ "$fx/gpt2_mini.safetensors" 60 \
    build provenance session-queued
  # A relative FIXTURE is read from benchmarks/, wherever the driver was started from.
  drive gh834-relative gap env -C "$TMP" bash "$tree/benchmarks/gh834_cells.sh" --box tuf cc @OUT@ \
    fixtures/gpt2_mini.safetensors 60 session-queued
  refused gh834-relative "$(want tuf "$fx" gpt2_mini)" gh834
  verdict 'gh834_cells.sh: a relative FIXTURE is checked where the sessions read it' $? gh834-relative
  # The crown steps read no fixture: a run of them needs no box.
  drive gh834-crown gap bash "$tree/benchmarks/gh834_cells.sh" cc @OUT@ "$fx/gpt2_mini.safetensors" 60 build crown-fwd
  started gh834-crown '' 'dune build'
  verdict 'gh834_cells.sh: a run of steps that read no fixture is not gated' $? gh834-crown
fi
if harness_leg 'gh612_cells.sh#shell'; then
  # The gate sits in the fixture check of the subcommands that load the fixture; OUT_ROOT is made
  # at startup, so it lives apart from the per-run OUT.
  pair gh612_cells.sh gh612 "$(want minix "$fx" gpt2_mini)" gh612_cells 'bench_gpt BENCH_FIXTURE=fixtures/gpt2_mini' \
    env OUT_ROOT="$TMP/gh612-root" FIXTURE_MD5= PIN=env \
    bash "$tree/benchmarks/gh612_cells.sh" --box minix search "$tree" cell 1
fi
if harness_leg 'gh1181_cells.py#python'; then
  pair gh1181_cells.py gh1181 "$(want rog-nv "$fx" gpt2_mini)" gh1181_cells.py out \
    python3 "$tree/benchmarks/gh1181_cells.py" --backend cuda --box rog-nv --no-anchor harness \
    --out @OUT@ --workloads gpt2_mini
fi
if harness_leg 'gh1002_cells.py#python'; then
  pair gh1002_cells.py gh1002 "$(want m4-max "$fx" gpt2_mini_train gpt2_mini_train_s512 \
    gpt2_mini_train_s1024 gpt2_mini_train_b1_s128 gpt2_mini_train_b1_s256 gpt2_mini_train_b1_s512)" \
    gh1002_cells.py out python3 "$tree/benchmarks/gh1002_cells.py" run --out @OUT@ --box m4-max \
    --phases preflight
fi
if harness_leg 'gh998_cells.py#python'; then
  pair gh998_cells.py gh998 "$(want m4-max "$fx" lenet gpt2_mini)" gh998_cells.py out \
    python3 "$tree/benchmarks/gh998_cells.py" prepare --out @OUT@ --box m4-max --workloads lenet,gpt2_mini
fi
if harness_leg 'gh675_cells.py#python'; then
  # --repeats 0 plans no pair: past the gate the driver makes its caches under GH675_OUT and stops.
  pair gh675_cells.py gh675 "$(want rog-nv "$fx" mlp_small)" gh675_cells.py out \
    env GH675_OUT=@OUT@ python3 "$tree/benchmarks/gh675_cells.py" --box rog-nv --arms C_eager \
    --workloads mlp_small --repeats 0
fi

if harness_leg 'the gate surface#surface'; then
  drive no-box gap gh514
  [ "$got" = 2 ] && grep -q '^gh514_cells: --box BOX is required to measure' "$TMP/runs/no-box/stdout" &&
    [ ! -s "$TMP/runs/no-box/calls" ] && [ ! -s "$TMP/runs/no-box/started" ]
  verdict 'no --box: exit 2 before the check and before the first action' $? no-box
  drive skip gap gh514 --skip-box-check
  started skip '' 'dune build' &&
    grep -q '^gh514_cells: --skip-box-check: measuring without benchmarks/check-box.sh' "$TMP/runs/skip/stdout"
  verdict '--skip-box-check: no check, said on the output, and the driver starts' $? skip
  drive skip-box gap gh514 --skip-box-check --box m4-max
  started skip-box '' 'dune build'
  verdict '--skip-box-check wins over --box' $? skip-box
  drive nothing judged gh514 --box m4-max
  [ "$got" = 2 ] && grep -q '^gh514_cells: refusing to start: benchmarks/check-box.sh judged nothing (exit 2)$' \
    "$TMP/runs/nothing/stdout" && [ "$(cat "$TMP/runs/nothing/calls")" = "$gh514_want" ] &&
    [ ! -s "$TMP/runs/nothing/started" ]
  verdict 'a check that judged nothing: exit 2, no start' $? nothing
  drive twice ready gh514 --box m4-max --box tuf
  [ "$got" = 2 ] && grep -q -- '--box given twice' "$TMP/runs/twice/stdout" && [ ! -s "$TMP/runs/twice/calls" ]
  verdict 'a repeated --box: exit 2' $? twice
  drive bare ready bash "$tree/benchmarks/gh514_cells.sh" --box
  [ "$got" = 2 ] && grep -q -- '--box needs a value' "$TMP/runs/bare/stdout" && [ ! -s "$TMP/runs/bare/calls" ]
  verdict 'a --box with no value: exit 2' $? bare
  drive unmeasured gap bash "$tree/benchmarks/gh1133_cells.sh" @OUT@ "$TMP" 60 summary
  [ ! -s "$TMP/runs/unmeasured/calls" ] && ! grep -q -- '--box BOX is required' "$TMP/runs/unmeasured/stdout"
  verdict 'a run with no measuring step needs no box' $? unmeasured
fi

if harness_leg 'negative controls#mutants'; then
  # [driver_refuses SUBJECT RUN]: SUBJECT, installed as a driver of the scratch tree, refused on a GAP.
  driver_refuses() {
    cp "$1" "$tree/benchmarks/subject.sh" || return 2
    drive "$2" gap bash "$tree/benchmarks/subject.sh" --box m4-max cc f32 "$tree"
    refused "$2" "$gh514_want" gh514_cells
  }
  # [gate_refuses SUBJECT RUN]: SUBJECT, installed as the scratch tree's box-gate.sh, has gh514 refused.
  gate_refuses() {
    cp "$1" "$gate" || return 2
    drive "$2" gap gh514 --box m4-max
    local rc=0
    refused "$2" "$gh514_want" gh514_cells || rc=1
    cp "$TMP/box-gate.sh.orig" "$gate"
    return "$rc"
  }
  driver_refuses "$here/gh514_cells.sh" control-driver
  verdict 'the oracle accepts the shipping driver' $? control-driver
  gate_refuses "$here/box-gate.sh" control-gate
  verdict 'the oracle accepts the shipping gate' $? control-gate
  SRC=$here/gh514_cells.sh
  ungated=$(mutant ungated '/^box_gate gh514_cells / { next } { print }') || exit 2
  cmp -s "$SRC" "$ungated" && { echo "mutant ungated changed nothing" >&2; exit 2; }
  expect_rejected 'a driver without its gate call' "$ungated" driver_refuses 'BUILD FAILED'
  SRC=$here/box-gate.sh
  lenient=$(mutant lenient '/^    1\)$/ { f = 1 } f && /^      exit 1$/ { print "      return 0"; f = 0; next } { print }') || exit 2
  cmp -s "$SRC" "$lenient" && { echo "mutant lenient changed nothing" >&2; exit 2; }
  expect_rejected 'a gate that lets a GAP through' "$lenient" gate_refuses 'BUILD FAILED'
fi
finish
