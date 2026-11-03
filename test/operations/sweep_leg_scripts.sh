#!/usr/bin/env bash
# Sourced by sweep_harness.sh: the diagnostic legs' own scripts (gh-ocannl-1705) -- the real
# tools/vendor-idents-check.sh, tools/cuda-tf32-parity.sh and tools/cuda-native-ptx-rerun.sh, beside
# the sweep under test -- run against a fake `dune` on PATH, which records each call's arguments and
# the environment the leg pins, and answers with a canned report and status. Each leg's verdict line
# and status are held to its documented table, the WARNING arm (status 1) firing exactly where its
# condition holds and staying quiet otherwise, and no credential reaches dune. No backend runs.
leg_tools=$(cd "$(dirname "$sweep")" && pwd)
leg_root=$tmp/leg-scripts
leg_bin=$tmp/leg-bin
mkdir -p "$leg_root/tools" "$leg_bin"
for leg in vendor-idents-check.sh cuda-tf32-parity.sh cuda-native-ptx-rerun.sh credential-env.sh; do
  cp "$leg_tools/$leg" "$leg_root/tools/$leg"
done
leg_calls=$tmp/leg-dune.calls
cat >"$leg_bin/dune" <<'EOF_LEG_DUNE'
#!/bin/sh
printf '%s | backend=%s ptx=%s tf32=%s parity=%s materialize=%s fixture=%s credentials=%s\n' "$*" \
  "${OCANNL_BACKEND:-}" "${OCANNL_CUDA_PTX_TARGET:-}" "${OCANNL_TF32_MATMULS:-}" \
  "${BENCH_PARITY:-}" "${BENCH_MATERIALIZE:-}" "${BENCH_FIXTURE:-}" \
  "${GH_TOKEN+GH_TOKEN}${FOO_API_KEY+FOO_API_KEY}" >>"$LEG_DUNE_CALLS"
# The parity's separate build step answers on its own.
case " $* " in
  " build --no-print-directory "*) exit "${LEG_DUNE_PREBUILD_RC:-0}" ;;
esac
[ -z "${LEG_DUNE_OUT:-}" ] || printf '%s\n' "$LEG_DUNE_OUT"
exit "${LEG_DUNE_RC:-0}"
EOF_LEG_DUNE
chmod +x "$leg_bin/dune"
run_leg() { # script args... -- its output in leg_out, its status in leg_rc
  # Unset what the legs pin: an ambient one (the native-PTX leg reruns this very harness) would
  # otherwise read as the leg's own.
  leg_rc=0
  leg_out=$(env -u OCANNL_BACKEND -u OCANNL_CUDA_PTX_TARGET -u OCANNL_TF32_MATMULS -u BENCH_PARITY \
    -u BENCH_MATERIALIZE -u BENCH_FIXTURE PATH="$leg_bin:$PATH" LEG_DUNE_CALLS="$leg_calls" \
    GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key \
    bash "$leg_root/tools/$1" "${@:2}" 2>&1) || leg_rc=$?
}
leg_verdict() { printf '%s\n' "$leg_out" | tail -n 1; }

# vendor-idents: the tool's own verdict and counts, its status mapped; a status with no verdict line
# of the tool's is a build's or a crash's, never a stale table.
: >"$leg_calls"
LEG_DUNE_OUT='fail, yet kernel_ident passes them through (missing from the table): 0
in the table, yet compile as kernel names (superfluous): 0
verdict: the cuda table holds for this toolchain' LEG_DUNE_RC=0 run_leg vendor-idents-check.sh cuda
[ "$leg_rc" -eq 0 ]
[ "$(leg_verdict)" = 'vendor-idents: verdict: holds (the cuda table holds for this toolchain)' ]
grep -qx 'exec --no-print-directory tools/vendor_idents.exe -- cuda | .* credentials=' "$leg_calls"
LEG_DUNE_OUT='fail, yet kernel_ident passes them through (missing from the table): 2
in the table, yet compile as kernel names (superfluous): 1
verdict: the hip table differs from this toolchain'"'"'s measurement' LEG_DUNE_RC=1 \
  run_leg vendor-idents-check.sh hip
[ "$leg_rc" -eq 1 ]
[ "$(leg_verdict)" = "vendor-idents: verdict: TABLE STALE (the hip table differs from this toolchain's measurement: 2 missing, 1 superfluous; regenerate arrayjit/lib/vendor_idents_hip.ml from this report)" ]
LEG_DUNE_OUT='Error: Library "cudajit" not found.' LEG_DUNE_RC=1 run_leg vendor-idents-check.sh cuda
[ "$leg_rc" -eq 2 ]
[ "$(leg_verdict)" = 'vendor-idents: verdict: unavailable: dune exec tools/vendor_idents.exe exited 1 before a verdict' ]
LEG_DUNE_OUT='verdict: no measurement of the cuda headers was taken' LEG_DUNE_RC=2 \
  run_leg vendor-idents-check.sh cuda
[ "$leg_rc" -eq 2 ]
[ "$(leg_verdict)" = 'vendor-idents: verdict: unavailable: no measurement of the cuda headers was taken' ]
absent 'credentials=[A-Z]' "$leg_calls"

# tf32 parity: the fixture first, then the build on its own, then the step under the pinned regime.
: >"$leg_calls"
leg_fixtures=$tmp/leg-fixtures
mkdir -p "$leg_fixtures"
run_leg cuda-tf32-parity.sh --fixture-dir "$leg_fixtures"
[ "$leg_rc" -eq 2 ]
[ "$(leg_verdict)" = "cuda-tf32-parity: verdict: unavailable: no $leg_fixtures/cifar_resnet.safetensors" ]
[ ! -s "$leg_calls" ]
: >"$leg_fixtures/cifar_resnet.safetensors"
LEG_DUNE_OUT='parity seed 1 F_sketch[x]: MATCH worst 1e-06
parity (seeds as proposed): backend cuda, 32 seeds, 0 failed at tolerance 0.001' LEG_DUNE_RC=0 \
  run_leg cuda-tf32-parity.sh --fixture-dir "$leg_fixtures"
[ "$leg_rc" -eq 0 ]
[ "$(leg_verdict)" = 'cuda-tf32-parity: verdict: holds ((seeds as proposed): backend cuda, 32 seeds, 0 failed at tolerance 0.001)' ]
grep -qx "exec --no-build --no-print-directory benchmarks/runners/ocannl/bench_conv_diag.exe | backend=cuda ptx= tf32=true parity=1 materialize=1 fixture=$leg_fixtures/cifar_resnet.safetensors credentials=" \
  "$leg_calls"
LEG_DUNE_OUT='autotune-seeding: census only' LEG_DUNE_RC=0 \
  run_leg cuda-tf32-parity.sh --fixture-dir "$leg_fixtures"
[ "$leg_rc" -eq 1 ]
[ "$(leg_verdict)" = 'cuda-tf32-parity: verdict: NO PARITY SUMMARY (bench_conv_diag exited 0 without a `parity (` line)' ]
LEG_DUNE_OUT='parity seed 1 F_sketch[x]: MISMATCH worst 0.7
parity (seeds as proposed): backend cuda, 32 seeds, 1 failed at tolerance 0.001' LEG_DUNE_RC=4 \
  run_leg cuda-tf32-parity.sh --fixture-dir "$leg_fixtures"
[ "$leg_rc" -eq 1 ]
[ "$(leg_verdict)" = 'cuda-tf32-parity: verdict: SEED PARITY FAILED ((seeds as proposed): backend cuda, 32 seeds, 1 failed at tolerance 0.001)' ]
LEG_DUNE_OUT='parity (seeds as proposed): backend cuda, 0 seeds' LEG_DUNE_RC=5 \
  run_leg cuda-tf32-parity.sh --fixture-dir "$leg_fixtures"
[ "$leg_rc" -eq 1 ]
[ "$(leg_verdict)" = 'cuda-tf32-parity: verdict: NO SEED EXECUTED ((seeds as proposed): backend cuda, 0 seeds)' ]
LEG_DUNE_OUT='Fatal error: exception Shape_error: occurs check failed
Raised at Fixture' LEG_DUNE_RC=2 run_leg cuda-tf32-parity.sh --fixture-dir "$leg_fixtures"
[ "$leg_rc" -eq 1 ]
[ "$(leg_verdict)" = 'cuda-tf32-parity: verdict: STEP BROKE (bench_conv_diag exited 2: Fatal error: exception Shape_error: occurs check failed)' ]
: >"$leg_calls"
LEG_DUNE_PREBUILD_RC=1 run_leg cuda-tf32-parity.sh --fixture-dir "$leg_fixtures"
[ "$leg_rc" -eq 2 ]
[ "$(leg_verdict)" = 'cuda-tf32-parity: verdict: unavailable: bench_conv_diag.exe did not build' ]
absent 'exec --no-build' "$leg_calls"

# native ptx: the whole suite (by default) at the native target, its failing files named.
: >"$leg_calls"
LEG_DUNE_RC=0 run_leg cuda-native-ptx-rerun.sh
[ "$leg_rc" -eq 0 ]
[ "$(leg_verdict)" = 'cuda-native-ptx: verdict: holds (dune build @runtest @train: the suite passes at the native PTX target)' ]
grep -qx 'build @runtest @train | backend=cuda ptx=native tf32= parity= materialize= fixture= credentials=' \
  "$leg_calls"
LEG_DUNE_OUT='File "test/operations/a.expected", line 1, characters 0-0:
File "test/operations/a.expected", line 9, characters 0-0:
File "test/operations/dune", line 3, characters 0-60:' LEG_DUNE_RC=1 \
  run_leg cuda-native-ptx-rerun.sh -j 3 @test/operations/runtest
[ "$leg_rc" -eq 1 ]
[ "$(leg_verdict)" = 'cuda-native-ptx: verdict: NATIVE TARGET RED (dune exited 1; 2 failing files: test/operations/a.expected, test/operations/dune)' ]
grep -q '^build -j 3 @test/operations/runtest | backend=cuda ptx=native ' "$leg_calls"
LEG_DUNE_OUT='File "a", line 1:
File "b", line 1:
File "c", line 1:
File "d", line 1:' LEG_DUNE_RC=1 run_leg cuda-native-ptx-rerun.sh
[ "$(leg_verdict)" = 'cuda-native-ptx: verdict: NATIVE TARGET RED (dune exited 1; 4 failing files: a, b, c, ...)' ]
absent 'credentials=[A-Z]' "$leg_calls"
printf 'sweep diagnostic leg scripts: PASS\n'
