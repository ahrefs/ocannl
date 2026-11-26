#!/usr/bin/env bash
# ocannl-harness: standalone
# Controls for benchmarks/check-box.sh (gh-ocannl-1062): a fabricated box that passes every item
# reads READY with exit 0, and each item's gap -- no venv, a failed import, a drifted or missing
# requirements record, a torch build for the wrong device, a device that refuses, crashes or hangs,
# no Python.h, drivers resolving another interpreter, no toolchain, an absent or unrecorded
# fixture, a fixture name the box has no DIGESTS row for (by name, without the bytes) -- reads GAP on
# that item with exit 1. Usage errors exit 2, and every box the checked-in DIGESTS.txt declares has
# a profile (the relationship, not a copy of the box list).
# gh-ocannl-2004: the default kernel's initrd and the dpkg states, against a stand-in Linux every
# leg sees (uname, /boot, dpkg), so the item reads alike on any host; skip off Linux and in WSL.
# Generated-input readiness checks the same device and toolchain on a box with no fixture store.
# The absent-fixture control covers fixture-loading readiness when recorded bytes are missing.
# gh-ocannl-1433: a fresh worktree's drivers find the box's shared venv (route named, and its
# absence still a GAP), and the fleet's names for the boxes are taken -- pinned against wake-lab.sh's
# endpoint map and FLEET_BOXES where this host can read them; a fleet box that measures nothing
# (mapped to `-` in fleet_row) is refused by name.
# Usage: benchmarks/test-check-box.sh [--keep|--only REGEX|--help]
#
# The subject runs in place against a scratch venv (python3 -m venv, so it has a sys.prefix of
# its own) whose site-packages hold stub numpy/safetensors/tinygrad/torch/pip modules; FAKE_TORCH
# picks the stub torch's behaviour, and a .pth hook points INCLUDEPY at a scratch Python.h.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3 perl
harness_scratch test-check-box
SUBJECT=$here/check-box.sh

venv=$TMP/venv
python3 -m venv --without-pip "$venv" >/dev/null 2>&1 || {
  skip "every leg" "python3 -m venv cannot build a scratch venv here"
  finish
}
site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || exit 2
mkdir -p "$site/pip" "$TMP/include" "$TMP/bin" "$TMP/fixtures" "$TMP/runs"
: >"$TMP/include/Python.h"
for m in numpy safetensors tinygrad; do
  printf '__version__ = "1.0"\n' >"$site/$m.py"
done
for d in numpy:2.5.3 safetensors:0.8.0 tinygrad:0.14.0 torch:2.14.0; do
  mkdir -p "$site/${d%%:*}-${d#*:}.dist-info"
  printf 'Metadata-Version: 2.1\nName: %s\nVersion: %s\n' "${d%%:*}" "${d#*:}" \
    >"$site/${d%%:*}-${d#*:}.dist-info/METADATA"
done
cat >"$site/torch.py" <<'PY'
import os, time, types
__version__ = "2.14.0+fake"
mode = os.environ.get("FAKE_TORCH", "cuda")
if mode == "import-error":
    raise ImportError("libtorch_fake.so: cannot open shared object file")
version = types.SimpleNamespace(
    cuda="13.0" if mode in ("cuda", "unavailable", "crash", "hang") else None,
    hip="7.2" if mode == "hip" else None)
class _T:
    def cuda(self):
        if mode == "crash":
            os.kill(os.getpid(), 11)
        if mode == "hang":
            time.sleep(3600)
        return self
    def cpu(self):
        return self
    def item(self):
        return 1.0
def ones(n, device=None):
    return _T()
cuda = types.SimpleNamespace(is_available=lambda: mode != "unavailable",
                             synchronize=lambda: None, get_device_name=lambda i: "Fake GPU")
backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mode == "mps"))
mps = types.SimpleNamespace(synchronize=lambda: None)
PY
printf 'import os, sys\nsys.stdout.write(open(os.environ["FAKE_FREEZE"]).read())\n' >"$site/pip/__main__.py"
: >"$site/pip/__init__.py"
# A .pth import line rather than a sitecustomize, which a distribution's own (Homebrew's) shadows.
printf 'import sysconfig, os\n_g = sysconfig.get_config_var\nsysconfig.get_config_var = lambda k: os.environ.get("FAKE_INCLUDEPY", _g(k)) if k == "INCLUDEPY" else _g(k)\n' \
  >"$site/_fake_includepy.py"
echo 'import _fake_includepy' >"$site/_fake_includepy.pth"
printf 'numpy==2.5.3\nsafetensors==0.8.0\ntinygrad==0.14.0\ntorch==2.14.0\n' >"$venv.requirements.txt"
cp "$venv.requirements.txt" "$TMP/freeze"
for tool in nvcc hipcc; do
  printf '#!/bin/sh\necho "%s fake, release 1.0"\n' "$tool" >"$TMP/bin/$tool"
done
printf '#!/bin/sh\necho /fake/MacOSX.sdk\n' >"$TMP/bin/xcrun"
# The default-kernel item reads `uname -s`, `/boot` and `dpkg -l`: every leg sees a stand-in Linux
# box with a clean /boot and dpkg database, whatever this host is, so the item reads ok on macOS
# and on a CI runner alike; its own legs change FAKE_UNAME, FAKE_DPKG and the boot directory.
real_uname=$(command -v uname) || exit 2
printf '#!/bin/sh\n[ $# = 1 ] && case $1 in\n  -s) echo "${FAKE_UNAME:-Linux}"; exit 0 ;;\n  -r) echo "${FAKE_UNAME_R:-7.0.0-43-generic}"; exit 0 ;;\nesac\nexec %s "$@"\n' \
  "$real_uname" >"$TMP/bin/uname"
printf '#!/bin/sh\n[ "$1" = -l ] || exit 2\ncat "$FAKE_DPKG"\n' >"$TMP/bin/dpkg"
chmod +x "$TMP/bin/"*
# [fake_boot DIR VERSION [initrd|no-initrd|dangling]]: Ubuntu's /boot layout for VERSION.
fake_boot() {
  mkdir -p "$1"
  [ "${3:-initrd}" = dangling ] || echo kernel >"$1/vmlinuz-$2"
  ln -sfn "vmlinuz-$2" "$1/vmlinuz"
  ln -sfn "initrd.img-$2" "$1/initrd.img"
  [ "${3:-initrd}" != initrd ] || echo initramfs >"$1/initrd.img-$2"
}
fake_boot "$TMP/boot" 7.0.0-43-generic
# A database as `dpkg -l` prints it: headers, an installed package, a removed one's leftovers.
cat >"$TMP/dpkg-clean" <<'DPKG'
Desired=Unknown/Install/Remove/Purge/Hold
| Status=Not/Inst/Conf-files/Unpacked/halF-conf/Half-inst/trig-aWait/Trig-pend
|/ Err?=(none)/Reinst-required (Status,Err: uppercase=bad)
||/ Name                          Version        Architecture Description
+++-=============================-==============-============-==============================
ii  linux-image-7.0.0-43-generic  7.0.0-43.43    amd64        Signed kernel image generic
rc  linux-image-7.0.0-31-generic  7.0.0-31.31    amd64        Signed kernel image generic
DPKG

# One tiny valid safetensors file under a real workload's name, recorded as rog-nv's bytes in a
# scratch DIGESTS that declares the real file's boxes, and as minix's, m4-max's and m4-pro's copies
# of them -- so tuf is the declared box with no row for it (the by-name check's gap).
python3 - "$TMP/fixtures/mlp_small.safetensors" <<'PY' || exit 2
import json, struct, sys
head = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                   "__metadata__": {"k": "v"}}).encode()
open(sys.argv[1], "wb").write(struct.pack("<Q", len(head)) + head + struct.pack("<f", 1.0))
PY
{ grep '^# measurement-boxes:' "$here/fixtures/DIGESTS.txt"; echo '#'; } >"$TMP/DIGESTS.txt"
python3 "$here/fixture_digest.py" --record --origin rog-nv --digests "$TMP/DIGESTS.txt" \
  "$TMP/fixtures/mlp_small.safetensors" >/dev/null || exit 2
# rog-nv's row alone, for the leg whose DIGESTS stops declaring minix.
cp "$TMP/DIGESTS.txt" "$TMP/DIGESTS-rog-only.txt"
for b in minix m4-max m4-pro; do
  python3 "$here/fixture_digest.py" --record-matching --origin "$b" --digests "$TMP/DIGESTS.txt" \
    "$TMP/fixtures/mlp_small.safetensors" >/dev/null || exit 2
done

export FAKE_FREEZE=$TMP/freeze FAKE_INCLUDEPY=$TMP/include BENCH_VENV_PY=$venv/bin/python
export PATH="$TMP/bin:$PATH" OCANNL_TOOL_CHECK_BOX_DEVICE_CAP=5
export OCANNL_TOOL_CHECK_BOX_BOOT_DIR=$TMP/boot FAKE_DPKG=$TMP/dpkg-clean
unset FAKE_UNAME FAKE_UNAME_R
unset FAKE_TORCH

# [check RUN ARG...]: the subject on the scratch box; output in runs/RUN, status in $rc.
check() {
  local run=$1
  shift
  rc=0
  "$SUBJECT" --venv "$venv" --fixture-dir "$TMP/fixtures" --digests "$TMP/DIGESTS.txt" "$@" \
    >"$TMP/runs/$run" 2>&1 || rc=$?
}
# [gap_leg LABEL RUN ITEM] after a check: exit 1, ITEM's line says GAP, and the verdict counts it.
gap_leg() {
  if [ "$rc" = 1 ] && grep -q "^check-box: $3: GAP -- " "$TMP/runs/$2" &&
     grep -q '^check-box: [a-z0-9-]*: GAP ([1-9][0-9]* of [0-9]* items)$' "$TMP/runs/$2"; then
    report 0 "$1"
  else
    report 1 "$1" "rc=$rc; see $TMP/runs/$2 (--keep)"
    sed 's/^/      | /' "$TMP/runs/$2"
  fi
}
# [usage_leg LABEL RUN PATTERN] after a check: exit 2 with PATTERN on its output.
usage_leg() {
  if [ "$rc" = 2 ] && grep -qE -- "$3" "$TMP/runs/$2"; then report 0 "$1"
  else report 1 "$1" "rc=$rc; see $TMP/runs/$2"; sed 's/^/      | /' "$TMP/runs/$2"; fi
}

if harness_leg 'ready box'; then
  check ready --box rog-nv
  if [ "$rc" = 0 ] && ! grep -q 'GAP' "$TMP/runs/ready" &&
     [ "$(grep -c ': ok -- ' "$TMP/runs/ready")" = 10 ] &&
     grep -qx "check-box: fixture-records: ok -- $TMP/DIGESTS.txt has a row of rog-nv's for each of the 1 fixture names checked (by name)" \
       "$TMP/runs/ready" &&
     tail -n 1 "$TMP/runs/ready" | grep -qx 'check-box: rog-nv: READY (10 items)'; then
    report 0 'a box meeting every item is READY, exit 0'
  else
    report 1 'a box meeting every item is READY, exit 0' "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/ready"
  fi
fi
if harness_leg 'box-only'; then
  check box-only --box rog-nv --box-only
  if [ "$rc" = 0 ] && grep -q '^check-box: driver-path: skip -- ' "$TMP/runs/box-only"; then
    report 0 '--box-only skips the driver path and stays READY'
  else report 1 '--box-only skips the driver path and stays READY' "rc=$rc"; fi
fi

if harness_leg 'generated inputs'; then
  generated=$TMP/generated/benchmarks
  mkdir -p "$generated"
  cp "$SUBJECT" "$here/bench_venv.py" "$here/fixture_digest.py" "$generated/"
  grep '^# measurement-boxes:' "$here/fixtures/DIGESTS.txt" >"$TMP/DIGESTS-boxes.txt"
  generated_check() {
    local run=$1
    shift
    rc=0
    bash "$generated/check-box.sh" --venv "$venv" --digests "$TMP/DIGESTS-boxes.txt" "$@" \
      --box rog-nv --generated-inputs >"$TMP/runs/$run" 2>&1 || rc=$?
  }
  check generated-reference --box rog-nv --fixture mlp_small
  fixture_rc=$rc
  fixture_items=$(grep -c '^check-box: .*: ok -- ' "$TMP/runs/generated-reference")
  generated_check generated
  # The fixture item disappears and fixture-records changes from ok to skip.
  if [ "$fixture_rc" = 0 ] && [ "$rc" = 0 ] &&
     [ "$(grep -c '^check-box: .*: ok -- ' "$TMP/runs/generated")" = "$((fixture_items - 2))" ] &&
     grep -q '^check-box: torch-device: ok -- .*round-tripped through cuda' "$TMP/runs/generated" &&
     grep -q '^check-box: toolchain: ok -- .*nvcc' "$TMP/runs/generated" &&
     grep -q '^check-box: fixture-records: skip -- ' "$TMP/runs/generated" &&
     ! grep -q '^check-box: fixture ' "$TMP/runs/generated" &&
     grep -Fqx "check-box: rog-nv: READY ($((fixture_items - 1)) items)" "$TMP/runs/generated"; then
    report 0 'generated inputs are READY without a fixture store or digest rows'
  else
    report 1 'generated inputs are READY without a fixture store or digest rows' \
      "generated rc=$rc; fixture reference rc=$fixture_rc"
    sed 's/^/      | /' "$TMP/runs/generated"
  fi
  FAKE_TORCH=unavailable generated_check generated-device
  gap_leg 'generated inputs still need a working device' generated-device torch-device
  generated_check generated-fixture --fixture mlp_small
  usage_leg '--generated-inputs refuses --fixture' generated-fixture 'cannot combine .*--fixture'
  generated_check generated-store --fixture-dir "$TMP/fixtures"
  usage_leg '--generated-inputs refuses --fixture-dir' generated-store 'cannot combine .*--fixture-dir'
fi

if harness_leg 'gap: no venv'; then
  rc=0
  "$SUBJECT" --venv "$TMP/absent" --fixture-dir "$TMP/fixtures" --digests "$TMP/DIGESTS.txt" \
    --box rog-nv >"$TMP/runs/no-venv" 2>&1 || rc=$?
  gap_leg 'no venv is a venv GAP' no-venv venv
  if grep -q '^check-box: torch-device: skip -- no venv' "$TMP/runs/no-venv" &&
     grep -q 'install-linux.sh --bench-venv nvidia' "$TMP/runs/no-venv"; then
    report 0 'its dependants skip, and the repair names the provisioning'
  else report 1 'its dependants skip, and the repair names the provisioning'; fi
fi
if harness_leg 'gap: import'; then
  FAKE_TORCH=import-error check import --box rog-nv
  gap_leg 'a failed import is a packages GAP' import packages
fi
if harness_leg 'gap: requirements drift'; then
  printf 'numpy==2.5.3\nsafetensors==0.8.0\ntinygrad==0.14.0\ntorch==2.13.0\n' >"$TMP/freeze-drift"
  FAKE_FREEZE=$TMP/freeze-drift check drift --box rog-nv
  gap_leg 'a freeze differing from the record is a requirements GAP' drift requirements
  if grep -qx '    < torch==2.14.0' "$TMP/runs/drift" && grep -qx '    > torch==2.13.0' "$TMP/runs/drift"; then
    report 0 'the drift names both sides'
  else report 1 'the drift names both sides'; fi
fi
if harness_leg 'requirements: editable spelling'; then
  printf -- '-e /src/tinygrad\n' >>"$venv.requirements.txt"
  { cat "$TMP/freeze"; printf -- '-e file:///src/tinygrad\n'; } >"$TMP/freeze-editable"
  FAKE_FREEZE=$TMP/freeze-editable check editable --box rog-nv
  if [ "$rc" = 0 ]; then report 0 "uv's file:// editable spelling equals pip's"
  else report 1 "uv's file:// editable spelling equals pip's" "rc=$rc"; fi
  FAKE_FREEZE=$TMP/freeze check editable-missing --box rog-nv
  gap_leg 'a recorded editable that is not installed is still a gap' editable-missing requirements
  cp "$TMP/freeze" "$venv.requirements.txt"
fi
if harness_leg 'gap: no requirements record'; then
  mv "$venv.requirements.txt" "$TMP/record-aside"
  check no-record --box rog-nv
  mv "$TMP/record-aside" "$venv.requirements.txt"
  gap_leg 'no recorded freeze is a requirements GAP' no-record requirements
fi
for mode in cpu-build unavailable crash hang; do
  if harness_leg "gap: device $mode"; then
    FAKE_TORCH=$mode check "device-$mode" --box rog-nv
    gap_leg "a $mode torch is a torch-device GAP" "device-$mode" torch-device
    if [ "$mode" = hang ]; then
      grep -q 'no answer within 5s' "$TMP/runs/device-hang"
      report $? 'a hung device smoke is cut at the cap and says so'
    fi
  fi
done
if harness_leg 'device: hip and mps profiles'; then
  FAKE_TORCH=hip check hip --box minix --fixture mlp_small
  if [ "$rc" = 0 ] && grep -q 'round-tripped through hip 7.2' "$TMP/runs/hip"; then
    report 0 'a hip box needs a hip build'
  else report 1 'a hip box needs a hip build' "rc=$rc"; fi
  FAKE_TORCH=cuda check hip-on-cuda --box tuf --fixture mlp_small
  gap_leg 'a cuda build on a hip box is a GAP' hip-on-cuda torch-device
  FAKE_TORCH=mps check mps --box m4-max --fixture mlp_small
  if [ "$rc" = 0 ] && grep -q '^check-box: toolchain: ok -- macOS SDK /fake/MacOSX.sdk' "$TMP/runs/mps"; then
    report 0 'an mps box reads the macOS SDK as its toolchain'
  else report 1 'an mps box reads the macOS SDK as its toolchain' "rc=$rc"; fi
  FAKE_TORCH=mps check mps-pro --box m4-pro --fixture mlp_small
  if [ "$rc" = 0 ] && grep -q '^check-box: toolchain: ok -- macOS SDK /fake/MacOSX.sdk' "$TMP/runs/mps-pro"; then
    report 0 'm4-pro uses the mps and macOS SDK profile'
  else report 1 'm4-pro uses the mps and macOS SDK profile' "rc=$rc"; fi
fi
if harness_leg 'gap: python headers'; then
  FAKE_INCLUDEPY=$TMP/absent check headers --box rog-nv
  gap_leg 'no Python.h is a python-headers GAP' headers python-headers
fi
if harness_leg 'gap: driver path'; then
  BENCH_VENV_PY=$(command -v python3) check driver --box rog-nv
  gap_leg 'drivers resolving another interpreter is a driver-path GAP' driver driver-path
fi
# A fresh worktree: no BENCH_VENV_PY and no benchmarks/.venv, from a copy of the subject in a tree
# of its own (this checkout may have a .venv link, which outranks the shared venv), with HOME
# holding the shared venv as a link to the scratch one -- or holding nothing.
if harness_leg 'driver path: the shared venv'; then
  fresh=$TMP/fresh/benchmarks
  mkdir -p "$fresh/workloads" "$TMP/home/.venvs" "$TMP/home-bare"
  cp "$SUBJECT" "$here/bench_venv.py" "$here/fixture_digest.py" "$fresh/"
  cp "$here"/workloads/*.json "$fresh/workloads/"
  ln -s "$venv" "$TMP/home/.venvs/ocannl-bench"
  fresh_check() { # RUN HOME
    rc=0
    env -u BENCH_VENV_PY HOME="$2" bash "$fresh/check-box.sh" --venv "$venv" --fixture-dir "$TMP/fixtures" \
      --digests "$TMP/DIGESTS.txt" --box rog-nv >"$TMP/runs/$1" 2>&1 || rc=$?
  }
  fresh_check fresh-shared "$TMP/home"
  if [ "$rc" = 0 ] &&
     grep -q '^check-box: driver-path: ok -- drivers here run .*/home/\.venvs/ocannl-bench/bin/python (route: shared venv .*/home/\.venvs/ocannl-bench)$' \
       "$TMP/runs/fresh-shared"; then
    report 0 'a fresh worktree reaches the shared venv, and the ok line names that route'
  else
    report 1 'a fresh worktree reaches the shared venv, and the ok line names that route' "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/fresh-shared"
  fi
  fresh_check fresh-bare "$TMP/home-bare"
  gap_leg 'a fresh worktree on a box with no shared venv is still a driver-path GAP' fresh-bare driver-path
  grep -q '(route: none)' "$TMP/runs/fresh-bare"
  report $? 'its GAP line says no route resolved'
fi
if harness_leg 'fixture records: judged nothing'; then
  # A by-name check that cannot answer (here: a workload spec it cannot read, which the subject's
  # own spec listing never parses) exits 2 before any item. fixture_digest.py exits 2 on such an
  # unexpected error (gh-ocannl-1797), its last line naming the error, which check-box quotes.
  broken=$TMP/broken/benchmarks
  mkdir -p "$broken/workloads"
  cp "$SUBJECT" "$here/bench_venv.py" "$here/fixture_digest.py" "$broken/"
  cp "$here"/workloads/*.json "$broken/workloads/"
  printf '{ not json\n' >"$broken/workloads/zz_broken.json"
  rc=0
  bash "$broken/check-box.sh" --venv "$venv" --fixture-dir "$TMP/fixtures" --digests "$TMP/DIGESTS.txt" \
    --box rog-nv >"$TMP/runs/records-broken" 2>&1 || rc=$?
  usage_leg 'a by-name check that judged nothing exits 2' records-broken \
    '^check-box: fixture_digest.py --check-names --expect-recorded rog-nv judged nothing \(exit 2\): fixture_digest.py: unexpected error, exit 2 -- not a verdict about any fixture: ValueError: '
  ! grep -q '^check-box: [a-z-]*: \(ok\|GAP\|skip\) -- ' "$TMP/runs/records-broken"
  report $? 'and prints no item first'
fi

if harness_leg 'gap: toolchain'; then
  if [ -e /usr/bin/nvcc ] || [ -e /bin/nvcc ]; then
    skip 'no nvcc is a toolchain GAP' 'this host has nvcc on /usr/bin or /bin'
  else
    mkdir -p "$TMP/bin2"
    for t in python3 perl; do ln -s "$(command -v "$t")" "$TMP/bin2/$t"; done
    PATH="$TMP/bin2:/usr/bin:/bin" check toolchain --box rog-nv
    gap_leg 'no nvcc is a toolchain GAP' toolchain toolchain
  fi
  mkdir -p "$TMP/bin3"
  printf '#!/bin/sh\necho "nvcc: error while loading shared libraries" >&2\nexit 127\n' >"$TMP/bin3/nvcc"
  chmod +x "$TMP/bin3/nvcc"
  PATH="$TMP/bin3:$PATH" check broken-toolchain --box rog-nv
  gap_leg 'an nvcc that cannot run is a toolchain GAP' broken-toolchain toolchain
fi

# gh-ocannl-2004: rog's half-configured kernel package left its default kernel without an initrd.
if harness_leg 'default kernel'; then
  check kernel-clean --box rog-nv
  if [ "$rc" = 0 ] &&
     grep -Fqx "check-box: default-kernel: ok -- 7.0.0-43-generic has $TMP/boot/initrd.img-7.0.0-43-generic; no half-configured or half-installed package" \
       "$TMP/runs/kernel-clean"; then
    report 0 'a default kernel with its initrd and a clean dpkg database is ok'
  else
    report 1 'a default kernel with its initrd and a clean dpkg database is ok' "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/kernel-clean"
  fi

  # The 2026-11-06 fault: vmlinuz and initrd.img point at -42, whose initrd was never built.
  fake_boot "$TMP/boot-no-initrd" 7.0.0-43-generic
  fake_boot "$TMP/boot-no-initrd" 7.0.0-42-generic no-initrd
  OCANNL_TOOL_CHECK_BOX_BOOT_DIR=$TMP/boot-no-initrd check kernel-no-initrd --box rog-nv
  gap_leg 'a default kernel whose initrd.img link dangles is a default-kernel GAP' kernel-no-initrd default-kernel
  if grep -Fq "the default kernel 7.0.0-42-generic has no $TMP/boot-no-initrd/initrd.img-7.0.0-42-generic" \
       "$TMP/runs/kernel-no-initrd" &&
     grep -q '^  repair: needs root on the box.*sudo dpkg --configure -a' "$TMP/runs/kernel-no-initrd"; then
    report 0 'its line names the kernel and the missing initrd, and its repair needs root'
  else report 1 'its line names the kernel and the missing initrd, and its repair needs root'; fi

  { cat "$TMP/dpkg-clean"
    echo 'iF  linux-image-7.0.0-42-generic  7.0.0-42.42    amd64        Signed kernel image generic'
  } >"$TMP/dpkg-half-configured"
  FAKE_DPKG=$TMP/dpkg-half-configured check kernel-iF --box rog-nv
  gap_leg 'a half-configured (iF) package is a default-kernel GAP' kernel-iF default-kernel
  if grep -q '^check-box: default-kernel: GAP -- dpkg -l shows packages a maintainer script left unfinished: linux-image-7.0.0-42-generic (iF)$' \
       "$TMP/runs/kernel-iF"; then
    report 0 'its line names the package and its state'
  else report 1 'its line names the package and its state'; fi

  { cat "$TMP/dpkg-clean"
    echo 'iH  nvidia-dkms-open              620.70.06-1    amd64        NVIDIA DKMS package'
  } >"$TMP/dpkg-half-installed"
  FAKE_DPKG=$TMP/dpkg-half-installed check kernel-iH --box rog-nv
  gap_leg 'a half-installed (iH) package is a default-kernel GAP' kernel-iH default-kernel
  if grep -q 'nvidia-dkms-open (iH)' "$TMP/runs/kernel-iH" &&
     grep -q '^  repair: .*half-installed (?H) package is reinstalled' "$TMP/runs/kernel-iH"; then
    report 0 'its repair reinstalls a half-installed package rather than only configuring it'
  else report 1 'its repair reinstalls a half-installed package rather than only configuring it'; fi

  fake_boot "$TMP/boot-dangling" 7.0.0-44-generic dangling
  OCANNL_TOOL_CHECK_BOX_BOOT_DIR=$TMP/boot-dangling check kernel-dangling --box rog-nv
  gap_leg 'a vmlinuz link to no kernel image is a default-kernel GAP' kernel-dangling default-kernel
  # The link names an image outside /boot that is gone, while /boot holds one of that name and
  # its initrd: the check follows the link, not the name.
  fake_boot "$TMP/boot-elsewhere" 7.0.0-43-generic
  ln -sfn "$TMP/elsewhere/vmlinuz-7.0.0-43-generic" "$TMP/boot-elsewhere/vmlinuz"
  OCANNL_TOOL_CHECK_BOX_BOOT_DIR=$TMP/boot-elsewhere check kernel-elsewhere --box rog-nv
  gap_leg 'a link to an absent image elsewhere is a GAP, whatever /boot holds under its name' \
    kernel-elsewhere default-kernel
  OCANNL_TOOL_CHECK_BOX_BOOT_DIR=$TMP/absent-boot check kernel-no-link --box rog-nv
  gap_leg 'no vmlinuz link names no default kernel: a GAP, not a pass' kernel-no-link default-kernel

  # Off Linux the item judges nothing, even over a /boot that would fail.
  FAKE_UNAME=Darwin OCANNL_TOOL_CHECK_BOX_BOOT_DIR=$TMP/boot-no-initrd check kernel-darwin --box rog-nv
  if [ "$rc" = 0 ] &&
     grep -qx 'check-box: default-kernel: skip -- not Linux (Darwin): no /boot or dpkg to check' "$TMP/runs/kernel-darwin"; then
    report 0 'on another OS the item reads skip and the box stays READY'
  else
    report 1 'on another OS the item reads skip and the box stays READY' "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/kernel-darwin"
  fi
  # A WSL guest boots the kernel the Windows host picks: its /boot judges nothing (the sweep's
  # -wsl lanes).
  FAKE_UNAME_R=6.18.33.2-microsoft-standard-WSL2 OCANNL_TOOL_CHECK_BOX_BOOT_DIR=$TMP/absent-boot \
    check kernel-wsl --box rog-nv
  if [ "$rc" = 0 ] &&
     grep -qx 'check-box: default-kernel: skip -- a WSL guest (kernel 6.18.33.2-microsoft-standard-WSL2): the Windows host picks its kernel, not /boot' \
       "$TMP/runs/kernel-wsl"; then
    report 0 'in a WSL guest the item reads skip and the box stays READY'
  else
    report 1 'in a WSL guest the item reads skip and the box stays READY' "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/kernel-wsl"
  fi
fi
if harness_leg 'gap: fixtures'; then
  mkdir -p "$TMP/empty-store"
  check absent --box rog-nv --fixture mlp_small --fixture-dir "$TMP/empty-store"
  gap_leg 'an absent fixture is a GAP' absent 'fixture mlp_small'
  cp "$TMP/fixtures/mlp_small.safetensors" "$TMP/mlp_small.orig"
  printf 'x' >>"$TMP/fixtures/mlp_small.safetensors"
  check mismatch --box rog-nv
  cp "$TMP/mlp_small.orig" "$TMP/fixtures/mlp_small.safetensors"
  gap_leg 'unrecorded bytes are a GAP' mismatch 'fixture mlp_small'
  # minix's row records the bytes rog-nv's does; once the file changes, minix's row no longer
  # describes it, and a new rog-nv row makes it rog-nv's bytes alone (rog-nv's lenet after the
  # 2026-10-12 copy, in miniature).
  cp -R "$TMP/fixtures" "$TMP/fixtures-other"
  cp "$TMP/DIGESTS.txt" "$TMP/DIGESTS-other.txt"
  python3 - "$TMP/fixtures-other/mlp_small.safetensors" <<'PY2' || exit 2
import json, struct, sys
head = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                   "__metadata__": {"k": "other"}}).encode()
open(sys.argv[1], "wb").write(struct.pack("<Q", len(head)) + head + struct.pack("<f", 1.0))
PY2
  python3 "$here/fixture_digest.py" --record --origin rog-nv --digests "$TMP/DIGESTS-other.txt" \
    "$TMP/fixtures-other/mlp_small.safetensors" >/dev/null || exit 2
  rc=0
  FAKE_TORCH=hip "$SUBJECT" --venv "$venv" --fixture-dir "$TMP/fixtures-other" \
    --digests "$TMP/DIGESTS-other.txt" --box minix --fixture mlp_small >"$TMP/runs/other-origin" 2>&1 || rc=$?
  if [ "$rc" = 0 ] &&
     grep -q "^check-box: fixture mlp_small: ok -- .*(not minix's recorded bytes)$" "$TMP/runs/other-origin"; then
    report 0 "another box's recorded bytes, under a name the box has a row for, are measurable and say whose"
  else
    report 1 "another box's recorded bytes, under a name the box has a row for, are measurable and say whose" "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/other-origin"
  fi
fi
if harness_leg 'gap: fixture records'; then
  # tuf has no row for mlp_small: named with --fixture, it is a fixture-records GAP although its
  # bytes are measurable (another box's), and the repair is --record-matching.
  FAKE_TORCH=hip check records-named --box tuf --fixture mlp_small
  gap_leg 'a named fixture the box has no DIGESTS row for is a fixture-records GAP' records-named fixture-records
  if grep -qx "check-box: fixture-records: GAP -- $TMP/DIGESTS.txt has no row of tuf's for: mlp_small" \
       "$TMP/runs/records-named" &&
     grep -q '^  repair: .*fixture_digest.py --record-matching --origin tuf ' "$TMP/runs/records-named" &&
     grep -q "^check-box: fixture mlp_small: ok -- .*(not tuf's recorded bytes)$" "$TMP/runs/records-named" &&
     tail -n 1 "$TMP/runs/records-named" | grep -qx 'check-box: tuf: GAP (1 of 10 items)'; then
    report 0 'it names the unrecorded fixture and the record-matching repair, and is the only gap'
  else
    report 1 'it names the unrecorded fixture and the record-matching repair, and is the only gap'
    sed 's/^/      | /' "$TMP/runs/records-named"
  fi
  # Without --fixture, the required fixtures are the box's recorded names; a workload's fixture in
  # the store that the box has no row for (a copy that landed, never recorded) is the gap.
  cp "$TMP/fixtures/mlp_small.safetensors" "$TMP/fixtures/mlp_wide.safetensors"
  check records-store --box rog-nv
  rm -f "$TMP/fixtures/mlp_wide.safetensors"
  gap_leg "a store copy the box has no row for is a fixture-records GAP, without --fixture" records-store fixture-records
  if grep -qx "check-box: fixture-records: GAP -- $TMP/DIGESTS.txt has no row of rog-nv's for: mlp_wide" \
       "$TMP/runs/records-store" && ! grep -q '^check-box: fixture mlp_wide:' "$TMP/runs/records-store"; then
    report 0 'it names the store copy, which is no required fixture of its own'
  else
    report 1 'it names the store copy, which is no required fixture of its own'
    sed 's/^/      | /' "$TMP/runs/records-store"
  fi
  # The by-name check reads names, never bytes: a named fixture with no row anywhere and no file is
  # both gaps, each on its own item.
  check records-absent --box rog-nv --fixture mlp_small --fixture lenet
  if grep -q '^check-box: fixture-records: GAP -- .* for: lenet$' "$TMP/runs/records-absent" &&
     grep -q '^check-box: fixture lenet: GAP -- absent' "$TMP/runs/records-absent"; then
    report 0 'a named fixture with neither bytes nor a row is both gaps'
  else
    report 1 'a named fixture with neither bytes nor a row is both gaps'
    sed 's/^/      | /' "$TMP/runs/records-absent"
  fi
fi

if harness_leg 'usage'; then
  rc=0; "$SUBJECT" >"$TMP/runs/no-box" 2>&1 || rc=$?
  usage_leg 'no --box exits 2' no-box '--box BOX is required'
  check undeclared --box rog-nvidia
  usage_leg 'an undeclared box exits 2' undeclared "not a measurement box declared"
  check bad-fixture --box rog-nv --fixture mlp_small.safetensors
  usage_leg 'a fixture that is no workload stem exits 2' bad-fixture 'names no spec'
  check unknown --box rog-nv --frobnicate
  usage_leg 'an unknown argument exits 2' unknown "unknown argument '--frobnicate'"
  OCANNL_TOOL_CHECK_BOX_DEVICE_CAP=soon check bad-cap --box rog-nv
  usage_leg 'a non-numeric device cap exits 2' bad-cap 'positive number'
  sed 's/^# measurement-boxes: /&newbox /' "$TMP/DIGESTS.txt" >"$TMP/DIGESTS-newbox.txt"
  rc=0
  "$SUBJECT" --venv "$venv" --fixture-dir "$TMP/fixtures" --digests "$TMP/DIGESTS-newbox.txt" \
    --box newbox >"$TMP/runs/newbox" 2>&1 || rc=$?
  usage_leg 'a declared box with no profile exits 2, not a default' newbox 'has no box_profile entry'
fi
if harness_leg 'fleet names'; then
  check fleet-rog --box rog-nv-linux
  if [ "$rc" = 0 ] && head -n 1 "$TMP/runs/fleet-rog" |
       grep -qx "check-box: --box rog-nv-linux is the fleet's name for measurement box rog-nv" &&
     tail -n 1 "$TMP/runs/fleet-rog" | grep -qx 'check-box: rog-nv: READY (10 items)'; then
    report 0 "the fleet's name for a box is that box, said before the items, and the verdict names the box"
  else
    report 1 "the fleet's name for a box is that box, said before the items, and the verdict names the box" "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/fleet-rog"
  fi
  FAKE_TORCH=hip check fleet-minix --box minix-amd-linux --fixture mlp_small
  [ "$rc" = 0 ] && tail -n 1 "$TMP/runs/fleet-minix" | grep -qx 'check-box: minix: READY (10 items)'
  report $? "minix-amd-linux is minix, a hip box" "rc=$rc"
  FAKE_TORCH=mps check fleet-macbook --box macbook-pro --fixture mlp_small
  if [ "$rc" = 0 ] && head -n 1 "$TMP/runs/fleet-macbook" |
       grep -qx "check-box: --box macbook-pro is the fleet's name for measurement box m4-pro" &&
     tail -n 1 "$TMP/runs/fleet-macbook" | grep -qx 'check-box: m4-pro: READY (10 items)'; then
    report 0 "macbook-pro is m4-pro, an mps box"
  else
    report 1 "macbook-pro is m4-pro, an mps box" "rc=$rc"
    sed 's/^/      | /' "$TMP/runs/fleet-macbook"
  fi
  got=$("$SUBJECT" --canonical-box macbook-pro)
  [ "$got" = m4-pro ]
  report $? 'a driver given macbook-pro records m4-pro' "got $got"
  # A fleet box mapped to `-` in fleet_row is still refused by name (exercised on a scratch copy
  # now that every current fleet box measures).
  non_meas=$TMP/non-measuring/benchmarks
  mkdir -p "$non_meas"
  cp "$here/bench_venv.py" "$here/fixture_digest.py" "$non_meas/"
  sed 's/macbook-pro) echo m4-pro ;;/macbook-pro) echo - ;;/' "$SUBJECT" >"$non_meas/check-box.sh"
  chmod +x "$non_meas/check-box.sh"
  rc=0
  "$non_meas/check-box.sh" --venv "$venv" --fixture-dir "$TMP/fixtures" --digests "$TMP/DIGESTS.txt" \
    --box macbook-pro >"$TMP/runs/fleet-non-measuring" 2>&1 || rc=$?
  usage_leg 'a fleet box mapped to - is refused by name, as no measurement box' fleet-non-measuring \
    "^check-box: 'macbook-pro' is a fleet box that measures nothing, so not a measurement box: there is nothing to check"
  ! grep -q '^check-box: [a-z-]*: \(ok\|GAP\|skip\) -- ' "$TMP/runs/fleet-non-measuring"
  report $? 'and nothing is judged for it'
  got=$("$non_meas/check-box.sh" --canonical-box macbook-pro)
  [ "$got" = macbook-pro ]
  report $? 'a driver given a non-measuring fleet box records the name it was given, which check-box refuses' "got $got"
  check fleet-wsl --box rog-nv-wsl
  usage_leg 'a WSL alias is no measurement box, and the refusal lists the fleet names' fleet-wsl \
    "nor the fleet's name for one \(mac-studio rog-nv-linux"
  # rog-nv's row alone, so the file can stop declaring minix but not rog-nv.
  sed 's/^# measurement-boxes: .*/# measurement-boxes: rog-nv/' "$TMP/DIGESTS-rog-only.txt" >"$TMP/DIGESTS-rog.txt"
  rc=0
  "$SUBJECT" --venv "$venv" --fixture-dir "$TMP/fixtures" --digests "$TMP/DIGESTS-rog.txt" \
    --box minix-amd-linux >"$TMP/runs/fleet-undeclared" 2>&1 || rc=$?
  usage_leg "a fleet name for a box the DIGESTS file does not declare exits 2" fleet-undeclared \
    "is the fleet's name for minix, which .* does not declare"
fi
if harness_leg 'fleet names: what a driver records'; then
  # --canonical-box is the gates' reading of --box (box-gate.sh, box_gate.py): a driver records
  # the box's ID, not the fleet's name for it (Codex review round 1). The shell gate, sourced here
  # against the real subject, holds the ID; the Python one is test_orchestrate's BoxNameTest.
  got=$("$SUBJECT" --canonical-box rog-nv-linux),$("$SUBJECT" --canonical-box tuf),$("$SUBJECT" --canonical-box elsewhere)
  [ "$got" = rog-nv,tuf,elsewhere ]
  report $? "--canonical-box maps a fleet name to its box and leaves any other name" "got $got"
  got=$(. "$here/box-gate.sh" && box_gate_args --box minix-amd-linux --other && printf '%s/%s' "$box_gate_box" "$box_gate_shift")
  [ "$got" = minix/2 ]
  report $? "box-gate.sh holds the box's ID for a fleet name" "got $got"
fi
if harness_leg 'fleet names: the relationships'; then
  # Every fleet name resolves to a box the checked-in DIGESTS.txt declares and box_profile knows,
  # and every declared box has a fleet name -- derived from the subject, not listed here.
  fleet=$("$SUBJECT" --fleet-names) || fleet=
  bad=
  [ -n "$fleet" ] || bad=" (--fleet-names printed nothing)"
  declared_boxes=$(python3 "$here/fixture_digest.py" --list-declared-measurement-boxes)
  while read -r name box; do
    [ -n "$name" ] || continue
    if [ "$box" = - ]; then
      # A fleet box that measures nothing: refused by name, and no declared box goes by its name.
      check "fleet-$name" --box "$name" --fixture mlp_small
      { [ "$rc" = 2 ] && grep -q 'measures nothing, so not a measurement box' "$TMP/runs/fleet-$name"; } ||
        bad="$bad $name(non-measuring-not-refused)"
      ! grep -qxF -- "$name" <<<"$declared_boxes" || bad="$bad $name(is-itself-declared)"
      continue
    fi
    check "fleet-$name" --box "$name" --fixture mlp_small
    [ "$rc" != 2 ] || bad="$bad $name"
    grep -qxF -- "$box" <<<"$declared_boxes" || bad="$bad $name->$box(undeclared)"
    # --canonical-box, which answers without reading DIGESTS, relies on this.
    ! grep -qxF -- "$name" <<<"$declared_boxes" || bad="$bad $name(is-itself-declared)"
  done <<<"$fleet"
  while IFS= read -r box; do
    awk -v b="$box" '$2 == b { f = 1 } END { exit !f }' <<<"$fleet" || bad="$bad $box(no-fleet-name)"
  done <<<"$declared_boxes"
  [ -z "$bad" ]
  report $? "every fleet name is a declared box with a profile or a refused non-measuring box, and every declared box has one" "refused:$bad"
  # The map's native boots are the fleet's Linux names: each `-linux` alias on a row of wake-lab.sh's
  # endpoint map is a fleet name here, and each `-linux` fleet name is an alias on the map.
  wake_lab=${OCANNL_TOOL_SWEEP_WAKE_LAB:-$HOME/bin/wake-lab.sh}
  if [ -x "$wake_lab" ] && map=$(perl -e 'alarm 60; exec @ARGV' "$wake_lab" endpoint-map 2>/dev/null) &&
     [ -n "$map" ]; then
    map_linux=$(tr ' ' '\n' <<<"$map" | grep -e '-linux$' | sort)
    ours_linux=$(awk '{ print $1 }' <<<"$fleet" | grep -e '-linux$' | sort)
    [ "$map_linux" = "$ours_linux" ]
    report $? "the fleet's Linux names are the native boots on wake-lab.sh's endpoint map" \
      "map: $(echo $map_linux); check-box.sh: $(echo $ours_linux)"
  else
    skip "the fleet's Linux names are the native boots on wake-lab.sh's endpoint map" \
      "no readable wake-lab.sh endpoint-map at $wake_lab (only the fleet's anchor has one)"
  fi
  if [ -n "${FLEET_BOXES:-}" ]; then
    [ "$(tr ' ' '\n' <<<"$FLEET_BOXES" | grep -v '^$' | sort)" = "$(awk '{ print $1 }' <<<"$fleet" | sort)" ]
    report $? "the fleet names are this host's FLEET_BOXES" "FLEET_BOXES: $FLEET_BOXES; check-box.sh: $(awk '{ print $1 }' <<<"$fleet" | tr '\n' ' ')"
  else
    skip "the fleet names are this host's FLEET_BOXES" "FLEET_BOXES is unset (not a fleet host)"
  fi
fi
if harness_leg 'every declared box has a profile'; then
  bad=
  while IFS= read -r box; do
    check "declared-$box" --box "$box" --fixture mlp_small
    [ "$rc" != 2 ] || bad="$bad $box"
  done < <(python3 "$here/fixture_digest.py" --list-declared-measurement-boxes)
  [ -z "$bad" ]
  report $? "every box benchmarks/fixtures/DIGESTS.txt declares is checkable" "refused:$bad"
fi
finish
