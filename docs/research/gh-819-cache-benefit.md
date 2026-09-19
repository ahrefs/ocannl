# gh-819: cache reuse from per-segment concat symbols

gh-ocannl-765 changed concat lowering so every segment loop receives a fresh binder. The change was
already justified and tested as a correctness hardening: a shared binder made flat symbol-keyed
analyses conflate sibling loops with different bounds. This report answers the narrower open
question from gh-ocannl-819: did removing the shadowed binder produce real analysis-cache and
schedule-cache reuse?

Yes, for the concat-heavy feature block below. At the commit immediately before the change, both
caches declined the routine: two equivalent analysis passes recorded no lookup, and two autotune
calls both searched. At the change commit, the first analysis missed and the second hit; the first
autotune call searched and the second replayed its saved schedule without timing a candidate.

This is cache-counter evidence, not a wall-time result. The experiment did not record elapsed time
and does not establish a compile-time or execution-time speedup. Both tuned routines returned 896
finite values, but the probe did not retain the values or a value digest, so that observation is not
a cross-commit numerical-equivalence claim. The executed correctness oracle for the symbol change
remains `test/operations/concat_loop_symbols` from gh-ocannl-765.

## Comparison

The two revisions are adjacent commits in staging PR #486:

| leg | commit | relationship |
| --- | --- | --- |
| before | `e1bfc51226fb54373b2f1579c9b9ebea03a19e1b` | parent of the symbol change |
| after | `5b1e26418563ac75af8abd487895be7ce9d16796` | `Mint a fresh loop symbol per concat segment, not per product component` |

Both legs used the exact same probe and Dune stanza:

| artifact | SHA-256 |
| --- | --- |
| [concat_cache_benefit_probe.ml.txt](gh-819-cache-benefit/concat_cache_benefit_probe.ml.txt) | `074631524d0a88fad26e14a68bf401481cbfaf32a452d5186682098686a82ad8` |
| [dune-stanza.txt](gh-819-cache-benefit/dune-stanza.txt) | `a975ec5f9e260ea83ad41daef639fc2f88fdc206c69bab72914658425405e37a` |

The model joins three differently sized inputs (64, 40 and 24 elements), transforms the joined
features, joins the original and two derived feature maps, transforms that expansion, then performs
a third join. Thus the measured routine carries several concat segment-loop families and
downstream consumers rather than an isolated renderer fixture.

The inspect leg lowered and optimized the same assignments twice. It recorded
`Low_level.analysis_cache_stats` deltas and canonical completeness. The measurement leg invoked
`Autotune.tune` twice against one initially absent cache directory with beam width 1, zero expansion
rounds and one repeat. The first call was the cold control; the second was eligible to replay only
what the first call had stored. The probe refuses to run if its cache path already exists.

## Results

| observation | before | after |
| --- | --- | --- |
| first canonical form complete | false | true |
| second canonical form complete | false | true |
| equivalent canonical digests | true | true |
| first direct analysis | 0 hits, 0 misses | 0 hits, 1 miss |
| second direct analysis | 0 hits, 0 misses | 1 hit, 0 misses |
| first autotune call | searched; 2 candidates timed | searched; 2 candidates timed |
| second autotune call | searched; 2 candidates timed | cache replay; 0 candidates timed |
| analysis during first tune | 0 hits, 0 misses | 5 hits, 1 miss |
| analysis during second tune | 0 hits, 0 misses | 2 hits, 0 misses |
| output observation, both calls | length 896; all finite | length 896; all finite |

Raw inspect output before:

```text
canonical_complete first=false second=false digest_equal=true
analysis_first analysis_hits=0 analysis_misses=0
analysis_second analysis_hits=0 analysis_misses=0
```

Raw inspect output after:

```text
canonical_complete first=true second=true digest_equal=true
analysis_first analysis_hits=0 analysis_misses=1
analysis_second analysis_hits=1 analysis_misses=0
```

Raw cold/warm output before:

```text
schedule_first outcome=searched candidates_timed=2 output_len=896 finite=true
schedule_second outcome=searched candidates_timed=2 output_len=896 finite=true
tune_first analysis_hits=0 analysis_misses=0
tune_second analysis_hits=0 analysis_misses=0
```

Raw cold/warm output after:

```text
schedule_first outcome=searched candidates_timed=2 output_len=896 finite=true
schedule_second outcome=cache-replay candidates_timed=0 output_len=896 finite=true
tune_first analysis_hits=5 analysis_misses=1
tune_second analysis_hits=2 analysis_misses=0
```

## Provenance

The runs used an Apple Silicon host (`arm64`) on macOS 26.6.2 build 25G83, OCaml 5.5.1, Dune
3.24.2, Base v0.17.3, Stdio v0.17.0 and ctypes 0.24.0. Both revisions used the same tracked
`test/config/ocannl_config` blob (`b9024063bc8e49c435537f2db9b242caa8d3b130`, SHA-256
`1bfa347c5f1f77d6e8812228ddc97860a18fab1674c00ee289a683067aa015ae`). The copied config selected
the `cc` backend, single precision and strict failure classification; the command also set
`OCANNL_BACKEND=cc`. The probe explicitly replaced the config's initialization state 42 with 819.

Every run went through `tools/test-run.sh`; the cold/warm runs had a 180-second cap. Retained local
run directories on the measuring host:

| leg | before | after |
| --- | --- | --- |
| inspect | `/Users/lukstafi/.ocannl-test-runs/20260919T222655Z-55373` | `/Users/lukstafi/.ocannl-test-runs/20260919T222718Z-57856` |
| cold/warm | `/Users/lukstafi/.ocannl-test-runs/20260919T223615Z-81317` | `/Users/lukstafi/.ocannl-test-runs/20260919T223630Z-83187` |

All four runs exited 0. The parser and linker emitted their known warnings; neither leg emitted an
OCANNL failure or a test-runner timeout.

## Reproduction

Create separate worktrees at the two commits above. In each worktree, copy
`concat_cache_benefit_probe.ml.txt` to `test/operations/concat_cache_benefit_probe.ml` and append
`dune-stanza.txt` to `test/operations/dune`. Verify the two SHA-256 values before running. From each
worktree root, run the inspect alias first:

```bash
OCANNL_BACKEND=cc tools/test-run.sh run --cap 180 build \
  @test/operations/probe-concat-cache-inspect -j 4
```

Then run the cold/warm alias exactly once. Its relative cache directory must not exist; the probe
checks this and stops rather than silently turning the cold control into a replay:

```bash
OCANNL_BACKEND=cc tools/test-run.sh run --cap 180 build \
  @test/operations/probe-concat-cache-measure -j 4
```

The historical worktrees contain only temporary instrumentation. No compiler or test-suite source
change is needed to retain the result; this report and its two text attachments are the durable
record.
