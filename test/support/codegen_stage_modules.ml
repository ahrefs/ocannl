(** Modules that run AFTER lowering: rendering a kernel, compiling it, dispatching it, and building
    the cache identity itself.

    A key whose only reads live here is invisible to the canonical digest by construction.
    [schedule_cache.ml] belongs here for the same reason as the backends (Codex P2 on PR #337): it
    reads its keys while assembling the codegen component, downstream of the code the digest names,
    so a [Code_borne] classification of one of them would be wrong in the exact way
    [digest_completeness]'s check 3 exists to catch.

    Shared by [digest_completeness] (no key read here may be classified [Code_borne]) and
    [cache_key_agreement] (every key read here whose class promises an unchanged kernel must have
    oracle values in [emitted_flips] or a documented exemption in [emitted_exemptions], even when
    that backend is not running; gh-ocannl-2440). *)

let modules =
  [
    "c_syntax.ml";
    "cc_backend.ml";
    "cuda_backend.ml";
    "cuda_like_config.ml";
    (* CUDA's arm tables and descriptor, extracted from cuda_backend.ml (gh-ocannl-1214). *)
    "cuda_mma.ml";
    "hip_backend.ml";
    "metal_backend.ml";
    "schedule_cache.ml";
  ]
