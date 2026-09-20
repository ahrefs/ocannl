(* gh-ocannl-572, calibration: the classification is real, not just declared.

   [digest_completeness] checks the registry's shape — every key classified, every claimed cache-key
   component existing. Shape is not substance: a key could be classified code-borne and reach
   nothing. So this test flips one representative key of each class and observes the identity a
   cache would key on, which is what the classes are statements about:

   - code-borne: the canonical digest changes (and with it the whole key), - keyed: the digest is
   UNCHANGED — the flip is invisible to the lowered code, which is the hazard — while the cache key
   separates the regimes, - search-shaping / execution-neutral: neither changes.

   The issue's own caution applies: flips are calibration, not the mechanism. Absence of change is
   weak evidence in general, so this covers a handful of representatives rather than every key, and
   the declarative registry stays the invariant. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module SC = Ir.Schedule_cache

let timing_identity =
  Some
    {
      Ir.Backend_intf.device_signature = "synthetic-device";
      toolchain_signature = Some "synthetic-compiler";
    }

open Verdict.Claims

(* Config values resolve per lookup, so a config-file entry poked in is what later reads see —
   unless the commandline or the environment states the key, which take precedence. Each flip below
   asserts it actually took effect before drawing a conclusion from it. *)
let set_config key value = Hashtbl.set Utils.config_file_args ~key ~data:value
let unset_config key = Hashtbl.remove Utils.config_file_args key

(* A fresh computation per compile: tensors carry their memory modes and lowering results, so a
   second compile of the same graph would not re-decide anything. [a]'s eight values are small
   enough to be filled by generated code under the default [limit_constant_fill_size]; read twice
   per cell, placement materializes it, and gh-ocannl-633 then moves its initialization to the
   link-time host-init upload in EVERY regime — the convergence probe below. [b] read twice by [c]
   is the code-borne flip's substrate: the visit cap decides whether it inlines into its readers or
   materializes. *)
let canonical_of_comp ctx comp =
  let canon = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        canon := Some (SC.canonicalize ~static_indices:[] opt);
        [ opt ])
      ctx comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !canon

let canonical_of ?(materialized_constant = false) ctx =
  let a =
    TDSL.ndarray [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] ~label:[ "dif_a" ] ~output_dims:[ 8 ] ()
  in
  let b = TDSL.O.(if materialized_constant then a *. a else relu a) in
  let c = TDSL.O.(b + b) in
  canonical_of_comp ctx (Train.forward c)

(* The key is a function of the canonical form and the current configuration, so a knob that cannot
   touch the code is answered without recompiling. *)
let key_of ctx canon =
  Option.value_exn
    (SC.cache_key ~timing_identity ~limits:(Context.hardware_limits ctx) canon
       ~backend:(Context.backend_name ctx))

let identity_of ?materialized_constant ctx =
  let canon = canonical_of ?materialized_constant ctx in
  (SC.digest canon, key_of ctx canon)

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let is_cc =
    String.is_prefix backend ~prefix:"cc" || String.is_prefix backend ~prefix:"multidev"
  in
  let digest0, key0 = identity_of ctx in
  p "the identity is stable across two compiles of the same program"
    (let digest1, key1 = identity_of ctx in
     String.equal digest0 digest1 && String.equal key0 key1);

  (* Code-borne: the default precision shapes every fresh tensor the front end builds, hence the
     code they lower to. Mutated in the ref, not the config file: the key is copied into
     [Tensor.default_value_prec] at startup, so a config poke would flip nothing (the Codex P2
     caution below). The flip must be to a precision EVERY backend can store — this test is
     backend-agnostic and Metal has no f64 — so it is half, not double (gh-ocannl-632). *)
  let prec0 = !Tensor.default_value_prec in
  Tensor.default_value_prec := Ir.Ops.half;
  let digest_v, key_v = identity_of ctx in
  Tensor.default_value_prec := prec0;
  p "a code-borne knob (default_prec) changes the digest" (not (String.equal digest0 digest_v));
  p "and therefore the cache key" (not (String.equal key0 key_v));

  (* gh-ocannl-633: [limit_constant_fill_size] is classified code-borne — it changes the assignments
     the front end builds — but once placement MATERIALIZES a small constant (two reads per cell
     exceed the default [virtualize_max_visits]), the in-kernel init moves to the link-time
     host-init upload, so both regimes compile the same program and the identity converges — a
     correct cache hit, not a missed separation. *)
  let digest_m0, key_m0 = identity_of ~materialized_constant:true ctx in
  set_config "limit_constant_fill_size" "1";
  let digest_m1, key_m1 = identity_of ~materialized_constant:true ctx in
  unset_config "limit_constant_fill_size";
  p "a materialized constant compiles identically in both constant-fill regimes (gh-633)"
    (String.equal digest_m0 digest_m1 && String.equal key_m0 key_m1);

  (* Keyed, backend-independent: the index/pool-slot width is read when a kernel is emitted. *)
  let large_models0 = Utils.settings.large_models in
  Utils.settings.large_models <- not large_models0;
  let digest_l, key_l = identity_of ctx in
  Utils.settings.large_models <- large_models0;
  p "a keyed codegen knob (large_models) leaves the digest alone" (String.equal digest0 digest_l);
  p "while the cache key separates the regimes" (not (String.equal key0 key_l));

  (* Keyed, backend-specific: the cc backend's vector width. On other backends the knob reaches no
     codegen, and the key is expected NOT to move — the same statement, read the other way. *)
  set_config "cc_vector_bytes" "0";
  let digest_c, key_c = identity_of ctx in
  unset_config "cc_vector_bytes";
  p "a cc codegen knob (cc_vector_bytes) leaves the digest alone" (String.equal digest0 digest_c);
  p "and separates the key exactly on the backend that reads it"
    (Bool.equal is_cc (not (String.equal key0 key_c)));

  (* Search-shaping and execution-neutral: neither touches the identity. Each flip has to be one
     that is actually RESOLVED per lookup (Codex P2 on PR #337): [print_decimals_precision] is
     copied into [Utils.settings] at startup, so poking the config file would have flipped nothing
     and left an assertion that cannot go red — worse than no assertion. So the settings-borne one
     is mutated in the record, and the config-borne ones are keys a compile really reads:
     [cpu_schedule_min_parallel] shapes the untuned default pipeline, and [ll_ident_style] renames
     identifiers in the emitted code, which the canonical rendering alpha-renames away. *)
  set_config "cpu_schedule_min_parallel" "1";
  set_config "ll_ident_style" "name_only";
  let precision0 = Utils.settings.print_decimals_precision in
  Utils.settings.print_decimals_precision <- precision0 + 7;
  let digest_n, key_n = identity_of ctx in
  Utils.settings.print_decimals_precision <- precision0;
  unset_config "cpu_schedule_min_parallel";
  unset_config "ll_ident_style";
  p "search-shaping and execution-neutral knobs leave the identity untouched"
    (String.equal digest0 digest_n && String.equal key0 key_n);

  (* The debug gates bite only at log_level > 1, so the codegen component hashes the EFFECTIVE
     predicates (Codex P1 on PR #337): the regimes must separate where they change the kernel, and
     an ordinary verbosity bump must not churn cache keys. *)
  let canon = canonical_of ctx in
  let key_with ~logs ~level =
    let logs0 = Utils.settings.debug_log_from_routines and level0 = Utils.settings.log_level in
    Utils.settings.debug_log_from_routines <- logs;
    Utils.settings.log_level <- level;
    let key = key_of ctx canon in
    Utils.settings.debug_log_from_routines <- logs0;
    Utils.settings.log_level <- level0;
    key
  in
  let quiet = key_with ~logs:false ~level:0 in
  p "raising the log level alone does not churn the key"
    (String.equal quiet (key_with ~logs:false ~level:2));
  p "nor does asking for routine logs the log level keeps switched off"
    (String.equal quiet (key_with ~logs:true ~level:0));
  p "while logs that actually reach the kernel separate the key"
    (not (String.equal quiet (key_with ~logs:true ~level:2)));

  (* Buffer aliasing drops the [restrict] qualifier from an alias candidate's kernel parameter
     (Codex P1 on PR #337): emitted C, not lowered code. *)
  set_config "buffer_aliasing" "true";
  let key_a = key_of ctx canon in
  unset_config "buffer_aliasing";
  p "an aliasing flip separates the key without touching the digest" (not (String.equal key0 key_a));

  (* The OpenMP runtime's controls decide the team every Grid loop executes on, and the pool tag —
     derived from process affinity — cannot see them (Codex P1 on PR #337). Only meaningful where
     Grid loops render as OpenMP, so the setting is forced for the key computation alone; no compile
     runs under it. *)
  set_config "cc_parallel_grid" "openmp";
  Unix.putenv "OMP_NUM_THREADS" "1";
  let key_omp1 = key_of ctx canon in
  Unix.putenv "OMP_NUM_THREADS" "16";
  let key_omp16 = key_of ctx canon in
  Unix.putenv "OMP_NUM_THREADS" "";
  unset_config "cc_parallel_grid";
  p "the OpenMP team size separates the key on the backend that runs OpenMP Grid loops"
    (Bool.equal is_cc (not (String.equal key_omp1 key_omp16)));

  (* And the numerics component, whose omission was gh-ocannl-568. *)
  let base = Ir.Numerics.get () in
  Ir.Numerics.set_policy { base with Ir.Numerics.tf32_matmuls = not base.Ir.Numerics.tf32_matmuls };
  let digest_t, key_t = identity_of ctx in
  Ir.Numerics.set_policy base;
  p "a numerics knob (tf32_matmuls) leaves the digest alone" (String.equal digest0 digest_t);
  p "while the cache key separates the policies" (not (String.equal key0 key_t))
