(* gh-ocannl-596: the classification and the cache key agree, key by key.

   [digest_completeness] checks the registry's SHAPE: every key classified, every claimed component
   existing, no codegen-stage read classified code-borne. What review kept finding instead was the
   registry and the cache key disagreeing about one key -- a key classified [Keyed "backend"] that
   moved only the codegen tag, a [Keyed] key whose component never hashed it (its read in a
   codegen-stage module was all the stage check saw), a gate hashed raw where only its effective
   predicate bites. Each of those is mechanically visible: state the key, recompute the identity,
   and compare against what the classification promises. [digest_identity_flips] does that for a
   handful of representatives; this sweeps the classification itself.

   Over a fixed canonical form (the digest cannot move, so only the configuration does):

   - every [Keyed c] key is flipped and must move component [c] -- on the backends whose key
   computation reads it, which the table below names by the predicate the code itself uses, and
   nothing at all on the others; - every other key must not be READ while the key is computed: the
   computation runs with [Utils.accessed_global_args] emptied, in the plain regime and in the one
   where routine logs reach the kernel (where the gated reads happen), and what it records is
   compared against the classification. A read is the over-keying hazard whether or not the value
   happens to move anything: a non-[Keyed] key that reaches the key costs a full retune per flip; -
   the keys resolved once at startup, which a trace cannot see, are flipped through their resolver
   ([Utils.restore_settings], or the memoized numerics policy) and held to the same promise; the
   settings-borne set is DERIVED from [utils.ml]'s [settings.field <- ...] assignments, so a new one
   cannot be swept by a config poke that changes nothing; - the search-shaping keys, whose defaults
   and kinds [Utils.config_key_defaults] owns, are flipped off their default too and must move
   nothing.

   That is stage 1: the registry against the key. Stage 2 is the oracle for the half both can get
   wrong together -- a key that changes the emitted kernel, is classified as leaving it alone
   (execution-neutral, placement-shaping, untimed-model), and is not hashed: every such key a
   compile reads is flipped, the probe recompiled, and a changed kernel under an unchanged key is
   reported by name.

   A [Keyed] key the sweep cannot flip in-process says why ([unflippable]), is varied another way --
   the backend name through [cache_key_components]' argument, the two lazily resolved cc settings in
   child processes -- and the reason stays true: such a key must not be read per lookup by the key
   computation, or it could have been flipped. That table and the oracle's exemptions are the
   reviewable artifacts, printed below. *)

open Base
open Stdio
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module SC = Ir.Schedule_cache

let timing_identity =
  Some
    {
      Ir.Backend_intf.device_signature = "synthetic-device";
      toolchain_signature = Some "synthetic-compiler";
    }

(* Which backends' key computation reads a key: the predicates the code that hashes it uses. *)
type scope =
  | Every_backend
  | Cpu
      (** the cc backend's own tag ([Schedulers.cpu_mma_limits]): not [Schedule.backend_is_gpu] *)
  | Gpu  (** [Schedule.backend_is_gpu], which [Schedule_cache.codegen_tag] gates on *)
  | Cuda_hip  (** the CUDA and HIP backends' own tags *)

let in_scope scope backend =
  match scope with
  | Every_backend -> true
  | Cpu -> not (Ir.Schedule.backend_is_gpu backend)
  | Gpu -> Ir.Schedule.backend_is_gpu backend
  | Cuda_hip -> List.mem [ "cuda"; "hip" ] (String.lowercase backend) ~equal:String.equal

let scope_name = function
  | Every_backend -> "every backend"
  | Cpu -> "CPU backends"
  | Gpu -> "GPU backends"
  | Cuda_hip -> "CUDA and HIP"

(* How a flip reaches the reader. *)
type resolver =
  | Per_lookup  (** stated in the config file; the reader resolves it on every call *)
  | Settings  (** resolved into [Utils.settings] at startup: re-resolved by [restore_settings] *)
  | Numerics_policy  (** resolved into the memoized [Numerics] policy: re-resolved *)
  | Log_level_field
      (** resolved once into [Utils.original_log_level], which [restore_settings] reapplies: the
          field is set directly *)

(* The regime a flip is compared in: what is set before the baseline and the flipped key are
   computed alike. A gate's own keys bite only behind it, so they are flipped behind it. *)
type setup =
  | Routine_logs of bool
  | Log_level of int
  | Stated of string * string
      (** a key stated in the config file for the comparison: a gate that is itself a setting *)

(* A [Stated] setup that a commandline or environment statement outranks: the regime cannot be
   established in this process, so the flips that need it are not made. *)
exception Setup_outranked of string

type flip = {
  resolver : resolver;
  candidates : string list;
      (** values to state; the claim needs ONE that moves the component, so a box whose baseline
          already sits at one value still has the other to flip to *)
  under : setup list;
  scope : scope;
}

let flag = [ "true"; "false" ]

let config ?(under = []) ?(scope = Every_backend) candidates =
  { resolver = Per_lookup; candidates; under; scope }

let logging = [ Routine_logs true; Log_level 2 ]

(* Every [Keyed] key, with how it is flipped -- or, in [unflippable], why it cannot be. Held equal
   to the class by the first claim below. *)
let keyed_flips =
  [
    ("tf32_matmuls", { (config flag) with resolver = Numerics_policy });
    ("narrow_compute_f32", { (config flag) with resolver = Numerics_policy });
    ("fp16_arithmetic", { (config [ "true"; "false" ]) with resolver = Numerics_policy });
    ("bf16_arithmetic", { (config [ "true"; "false" ]) with resolver = Numerics_policy });
    ("cc_backend_optimization_level", config ~scope:Cpu [ "1"; "2" ]);
    ("cc_backend_compiler_command", config ~scope:Cpu [ "flipped-cc" ]);
    ("cc_backend_arch_flags", config ~scope:Cpu [ "-mflipped-arch" ]);
    ("cc_backend_simd_flags", config ~scope:Cpu [ "-mflipped-simd" ]);
    ("cc_backend_fp_contract", config ~scope:Cpu [ "off"; "on" ]);
    ("cc_backend_fast_math", config ~scope:Cpu flag);
    ("cc_vector_bytes", config ~scope:Cpu [ "0"; "64" ]);
    ("cc_vector_registers", config ~scope:Cpu [ "7"; "8" ]);
    ("cc_fp16_arithmetic", config ~scope:Cpu [ "none"; "promoted" ]);
    ("cc_parallel_grid", config ~scope:Cpu [ "none"; "openmp" ]);
    (* Read per compile, and only behind a parallel grid, so flipped behind one like the chunk
       count. *)
    ( "cc_grid_min_updates",
      config ~scope:Cpu ~under:[ Stated ("cc_parallel_grid", "openmp") ] [ "0"; "4096" ] );
    (* The chunk count reaches the emitted code only through a parallel grid, so it is flipped
       behind one: stating the mode probes nothing, and nothing is compiled here, so OpenMP serves
       on a box whose own [auto] resolves to no grid. *)
    ( "cc_parallel_chunks",
      config ~scope:Cpu ~under:[ Stated ("cc_parallel_grid", "openmp") ] [ "3"; "5" ] );
    ("large_models", { (config flag) with resolver = Settings });
    ("big_models", { (config flag) with resolver = Settings });
    ( "log_level",
      { (config [ "2"; "0" ]) with resolver = Log_level_field; under = [ Routine_logs true ] } );
    ("debug_log_from_routines", { (config flag) with resolver = Settings; under = [ Log_level 2 ] });
    ("debug_log_to_stream_files", config ~under:logging flag);
    ( "output_debug_files_in_build_directory",
      { (config flag) with resolver = Settings; under = [ Log_level 2 ]; scope = Cuda_hip } );
    ("prefer_backend_uniformity", config ~under:logging flag);
    ("buffer_aliasing", config flag);
    ("gpu_graph_capture", config ~scope:Cuda_hip flag);
    ("gpu_elide_covered_zeros", config ~scope:Gpu flag);
    ("gpu_default_seed_narrow_storage", config ~scope:Cuda_hip flag);
    ("autotune_timing", config [ "isolated"; "queued" ]);
  ]

(* Components that move WITH a classified one by derivation, not by a second reading of the key: the
   codegen tag hashes what the numerics policy resolves to on this backend
   ([Backend_intf.codegen_capabilities_fingerprint], gh-ocannl-1117), so a policy flip may move it
   too. And the codegen tag hashes the device description's Key-class projection
   ([Backend_intf.limits_key]), whose [worker_pool_tag] field is the very pool decision the pool
   component renders, and the cc tag hashes the parallel-grid chunk count, whose auto value is four
   times the decided pool's width ([Cc_backend.parallel_grid_chunks_setting]) -- so a pool decision
   that changes moves codegen too, by those two routes and no other, which the pool-class children
   pin exactly (rog-nv-linux, a hybrid Core Ultra 9 275HX, is where both showed). Each listed pair
   is one fact under two views, which partitions the cache no finer than the classified component
   alone. *)
let derived_components = [ ("numerics", [ "codegen" ]); ("pool", [ "codegen" ]) ]

(* The [Keyed] keys no in-process flip reaches, each with how it is varied instead (Codex P2, round
   4): the backend name through [cache_key_components]' own argument, and the two lazily resolved cc
   settings in child processes of this executable ([child_flag]), each stating its value on the
   commandline before anything forces the lazy. *)
let unflippable =
  [
    ( "backend",
      "a parameter, not a lookup: backends are process-wide singletons, and the component is the \
       backend name the caller passes -- varied through that argument" );
    ( "cc_pool_core_class",
      "resolved once, by the lazy [Cc_backend.pool_restriction] the first compile forces; the pool \
       tag it decides is what the component hashes -- varied in child processes, held to \
       separating exactly the pool decisions that differ and to moving codegen only through that \
       decision: its tag in the limits record, and the auto chunk count its width sets" );
    ( "cc_grid_private_bytes_cap",
      "resolved once, by the lazy [C_syntax.per_chunk_private_bytes_cap] the first compile forces; \
       the codegen tag hashes that resolved value -- varied in child processes" );
  ]

(* What the backend name reaches besides its own component: the codegen tag gates on
   [Schedule.backend_is_gpu] of it, and the timing component spells the CUDA/HIP queued policy's
   generation from it. *)
let backend_derived_components = [ "codegen"; "timing" ]

(* Run as a child with this flag, the executable prints its cache-key components (and, on a CPU
   backend, the pool width the cc backend decided) instead of sweeping. *)
let child_flag = "--components"

(* The keys resolved into [Utils.settings] but classified other than [Keyed]: flipped the same way,
   and held to moving nothing. With [keyed_flips]' [Settings] entries they must cover exactly what
   [utils.ml] assigns from configuration. *)
let settings_flips =
  [
    ("fixed_state_for_init", [ "7"; "8" ]);
    ("print_decimals_precision", [ "9"; "3" ]);
    ("check_half_prec_constants_cutoff", [ "100.0"; "200.0" ]);
    ("default_prng_variant", [ "crypto"; "light" ]);
  ]

(* Values off a registered key's default, by its kind: [Utils.config_key_defaults] owns both. A
   keyword the read site's [~parse] rejects is fine: a key read while the cache key is computed is
   the finding, whether it raises or not. *)
let off_default (kind : Utils.config_value_kind) default =
  (* The default itself as well: where a source already states the off-default value, the default is
     the candidate that flips it. *)
  (fun off -> off @ [ default ])
  @@
  match kind with
  | Flag -> if String.equal default "true" then [ "false" ] else [ "true" ]
  | Integer -> [ Int.to_string (Int.of_string default + 1) ]
  | Float_number -> [ Float.to_string ((Float.of_string default *. 2.) +. 1.) ]
  | Keyword -> [ "flipped-mode" ]
  | Text -> [ "flipped-text" ]
  | Positive_float_or_unset -> [ "1.5" ]
  | Float_at_least { floor } -> [ Float.to_string (floor +. 3.) ]

(* Stage 2 (gh-ocannl-596's codegen-diff oracle). Stage 1 checks the registry against the key; it
   cannot see a key that changes the emitted kernel, is classified neither [Keyed] nor
   search-shaping, AND is not hashed -- both halves agree, and both are wrong. [buffer_aliasing] was
   exactly that until gh-ocannl-572's review: dropping the [restrict] qualifier is a textual change
   in the emitted C no part of the registry knew about. So every key a compile reads whose class
   promises an unchanged kernel -- execution-neutral, placement-shaping, untimed-model -- is flipped
   here, the kernel recompiled, and a changed kernel under an unchanged key is an under-keying,
   reported by name. The other classes are out by their own statements: [Search_shaping] shapes the
   untuned default pipeline, which a cached winner replays its own ops over, so a different default
   kernel under one key is what the class says; and a [Code_borne] key changes the lowered code, so
   the digest carries it, and a flip of one the probe does not exercise (online softmax on a probe
   with no attention) could never fail here (Codex P2, round 2). That class's guards are
   [digest_completeness]'s check 3 (no code-borne key read at codegen) and the representatives of
   [digest_identity_flips].

   Which keys a compile reads is TRACED, not listed, so a key that starts being read at compile time
   joins the oracle the day it does; this table must then give it values, or the first claim of the
   stage names it. Rows for keys this backend's compile does not read are other backends'. The
   settings-borne keys take their values from [settings_flips]. *)
let emitted_flips =
  [
    ("cc_backend_post_compile_timeout", [ "2.5" ]);
    ("max_shape_error_origins", [ "3"; "9" ]);
    ("output_dlls_in_build_directory", flag);
    ("output_prec_in_ll_files", flag);
    ("cd_ident_style", [ "name_only"; "name_and_label" ]);
  ]

(* Read at compile time, in the classes the oracle covers, and deliberately not flipped. *)
let emitted_exemptions =
  [
    ( "ll_ident_style",
      "identifier spelling: the canonical rendering alpha-renames identifiers away, and code that \
       differs only in their names compiles to the same kernel -- a changed text is what the key \
       is right to ignore" );
    ( "cc_backend_verify_codesign",
      "a check on the compiled library, which signs it or refuses it and never touches the source: \
       stated on a host without codesign (every Linux CI box) it refuses every compile, so no \
       kernel comes back to compare" );
    ( "build_files_prefix",
      "it moves the very artifacts this oracle reads, and Generated refuses a set prefix" );
  ]

(* [~lower_only] stops the compile once the lowered code is canonicalized, before anything is
   rendered or handed to a compiler. *)
exception Lowered

let canonical_of ?(lower_only = false) ctx =
  let a =
    TDSL.ndarray [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] ~label:[ "cka_a" ] ~output_dims:[ 8 ] ()
  in
  let c = TDSL.O.(relu a + a) in
  let canon = ref None in
  (try
     ignore
       (Context.compile
          ~lowered_transform:(fun opt ->
            canon := Some (SC.canonicalize ~static_indices:[] opt);
            if lower_only then raise Lowered;
            [ opt ])
          ctx (Train.forward c) Ir.Indexing.Empty
         : Context.t * Context.routine)
   with exn when lower_only && Option.is_some !canon -> ignore (exn : exn));
  Option.value_exn ~here:[%here] !canon

(* Whether the config file is where [key] now resolves from, stating [value]: a commandline or
   environment statement outranks it, and one that happens to state the same value is still not the
   flip (it was the baseline too). *)
let stated_in_file key value =
  match Utils.profile_key_source key with
  | Some (v, Utils.From_config_file) -> String.equal v value
  | Some _ | None -> false

let with_setup setups f =
  let level0 = Utils.settings.log_level and logs0 = Utils.settings.debug_log_from_routines in
  let stated =
    List.filter_map setups ~f:(function
      | Stated (key, value) -> Some (key, value, Hashtbl.find Utils.config_file_args key)
      | Routine_logs _ | Log_level _ -> None)
  in
  let restore () =
    Utils.settings.log_level <- level0;
    Utils.settings.debug_log_from_routines <- logs0;
    List.iter stated ~f:(fun (key, _, previous) ->
        match previous with
        | None -> Hashtbl.remove Utils.config_file_args key
        | Some data -> Hashtbl.set Utils.config_file_args ~key ~data)
  in
  Exn.protect ~finally:restore ~f:(fun () ->
      List.iter setups ~f:(function
        | Routine_logs on -> Utils.settings.debug_log_from_routines <- on
        | Log_level level -> Utils.settings.log_level <- level
        | Stated (key, value) ->
            Hashtbl.set Utils.config_file_args ~key ~data:value;
            if not (stated_in_file key value) then raise (Setup_outranked key));
      f ())

let resync = function
  | Per_lookup -> ()
  | Settings -> Utils.restore_settings ()
  | Numerics_policy -> Ir.Numerics.set_policy (Ir.Numerics.default ())
  | Log_level_field -> ()

(* Whether stating [value] would change nothing: the key already resolves to it, from whatever
   source (Codex P2, round 5) -- a candidate equal to its baseline is no flip, and a claim that
   nothing moved under it would be vacuous. A key no source states resolves to its read site's
   default, which this cannot see; every table therefore carries a candidate off that default. *)
let same_as_baseline key value =
  let norm v = String.lowercase (String.strip v) in
  match Utils.profile_key_source key with
  | Some (current, _) -> String.equal (norm current) (norm value)
  | None -> false

(* [f] under [key] stated as [value]; [None] when the flip cannot take effect in this process: a
   commandline or environment statement outranks the config file, or the key already resolves to
   [value]. *)
let with_stated key value resolver f =
  match resolver with
  | Log_level_field ->
      let level0 = Utils.settings.log_level in
      if Int.of_string value = level0 then None
      else (
        Utils.settings.log_level <- Int.of_string value;
        Some (Exn.protect ~f ~finally:(fun () -> Utils.settings.log_level <- level0)))
  | _ when same_as_baseline key value -> None
  | _ ->
      let previous = Hashtbl.find Utils.config_file_args key in
      Hashtbl.set Utils.config_file_args ~key ~data:value;
      let restore () =
        (match previous with
        | None -> Hashtbl.remove Utils.config_file_args key
        | Some data -> Hashtbl.set Utils.config_file_args ~key ~data);
        resync resolver
      in
      if not (stated_in_file key value) then (
        restore ();
        None)
      else
        Some
          (Exn.protect
             ~f:(fun () ->
               resync resolver;
               f ())
             ~finally:restore)

let components_of ctx canon ~backend =
  Option.value_exn ~here:[%here]
    (SC.cache_key_components ~timing_identity ~limits:(Context.hardware_limits ctx)
       ~capabilities:(Context.codegen_capabilities ctx)
       canon ~backend)

let () =
  if Array.mem Stdlib.Sys.argv child_flag ~equal:String.equal then (
    let ctx = Context.auto () in
    let backend = Context.backend_name ctx in
    (* Lowered and canonicalized, never compiled: a child states a cc grid mode the host's compiler
       may lack (clang without libomp), and the components need no kernel -- the codegen and pool
       tags force the two lazies themselves. *)
    let canon = canonical_of ~lower_only:true ctx in
    List.iter (components_of ctx canon ~backend) ~f:(fun (name, value) ->
        printf "component\t%s\t%s\n" name value);
    if not (Ir.Schedule.backend_is_gpu backend) then
      printf "pool-tag\t%s\n" (Context.Cc_backend.pool_tag ());
    (* The codegen tag as it would read without the pool decision in the limits record: what must
       not move across pool classes, if the decision is all a pool class changes. *)
    if not (Ir.Schedule.backend_is_gpu backend) then
      printf "pool-width\t%d\n" (Context.Cc_backend.effective_pool_width ());
    printf "codegen-sans-pool\t%s\n"
      (SC.codegen_tag ~backend
         ~limits:{ (Context.hardware_limits ctx) with Ir.Backend_intf.worker_pool_tag = None }
         ~capabilities:(Context.codegen_capabilities ctx)
         ());
    Out_channel.flush stdout;
    Stdlib.exit 0)

let () =
  let ctx = Context.auto () in
  (* A flip a commandline or environment statement outranks is not made, and says so as a skip
     rather than leaving a claim that reads as if it had been (Codex P2, round 3): rerunning a test
     under an [OCANNL_<KEY>] is supported, and the key it states is the one this run cannot flip. *)
  let skip_outranked ~what key =
    Verdict.skipped ~aggregation:`Environment ~backend:(Context.backend_name ctx)
      (Printf.sprintf
         "%s of %s: the key, or the regime it is flipped in, is stated on the commandline or in \
          the environment, or already resolves to every candidate value"
         what key)
  in
  let backend = Context.backend_name ctx in
  (* A backend-uniform golden cannot say which backend the sweep ran on (gh-ocannl-622). *)
  eprintf "cache_key_agreement: sweeping on the %s backend (not part of the golden)\n%!" backend;
  (* Before the first compile, as [Generated] requires: [canonical_of] compiles, and with debug
     artifacts configured on it would write through [build_files/] before [init]'s cleanup and
     symlink refusals ran (Codex P1, round 1). *)
  Test_utils.Generated.init ~backend_name:backend;
  let canon = canonical_of ctx in
  let components () = components_of ctx canon ~backend in
  let moved before after =
    List.filter_map (List.zip_exn before after) ~f:(fun ((name, a), (_, b)) ->
        Option.some_if (not (String.equal a b)) name)
  in
  (* What the key computation reads per lookup, in [setups]' regime. *)
  let reads setups =
    let saved = Hash_set.to_list Utils.accessed_global_args in
    Hash_set.clear Utils.accessed_global_args;
    let read =
      Exn.protect
        ~f:(fun () ->
          ignore (with_setup setups components : (string * string) list);
          Hash_set.to_list Utils.accessed_global_args)
        ~finally:(fun () -> List.iter saved ~f:(Hash_set.add Utils.accessed_global_args))
    in
    Set.of_list (module String) read
  in
  let keyed =
    List.concat_map Utils.config_key_classification ~f:(fun (c, _, keys) ->
        match c with Utils.Keyed component -> List.map keys ~f:(fun k -> (k, component)) | _ -> [])
  in
  let keyed_names = Set.of_list (module String) (List.map keyed ~f:fst) in
  let tabled =
    Set.of_list (module String) (List.map keyed_flips ~f:fst @ List.map unflippable ~f:fst)
  in
  (* 1. The tables cover the [Keyed] class exactly: a key added to it is flipped from the day it is
     classified, and a key that leaves it does not keep a stale row. *)
  p_empty ~min:20 "the flip and unflippable tables cover exactly the Keyed class"
    ~over:(Set.to_list keyed_names)
    (Set.to_list (Set.union (Set.diff keyed_names tabled) (Set.diff tabled keyed_names)));
  (* 2. Settings-borne keys, derived from [utils.ml] (the rule's argument): each one is flipped
     through [restore_settings], never by a config poke that would change nothing. *)
  let settings_borne =
    match Stdlib.Sys.argv with
    | [| _; utils_ml |] ->
        Test_utils.Config_key_scan.settings_field_initializers (In_channel.read_all utils_ml)
        |> List.concat_map ~f:snd
        |> Set.of_list (module String)
    | _ ->
        Verdict.fail "usage: cache_key_agreement <arrayjit/lib/utils.ml>";
        Set.empty (module String)
  in
  let flipped_through_settings =
    List.filter_map keyed_flips ~f:(fun (key, flip) ->
        Option.some_if (Poly.equal flip.resolver Settings) key)
    @ List.map settings_flips ~f:fst
    |> Set.of_list (module String)
  in
  p_empty ~min:6
    "every key utils.ml resolves into Utils.settings is flipped through restore_settings, and only \
     those"
    ~over:(Set.to_list settings_borne)
    (Set.to_list
       (Set.union
          (Set.diff settings_borne flipped_through_settings)
          (Set.diff flipped_through_settings settings_borne)));
  (* 3. Each [Keyed] key moves its own component where its scope says the key computation reads it,
     and nothing elsewhere. *)
  let outcome (key, flip) =
    let component = List.Assoc.find_exn keyed key ~equal:String.equal in
    (* The regime is set inside the flip as well: re-resolving the settings record resets the log
       level the regime raised. *)
    let in_regime () = with_setup flip.under components in
    let flips =
      match in_regime () with
      | exception Setup_outranked gate ->
          eprintf
            "%s: its regime needs %s, which is stated on the commandline or in the environment\n"
            key gate;
          []
      | before ->
          List.filter_map flip.candidates ~f:(fun value ->
              match with_stated key value flip.resolver in_regime with
              | None -> None
              | Some after -> Some (value, moved before after)
              | exception exn ->
                  eprintf "%s=%s: computing the key raised %s\n" key value (Exn.to_string exn);
                  Some (value, [ "<raised>" ]))
    in
    (key, component, flip.scope, flips)
  in
  let outcomes = List.map keyed_flips ~f:outcome in
  let took_effect =
    List.filter outcomes ~f:(fun (key, _, _, flips) ->
        if List.is_empty flips then skip_outranked ~what:"the Keyed flip" key;
        not (List.is_empty flips))
  in
  let here, elsewhere =
    List.partition_tf took_effect ~f:(fun (_, _, scope, _) -> in_scope scope backend)
  in
  let describe flips =
    String.concat ~sep:"; "
      (List.map flips ~f:(fun (value, moved) ->
           value ^ " moved [" ^ String.concat ~sep:", " moved ^ "]"))
  in
  p_all ~min:8 "every Keyed key the running backend hashes moves its own component when flipped"
    here ~f:(fun (key, component, _, flips) ->
      let ok =
        List.exists flips ~f:(fun (_, moved) -> List.mem moved component ~equal:String.equal)
      in
      if not ok then
        eprintf "%s is classified Keyed %S, but no flip moved it: %s\n" key component
          (describe flips);
      ok);
  (* And no other component, beyond what [derived_components] says follows from it: a flip that also
     moves a component its classification does not name is over-keying there, or a misclassification
     (Codex P2, round 1). A candidate equal to the baseline moves nothing and passes. *)
  p_all ~min:8 "and no Keyed flip moves a component its classification does not account for" here
    ~f:(fun (key, component, _, flips) ->
      let allowed =
        component
        :: Option.value ~default:[]
             (List.Assoc.find derived_components component ~equal:String.equal)
      in
      let ok =
        List.for_all flips ~f:(fun (_, moved) ->
            List.for_all moved ~f:(List.mem allowed ~equal:String.equal))
      in
      if not ok then
        eprintf "%s is classified Keyed %S, but a flip moved another component: %s\n" key component
          (describe flips);
      ok);
  p_all ~min:1 "every Keyed key outside the running backend's scope moves nothing when flipped"
    elsewhere ~f:(fun (key, _, scope, flips) ->
      let ok = List.for_all flips ~f:(fun (_, moved) -> List.is_empty moved) in
      if not ok then
        eprintf "%s is hashed only on %s, yet a flip moved the %s key\n" key (scope_name scope)
          backend;
      ok);
  (* 4. Nothing outside the [Keyed] class is read per lookup while the key is computed -- in the
     plain regime, and behind the routine-log gate where the gated reads happen. *)
  let read = Set.union (reads []) (reads logging) in
  let non_keyed =
    Set.diff (Set.of_list (module String) (Set.to_list Utils.known_config_keys)) keyed_names
    |> Set.to_list
  in
  p_none ~min:60 "no key outside the Keyed class is read while the cache key is computed" non_keyed
    ~f:(fun key ->
      let found = Set.mem read key in
      if found then eprintf "%s is read while the cache key is computed\n" key;
      found);
  p_exists "the key computation reads its Keyed keys per lookup, so the trace sees reads"
    (Set.to_list read) ~f:(Set.mem keyed_names);
  p_none ~min:1 "no unflippable Keyed key is read per lookup by the key computation" unflippable
    ~f:(fun (key, _) ->
      let found = Set.mem read key in
      if found then eprintf "%s is unflippable, yet read per lookup -- flip it instead\n" key;
      found);
  (* 4b. Varying what no in-process flip reaches. The backend name, through the argument: it must
     move its own component, and nothing it does not reach. *)
  (let before = components () in
   let after = components_of ctx canon ~backend:(backend ^ "_flipped") in
   let moved = moved before after in
   p "the backend name moves its own component" (List.mem moved "backend" ~equal:String.equal);
   p_all ~min:1 "and only components it reaches" moved ~f:(fun c ->
       String.equal c "backend" || List.mem backend_derived_components c ~equal:String.equal));
  (* The two lazily resolved cc settings, in children that state them on the commandline (which
     outranks every other source) before anything forces the lazy. A child reports its components
     and the pool the cc backend decided. *)
  let child args =
    let ((status, out, _) as run) = Fresh_process.run (child_flag :: args) in
    match status with
    | Unix.WEXITED 0 ->
        let lines = List.map (String.split_lines out) ~f:(String.split ~on:'\t') in
        Some
          ( List.filter_map lines ~f:(function
              | [ "component"; name; value ] -> Some (name, value)
              | _ -> None),
            List.find_map lines ~f:(function [ "pool-tag"; tag ] -> Some tag | _ -> None),
            ( List.find_map lines ~f:(function
                | [ "codegen-sans-pool"; tag ] -> Some tag
                | _ -> None),
              List.find_map lines ~f:(function
                | [ "pool-width"; width ] -> Int.of_string_opt width
                | _ -> None) ) )
    | _ ->
        Fresh_process.report ~label:"cache_key_agreement child" run;
        None
  in
  let grid = "--ocannl_cc_parallel_grid=openmp" in
  let cpu = not (Ir.Schedule.backend_is_gpu backend) in
  let cap_flip =
    (* Both values stated, so neither child resolves the cap from an ambient source that might equal
       the other's (Codex P2, round 5). *)
    match
      ( child [ grid; "--ocannl_cc_grid_private_bytes_cap=4096" ],
        child [ grid; "--ocannl_cc_grid_private_bytes_cap=1024" ] )
    with
    | Some (before, _, _), Some (after, _, _) -> Some (moved before after)
    | _ -> None
  in
  p
    "cc_grid_private_bytes_cap, stated in a child, moves the codegen component alone on a CPU \
     backend, and nothing on a GPU one"
    (Option.value_map cap_flip ~default:false ~f:(fun moved ->
         if cpu then List.equal String.equal moved [ "codegen" ] else List.is_empty moved));
  (* Each pool class twice: with the chunk count left to its auto value, which follows the decided
     pool's width, and with it stated, which takes that route away. *)
  let pool_children chunks =
    List.filter_map [ "all"; "performance"; "efficiency" ] ~f:(fun cls ->
        Option.map
          (child ([ grid; "--ocannl_cc_pool_core_class=" ^ cls ] @ chunks))
          ~f:(fun (components, decision, (sans_pool, width)) ->
            (cls, components, (decision, sans_pool, width))))
  in
  (* Auto stated as such ([0]), so an ambient or configured count cannot stand in for it. *)
  let pools = pool_children [ "--ocannl_cc_parallel_chunks=0" ] in
  let pools_fixed_chunks = pool_children [ "--ocannl_cc_parallel_chunks=8" ] in
  p "every child run of the pool-class flips completed"
    (List.length pools = 3 && List.length pools_fixed_chunks = 3);
  let pairs_of pools =
    List.concat_mapi pools ~f:(fun i a ->
        List.filteri pools ~f:(fun j _ -> j > i) |> List.map ~f:(fun b -> (a, b)))
  in
  let pairs = pairs_of pools in
  let pool_allowed =
    "pool"
    :: Option.value ~default:[] (List.Assoc.find derived_components "pool" ~equal:String.equal)
  in
  p_all "no pool-class flip moves a component other than the pool and what follows from it" pairs
    ~f:(fun ((_, ca, _), (_, cb, _)) ->
      List.for_all (moved ca cb) ~f:(List.mem pool_allowed ~equal:String.equal));
  (* The two routes from the pool decision to codegen, each pinned: with [worker_pool_tag] blanked
     from the limits record, the codegen tags of two classes differ exactly when their pools' widths
     do (the auto chunk count); and with the chunk count stated as well, they are one tag. A pool
     class that reached anything else the codegen tag reads would split it. *)
  p_all ~min:3
    "with the chunk count auto, codegen beyond the pool tag moves exactly when the pool width does"
    pairs ~f:(fun ((_, _, (_, sa, wa)), (_, _, (_, sb, wb))) ->
      Option.is_some sa
      && Bool.equal (Option.equal String.equal sa sb) (Option.equal Int.equal wa wb));
  p_all ~min:3
    "with the chunk count stated, a pool-class flip moves codegen only through the pool tag"
    (pairs_of pools_fixed_chunks) ~f:(fun ((_, _, (_, sa, _)), (_, _, (_, sb, _))) ->
      Option.is_some sa && Option.equal String.equal sa sb);
  (* The pool component separates exactly the pools the cc backend decided apart: two classes whose
     decisions ([Cc_backend.pool_tag], what the restriction settled on) differ get different pool
     components, and two whose decisions agree share one. Which classes the host can tell apart is
     its own business -- a host without core classes to restrict to decides one pool for all three
     -- so the claim is over the pairs, whatever they decided. A GPU backend has no pool. *)
  p_all ~min:3 "the pool component separates exactly the pool decisions that differ" pairs
    ~f:(fun ((_, ca, (ta, _, _)), (_, cb, (tb, _, _))) ->
      Bool.equal (Option.equal String.equal ta tb)
        (not (List.mem (moved ca cb) "pool" ~equal:String.equal)));
  (* 5. The keys a trace cannot see -- resolved once into [Utils.settings] -- flipped through their
     resolver: each flip must change the settings record (or the flip proves nothing) and move no
     component. *)
  let settings_outcomes =
    List.map settings_flips ~f:(fun (key, candidates) ->
        let before = components () and settings0 = Utils.sexp_of_settings Utils.settings in
        let flips =
          List.filter_map candidates ~f:(fun value ->
              with_stated key value Settings (fun () ->
                  ( not (Sexp.equal settings0 (Utils.sexp_of_settings Utils.settings)),
                    moved before (components ()) )))
        in
        (key, flips))
    |> List.filter ~f:(fun (key, flips) ->
        if List.is_empty flips then skip_outranked ~what:"the settings flip" key;
        not (List.is_empty flips))
  in
  p_all ~min:4 "each settings-borne key outside Keyed is flipped by restore_settings"
    settings_outcomes ~f:(fun (key, flips) ->
      let ok = List.exists flips ~f:fst in
      if not ok then eprintf "%s: no flip changed Utils.settings\n" key;
      ok);
  p_all ~min:4 "and moves no component of the cache key" settings_outcomes ~f:(fun (key, flips) ->
      let ok = List.for_all flips ~f:(fun (_, moved) -> List.is_empty moved) in
      if not ok then eprintf "%s is not Keyed, yet its flip moved the cache key\n" key;
      ok);
  (* 6. The search-shaping keys, flipped off the defaults [Utils.config_key_defaults] registers. *)
  let registered =
    List.filter Utils.config_key_defaults ~f:(fun (key, _) -> not (Set.mem keyed_names key))
  in
  let registered =
    List.filter registered ~f:(fun (key, (kind, default)) ->
        let flippable =
          List.exists (off_default kind default) ~f:(fun value ->
              Option.is_some (with_stated key value Per_lookup (fun () -> ())))
        in
        if not flippable then skip_outranked ~what:"the registered flip" key;
        flippable)
  in
  p_none ~min:20 "no registered non-Keyed key moves the cache key when flipped off its default"
    registered ~f:(fun (key, (kind, default)) ->
      let before = components () in
      List.exists (off_default kind default) ~f:(fun value ->
          match with_stated key value Per_lookup (fun () -> moved before (components ())) with
          | None | Some [] -> false
          | Some moved ->
              eprintf "%s=%s moved [%s]\n" key value (String.concat ~sep:", " moved);
              true
          | exception exn ->
              eprintf "%s=%s: the key computation read it and raised %s\n" key value
                (Exn.to_string exn);
              true));
  (* 7. Stage 2: the codegen-diff oracle. A fresh graph per compile, under one routine name with
     debug artifacts on, so that [Generated] can read the kernel each compile emits: a flip then
     acts on the whole pipeline -- the front end, the placements, the lowering and the rendering --
     so a key read anywhere on it enters the oracle. Fresh builds of one program emit one kernel up
     to the numbers minted into identifiers (gh-ocannl-1473), which [Generated.normalize_minted]
     renames away. The population is traced on the second compile, so it holds exactly the keys a
     compile of a fresh build reads. *)
  let routine = "cka_probe" in
  let probe () =
    let a =
      TDSL.ndarray [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] ~label:[ "cka_p" ] ~output_dims:[ 8 ] ()
    in
    Train.forward TDSL.O.(relu a + (a *. a))
  in
  let compile_probe () =
    Test_utils.Generated.arm routine;
    let canon = ref None in
    let _ctx, _routine =
      Context.compile ~name:routine
        ~lowered_transform:(fun opt ->
          canon := Some (SC.canonicalize ~static_indices:[] opt);
          [ opt ])
        ctx (probe ()) Ir.Indexing.Empty
    in
    let canon = Option.value_exn ~here:[%here] !canon in
    let key =
      Option.value_exn ~here:[%here]
        (SC.cache_key ~timing_identity ~limits:(Context.hardware_limits ctx)
           ~capabilities:(Context.codegen_capabilities ctx)
           canon ~backend)
    in
    (key, Test_utils.Generated.normalize_minted (Test_utils.Generated.read routine))
  in
  let oracle () =
    let baseline_key, baseline_src = compile_probe () in
    let traced_reads =
      let saved = Hash_set.to_list Utils.accessed_global_args in
      Hash_set.clear Utils.accessed_global_args;
      let again_key, again_src = compile_probe () in
      let r = Hash_set.to_list Utils.accessed_global_args in
      List.iter saved ~f:(Hash_set.add Utils.accessed_global_args);
      let stable = String.equal again_key baseline_key && String.equal again_src baseline_src in
      if not stable then
        eprintf "two compiles of the probe, normalized:\n%s\n----\n%s\n" baseline_src again_src;
      p "two fresh builds of the probe compile to the same kernel under the same key" stable;
      Set.of_list (module String) r
    in
    let oracle_class key =
      match Utils.classify_config_key key with
      | Some ((Utils.Execution_neutral | Utils.Placement_shaping | Utils.Untimed_model), _) -> true
      | Some ((Utils.Code_borne | Utils.Keyed _ | Utils.Search_shaping | Utils.Aggregate), _) | None
        ->
          false
    in
    let settings_rows =
      Set.of_list (module String) (List.map settings_flips ~f:fst) |> Set.filter ~f:oracle_class
    in
    let population =
      Set.union (Set.filter traced_reads ~f:oracle_class) settings_rows |> Set.to_list
    in
    let exempt key = List.Assoc.mem emitted_exemptions key ~equal:String.equal in
    let values key =
      match List.Assoc.find settings_flips key ~equal:String.equal with
      | Some candidates -> Some (Settings, candidates)
      | None ->
          Option.map (List.Assoc.find emitted_flips key ~equal:String.equal) ~f:(fun c ->
              (Per_lookup, c))
    in
    p_all ~min:4
      "every key a compile reads whose class promises an unchanged kernel has oracle values or an \
       exemption"
      population ~f:(fun key ->
        let ok = exempt key || Option.is_some (values key) in
        if not ok then eprintf "%s is read during a compile; give it values in emitted_flips\n" key;
        ok);
    let flipped =
      List.filter population ~f:(fun key ->
          (not (exempt key))
          &&
          match values key with
          | None -> true
          | Some (resolver, candidates) ->
              let flippable =
                List.exists candidates ~f:(fun value ->
                    Option.is_some (with_stated key value resolver (fun () -> ())))
              in
              if not flippable then skip_outranked ~what:"the codegen-diff flip" key;
              flippable)
    in
    p_none ~min:4 "no such key changes the emitted kernel while leaving the cache key unchanged"
      flipped ~f:(fun key ->
        match values key with
        | None -> false
        | Some (resolver, candidates) ->
            List.exists candidates ~f:(fun value ->
                match with_stated key value resolver compile_probe with
                | None -> false
                | Some (key', src) ->
                    let under =
                      String.equal key' baseline_key && not (String.equal src baseline_src)
                    in
                    if under then
                      eprintf "%s=%s changes the emitted kernel but not the cache key\n" key value;
                    under
                | exception exn ->
                    eprintf "%s=%s: the probe compile raised %s\n" key value (Exn.to_string exn);
                    true))
  in
  (match with_stated "output_debug_files_in_build_directory" "true" Settings oracle with
  | Some () -> ()
  | None ->
      Verdict.fail
        "output_debug_files_in_build_directory is stated on the commandline or in the environment, \
         so the oracle cannot read the kernels it compiles");
  (* The reviewable part: what this sweep takes on trust, and why. *)
  printf "\nKeyed keys this sweep cannot flip in-process:\n";
  List.iter unflippable ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nKeys a compile reads that the codegen-diff oracle does not flip:\n";
  List.iter emitted_exemptions ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nKeyed keys hashed only on some backends:\n";
  List.iter keyed_flips ~f:(fun (key, flip) ->
      match flip.scope with
      | Every_backend -> ()
      | scope -> printf "  %s -- %s\n" key (scope_name scope))
