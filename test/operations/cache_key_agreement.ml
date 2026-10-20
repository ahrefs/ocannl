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
   wrong together -- a key that changes the emitted kernel, is classified neither [Keyed] nor
   search-shaping, and is not hashed: every key a compile reads outside those classes is flipped,
   the probe recompiled, and a changed kernel under an unchanged key is reported by name.

   A [Keyed] key the sweep cannot flip in-process says why ([unflippable]), and the reason stays
   true: such a key must not be read per lookup by the key computation, or it could have been
   flipped. That table and the oracle's exemptions are the reviewable artifacts, printed below. *)

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
    ("autotune_timing", config [ "isolated"; "queued" ]);
  ]

(* Components that move WITH a classified one by derivation, not by a second reading of the key: the
   codegen tag hashes what the numerics policy resolves to on this backend
   ([Backend_intf.codegen_capabilities_fingerprint], gh-ocannl-1117), so a policy flip may move it
   too. Each listed pair is one fact under two views, which partitions the cache no finer than the
   classified component alone. *)
let derived_components = [ ("numerics", [ "codegen" ]) ]

let unflippable =
  [
    ( "backend",
      "a parameter, not a lookup: backends are process-wide singletons, and the component is the \
       backend name the caller passes" );
    ( "cc_pool_core_class",
      "resolved once, by the lazy [Cc_backend.pool_restriction] the first compile forces; the pool \
       tag it decides is what the component hashes" );
    ( "cc_grid_private_bytes_cap",
      "resolved once, by the lazy [C_syntax.per_chunk_private_bytes_cap] the first compile forces; \
       the codegen tag hashes that resolved value" );
  ]

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
   in the emitted C no part of the registry knew about. So every key a compile reads outside those
   two classes is flipped here, the kernel recompiled, and a changed kernel under an unchanged key
   is an under-keying, reported by name. [Search_shaping] is out by its class's own reason: it
   shapes the untuned default pipeline, which a cached winner replays its own ops over, so a
   different default kernel under one key is what the class says.

   Which keys a compile reads is TRACED, not listed, so a key that starts being read at compile time
   joins the oracle the day it does; this table must then give it values, or the first claim of the
   stage names it. Rows for keys this backend's compile does not read are other backends'. The
   settings-borne keys take their values from [settings_flips]. *)
let emitted_flips =
  [
    ("cc_backend_post_compile_timeout", [ "2.5" ]);
    ("cc_backend_verify_codesign", flag);
    ("fp_contract", flag);
    ("fp_reassociate", flag);
    ("fp_finite_math", flag);
    ("limit_constant_fill_size", [ "4096" ]);
    ("max_shape_error_origins", [ "3"; "9" ]);
    ("online_softmax", flag);
    ("output_dlls_in_build_directory", flag);
    ("output_prec_in_ll_files", flag);
    ("cd_ident_style", [ "name_only"; "name_and_label" ]);
  ]

(* Read at compile time, outside the two classes, and deliberately not flipped. *)
let emitted_exemptions =
  [
    ( "ll_ident_style",
      "identifier spelling: the canonical rendering alpha-renames identifiers away, and code that \
       differs only in their names compiles to the same kernel -- a changed text is what the key \
       is right to ignore" );
    ( "build_files_prefix",
      "it moves the very artifacts this oracle reads, and Generated refuses a set prefix" );
  ]

(* A kernel's text with its numbered identifiers renamed by first appearance: a fresh build of the
   program numbers its tensors and loop indices afresh ([n62], [i110]), and two texts that differ
   only there are one kernel. Number literals are copied whole, so a hex float's letters are not
   read as an identifier. *)
let alpha_normalized text =
  let names = Hashtbl.create (module String) in
  let out = Buffer.create (String.length text) in
  let n = String.length text in
  let is_ident_char c = Char.is_alphanum c || Char.equal c '_' in
  let rec scan i =
    if i < n then
      let c = text.[i] in
      if Char.is_alpha c || Char.equal c '_' then (
        let j = ref i in
        while !j < n && is_ident_char text.[!j] do
          Int.incr j
        done;
        let ident = String.sub text ~pos:i ~len:(!j - i) in
        if Char.is_digit ident.[String.length ident - 1] then
          let fresh = Hashtbl.length names in
          Buffer.add_string out
            ("@" ^ Int.to_string (Hashtbl.find_or_add names ident ~default:(fun () -> fresh)))
        else Buffer.add_string out ident;
        scan !j)
      else if Char.is_digit c then (
        let j = ref i in
        while !j < n && (is_ident_char text.[!j] || Char.equal text.[!j] '.') do
          Int.incr j
        done;
        Buffer.add_string out (String.sub text ~pos:i ~len:(!j - i));
        scan !j)
      else (
        Buffer.add_char out c;
        scan (i + 1))
  in
  scan 0;
  Buffer.contents out

let canonical_of ctx =
  let a =
    TDSL.ndarray [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] ~label:[ "cka_a" ] ~output_dims:[ 8 ] ()
  in
  let c = TDSL.O.(relu a + a) in
  let canon = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        canon := Some (SC.canonicalize ~static_indices:[] opt);
        [ opt ])
      ctx (Train.forward c) Ir.Indexing.Empty
  in
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

(* [f] under [key] stated as [value]; [None] when a commandline or environment statement outranks
   the config file, so the flip cannot take effect in this process. *)
let with_stated key value resolver f =
  match resolver with
  | Log_level_field ->
      let level0 = Utils.settings.log_level in
      Utils.settings.log_level <- Int.of_string value;
      Some (Exn.protect ~f ~finally:(fun () -> Utils.settings.log_level <- level0))
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

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  (* A backend-uniform golden cannot say which backend the sweep ran on (gh-ocannl-622). *)
  eprintf "cache_key_agreement: sweeping on the %s backend (not part of the golden)\n%!" backend;
  (* Before the first compile, as [Generated] requires: [canonical_of] compiles, and with debug
     artifacts configured on it would write through [build_files/] before [init]'s cleanup and
     symlink refusals ran (Codex P1, round 1). *)
  Test_utils.Generated.init ~backend_name:backend;
  let canon = canonical_of ctx in
  let components () =
    Option.value_exn ~here:[%here]
      (SC.cache_key_components ~timing_identity ~limits:(Context.hardware_limits ctx)
         ~capabilities:(Context.codegen_capabilities ctx)
         canon ~backend)
  in
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
        if List.is_empty flips then
          eprintf
            "%s: not flipped here -- it, or its regime, is stated on the commandline or in the \
             environment\n"
            key;
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
  (* 7. Stage 2: the codegen-diff oracle. The program is rebuilt per compile (memory modes are
     decided per tensor), under one routine name, with debug artifacts on so that [Generated] can
     read the kernel each compile emits. *)
  let routine = "cka_probe" in
  let probe =
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
        ctx probe Ir.Indexing.Empty
    in
    let canon = Option.value_exn ~here:[%here] !canon in
    let key =
      Option.value_exn ~here:[%here]
        (SC.cache_key ~timing_identity ~limits:(Context.hardware_limits ctx)
           ~capabilities:(Context.codegen_capabilities ctx)
           canon ~backend)
    in
    (key, alpha_normalized (Test_utils.Generated.read routine))
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
      p "the probe compiles to the same kernel under the same key twice over" stable;
      Set.of_list (module String) r
    in
    let oracle_class key =
      match Utils.classify_config_key key with
      | Some
          ( ( Utils.Code_borne | Utils.Execution_neutral | Utils.Placement_shaping
            | Utils.Untimed_model ),
            _ ) ->
          true
      | Some ((Utils.Keyed _ | Utils.Search_shaping | Utils.Aggregate), _) | None -> false
    in
    let settings_rows = Set.of_list (module String) (List.map settings_flips ~f:fst) in
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
    p_all ~min:5
      "every key a compile reads outside Keyed and search-shaping has oracle values or an exemption"
      population ~f:(fun key ->
        let ok = exempt key || Option.is_some (values key) in
        if not ok then eprintf "%s is read during a compile; give it values in emitted_flips\n" key;
        ok);
    let flipped = List.filter population ~f:(Fn.non exempt) in
    p_none ~min:5 "no such key changes the emitted kernel while leaving the cache key unchanged"
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
