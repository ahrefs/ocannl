(* gh-ocannl-1253: every field of the device description has a cache-identity class, and the cache
   key hashes the Key-class projection only.

   [Backend_intf.limits_key] IS the classification: it destructures the whole [hardware_limits]
   record with no [; _], copying each [Key] field into [Limits_key.t] and binding each [Untimed] one
   to [_], so a new field is a compile error until it is classified. This test holds what the
   compiler cannot:

   The classes are read off the code rather than restated: a field is [Key] exactly when the
   projection's sexp names it, and the field set is what the derived sexp of a record with every
   field present names; a toggle per field (below) must cover that set exactly.

   Toggling an [Untimed] field leaves the schedule cache's codegen component
   ([Schedule_cache.codegen_tag], which hashes the projection) unchanged; toggling a [Key] field
   changes it.

   The projection renders byte-identically to the full record with its [Untimed] fields absent (they
   are [[@sexp.option]]), so the key of every record that existed before they did is the key it had,
   on every preset here ([schedule_cache_device] makes the same claim on the running backend's own
   limits; this scan holds no backend).

   No [Untimed] field is read outside [backend_intf.ml] (record literals SET them; a field access or
   a binding record pattern READS), the accessor [Backend_intf.untimed_limits] is referenced only
   from [autotune.ml], and inside [autotune.ml] no top-level definition [Autotune.tune] reaches
   references one that (transitively) reaches the accessor. The codegen-stage modules are thereby
   covered by the first rule. The reach is a call graph over [autotune.ml]'s top-level bindings and
   modules by name, which over-approximates (a local shadowing a top-level name adds an edge), so it
   can refuse wrongly but not pass wrongly within the file; other modules cannot reach the fields at
   all.

   The scan rules are pure functions over source text, run on synthetic violations and their nearest
   honest neighbours before the repository. *)

open Base
open Stdio
open Verdict.Claims
module BI = Ir.Backend_intf
module SC = Ir.Schedule_cache
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse

(* A record with every field present (options [Some], the [[@sexp.option]] ones included), so its
   derived sexp names every field. A full literal, not a [with] update: a new field fails to compile
   here until it is given a value -- and a toggle below. *)
let full =
  {
    BI.max_bound_pools = Some 31;
    max_threads_per_workgroup = Some 1024;
    max_workgroup_memory_bytes = Some 49152;
    max_thread_private_bytes = Some 4096;
    max_workgroup_dims = Some (1024, 1024, 64);
    max_grid_yz = Some 65535;
    mma = Some BI.minimal_mma_capability;
    simd_vector = { Ir.Vector_file.bytes = 16; registers = 32 };
    peak_flops = Some 1e13;
    peak_memory_bandwidth = Some 1e12;
    cache_capacity_bytes = Some (1 lsl 30);
    max_resident_threads = Some 65536;
    cache_bandwidth = Some 2e12;
    issue_rate = Some 3e12;
    dram_bandwidth = Some 5e11;
    native_fp16_arithmetic = true;
    simdgroup_width = Some 32;
    lane_scalar_recompute_cheap = true;
    lane_all_reduce_simdgroups = 4;
    online_softmax_auto_block = 16;
    worker_pool_tag = Some "w8";
    codegen_tag = Some "tag";
  }

(* One change per field, each to a value the field cannot already hold in [full] -- two for the
   vector file, one per component. *)
let toggles : (string * (BI.hardware_limits -> BI.hardware_limits)) list =
  [
    ("max_bound_pools", fun l -> { l with max_bound_pools = Some 7 });
    ("max_threads_per_workgroup", fun l -> { l with max_threads_per_workgroup = Some 512 });
    ("max_workgroup_memory_bytes", fun l -> { l with max_workgroup_memory_bytes = Some 1 });
    ("max_thread_private_bytes", fun l -> { l with max_thread_private_bytes = Some 1 });
    ("max_workgroup_dims", fun l -> { l with max_workgroup_dims = Some (1, 1, 1) });
    ("max_grid_yz", fun l -> { l with max_grid_yz = Some 1 });
    ("mma", fun l -> { l with mma = None });
    (* Both halves of the vector file (gh-ocannl-1399): each moves the key on its own. *)
    ("simd_vector", fun l -> { l with simd_vector = { l.simd_vector with bytes = 64 } });
    ("simd_vector", fun l -> { l with simd_vector = { l.simd_vector with registers = 16 } });
    ("peak_flops", fun l -> { l with peak_flops = Some 1. });
    ("peak_memory_bandwidth", fun l -> { l with peak_memory_bandwidth = Some 1. });
    ("cache_capacity_bytes", fun l -> { l with cache_capacity_bytes = Some 1 });
    ("max_resident_threads", fun l -> { l with max_resident_threads = Some 1 });
    ("cache_bandwidth", fun l -> { l with cache_bandwidth = Some 1. });
    ("issue_rate", fun l -> { l with issue_rate = Some 1. });
    ("dram_bandwidth", fun l -> { l with dram_bandwidth = Some 1. });
    ("native_fp16_arithmetic", fun l -> { l with native_fp16_arithmetic = false });
    ("simdgroup_width", fun l -> { l with simdgroup_width = Some 64 });
    ("lane_scalar_recompute_cheap", fun l -> { l with lane_scalar_recompute_cheap = false });
    ("lane_all_reduce_simdgroups", fun l -> { l with lane_all_reduce_simdgroups = 1 });
    ("online_softmax_auto_block", fun l -> { l with online_softmax_auto_block = 0 });
    ("worker_pool_tag", fun l -> { l with worker_pool_tag = Some "w24" });
    ("codegen_tag", fun l -> { l with codegen_tag = Some "other" });
  ]

let field_names sexp =
  match sexp with
  | Sexp.List fields ->
      List.filter_map fields ~f:(function
        | Sexp.List (Sexp.Atom name :: _) -> Some name
        | _ -> None)
  | Sexp.Atom _ -> []

let all_fields = field_names (BI.sexp_of_hardware_limits full)
let key_fields = field_names (BI.Limits_key.sexp_of_t (BI.limits_key full))
let is_key name = List.mem key_fields name ~equal:String.equal
let untimed_fields = List.filter all_fields ~f:(Fn.non is_key)

let key_of limits =
  SC.codegen_tag ~backend:"cuda" ~limits ~capabilities:BI.no_codegen_capabilities ()

let () =
  let names = List.map toggles ~f:fst in
  p_all ~min:1 "every field of the record has a toggle here" all_fields ~f:(fun f ->
      List.mem names f ~equal:String.equal);
  p_all ~min:1 "every toggle here names a field of the record" names ~f:(fun f ->
      List.mem all_fields f ~equal:String.equal);
  (* The ranking score's capacity, residency, and its cache, issue and DRAM rates. *)
  p "the Untimed fields are the ranking score's capacity, residency and three rates"
    (List.equal String.equal
       (List.sort untimed_fields ~compare:String.compare)
       [
         "cache_bandwidth";
         "cache_capacity_bytes";
         "dram_bandwidth";
         "issue_rate";
         "max_resident_threads";
       ]);
  let base = key_of full in
  p_all ~min:(List.length all_fields)
    "toggling a Key field changes the cache key, toggling an Untimed field leaves it unchanged"
    toggles ~f:(fun (name, toggle) ->
      let moved = not (String.equal base (key_of (toggle full))) in
      if Bool.equal moved (is_key name) then true
      else (
        eprintf "field %s: key %s, class %s\n" name
          (if moved then "moved" else "unchanged")
          (if is_key name then "Key" else "Untimed");
        false))

let presets =
  [
    ("no_hardware_limits", BI.no_hardware_limits);
    ("every field present", full);
    ( "a CUDA-shaped record",
      {
        BI.no_hardware_limits with
        max_threads_per_workgroup = Some 1024;
        max_workgroup_memory_bytes = Some 49152;
        max_workgroup_dims = Some (1024, 1024, 64);
        max_grid_yz = Some 65535;
        mma = Some BI.minimal_mma_capability;
        simdgroup_width = Some 32;
        lane_scalar_recompute_cheap = true;
        lane_all_reduce_simdgroups = 4;
        peak_flops = Some 1.5e13;
        peak_memory_bandwidth = Some 8.0e12;
        cache_capacity_bytes = Some (1 lsl 30);
        max_resident_threads = Some 71680;
        cache_bandwidth = Some 3e12;
        issue_rate = Some 3.4e12;
        dram_bandwidth = Some 5.8e11;
        codegen_tag = Some "graph-capture/no-device-debug";
      } );
  ]

let () =
  p_all ~min:(List.length presets)
    "the Key projection renders as the full record with its Untimed fields absent, on every preset"
    presets ~f:(fun (name, l) ->
      let without =
        {
          l with
          cache_capacity_bytes = None;
          max_resident_threads = None;
          cache_bandwidth = None;
          issue_rate = None;
          dram_bandwidth = None;
        }
      in
      let ok =
        Sexp.equal (BI.Limits_key.sexp_of_t (BI.limits_key l)) (BI.sexp_of_hardware_limits without)
      in
      if not ok then eprintf "preset %s: projection differs\n" name;
      ok)

(* {2 The source scan} *)

let accessor = "untimed_limits"

(* Reads of an Untimed field in one source: field accesses, and record patterns binding the field to
   anything but [_]. Record expressions set fields and are not reads. *)
let field_reads ~fields (source : string) : string list =
  let found = ref [] in
  let last = function Ppxlib.Longident.Lident s | Ldot (_, s) -> Some s | Lapply _ -> None in
  let hit lid =
    Option.value_map (last lid) ~default:false ~f:(List.mem fields ~equal:String.equal)
  in
  let iter =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_field (_, { txt; _ }) when hit txt -> found := Option.value_exn (last txt) :: !found
        | _ -> ());
        super#expression e

      method! pattern p =
        (match p.ppat_desc with
        | Ppat_record (fields_pats, _) ->
            List.iter fields_pats ~f:(fun ({ Ppxlib.txt; _ }, (pat : pattern)) ->
                match pat.ppat_desc with
                | Ppat_any -> ()
                | _ -> if hit txt then found := Option.value_exn (last txt) :: !found)
        | _ -> ());
        super#pattern p
    end
  in
  iter#structure (Ppxlib.Parse.implementation (Lexing.from_string source));
  List.rev !found

(* Whether a source names [name] as a value (bare or qualified). *)
let references ~name (source : string) : bool =
  let found = ref false in
  let iter =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_ident { txt = Lident s | Ldot (_, s); _ } when String.equal s name -> found := true
        | _ -> ());
        super#expression e
    end
  in
  iter#structure (Ppxlib.Parse.implementation (Lexing.from_string source));
  !found

(* The top-level definitions of a source -- value bindings and modules, by name -- each with the
   names it references: bare identifiers, a qualified identifier's head module and its last
   component. A qualified [Other.f] thereby also counts as a reference to a top-level [f], one more
   way the reach over-approximates. *)
let top_level_graph (source : string) : (string * string list) list =
  let structure = Ppxlib.Parse.implementation (Lexing.from_string source) in
  let refs_of visit =
    let acc = ref [] in
    let iter =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_ident { txt = Lident s; _ } -> acc := s :: !acc
          | Pexp_ident { txt = Ldot (Lident m, s); _ } -> acc := m :: s :: !acc
          | Pexp_ident { txt = Ldot (_, s); _ } -> acc := s :: !acc
          | _ -> ());
          super#expression e

        method! module_expr m =
          (match m.pmod_desc with Pmod_ident { txt = Lident s; _ } -> acc := s :: !acc | _ -> ());
          super#module_expr m
      end
    in
    visit iter;
    !acc
  in
  List.concat_map structure ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.filter_map bindings ~f:(fun vb ->
              match vb.pvb_pat.ppat_desc with
              | Ppat_var { txt; _ } | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _) ->
                  Some (txt, refs_of (fun it -> it#expression vb.pvb_expr))
              | _ -> None)
      | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
          [ (name, refs_of (fun it -> it#module_expr pmb_expr)) ]
      | _ -> [])

(* The top-level definitions of [source] that [root] reaches (itself included), and those of them
   that (transitively) reach a definition referencing [reader]. *)
let reach ?(through = []) ~root ~reader (source : string) : string list * string list =
  let graph = top_level_graph source in
  let edges name =
    List.concat_map graph ~f:(fun (n, refs) -> if String.equal n name then refs else [])
    |> List.filter ~f:(fun r -> List.Assoc.mem graph r ~equal:String.equal)
  in
  let direct name =
    List.exists graph ~f:(fun (n, refs) ->
        String.equal n name && List.mem refs reader ~equal:String.equal)
  in
  let rec closure ?(cut = []) seen = function
    | [] -> seen
    | n :: rest ->
        if Set.mem seen n then closure ~cut seen rest
        else
          let next = List.filter (edges n) ~f:(fun r -> not (List.mem cut r ~equal:String.equal)) in
          closure ~cut (Set.add seen n) (next @ rest)
  in
  let names = List.dedup_and_sort (List.map graph ~f:fst) ~compare:String.compare in
  let reaches_reader n =
    Set.exists (closure ~cut:through (Set.empty (module String)) [ n ]) ~f:direct
  in
  (* [through]: sanctioned routes -- the root's reach does not continue into them. *)
  let reached =
    Set.to_list (closure ~cut:through (Set.empty (module String)) [ root ])
    |> List.filter ~f:(fun n -> List.mem names n ~equal:String.equal)
  in
  (reached, List.filter reached ~f:reaches_reader)

let reached_readers ?through ~root ~reader source = snd (reach ?through ~root ~reader source)

(* The routes by which an [Untimed] reading may reach [Autotune.tune]. The seeded untuned plan
   (gh-ocannl-1253), whose [auto] gate reads the capacity, and whose resolved content digest joins
   the search trajectory's identity -- held below by requiring [tune] to name the digest. And the
   seeded transform (gh-ocannl-1441), through which the untuned fallback of a call that does not
   search compiles what the recipes' untuned compile ([Autotune.seeded_default]) compiles: that call
   times, stores and replays nothing, so the reading reaches no record a search compares or a cache
   replays. *)
let sanctioned = [ "seeded_plan"; "seeded_transform" ]

let () =
  let fields = [ "cache_capacity_bytes" ] in
  (* One source per rule, each holding the violation and its nearest honest neighbours, so the claim
     is that the rule finds exactly the violation: neither blind nor firing on the neighbours. *)
  p "the read rule finds the field access and the binding pattern, not the literal or the discard"
    (List.equal String.equal
       (field_reads ~fields
          "let a l = l.Backend_intf.cache_capacity_bytes\n\
           let b { cache_capacity_bytes = c; _ } = c\n\
           let c l = { l with cache_capacity_bytes = Some 1 }\n\
           let d { cache_capacity_bytes = _; x } = x")
       [ "cache_capacity_bytes"; "cache_capacity_bytes" ]);
  let synthetic =
    "let ranking () = Backend_intf.untimed_limits ()\n\
     let helper () = ranking ()\n\
     module M = struct let r () = untimed_limits () end\n\
     let model_default () = ranking ()\n\
     let unrelated () = 0\n\
     let via_helper () = helper ()\n\
     let via_module () = M.r ()\n\
     let tune () = unrelated ()"
  in
  let reach root = reached_readers ~root ~reader:accessor synthetic in
  p
    "the reach rule refuses a root reaching the accessor through a helper or a module only, and \
     not through a sanctioned route"
    (List.equal String.equal (reach "via_helper") [ "helper"; "ranking"; "via_helper" ]
    && List.equal String.equal (reach "via_module") [ "M"; "via_module" ]
    && List.equal String.equal (reach "model_default") [ "model_default"; "ranking" ]
    && List.equal String.equal (reach "tune") []
    && List.equal String.equal
         (reached_readers ~through:[ "helper" ] ~root:"via_helper" ~reader:accessor synthetic)
         []
    && List.equal String.equal
         (reached_readers ~through:[ "helper" ] ~root:"via_module" ~reader:accessor synthetic)
         [ "M"; "via_module" ])

let () =
  let sources =
    Array.to_list (Array.subo (Sys.get_argv ()) ~pos:1)
    |> List.filter ~f:(String.is_suffix ~suffix:".ml")
    |> List.filter ~f:(fun path -> not (String.is_suffix path ~suffix:".pp.ml"))
  in
  let base path = Stdlib.Filename.basename path in
  let read path = In_channel.read_all path in
  let is_lib path = String.is_substring path ~substring:"arrayjit/lib/" in
  p_exists "the scan reads backend_intf.ml" sources ~f:(fun f ->
      String.equal (base f) "backend_intf.ml");
  let outside_intf =
    List.filter sources ~f:(fun f -> not (String.equal (base f) "backend_intf.ml"))
  in
  p_all ~min:20 "no source but backend_intf.ml reads an Untimed field" outside_intf ~f:(fun f ->
      match field_reads ~fields:untimed_fields (read f) with
      | [] -> true
      | reads ->
          eprintf "%s reads %s\n" f (String.concat ~sep:", " reads);
          false);
  p_all ~min:20 "only autotune.ml references the Untimed accessor"
    (List.filter outside_intf ~f:(fun f -> not (String.equal (base f) "autotune.ml")))
    ~f:(fun f ->
      let ok = not (references ~name:accessor (read f)) in
      if not ok then eprintf "%s references %s\n" f accessor;
      ok);
  let autotune = List.find sources ~f:(fun f -> is_lib f && String.equal (base f) "autotune.ml") in
  p "the scan reads arrayjit/lib/autotune.ml" (Option.is_some autotune);
  Option.iter autotune ~f:(fun f ->
      let source = read f in
      p "autotune.ml references the Untimed accessor" (references ~name:accessor source);
      p "model_default reaches the Untimed accessor (the reach is not blind)"
        (List.mem
           (reached_readers ~root:"model_default" ~reader:accessor source)
           "model_default" ~equal:String.equal);
      p "tune is a top-level definition of autotune.ml"
        (List.Assoc.mem (top_level_graph source) "tune" ~equal:String.equal);
      let reached, readers = reach ~through:sanctioned ~root:"tune" ~reader:accessor source in
      let refs_of n =
        List.concat_map (top_level_graph source) ~f:(fun (m, refs) ->
            if String.equal m n then refs else [])
      in
      let names r n = List.mem (refs_of n) r ~equal:String.equal in
      p_exists
        "the definition tune reaches the seeded plan through records the plan's digest in its \
         trajectory identity"
        reached ~f:(fun n -> names "seeded_plan" n && names "seeded_plan_digest" n);
      if not (List.is_empty readers) then
        eprintf "tune reaches: %s\n" (String.concat ~sep:", " readers);
      p_empty ~min:10 "nothing Autotune.tune reaches outside the seeded plan reads an Untimed field"
        ~over:reached readers);
  Test_utils.Refusal_control_manifest.print "limits_field_classes.ml"
