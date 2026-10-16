(* gh-ocannl-1272: the search-trajectory identity compares resolved VALUES against the built-in
   default, not whether some source states a key.

   [Utils.config_class_fingerprint Search_shaping] keys [Autotune.tune]'s cached trajectory (the
   incumbent's depth record and the [abandonment-] records). It used to render every key some source
   sets, so a key stated at its default -- [autotune_keep_fraction=1] against the read site's
   ["1."], or the [strict_failure_classification=true] this directory's own [ocannl_config] states
   -- moved the identity and cost a warm flip chain its records. [flip_abandonment] shows that end
   to end; this pins the semantics key by key, without a search:

   - every [Search_shaping] key has a registered default its kind parses, and the registry names no
   other key; - each key stated at its default in another spelling leaves the identity alone, and
   each stated off it renders, canonically spelled; - a default the device resolves ([auto] lane
   modes, the block-size clamp) is compared as resolved
   ([Ir.Schedule.resolve_search_shaping_value]), on synthetic limits on either side of the
   economics; - a read site whose [~default:] disagrees with the registry is refused; - under a
   commandline-picked profile (the second invocation in the dune rule), the profile's off-default
   value renders as a statement of it would, and a statement restoring the built-in default renders
   nothing. *)

open Base
open Stdio
open Verdict.Claims

let class_keys cls =
  List.concat_map Utils.config_key_classification ~f:(fun (c, _, keys) ->
      if Poly.equal c cls then keys else [])

let search_keys = class_keys Utils.Search_shaping

(* Config values resolve per lookup, so a config-file entry poked in is what later reads see --
   unless the commandline or the environment states the key, which take precedence; [with_stated]
   reports whether the statement took effect, so no claim is drawn from one that did not. This
   directory's [ocannl_config] states some keys itself, so the previous entry is restored, not
   dropped. *)
let with_stated stated f =
  let saved = List.map stated ~f:(fun (key, _) -> (key, Hashtbl.find Utils.config_file_args key)) in
  List.iter stated ~f:(fun (key, data) -> Hashtbl.set Utils.config_file_args ~key ~data);
  Exn.protect
    ~finally:(fun () ->
      List.iter saved ~f:(fun (key, previous) ->
          match previous with
          | Some data -> Hashtbl.set Utils.config_file_args ~key ~data
          | None -> Hashtbl.remove Utils.config_file_args key))
    ~f:(fun () ->
      let took_effect =
        List.for_all stated ~f:(fun (key, data) ->
            Option.equal String.equal (Option.map (Utils.profile_key_source key) ~f:fst) (Some data))
      in
      (took_effect, f ()))

let fingerprint ?resolve () = Utils.config_class_fingerprint ?resolve Utils.Search_shaping

(* Another spelling of the same value under [kind], and a value off it -- derived from the kind, so
   a key registered later is covered with no list to extend here. *)
let respelled (kind : Utils.config_value_kind) default =
  match kind with
  | Flag -> if String.equal (String.lowercase default) "true" then " TRUE" else "0"
  | Integer -> "0" ^ default
  (* Exponent notation at round-trip precision: a default already written with three decimals
     ([autotune_composite_playoff_margin]'s 0.005) is its own [%.3f] spelling. *)
  | Float_number | Float_at_least _ -> Printf.sprintf "%.17e" (Float.of_string default)
  | Keyword -> String.uppercase default ^ " "
  | Text -> " " ^ default ^ "  "
  | Positive_float_or_unset -> "0"

let off_default (kind : Utils.config_value_kind) default =
  match kind with
  | Flag -> if String.equal (String.lowercase default) "true" then "false" else "true"
  | Integer -> Int.to_string (Int.of_string default + 1)
  | Float_number -> Float.to_string (Float.of_string default /. 2.)
  (* Above the floor, so the clamp does not fold it back onto another value. *)
  | Float_at_least { floor; _ } ->
      Float.to_string (Float.max (floor +. 1.) (Float.of_string default *. 2.))
  | Keyword -> default ^ "-other"
  | Text -> default ^ "elsewhere"
  | Positive_float_or_unset -> "1000."

let without_profile () =
  let registered =
    List.map search_keys ~f:(fun key -> (key, Utils.registered_config_default key))
  in
  p_all ~min:20
    "every Search_shaping key has a registered default, spelled so its own kind parses it"
    registered ~f:(fun (_, entry) ->
      match entry with
      | Some (kind, default) -> Option.is_some (Utils.canonical_config_value kind default)
      | None -> false);
  p_all ~min:20 "the registry names no key outside the Search_shaping class"
    Utils.config_key_defaults ~f:(fun (key, _) -> List.mem search_keys key ~equal:String.equal);
  let defaults =
    List.filter_map registered ~f:(fun (key, entry) ->
        Option.map entry ~f:(fun (kind, default) -> (key, kind, default)))
  in
  let baseline = fingerprint () in
  (* This directory's ocannl_config states strict_failure_classification=true, its default: before
     gh-ocannl-1272 that alone made every test's trajectory identity non-empty. *)
  p "with every Search_shaping key unset or at its default, the identity is empty"
    (String.is_empty baseline
    && Option.is_some (Utils.profile_key_source "strict_failure_classification"));
  p_all ~min:20
    "each Search_shaping key stated at its default, spelled otherwise, leaves the identity alone"
    defaults ~f:(fun (key, kind, default) ->
      let spelling = respelled kind default in
      let took_effect, shape = with_stated [ (key, spelling) ] fingerprint in
      (not (String.equal spelling default)) && took_effect && String.equal shape baseline);
  p_all ~min:20 "each Search_shaping key stated off its default renders, canonically spelled"
    defaults ~f:(fun (key, kind, default) ->
      let value = off_default kind default in
      let expected = key ^ "=" ^ Option.value_exn (Utils.canonical_config_value kind value) ^ ";" in
      let took_effect, shape = with_stated [ (key, value) ] fingerprint in
      took_effect && String.equal shape expected);
  (* A clamped kind canonicalizes as its read site reads it (gh-ocannl-740): below the floor is the
     floor, and an unparsable or NaN value is the fallback -- for [autotune_pad_waste_max], the
     default, so stating it leaves the identity alone. Derived from the registry, so a key given the
     kind later is covered too. *)
  let clamped =
    List.filter_map defaults ~f:(fun (key, kind, default) ->
        match kind with
        | Utils.Float_at_least { floor; fallback } ->
            Some (key, Utils.canonical_config_value kind default, floor, fallback)
        | _ -> None)
  in
  (* What stating [value] renders: nothing when it canonicalizes onto the default. *)
  let renders key canonical_default value =
    if Option.equal String.equal (Some value) canonical_default then "" else key ^ "=" ^ value ^ ";"
  in
  p_all "each clamped key stated unparsable or NaN renders as its fallback" clamped
    ~f:(fun (key, canonical_default, _, fallback) ->
      List.for_all [ "junk"; "nan" ] ~f:(fun spelling ->
          let took_effect, shape = with_stated [ (key, spelling) ] fingerprint in
          took_effect
          && String.equal shape (renders key canonical_default (Float.to_string fallback))));
  p_all "each clamped key stated below its floor renders as the floor" clamped
    ~f:(fun (key, canonical_default, floor, _) ->
      let took_effect, shape = with_stated [ (key, Float.to_string (floor /. 2.)) ] fingerprint in
      took_effect && String.equal shape (renders key canonical_default (Float.to_string floor)));
  (* A value no kind parses has nothing to compare: it renders as stated. *)
  let took_effect, shape = with_stated [ ("autotune_rounds", "two") ] fingerprint in
  p "an unparsable value renders as stated"
    (took_effect && String.equal shape "autotune_rounds=two;");
  (* The device-resolved defaults, on synthetic limits either side of the lane economics. *)
  let base = Ir.Backend_intf.no_hardware_limits in
  let cheap = { base with lane_scalar_recompute_cheap = true; lane_all_reduce_simdgroups = 2 } in
  let costly =
    {
      base with
      lane_scalar_recompute_cheap = false;
      lane_all_reduce_simdgroups = 1;
      max_threads_per_workgroup = Some 512;
    }
  in
  let at_default limits stated =
    let resolve = Ir.Schedule.resolve_search_shaping_value limits in
    let unstated = fingerprint ~resolve () in
    let took_effect, shape = with_stated stated (fingerprint ~resolve) in
    took_effect && String.is_empty unstated && String.equal shape unstated
  in
  p "where the lanes' scalar recompute is cheap, auto is the cooperative preamble and cut lanes"
    (at_default cheap [ ("gpu_lane_preamble_reduction", "cooperative") ]
    && at_default cheap [ ("gpu_serial_lanes", "Cut") ]
    && at_default cheap [ ("gpu_lane_all_reduce_simdgroups", "2") ]);
  p "and there the refused preamble, admitted lanes and one simdgroup are not the default"
    ((not (at_default cheap [ ("gpu_lane_preamble_reduction", "refused") ]))
    && (not (at_default cheap [ ("gpu_serial_lanes", "admitted") ]))
    && not (at_default cheap [ ("gpu_lane_all_reduce_simdgroups", "1") ]));
  p "where it is costly, auto is the refused preamble and admitted lanes, and the reverse renders"
    (at_default costly [ ("gpu_lane_preamble_reduction", "refused") ]
    && at_default costly [ ("gpu_serial_lanes", "admitted") ]
    && at_default costly [ ("gpu_lane_all_reduce_simdgroups", "1") ]
    && (not (at_default costly [ ("gpu_lane_preamble_reduction", "cooperative") ]))
    && not (at_default costly [ ("gpu_serial_lanes", "cut") ]));
  p "a block size the device clamps to the default's clamp is the default, unclamped it renders"
    (at_default
       { costly with max_threads_per_workgroup = Some 256 }
       [ ("gpu_schedule_block_size", "1024") ]
    && not (at_default costly [ ("gpu_schedule_block_size", "1024") ]));
  p "an explicit mode renders as the device resolves it"
    (let resolve = Ir.Schedule.resolve_search_shaping_value cheap in
     let took_effect, shape =
       with_stated [ ("gpu_lane_preamble_reduction", " Refused") ] (fingerprint ~resolve)
     in
     took_effect && String.equal shape "gpu_lane_preamble_reduction=refused;");
  (* The read sites' defaults are held to the registry where both meet: the read itself. *)
  p "a read whose default means the registered value is served, in any spelling"
    (String.equal "08"
       (Utils.get_global_arg ~arg_name:"autotune_split_reduce_max_sites" ~default:"08"));
  p "a read whose default means another value is refused"
    (match Utils.get_global_arg ~arg_name:"autotune_split_reduce_max_sites" ~default:"9" with
    | _ -> false
    | exception Invalid_argument _ -> true)

(* Run by the dune rule with --ocannl_profile=performance --ocannl_autotune_rounds=2. The payload
   sets autotune_search=true (the default), autotune_beam_width=4 and autotune_rounds=4. *)
let under_profile () =
  let source key = Utils.profile_key_source key in
  let from_profile key value =
    match source key with Some (v, Utils.From_profile _) -> String.equal v value | _ -> false
  in
  p "the performance profile sets the beam width off its default and the search at it"
    (from_profile "autotune_beam_width" "4" && from_profile "autotune_search" "true");
  p "the commandline restores the rounds the profile raises to their built-in default"
    (match source "autotune_rounds" with Some ("2", Utils.From_cmdline _) -> true | _ -> false);
  p
    "the identity is the profile's off-default value alone, as a profile-less statement of it \
     would render"
    (String.equal (fingerprint ()) "autotune_beam_width=4;")

let () =
  match Utils.active_profile with
  | None ->
      printf "Without a profile:\n";
      without_profile ()
  | Some (_, name, _) ->
      printf "Under the commandline-picked profile %s:\n" name;
      under_profile ()
