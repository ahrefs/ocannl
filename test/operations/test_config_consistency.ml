open Base
open Stdio

(* Failures go through [Verdict]: reported on both channels, and the run exits nonzero from its
   teardown. A golden-diff test that prints its failures and exits 0 can be `dune promote`d into
   passing, blessing the FAIL text as the expected output (Codex P2, round 10 of PR #343); since a
   nonzero exit means dune never writes the redirected stdout, the same lines go to stderr, where
   they survive to be read (gh-ocannl-601). *)
open Verdict.Claims

let printf = Test_utils.Refusal_control_manifest.printf

module Config_key_scan = Test_utils.Config_key_scan

(* The reference file ships with every setting COMMENTED OUT (gh-ocannl-559), so that copying it
   wholesale to an `ocannl_config` states no settings. A commented-out setting is spelled `#key=…`
   with no space after the `#`; prose comments (and the verbatim profile-payload blocks at the end
   of the file) always use `# `, so the two are told apart mechanically. *)
let uncomment line =
  match String.chop_prefix line ~prefix:"#" with
  | Some rest -> (
      match String.lsplit2 rest ~on:'=' with
      | Some (key, _)
        when (not (String.is_empty key))
             && String.for_all key ~f:(fun c ->
                 Char.is_alphanum c || Char.equal '_' c || Char.equal '-' c) ->
          rest
      | _ -> line)
  | None -> line

let extract_keys filename =
  In_channel.read_lines filename
  |> List.map ~f:(fun line -> uncomment (String.strip line))
  |> List.filter_map ~f:(fun line ->
      if
        String.is_empty line || String.is_prefix ~prefix:"#" line
        || String.is_prefix ~prefix:"~~" line
      then None
      else
        match String.lsplit2 line ~on:'=' with
        | Some (key, _) ->
            let key =
              String.lowercase
              @@ String.strip ~drop:(fun c -> Char.equal '-' c || Char.equal ' ' c) key
            in
            let key =
              if String.is_prefix key ~prefix:"ocannl" then
                String.drop_prefix key 6 |> String.strip ~drop:(Char.equal '_')
              else key
            in
            if String.is_empty key then None else Some key
        | None -> None)
  |> Set.of_list (module String)

let () =
  if Array.length Stdlib.Sys.argv < 3 then (
    eprintf "Usage: %s <reference_file> <source_file...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let reference_file = Stdlib.Sys.argv.(1) in
  (* The rest is the rule's whole dependency set, globbed over the library directories rather than
     enumerated by hand (gh-ocannl-592). *)
  let source_files =
    Config_key_scan.sources_among (Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2))
  in
  if List.is_empty source_files then (
    eprintf "%s: no sources among the arguments -- the rule's globs match nothing\n"
      Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let file_keys = extract_keys reference_file in
  let code_keys = Utils.known_config_keys in
  let source_keys = Config_key_scan.keys_in_files source_files in
  (* 1. Source call-site keys must all appear in the reference file *)
  let missing_in_ref = Set.diff source_keys file_keys in
  if not (Set.is_empty missing_in_ref) then
    fail
      (Printf.sprintf "call-site keys missing from %s: %s" reference_file
         (String.concat ~sep:", " @@ Set.to_list missing_in_ref));
  (* 2. Source call-site keys must all appear in known_config_keys registry *)
  let missing_in_registry = Set.diff source_keys code_keys in
  if not (Set.is_empty missing_in_registry) then
    fail
      (Printf.sprintf "call-site keys missing from known_config_keys registry: %s"
         (String.concat ~sep:", " @@ Set.to_list missing_in_registry));
  (* 3. known_config_keys and reference file must agree (bidirectional) *)
  let extra_in_ref = Set.diff file_keys code_keys in
  let extra_in_registry = Set.diff code_keys file_keys in
  if not (Set.is_empty extra_in_ref) then
    fail
      (Printf.sprintf "reference file has keys not in known_config_keys: %s"
         (String.concat ~sep:", " @@ Set.to_list extra_in_ref));
  if not (Set.is_empty extra_in_registry) then
    fail
      (Printf.sprintf "known_config_keys has keys not in reference file: %s"
         (String.concat ~sep:", " @@ Set.to_list extra_in_registry));
  (* 4. Profile payloads (gh-ocannl-559) are partial config files: every key they set must be a
     known, documented setting -- a payload is not a place a new key can hide. *)
  let payload_keys =
    List.map Utils.profile_payloads ~f:(fun (name, text) ->
        let keys =
          Utils.parse_config_lines ~source:("profile " ^ name) (String.split_lines text)
          |> List.map ~f:fst
          |> Set.of_list (module String)
        in
        let unknown = Set.diff keys code_keys in
        if not (Set.is_empty unknown) then
          fail
            (Printf.sprintf "profile %S sets keys missing from known_config_keys registry: %s" name
               (String.concat ~sep:", " @@ Set.to_list unknown));
        let undocumented = Set.diff keys file_keys in
        if not (Set.is_empty undocumented) then
          fail
            (Printf.sprintf "profile %S sets keys missing from %s: %s" name reference_file
               (String.concat ~sep:", " @@ Set.to_list undocumented));
        (name, keys))
  in
  (* 5. The payload texts are reproduced in the reference file between BEGIN/END markers, each line
     prefixed with "# ". Documentation of a value that can drift from the value is worse than no
     documentation, so the copy is checked rather than trusted. *)
  let reference_lines = In_channel.read_lines reference_file in
  List.iter Utils.profile_payloads ~f:(fun (name, text) ->
      let marker kind = Printf.sprintf "# --- %s PROFILE PAYLOAD %s ---" kind name in
      let index_of m =
        List.findi reference_lines ~f:(fun _ l -> String.equal (String.strip l) m)
        |> Option.map ~f:fst
      in
      match (index_of (marker "BEGIN"), index_of (marker "END")) with
      | Some b, Some e when e > b ->
          let quoted =
            List.sub reference_lines ~pos:(b + 1) ~len:(e - b - 1)
            |> List.map ~f:(fun l ->
                let l = Option.value (String.chop_prefix l ~prefix:"#") ~default:l in
                Option.value (String.chop_prefix l ~prefix:" ") ~default:l)
          in
          let expected = String.split_lines text in
          if not (List.equal String.equal quoted expected) then
            fail
              (Printf.sprintf
                 "the %s payload quoted in %s differs from the embedded one in \
                  arrayjit/lib/utils.ml"
                 name reference_file)
      | _ ->
          fail
            (Printf.sprintf "%s has no '%s' … '%s' block" reference_file (marker "BEGIN")
               (marker "END")));
  (* 6. Both consistency tests (this one and digest_completeness) find a configuration read by
     scanning sources for the key spelled as a string literal at the call site
     (Test_utils.Config_key_scan). That convention is load-bearing: a helper that takes the key as a
     parameter hides every key routed through it from BOTH scanners -- on staging PR #337 one such
     helper hid three real keys, and a deliberately unregistered fake key stayed green until the
     helper was removed. So the label must carry a literal everywhere except in the functions that
     legitimately move a key around, named one by one below: - in utils.ml, the lookup plumbing
     itself, which forwards the name between get_global_arg, get_global_flag and
     get_global_arg_with_source and on into the boolean parser; - in tnode.ml, get_style, which
     takes the key as an optional parameter defaulting to a literal, re-passed by its callers with
     literals of their own. Naming the functions rather than exempting their files is the difference
     between "this forwarding is known" and "nothing in this file is ever checked" (Codex P2, round
     1): utils.ml and tnode.ml go on reading configuration in the ordinary way everywhere else, and
     a new forwarding helper in either of them fails like anywhere else. What counts as a use, and
     which function it sits in, are the parse tree's answers rather than a pattern's
     (Config_key_scan): every spelling of the label that OCaml accepts is one node, and prose about
     the convention -- including this comment's neighbours in the scanned files -- is not code at
     all. An exemption reaches a named TOP-LEVEL function and nothing nested inside it, so a local
     helper, however it is introduced, forwards keys on its own account. *)
  let forwarding_sites =
    Map.of_alist_exn
      (module String)
      [
        ( "utils.ml",
          Set.of_list
            (module String)
            [
              "bool_of_config_string";
              "resolve_config_value";
              "get_global_arg_with_source";
              "get_global_arg";
              "get_global_flag";
              (* gh-ocannl-719: resolves one key of a profile payload with its source, for
                 profile_payload_sources, whose key list is the payload's own. Top-level, because an
                 exemption does not reach a lambda nested in an exempted function. *)
              "profile_key_source";
            ] );
        ("tnode.ml", Set.of_list (module String) [ "get_style" ]);
      ]
  in
  (* Both halves of this check are the lexer's answer rather than a pattern's
     (Config_key_scan): which uses of the label are not literals -- covering every spelling OCaml
     accepts, including the optional forms a textual matcher missed (Codex P2, round 3) -- and
     which top-level definition an offset sits in, so that a documentation comment quoting a
     column-zero `let get_global_arg` cannot lend an exempt name to unrelated code (round 4). Only
     the line quoted back to the reader is sliced from the text, at the offset the lexer reported. *)
  (* Keying the exemptions by basename reads well and is what this list is written as -- but it is
     unambiguous only while no two scanned files share a name. Rather than let a future
     `tensor/utils.ml` inherit `arrayjit/lib/utils.ml`'s exemptions in silence (Codex P2, round 10),
     the ambiguity fails here, where the fix is either a rename or a switch to paths. Precautionary
     while the list was written by hand; load-bearing now that whole directories are globbed. *)
  let duplicate_basenames = Config_key_scan.duplicate_basenames source_files in
  if not (List.is_empty duplicate_basenames) then
    fail
      (Printf.sprintf
         "scanned files share a basename, which the exemption list is keyed by -- rename one or \
          key the list by path: %s"
         (String.concat ~sep:", " duplicate_basenames));
  let line_at original i =
    let line_start =
      match String.rfindi original ~pos:i ~f:(fun _ c -> Char.equal c '\n') with
      | Some k -> k + 1
      | None -> 0
    in
    let line_end =
      match String.lfindi original ~pos:i ~f:(fun _ c -> Char.equal c '\n') with
      | Some k -> k
      | None -> String.length original
    in
    String.strip (String.sub original ~pos:line_start ~len:(line_end - line_start))
  in
  let forwarding_hit = ref (Set.empty (module String)) in
  (* 6b. The OTHER spelling of a read -- a field of `Utils.settings` -- is recognised by that
     receiver, and a read spelled any other way (bare `settings.k` under an `open Utils`, or
     `U.settings.k` under an alias, at any depth) vanishes from the census: a key read only that way
     at codegen could then be classified code-borne and pass digest_completeness unchallenged (Codex
     P2, rounds 14 and 15). The READ is what is checked, not the scope, so a qualified record
     expression like row.ml's `Utils.{ value; unique_id }` is not a finding. utils.ml is where the
     record lives, so its own unqualified reads are the definition, not a hidden call site. *)
  List.iter source_files ~f:(fun fname ->
      let base = Stdlib.Filename.basename fname in
      if not (String.equal base "utils.ml") then
        let original = In_channel.read_all fname in
        List.iter (Config_key_scan.unqualified_settings_reads original) ~f:(fun offset ->
            fail
            @@ Printf.sprintf
                 "%s uses the settings record or one of its predicates somewhere the census cannot \
                  follow -- a read without the `Utils.settings` receiver, or either handed on to \
                  be read elsewhere: %s"
                 base (line_at original offset)));
  List.iter source_files ~f:(fun fname ->
      let base = Stdlib.Filename.basename fname in
      let known =
        Option.value (Map.find forwarding_sites base) ~default:(Set.empty (module String))
      in
      let original = In_channel.read_all fname in
      let definitions = Config_key_scan.definitions original in
      let enclosing offset = Config_key_scan.definition_at definitions offset in
      List.iter (Config_key_scan.label_uses original) ~f:(fun use ->
          match use.Config_key_scan.key with
          (* A literal, so the convention holds -- but no setting is named by the empty string, and
             the censuses drop it as not a key. Dropping and reporting are different things, and
             this is the one that reports (Codex P2, round 12). *)
          | Some "" ->
              fail
              @@ Printf.sprintf "%s names the empty string as a config key: %s" base
                   (line_at original use.Config_key_scan.offset)
          | Some _ -> ()
          | None -> (
              let offset = use.Config_key_scan.offset in
              let definition = enclosing offset in
              (* An exemption reaches a TOP-LEVEL binding of the file and nothing else: whatever
                 nesting a same-named binding hides in, it has to earn its own exemption. *)
              let exempt_name =
                match definition with
                | Some { Config_key_scan.name = Some name; top_level = true; _ }
                  when Set.mem known name ->
                    Some name
                | _ -> None
              in
              match exempt_name with
              | Some name -> forwarding_hit := Set.add !forwarding_hit (base ^ ":" ^ name)
              | None ->
                  let where =
                    match definition with
                    | None -> "<no enclosing definition>"
                    | Some { Config_key_scan.name; top_level; _ } ->
                        Option.value name ~default:"<anonymous function>"
                        ^ if top_level then "" else " (nested)"
                  in
                  fail
                  @@ Printf.sprintf
                       "%s does not spell the config key as a string literal, in %s: %s" base where
                       (line_at original offset))));
  (* An exemption is a claim that a named function forwards a key; a claim that stops being true is
     stale, not a free pass, so each one has to earn its place on every run. *)
  let exempted_sites =
    Map.to_alist forwarding_sites
    |> List.concat_map ~f:(fun (base, fns) ->
        List.map (Set.to_list fns) ~f:(fun fn -> base ^ ":" ^ fn))
    |> Set.of_list (module String)
  in
  let stale = Set.diff exempted_sites !forwarding_hit in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted functions that no longer forward a config key -- drop them from the exemption \
          list: %s"
         (String.concat ~sep:", " @@ Set.to_list stale));
  (* What the census covers, as a floor per scan root rather than as an exact count: the count was a
     tally of the repository, and every correct addition anywhere under the globbed directories owed
     a promote round for it (gh-ocannl-701). A floor keeps what the count was for -- a glob that
     stops matching goes to zero -- and the exact numbers go to stderr, which a `(test)` stanza's
     golden does not see. *)
  Config_key_scan.report_counts source_files;
  let floor_violations = Config_key_scan.floor_violations source_files in
  List.iter floor_violations ~f:fail;
  Verdict.p "every scanned root meets its source-count floor" (List.is_empty floor_violations);
  if not (Verdict.any_failed ()) then (
    printf
      "OK: %d call-site keys, all in reference file and registry; registry and reference agree on \
       %d keys.\n"
      (Set.length source_keys) (Set.length code_keys);
    printf "OK: %d profile payloads, documented and quoted verbatim: %s.\n"
      (List.length payload_keys)
      (String.concat ~sep:", "
      @@ List.map payload_keys ~f:(fun (name, keys) ->
          Printf.sprintf "%s (%d keys)" name (Set.length keys)));
    printf
      "OK: every scanned file spells every config key as a string literal, outside %d forwarding \
       functions: %s.\n"
      (Set.length exempted_sites)
      (String.concat ~sep:", " @@ Set.to_list exempted_sites);
    (* Which directories the census came from, so that the globs' reach is reviewable rather than
       implicit -- by name, since how many files each holds is a tally of the repository rather than
       a fact about it (gh-ocannl-701). The counts are on stderr, and the floors under them are
       asserted above. *)
    printf "OK: scanned %s.\n" (String.concat ~sep:", " (Config_key_scan.scan_roots source_files)));
  Test_utils.Refusal_control_manifest.print "test_config_consistency.ml"
