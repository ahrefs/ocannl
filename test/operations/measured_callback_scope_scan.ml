(* gh-ocannl-1465. A compiled test module that installs the measurement seam needs an
   uncontended-run witness or a read of one of the timing refusal records. Read the OCaml syntax
   tree of test and arrayjit/test modules. Recognize direct set/with_value applications to
   Autotune.on_candidate_measured, including locally opened Seam and partial applications; the
   approved witnesses are the uncontended scope, Scripted_search.search, and the refusal
   fields/accessors below. Comments, strings, simply mentioning a record label in a pattern,
   getters, callback aliases, inter-module control flow, and reachability of the witness from each
   application are outside this file-level check. *)

open Base
open Ppxlib
module Inventory = Test_scan.Source_inventory
module Manifest = Test_scan.Refusal_control_manifest

let identifier (e : expression) =
  match e.pexp_desc with Pexp_ident { txt; _ } -> Longident.flatten_exn txt | _ -> []

let scope_witness = function
  | [ "Autotune"; ("with_uncontended_test_windows" | "contention_refusals") ]
  | [ "Scripted_search"; "search" ] ->
      true
  | _ -> false

let refusal_field = function
  | "timings_contended" | "timings_unreadable" | "timings_unbatched" | "baseline_refusal"
  | "admission" | "discarded_window" ->
      true
  | _ -> false

let scan source =
  let setters = ref [] and witness = ref false in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_apply (f, args) -> (
            let path = identifier f in
            if scope_witness path then witness := true;
            match (path, args) with
            | ( ( [ ("set" | "with_value") ]
                | [ "Seam"; ("set" | "with_value") ]
                | [ "Autotune"; "Seam"; ("set" | "with_value") ] ),
                (Nolabel, seam) :: _ )
              when List.equal String.equal (identifier seam) [ "Autotune"; "on_candidate_measured" ]
              ->
                setters := e.pexp_loc.loc_start.pos_lnum :: !setters
            | _ -> ())
        | Pexp_field (_, { txt; _ }) ->
            if refusal_field (Longident.last_exn txt) then witness := true
        | _ -> ());
        super#expression e
    end
  in
  walker#structure (Parse.implementation (Lexing.from_string source));
  (!setters, !witness)

let () =
  let root, generated =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: root :: generated -> (root, generated)
    | [] | [ _ ] -> invalid_arg "expected a workspace root"
  in
  let check source = match scan source with _ :: _, false -> true | [], _ | _, true -> false in
  let setter = "let run = Autotune.Seam.with_value Autotune.on_candidate_measured f" in
  Verdict.p "a measurement setter with only a comment about the scope is reported"
    (check (setter ^ " (* Autotune.with_uncontended_test_windows run *)"));
  Verdict.p "a measured callback may use the uncontended scope or inspect the refusal record"
    ((not (check (setter ^ "\nlet () = Autotune.with_uncontended_test_windows run")))
    && (not (check (setter ^ "\nlet rejected r = r.Autotune.timings_contended > 0")))
    && not (check (setter ^ "\nlet () = Scripted_search.search ~tag comp")));
  Verdict.p "a getter does not require a timing witness"
    (not (check "let f = Autotune.Seam.get Autotune.on_candidate_measured"));
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  Inventory.select inventory ~f:(fun path ->
      String.is_suffix path ~suffix:".ml"
      && (String.is_prefix path ~prefix:"test/" || String.is_prefix path ~prefix:"arrayjit/test/"))
  |> List.filter_map ~f:(fun (file : Inventory.file) ->
      let source = Stdio.In_channel.read_all file.on_disk in
      if String.is_substring source ~substring:"on_candidate_measured" then
        match scan source with
        | [], _ -> None
        | lines, has_witness -> Some (file.path, List.rev lines, has_witness)
      else None)
  |> Verdict.p_all "test modules installing the measurement seam have a timing witness"
       ~f:(fun (path, lines, has_witness) ->
         if not has_witness then
           Stdio.eprintf "%s: no uncontended scope or refusal record for setter on line(s) %s\n"
             path
             (String.concat ~sep:", " (List.map lines ~f:Int.to_string));
         has_witness);
  Manifest.print "measured_callback_scope_scan.ml"
