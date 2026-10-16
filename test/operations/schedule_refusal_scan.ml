(* gh-ocannl-1323: a schedule op refuses typed, or the raise is an invariant -- never a bare
   [invalid_arg] in between.

   [Schedule.apply_classified] passes through only the refusals the ops raise typed
   ([Schedule.refuse], a [Schedule_outcome.Cause_at (Transform, Illegal_schedule _)]); any other
   exception escaping schedule application is a transform bug, fatal under
   [strict_failure_classification]. A new op precondition written the old way, as a bare
   [invalid_arg], would therefore end a search instead of declining one candidate -- loudly on the C
   backends, but on the GPU-only paths (Tensorize, cooperative Stage, Fold_mma) only on a GPU sweep.
   This scan makes the choice explicit where it is written: every untyped raise that schedule
   application can reach in [arrayjit/lib/schedule.ml] carries its own [(* invariant: ... *)]
   comment, saying why reaching it is a bug rather than a property of the candidate.

   What "can reach" means here: everything bound before [apply_typed], the body of both [apply] and
   [apply_classified]. OCaml's definition order makes that the whole of what application calls
   inside this file; what it calls in other modules ([Low_level.simplify_llc] and the like) is out
   of scope and stays fatal, which is the intent. An untyped raise is a [Pexp_ident] whose last
   component is [invalid_arg], [invalid_argf], [failwith], [failwithf] or [failwiths], however
   qualified, or a [Pexp_construct] of [Invalid_argument] or [Failure] -- read on the parse tree, so
   a comment or string naming them is not a site. The marker is the one textual part: a comment
   opening with the [marker] text on the site's line or within the [marker_window] lines above it,
   and each marker answers for one site only, so one comment cannot vouch for a neighbour it was not
   written about. *)

open Base
open Stdio
open Verdict.Claims
module Read = Test_utils.Config_key_scan
module Ast_traverse = Ppxlib.Ast_traverse

let bound = "apply_typed"
let marker = "(* invariant:"
let marker_window = 4
let untyped_idents = [ "invalid_arg"; "invalid_argf"; "failwith"; "failwithf"; "failwiths" ]
let untyped_constructors = [ "Invalid_argument"; "Failure" ]

type site = { line : int; what : string }

(* The line a top-level [name] is bound on, or [None] when the source binds no such value. *)
let binding_line structure name =
  List.find_map structure ~f:(fun (item : Ppxlib.structure_item) ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.find_map bindings ~f:(fun (vb : Ppxlib.value_binding) ->
              match vb.pvb_pat.ppat_desc with
              | Ppat_var { txt; _ } when String.equal txt name ->
                  Some item.pstr_loc.loc_start.pos_lnum
              | _ -> None)
      | _ -> None)

let untyped_sites structure =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        let line = expression.pexp_loc.loc_start.pos_lnum in
        (match expression.pexp_desc with
        | Pexp_ident _ -> (
            match Option.bind (Read.longident_of expression) ~f:List.last with
            | Some name when List.mem untyped_idents name ~equal:String.equal ->
                found := { line; what = name } :: !found
            | _ -> ())
        | Pexp_construct ({ txt; _ }, _) -> (
            match List.last (Read.flatten_longident txt) with
            | Some name when List.mem untyped_constructors name ~equal:String.equal ->
                found := { line; what = name } :: !found
            | _ -> ())
        | _ -> ());
        super#expression expression
    end
  in
  iterator#structure structure;
  List.sort !found ~compare:(fun a b -> Int.compare a.line b.line)

(* [Ok sites_in_region] when the bound is found, each site paired with whether a marker of its own
   answers for it; markers are consumed nearest-first, in source order. *)
let classify content =
  let structure = Read.structure_of content in
  match binding_line structure bound with
  | None -> Error ("the source binds no " ^ bound)
  | Some limit ->
      let lines = Array.of_list (String.split_lines content) in
      let has_marker n =
        n >= 1 && n <= Array.length lines && String.is_substring lines.(n - 1) ~substring:marker
      in
      let used = Hash_set.create (module Int) in
      let sites = List.filter (untyped_sites structure) ~f:(fun s -> s.line < limit) in
      Ok
        (List.map sites ~f:(fun site ->
             let candidates =
               List.init (marker_window + 1) ~f:(fun d -> site.line - d)
               |> List.filter ~f:(fun n -> has_marker n && not (Hash_set.mem used n))
             in
             match candidates with
             | n :: _ ->
                 Hash_set.add used n;
                 (site, true)
             | [] -> (site, false)))

let unmarked content =
  match classify content with
  | Error e -> Error e
  | Ok sites -> Ok (List.filter_map sites ~f:(fun (s, ok) -> Option.some_if (not ok) s))

(* Synthetic controls, each a source whose verdict is known: every spelling the reader catches, the
   marker's reach and its one-site budget, and the region bound. *)
let control_cases =
  let tail = "\nlet apply_typed x = x\n" in
  [
    ("a bare invalid_arg before the bound", "let f () = invalid_arg \"x\"" ^ tail, 1);
    ("a qualified Stdlib.invalid_arg", "let f () = Stdlib.invalid_arg \"x\"" ^ tail, 1);
    ("a failwith", "let f () = failwith \"x\"" ^ tail, 1);
    ("a Printf.ksprintf failwith", "let f () = Printf.ksprintf failwith \"%d\" 1" ^ tail, 1);
    ("a raised Invalid_argument", "let f () = raise (Invalid_argument \"x\")" ^ tail, 1);
    ("a raised Failure", "let f () = raise (Failure \"x\")" ^ tail, 1);
    ("a marked site", "(* invariant: unreachable. *)\nlet f () = invalid_arg \"x\"" ^ tail, 0);
    ( "two sites under one marker",
      "(* invariant: one of them. *)\nlet f () = invalid_arg \"x\"\nlet g () = failwith \"y\""
      ^ tail,
      1 );
    ( "a marker beyond the window",
      "(* invariant: too far. *)\n\n\n\n\n\nlet f () = invalid_arg \"x\"" ^ tail,
      1 );
    ("a site after the bound", "let apply_typed x = x\nlet f () = invalid_arg \"x\"\n", 0);
    ("a comment naming invalid_arg", "(* invalid_arg \"x\" *)\nlet f () = ()" ^ tail, 0);
  ]

let () =
  let source =
    match Array.to_list Stdlib.Sys.argv with
    | [ _; source ] -> source
    | _ ->
        eprintf "usage: schedule_refusal_scan <path to arrayjit/lib/schedule.ml>\n";
        Stdlib.exit 2
  in
  List.iter control_cases ~f:(fun (label, content, expected) ->
      pf "control: %s leaves %d unmarked site(s)" label expected
        (match unmarked content with Ok sites -> List.length sites = expected | Error _ -> false));
  p "control: a source without the bound is refused"
    (Result.is_error (unmarked "let f () = invalid_arg \"x\"\n"));
  let content = In_channel.read_all source in
  match classify content with
  | Error e -> fail (source ^ ": " ^ e)
  | Ok sites ->
      p_all "every untyped raise schedule application can reach carries its own invariant marker"
        sites ~f:(fun (site, ok) ->
          if not ok then
            eprintf
              "%s:%d: a bare %s schedule application can reach: raise a typed refusal with \
               [refuse], or mark it (* invariant: why reaching it is a bug *)\n"
              source site.line site.what;
          ok);
      let structure = Read.structure_of content in
      p "schedule.ml binds refuse, the typed alternative, before apply_typed"
        (match (binding_line structure "refuse", binding_line structure bound) with
        | Some r, Some b -> r < b
        | _ -> false)
