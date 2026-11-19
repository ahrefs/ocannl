(* gh-ocannl-2313: the post-op scope-grammar check refuses a real [Stage] that adds a tensor write
   to a pure [Local_scope]. [validate_loop_binders] requires unique loop symbols at optimize entry.
   Schedule rewrites can duplicate binders later, so the hand-built fixture uses [optimize_scoped]
   to give the Stage anchor a second binder inside the scope after optimize. Stage's anchor precheck
   follows staged reads, all outside the scope here, while the splice rewrites every occurrence of
   the anchor symbol. A distinct symbol for the scope's loop lets the same Stage write the tile
   outside the scope. *)

open Base
open Ocannl.Operation.DSL_modules
open Ll_test
open Verdict.Claims
module Sched = Ir.Schedule
module Outcome = Ir.Schedule_outcome

let node = node_factory ~first_id:2313000 ~dims:[| 2; 2 |] ()

let program ~duplicate_anchor name =
  let source = node "source" and out = node "out" in
  let local = node ~dims:[| 1 |] "local" and sink = node ~dims:[| 1 |] "sink" in
  virtualize local;
  let r = sym () and i = sym () in
  let scope_loop = if duplicate_anchor then r else sym () in
  let idcs = [| iter r; iter i |] and cell = [| fixed 0 |] in
  let copy = loop_n r 2 (loop_n i 2 (set out idcs (get source idcs))) in
  let id = LL.get_scope local in
  let scoped_value = scope_of id (loop_n scope_loop 2 (LL.Set_local (id, c 1.))) in
  let opt =
    optimize_scoped ~materialized:[ source; out; sink ] ~name
      ~raw:(seq copy (set sink cell (c 1.)))
      (seq copy (set sink cell scoped_value))
  in
  let stage =
    Sched.Stage
      {
        source;
        tile_loops = [ i ];
        shared = false;
        cooperative = None;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      }
  in
  (opt, stage)

let () =
  let opt, stage = program ~duplicate_anchor:true "op_scope_duplicate" in
  p "the entry with a duplicated Stage anchor has a pure scope body"
    (Option.is_none (LL.scope_purity_violation opt.llc));
  p "Stage is refused at transform time for leaving a tensor write inside a scope body"
    (match Sched.apply_classified [ stage ] opt with
    | exception Outcome.Cause_at (Outcome.Transform, Outcome.Illegal_schedule { detail; _ }) ->
        Stdio.eprintf "%s\n%!" detail;
        String.is_prefix detail
          ~prefix:"Schedule.Stage: leaves an impure scope body: Set writes the tensor node "
    | applied ->
        Option.iter (LL.scope_purity_violation applied.llc) ~f:(fun detail ->
            Stdio.eprintf "Stage applied with a scope purity violation: %s\n%!" detail);
        false);
  let opt, stage = program ~duplicate_anchor:false "op_scope_distinct" in
  let applied = Sched.apply_classified [ stage ] opt in
  let tile_writes =
    count_stmt ~in_scopes:false applied.llc ~f:(function
      | LL.Set { tn; _ } -> String.equal tn.namespace Sched.tile_namespace
      | _ -> false)
  in
  p "with a distinct scope-loop binder, Stage writes a tile outside the scope and leaves it pure"
    (tile_writes > 0 && Option.is_none (LL.scope_purity_violation applied.llc))
