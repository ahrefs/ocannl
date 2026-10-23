(* Where a [Schedule.Stage]'s loads go relative to a [Local_scope] a materializing [Unroll] minted
   (gh-ocannl-1534, following gh-ocannl-1467). A scope body is hoisted out of its position, so it
   may write no tensor node and hold no barrier; [Stage] places its load nest at the start of its
   anchor (the deepest loop an outer-part index of the staged read varies with), or, shared and
   unanchored, around its outermost tile loop.

   1. An anchor bound inside the scope is refused by [Stage] itself, typed and named for the decline
   census, before any tile is minted -- including the issue's own shape, a [Stage] over a loop
   OUTSIDE the scope whose staged read varies with loops inside it. Hoisting those loads ahead of
   the scope is unsound at this tile shape: the tile holds the cells of ONE anchor value, and ahead
   of the scope the anchor is unbound. This is the claim that fails if the loads are ever hoisted.

   2. The widened stage the refusal names -- the scope-bound loops of the staged index added to
   [tile_loops] -- stages the scope's whole footprint ahead of it: admitted, no tile write in a
   scope body, and it executes to the unscheduled program's value.

   3. A shared, unanchored stage whose outermost tile loop is inside the scope has a sound lowering:
   its loads and barriers wrap the statement holding the scope (with no anchor the tile depends on
   no loop between that statement and the tile loop). Admitted -- on one copy of the scope-nested
   loop and on two copies sharing one scope -- with nothing in a scope body, the load nest ahead of
   the statement, an [op_legality] probe that leaves the program's traced store alone, and executed
   parity with the unscheduled program.

   Inputs are dyadic and distinct per cell, so every sum is exact in f32 whatever the order, and a
   tile slot holding the wrong cell changes the result. *)

open Base
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Outcome = Ir.Schedule_outcome
open Verdict.Claims

let it = Ll_test.iter
let node = Ll_test.node_factory ~first_id:15340 ~dims:[| 1 |] ()
let floats l = String.concat ~sep:"," (List.map l ~f:Float.to_string)
let add a b = LL.Binop (Ir.Ops.Add, (a, Ir.Ops.single), (b, Ir.Ops.single))

(* Distinct dyadic cells: [(i + 1) / 8] in row-major order. *)
let cells n = Array.init n ~f:(fun i -> Float.of_int (i + 1) /. 8.)

(* [r { s { u { t { acc[0] += x[...] } } } }]; [index ~r ~s ~u ~t] is the read of [x]. *)
let build name ~s_extent ~x_dims ~index =
  let acc = node (name ^ "_acc") and x = node ~dims:x_dims (name ^ "_x") in
  Ll_test.materialize acc;
  Ll_test.materialize x;
  let r = Ll_test.sym () and s = Ll_test.sym () and u = Ll_test.sym () and t = Ll_test.sym () in
  let c0 = [| Ll_test.fixed 0 |] in
  let raw =
    Ll_test.loop_n r 2
      (Ll_test.loop_n s s_extent
         (Ll_test.loop_n u 2
            (Ll_test.loop_n t 2
               (Ll_test.set acc c0 (add (Ll_test.get acc c0) (Ll_test.get x (index ~r ~s ~u ~t)))))))
  in
  (Ll_test.optimize ~materialized:[ acc; x ] ~name raw, acc, x, (r, s, u, t))

(* The issue's fixture: [x[r, s, u, t]] with [s] of extent 1, so the materializing [Unroll] of [s]
   leaves [u] and [t] inside the minted scope while [r] stays outside it. *)
let build_rsut name =
  build name ~s_extent:1 ~x_dims:[| 2; 1; 2; 2 |] ~index:(fun ~r ~s ~u ~t ->
      [| it r; it s; it u; it t |])

(* A read that varies only with the scope-nested loops: [x[u, t]]. *)
let build_ut name ~s_extent =
  build name ~s_extent ~x_dims:[| 2; 2 |] ~index:(fun ~r:_ ~s:_ ~u ~t -> [| it u; it t |])

let stage ?(shared = false) x tile_loops =
  Sched.Stage
    {
      source = x;
      tile_loops;
      shared;
      cooperative = None;
      hoisted = false;
      swizzle = None;
      pad_stride = None;
      pipeline_depth = 1;
      tile_prec = None;
    }

let unroll s = Sched.Unroll { axis = s; materialize = true }

let exec ~name ~acc ~x ~n o =
  Ll_test.execute ~name o ~seed:[ (acc, [| 1.0 |]); (x, cells n) ] ~read:[ acc ]
  |> List.hd_exn |> Array.to_list

let keys (o : LL.optimized) = List.sort (Hashtbl.keys o.LL.traced_store) ~compare:Tn.compare
let is_tile tn = String.equal tn.Tn.namespace "tile"
let writes_tile = function LL.Set { tn; _ } -> is_tile tn | _ -> false

(* Nothing a [Stage] adds sits in a scope body: the scope grammar holds, and every tile write is a
   statement-level one. *)
let scopes_clean (llc : LL.t) =
  Option.is_none (LL.scope_purity_violation llc)
  && Option.is_none (LL.scope_hardware_loop llc)
  && Ll_test.count_scopes llc > 0
  && Ll_test.count_stmt ~f:writes_tile llc > 0
  && Ll_test.count_stmt ~f:writes_tile llc = Ll_test.count_stmt ~in_scopes:false ~f:writes_tile llc

let anchor_refusal = "is bound inside the body of local scope"

let claim_refused =
  "a Stage whose anchor is bound inside the scope a materializing Unroll minted -- over the loop \
   outside the scope (the issue's shape) or over a loop inside it -- is refused at the Stage, \
   typed, naming the anchor and the scope, before any tile is minted: the caller's traced store is \
   left as it was"

let claim_oracle =
  "op_legality asked of the program the Unroll left calls both Stages illegal with the same \
   refusal, and the probe leaves that program's traced store as it was"

let claim_widened =
  "the widened Stage the refusal names (the scope-bound loops of the staged index added to \
   tile_loops, anchored outside the scope or unanchored) is admitted, leaves no tile write in a \
   scope body, and executes to the unscheduled program's value"

let claim_holder =
  "a shared, unanchored Stage whose outermost tile loop is inside the scope is admitted, on one \
   copy of that loop and on two copies sharing the scope: nothing it adds is in a scope body, and \
   its load nest and barrier precede the statement holding the scope"

let claim_holder_oracle =
  "op_legality does not call that Stage illegal, and probing it leaves the program's traced store \
   as it was"

let claim_holder_exec =
  "that Stage executes to the unscheduled program's value, on both copy counts (a backend with \
   workgroup-shared memory)"

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* The C backends emit no workgroup-shared storage ([C_syntax.compile_proc] refuses a shared tile),
   so a shared stage executes only on a GPU backend. *)
let on_gpu = Sched.backend_is_gpu backend_name

let refusal ~name sched o =
  match Sched.apply_classified sched o with
  | (_ : LL.optimized) -> None
  | exception Outcome.Cause_at (Outcome.Transform, Outcome.Illegal_schedule { detail; _ }) ->
      Stdio.eprintf "stage_scope_anchor: %s refused: %s (not part of the golden)\n%!" name detail;
      Some detail

let names_anchor ~anchor detail =
  String.is_prefix detail ~prefix:"Schedule.Stage:"
  && String.is_substring detail ~substring:anchor_refusal
  && String.is_substring detail
       ~substring:("staging anchor " ^ Ir.Indexing.symbol_ident anchor ^ " ")

let refused_legs () =
  (* [Stage x over [r]]: the outer part is [u, t], so the anchor is [t]. [Stage x over [t]]: the
     anchor is [u]. Both are bound inside the scope. *)
  let leg name pick =
    let o, _, x, (r, s, u, t) = build_rsut name in
    let tile_loops, anchor = pick ~r ~u ~t in
    let before = keys o in
    let refused =
      match refusal ~name [ unroll s; stage x tile_loops ] o with
      | Some detail -> names_anchor ~anchor detail
      | None -> false
    in
    refused && List.equal Tn.equal (keys o) before
  in
  p claim_refused
    (leg "ssa_over_r" (fun ~r ~u:_ ~t -> ([ r ], t))
    && leg "ssa_over_t" (fun ~r:_ ~u ~t -> ([ t ], u)));
  let oracle name pick =
    let o, _, x, (r, s, u, t) = build_rsut name in
    let unrolled = Sched.apply [ unroll s ] o in
    let tile_loops, anchor = pick ~r ~u ~t in
    let before = keys unrolled in
    (match Sched.op_legality unrolled (stage x tile_loops) with
      | Sched.Op_illegal detail -> names_anchor ~anchor detail
      | Sched.Op_legal | Sched.Op_unknown _ -> false)
    && List.equal Tn.equal (keys unrolled) before
  in
  p claim_oracle
    (oracle "ssa_oracle_r" (fun ~r ~u:_ ~t -> ([ r ], t))
    && oracle "ssa_oracle_t" (fun ~r:_ ~u ~t -> ([ t ], u)))

let widened_legs () =
  let serial =
    let o, acc, x, _ = build_rsut "ssa_serial" in
    exec ~name:"ssa_serial" ~acc ~x ~n:8 o
  in
  let leg name pick =
    let o, acc, x, (r, s, u, t) = build_rsut name in
    match Sched.apply_classified [ unroll s; stage x (pick ~r ~u ~t) ] o with
    | exception Outcome.Cause_at _ -> (false, [])
    | staged -> (scopes_clean staged.LL.llc, exec ~name ~acc ~x ~n:8 staged)
  in
  let anchored_ok, anchored = leg "ssa_wide_ut" (fun ~r:_ ~u ~t -> [ u; t ]) in
  let root_ok, root = leg "ssa_wide_rut" (fun ~r ~u ~t -> [ r; u; t ]) in
  Stdio.eprintf
    "stage_scope_anchor: widened legs serial %s, [u; t] %s, [r; u; t] %s (not part of the golden)\n\
     %!"
    (floats serial) (floats anchored) (floats root);
  p claim_widened
    (anchored_ok && root_ok
    (* 1 + the sum of (i + 1) / 8 over the 8 cells. *)
    && List.equal Float.equal serial [ 5.5 ]
    && List.equal Float.equal anchored serial
    && List.equal Float.equal root serial)

(* The statements of [llc] at the top of its [Seq] spine. *)
let rec flat (llc : LL.t) = match llc with LL.Seq (a, b) -> flat a @ flat b | s -> [ s ]

(* In [r]'s body: a tile-writing statement, then a barrier, then the statement holding the scope, in
   that order, and no statement that holds a scope writes the tile. *)
let loads_precede_holder ~r (llc : LL.t) =
  match
    Ll_test.find_loop ~in_scopes:false
      ~f:(fun l -> Ir.Indexing.equal_symbol l.Ll_test.ls_index r)
      llc
  with
  | Some { Ll_test.ls_stmt = LL.For_loop { body; _ }; _ } -> (
      let stmts = flat body in
      let holds_scope s = Ll_test.count_scopes s > 0 in
      let writes s = Ll_test.count_stmt ~in_scopes:false ~f:writes_tile s > 0 in
      let pos f = List.findi stmts ~f:(fun _ s -> f s) |> Option.map ~f:fst in
      match
        (pos writes, pos (function LL.Workgroup_barrier -> true | _ -> false), pos holds_scope)
      with
      | Some l, Some b, Some h ->
          l < b && b < h
          && (not (List.is_empty stmts))
          && not (List.exists stmts ~f:(fun s -> holds_scope s && writes s))
      | _ -> false)
  | _ -> false

let holder_legs () =
  let leg ~s_extent =
    let name = "ssa_holder_" ^ Int.to_string s_extent in
    let o, acc, x, (r, s, u, t) = build_ut name ~s_extent in
    let structural, staged =
      match Sched.apply_classified [ unroll s; stage ~shared:true x [ u; t ] ] o with
      | exception Outcome.Cause_at (_, cause) ->
          Stdio.eprintf "stage_scope_anchor: %s refused: %s (not part of the golden)\n%!" name
            (Outcome.detail_of_cause cause);
          (false, None)
      | staged ->
          ( scopes_clean staged.LL.llc
            && Ll_test.count_loops u staged.LL.llc = s_extent
            && loads_precede_holder ~r staged.LL.llc,
            Some staged )
    in
    (* Evaluated only where the gate below is open. *)
    let executed () =
      let serial =
        let o, acc, x, _ = build_ut (name ^ "_serial") ~s_extent in
        exec ~name:(name ^ "_serial") ~acc ~x ~n:4 o
      in
      let values = Option.value_map staged ~default:[] ~f:(exec ~name ~acc ~x ~n:4) in
      Stdio.eprintf "stage_scope_anchor: %s serial %s, staged %s (not part of the golden)\n%!" name
        (floats serial) (floats values);
      (* 1 + 2 * s_extent * (1/8 + 2/8 + 3/8 + 4/8). *)
      let expected = 1.0 +. (2.5 *. Float.of_int s_extent) in
      List.equal Float.equal serial [ expected ] && List.equal Float.equal values serial
    in
    let probe =
      let o, _, x, (_, s, u, t) = build_ut (name ^ "_oracle") ~s_extent in
      let unrolled = Sched.apply [ unroll s ] o in
      let before = keys unrolled in
      (match Sched.op_legality unrolled (stage ~shared:true x [ u; t ]) with
        | Sched.Op_illegal _ -> false
        | Sched.Op_legal | Sched.Op_unknown _ -> true)
      && List.equal Tn.equal (keys unrolled) before
    in
    (structural, probe, executed)
  in
  let s1, p1, e1 = leg ~s_extent:1 in
  let s2, p2, e2 = leg ~s_extent:2 in
  p claim_holder (s1 && s2);
  p claim_holder_oracle (p1 && p2);
  if on_gpu then p claim_holder_exec (e1 () && e2 ())
  else skipped ~backend:backend_name claim_holder_exec

let () =
  refused_legs ();
  widened_legs ();
  holder_legs ()
