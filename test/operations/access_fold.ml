open Base
open Ll_builders
open Verdict.Claims
module F = LL.Access_fold

let policy : F.policy =
  {
    discarded_operands = Skip;
    gated_operands = Visit;
    dead_loops = Visit;
    local_scopes = Visit;
    guards = Track;
    scan_implicit = Visit;
  }

let mk = node_factory ~first_id:16300 ~dims:[| 4 |] ()
let a = mk "a"
let b = mk "b"
let out = mk "out"
let st = mk "state"
let at tn = get tn [| fixed 0 |]
let put sc = set_at out (fixed 0) sc
let scope_id = LL.get_scope st

let scope body : LL.scalar_t =
  Local_scope { id = scope_id; body; orig_indices = [| fixed 0 |]; mint = Inlined_computation }

let reads policy code =
  let hooks =
    {
      (F.hooks ()) with
      after_scalar =
        (fun ctx acc sc ->
          match sc with LL.Get (tn, _) | LL.Get_dynamic { tn; _ } -> (tn, ctx) :: acc | _ -> acc);
    }
  in
  List.rev (F.fold ~policy ~hooks ~init:[] code)

let nodes xs = List.map xs ~f:fst
let same_nodes got want = List.equal Tn.equal (nodes got) want

let () =
  let projected = binop Ops.Arg1 (at a) (scope (LL.Set_local (scope_id, at b))) in
  p "projection discards its unrendered scope body"
    (same_nodes (reads policy (put projected)) [ a ]);
  p "structural policy includes discarded operands"
    (same_nodes
       (reads { policy with discarded_operands = Visit } (put (binop Ops.Arg2 (at a) (at b))))
       [ a; b ]);
  p "second projection selects only its second operand"
    (same_nodes (reads policy (put (binop Ops.Arg2 (at a) (at b)))) [ b ]);
  p_all "every binop follows the operator classifier" Ops.all_of_binop ~f:(fun op ->
      let want =
        match Ops.binop_conditionality op with
        | Only_first -> [ a ]
        | Only_second -> [ b ]
        | Both_operands | Gated_second -> [ a; b ]
      in
      same_nodes (reads policy (put (binop op (at a) (at b)))) want);
  let conditional =
    LL.Ternop
      (Ops.Where, (at a, single), (scope (LL.Set_local (scope_id, at b)), single), (at out, single))
  in
  let rs = reads policy (if_ (at a) (put conditional)) in
  p "guard read precedes conditional operands and hoisted scope reads"
    (same_nodes rs [ a; a; b; out ]);
  p "scope hoisting clears operand gating but preserves statement guard"
    (match List.nth rs 2 with
    | Some (_, ctx) -> (not ctx.gated) && ctx.scope_depth = 1 && List.length ctx.guards = 1
    | None -> false);
  p "direct Where arm remains gated"
    (match List.last rs with Some (_, ctx) -> ctx.gated && ctx.scope_depth = 0 | None -> false);
  p_all "guard policy can suppress statement context"
    (reads { policy with guards = Ignore } (if_ (at a) (put (at b))))
    ~f:(fun (_, ctx) -> List.is_empty ctx.guards);
  p "gated skip is explicit subtree pruning"
    (same_nodes (reads { policy with gated_operands = Skip } (put conditional)) [ a ]);
  p "scope policy skips bodies"
    (same_nodes (reads { policy with local_scopes = Skip } (put conditional)) [ a; out ]);
  let i = sym () in
  let dead = seq (loop ~upto:(-1) i (put (at a))) (put (at b)) in
  p "structural dead reads register without becoming live"
    (match reads policy dead with [ (_, x); (_, y) ] -> (not x.live) && y.live | _ -> false);
  p "dead-loop skip omits only the dead body"
    (same_nodes (reads { policy with dead_loops = Skip } dead) [ b ]);
  let dynamic =
    scatter ~tn:out
      ~idcs:[| fixed 0 |]
      ~dyn_axis:0
      ~dyn_value:(at a, single)
      (gather ~tn:b ~idcs:[| fixed 0 |] ~dyn_axis:0 ~dyn_value:(at out, single))
  in
  let hooks =
    {
      (F.hooks ()) with
      after_scalar =
        (fun _ acc sc ->
          match sc with LL.Get (tn, _) | LL.Get_dynamic { tn; _ } -> ("r", tn) :: acc | _ -> acc);
      after_statement =
        (fun _ acc stmt ->
          match stmt with LL.Set_dynamic { tn; _ } -> ("w", tn) :: acc | _ -> acc);
    }
  in
  p "dynamic index reads precede gather read and final scatter write"
    (List.equal
       (fun (x, a) (y, b) -> String.equal x y && Tn.equal a b)
       (List.rev (F.fold ~policy ~hooks ~init:[] dynamic))
       [ ("r", a); ("r", out); ("r", b); ("w", out) ]);
  virtualize st;
  let cr = carry ~init:(at a) st in
  let scan_code =
    scan ~upto:3 i ~carried:[ cr ] (seq (set_next cr (add (prev cr) (embed i))) (put (next cr)))
  in
  let hooks =
    {
      (F.hooks ()) with
      after_statement =
        (fun ctx acc stmt ->
          match stmt with LL.Set_local (id, _) -> (id.scope_id, ctx.implicit) :: acc | _ -> acc);
    }
  in
  let assignments p = List.rev (F.fold ~policy:p ~hooks ~init:[] scan_code) in
  p "scan init precedes body update and trailing implicit rotation"
    (Poly.equal (assignments policy)
       [
         (cr.prev.scope_id, F.Scan_init);
         (cr.next.scope_id, F.Explicit);
         (cr.prev.scope_id, F.Scan_rotate);
       ]);
  p "scan implicit policy retains only explicit body assignments"
    (Poly.equal
       (assignments { policy with scan_implicit = Skip })
       [ (cr.next.scope_id, F.Explicit) ]);
  p "scan init reads remain visible when implicit statements are skipped"
    (same_nodes (reads { policy with scan_implicit = Skip } scan_code) [ a ]);
  let locals = LL.scope_value_syms scan_code in
  p "live scope census propagates scan update symbols through rotation"
    (match Hashtbl.find locals cr.prev.scope_id with
    | Some syms -> List.mem syms i ~equal:Idx.equal_symbol
    | None -> false);
  let hidden = sym () in
  let scopes = put (scope (loop ~upto:2 hidden (LL.Set_local (scope_id, embed hidden)))) in
  p "loop bounds sees scalar-position loops"
    (Ir.Affine.Loop_ranges.mem (LL.loop_ranges scopes) hidden);
  p "statement-local census excludes assignments instantiated inside scopes"
    (not (Hashtbl.mem (LL.scope_value_syms scopes) scope_id.scope_id));
  let tile =
    tile_mma ~d:(out, [| fixed 0 |]) ~a:(a, [| fixed 0 |]) ~b:(b, [| fixed 0 |]) (put (at a))
  in
  p "tile footprint comes from fallback" (same_nodes (reads policy tile) [ a ]);
  let hooks =
    {
      (F.hooks ()) with
      statement =
        (fun _ acc stmt -> match stmt with LL.Tile_mma _ -> F.Prune acc | _ -> F.Continue acc);
      after_scalar = (fun _ n _ -> n + 1);
    }
  in
  p "pruning an opaque construct skips descendants" (F.fold ~policy ~hooks ~init:0 tile = 0)

(* gh-ocannl-1451: the enclosing-binder context, the [branches] hook, the one read collector
   ([LL.evaluation_reads]) and the one statement-position walk ([LL.write_demand]). *)
let () =
  let c_ = mk "c" and d_ = mk "d" and e_ = mk "e" in
  let st2 = mk "state2" in
  virtualize st2;
  let i = sym () and j = sym () and k = sym () in
  let cr = carry ~init:(at b) st2 in
  let nest =
    loop ~upto:3 i
      (scan ~upto:2 j ~carried:[ cr ]
         (seq (set_next cr (prev cr)) (loop ~upto:1 k ~axis:LL.Unrolled (put (at a)))))
  in
  let hooks =
    {
      (F.hooks ()) with
      after_scalar =
        (fun ctx acc sc ->
          match sc with
          | LL.Get (tn, _) ->
              (tn, List.map ctx.loops ~f:(fun (b : F.binder) -> (b.index, b.axis))) :: acc
          | _ -> acc);
    }
  in
  let seen = List.rev (F.fold ~policy:{ policy with scan_implicit = Skip } ~hooks ~init:[] nest) in
  let loops_of tn = List.find_map seen ~f:(fun (tn', ls) -> Option.some_if (Tn.equal tn tn') ls) in
  let same_loops got want =
    match got with
    | None -> false
    | Some got ->
        List.equal
          (fun (s, ax) (s', ax') -> Idx.equal_symbol s s' && Option.equal LL.equal_axis_type ax ax')
          got want
  in
  p "a scan's init sits outside its binder" (same_loops (loops_of b) [ (i, Some LL.Serial) ]);
  p "a body read sees its enclosing binders innermost first, the scan's with no axis"
    (same_loops (loops_of a) [ (k, Some LL.Unrolled); (j, None); (i, Some LL.Serial) ]);
  let implicit_loops =
    F.fold ~policy
      ~hooks:
        {
          (F.hooks ()) with
          after_statement =
            (fun ctx acc stmt ->
              match (stmt, ctx.implicit) with
              | LL.Set_local _, (F.Scan_init | F.Scan_rotate) ->
                  (ctx.implicit, List.map ctx.loops ~f:(fun (b : F.binder) -> (b.index, b.axis)))
                  :: acc
              | _ -> acc);
        }
      ~init:[] nest
    |> List.rev
  in
  p "a scan's synthetic init sits outside its binder, its rotation inside"
    (match implicit_loops with
    | [ (F.Scan_init, init_loops); (F.Scan_rotate, rot_loops) ] ->
        same_loops (Some init_loops) [ (i, Some LL.Serial) ]
        && same_loops (Some rot_loops) [ (j, None); (i, Some LL.Serial) ]
    | _ -> false);
  let set_eq got want = Set.equal got (Set.of_list (module Tn) want) in
  let ev sc = LL.evaluation_reads ~self:out sc in
  p "a Where charges its condition and its wider arm, not the union"
    (set_eq (ev (where_ (at a) (add (at b) (at c_)) (at d_))) [ a; b; c_ ]);
  p "a Where whose arms tie charges the first" (set_eq (ev (where_ (at a) (at b) (at c_))) [ a; b ]);
  p "a scope body inside an arm charges the statement, not the arm"
    (set_eq
       (ev (where_ (at a) (scope (LL.Set_local (scope_id, add (at d_) (at e_)))) (at b)))
       [ a; b; d_; e_ ]);
  p "nested alternatives take the wider arm at each level"
    (set_eq
       (ev (where_ (at a) (where_ (at b) (add (at c_) (at d_)) (at e_)) (at e_)))
       [ a; b; c_; d_ ]);
  p "the setter's own reads and merge-buffer reads are no contributors"
    (set_eq (ev (add (at out) (LL.Get_merge_buffer (a, [| fixed 0 |])))) []);
  p "a gather charges its index and its table"
    (set_eq (ev (gather ~tn:b ~idcs:[| fixed 0 |] ~dyn_axis:0 ~dyn_value:(at a, single))) [ a; b ]);
  p "a projection charges only its rendered operand"
    (set_eq (ev (binop Ops.Arg2 (at a) (at b))) [ b ]);
  p "a stored computation's dead loop replays nothing"
    (set_eq
       (LL.evaluation_reads_of_proc ~self:out (seq (loop ~upto:(-1) i (put (at a))) (put (at b))))
       [ b ]);
  p "a stored computation's Tile_mma is opaque"
    (set_eq
       (LL.evaluation_reads_of_proc ~self:out
          (tile_mma
             ~d:(d_, [| fixed 0 |])
             ~a:(a, [| fixed 0 |])
             ~b:(b, [| fixed 0 |])
             (put (at c_))))
       []);
  let tile =
    tile_mma
      ~d:(d_, [| fixed 0 |])
      ~a:(a, [| fixed 0 |])
      ~b:(b, [| fixed 0 |])
      (set_at e_ (fixed 0) (at a))
  in
  let scoped = put (scope (seq (set_at e_ (fixed 0) (at a)) (LL.Set_local (scope_id, at b)))) in
  let demand = LL.write_demand (seq nest (seq tile (seq scoped (zero c_)))) in
  p
    "the write demand enters a scan body, counts a Tile_mma once as its accumulator, and skips \
     expressions"
    (List.equal Tn.equal demand.written [ out; d_; out; c_ ]);
  p "the write demand reports every binder in program order, a scan's with no axis"
    (same_loops
       (Some (List.map demand.binders ~f:(fun (b : F.binder) -> (b.index, b.axis))))
       [ (i, Some LL.Serial); (j, None); (k, Some LL.Unrolled) ])
