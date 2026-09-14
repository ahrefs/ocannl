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
    (List.mem (List.map (LL.loop_bounds scopes) ~f:fst) hidden ~equal:Idx.equal_symbol);
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
