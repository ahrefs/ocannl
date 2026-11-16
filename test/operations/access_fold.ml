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

let scope ?(id = scope_id) body : LL.scalar_t =
  Local_scope { id; body; orig_indices = [| fixed 0 |]; mint = Inlined_computation }

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
  (* gh-ocannl-2145: a [Where] arm's scope bodies render under its condition. *)
  p "a scope in a Where arm stays gated and keeps the statement guard"
    (match List.nth rs 2 with
    | Some (_, ctx) -> ctx.gated && ctx.arm && ctx.scope_depth = 1 && List.length ctx.guards = 1
    | None -> false);
  let gated_scope = binop Ops.Relu_gate (at a) (scope (LL.Set_local (scope_id, at b))) in
  p "scope hoisting clears a gate's operand gating"
    (match List.nth (reads policy (put gated_scope)) 1 with
    | Some (_, ctx) -> (not ctx.gated) && ctx.scope_depth = 1
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
  (* A charge resolves to its tied widest alternatives (gh-ocannl-2098): [alts_eq] compares them as
     a set of node sets, [set_eq] asks for exactly one. *)
  let canonical sets = List.sort sets ~compare:Set.compare_direct in
  let alts_eq got wants =
    List.equal Set.equal (canonical got) (canonical (List.map wants ~f:(Set.of_list (module Tn))))
  in
  let set_eq got want = alts_eq got [ want ] in
  let not_ sc = LL.Unop (Ops.Not, (sc, single)) in
  let ev sc = LL.evaluation_reads ~blends:false ~self:out sc in
  p "a Where charges its condition and its wider arm, not the union"
    (set_eq (ev (where_ (at a) (add (at b) (at c_)) (at d_))) [ a; b; c_ ]);
  (* gh-ocannl-2098: a tie keeps both arms' alternatives, so inverting the predicate and swapping
     the arms charges the same; keeping the first arm made the two spellings differ. *)
  p "a Where whose arms tie keeps both alternatives"
    (alts_eq (ev (where_ (at a) (at b) (at c_))) [ [ a; b ]; [ a; c_ ] ]);
  p "a predicate-inverted, arm-swapped tie keeps the same alternatives"
    (alts_eq (ev (where_ (not_ (at a)) (at c_) (at b))) [ [ a; b ]; [ a; c_ ] ]);
  p "a Where whose arms read the same nodes keeps one alternative"
    (set_eq (ev (where_ (at a) (add (at b) (at c_)) (add (at c_) (at b)))) [ a; b; c_ ]);
  p "a nested tie under the wider arm keeps both nested alternatives"
    (alts_eq
       (ev (where_ (at a) (where_ (at b) (at c_) (at d_)) (at e_)))
       [ [ a; b; c_ ]; [ a; b; d_ ] ]);
  (* gh-ocannl-2098: every tied alternative continues to the next select, so the first select's
     spelling cannot decide the second's charge. From [{a,d}], the first select ties at [{a,d,b}]
     and [{a,d,c}]; the second's [b+e] arm adds two nodes to [{a,d,c}] but one to [{a,d,b}]. *)
  let successive first = add first (where_ (at d_) (add (at b) (at e_)) (c 0.)) in
  p "successive selects charge the widest path whichever way a tie is spelled"
    (set_eq (ev (successive (where_ (at a) (at b) (at c_)))) [ a; b; c_; d_; e_ ]
    && set_eq (ev (successive (where_ (not_ (at a)) (at c_) (at b)))) [ a; b; c_; d_; e_ ]);
  (* gh-ocannl-1775: on a backend that blends value selects, a select [LL.speculable_select] admits
     evaluates both arms, so it is charged their union; one it does not admit still short-circuits
     there and keeps the wider arm. *)
  let evb sc = LL.evaluation_reads ~blends:true ~self:out sc in
  p "a blended speculable Where charges the union of its arms"
    (set_eq (evb (where_ (at a) (add (at b) (at c_)) (at d_))) [ a; b; c_; d_ ]);
  p "a blended Where whose arms read the same node charges it once"
    (set_eq (evb (where_ (at a) (add (at b) (at c_)) (add (at c_) (at b)))) [ a; b; c_ ]);
  p "a blended Where over an index condition (an inliner guard) keeps the wider arm"
    (set_eq (evb (where_ (lt (embed i) (ic 2)) (add (at b) (at c_)) (at d_))) [ b; c_ ]);
  (* The refused selects short-circuit: their arms tie at 2 (the gather charges its table and its
     index), so each arm is its own alternative, never their union. *)
  p "a blended Where with a gather arm short-circuits: one arm per alternative"
    (alts_eq
       (evb
          (where_ (at a)
             (add (at b) (at c_))
             (gather ~tn:d_ ~idcs:[| fixed 0 |] ~dyn_axis:0 ~dyn_value:(at e_, single))))
       [ [ a; b; c_ ]; [ a; d_; e_ ] ]);
  p "a blended Where with an integer-division arm short-circuits: one arm per alternative"
    (alts_eq
       (evb
          (LL.Ternop
             ( Ops.Where,
               (at a, single),
               (add (at b) (at c_), Ops.int32),
               (LL.Binop (Ops.Div, (at d_, Ops.int32), (at e_, Ops.int32)), Ops.int32) )))
       [ [ a; b; c_ ]; [ a; d_; e_ ] ]);
  p "nested blended selects charge every arm"
    (set_eq (evb (where_ (at a) (where_ (at b) (at c_) (at d_)) (at e_))) [ a; b; c_; d_; e_ ]);
  p "a scope body inside an arm charges the statement, not the arm"
    (set_eq
       (ev (where_ (at a) (scope (LL.Set_local (scope_id, add (at d_) (at e_)))) (at b)))
       [ a; b; d_; e_ ]);
  p "nested alternatives take the wider arm at each level"
    (set_eq
       (ev (where_ (at a) (where_ (at b) (add (at c_) (at d_)) (at e_)) (at e_)))
       [ a; b; c_; d_ ]);
  (* gh-ocannl-1874: an arm is charged by what it adds to the evaluation, not by its own size: the
     then-arm re-reads the condition's node, so the else-arm is the wider one. *)
  p "a Where whose arm re-reads the condition charges the arm that adds more"
    (set_eq (ev (where_ (at a) (at a) (at b))) [ a; b ]);
  (* gh-ocannl-1874: the arm is chosen after each load is expanded, as the transitive guard expands
     a virtual dependency into its own fan-in. Here [c_] stands for a virtual node over [b] and [d_]
     for one over [d_] and [e_]: the then-arm loads more nodes directly (2 to 1) but fewer once
     expanded (1 to 2), so a short-circuiting select is charged the else-arm. *)
  let expand acc q =
    if Tn.equal q c_ then Set.add acc b
    else if Tn.equal q d_ then Set.add (Set.add acc d_) e_
    else Set.add acc q
  in
  let charge = LL.read_charge ~self:out (where_ (at a) (add (at b) (at c_)) (at d_)) in
  p "a short-circuiting Where charges the arm wider after expansion, not before"
    (set_eq (LL.resolve_charge ~blends:false ~expand charge) [ a; d_; e_ ]);
  p "a blended Where charges the union of its arms' expansions"
    (set_eq (LL.resolve_charge ~blends:true ~expand charge) [ a; b; d_; e_ ]);
  (* gh-ocannl-2098: arms unequal on their direct loads (1 to 2) that tie once expanded keep
     both. *)
  let tie_after = LL.read_charge ~self:out (where_ (at a) (at d_) (add (at b) (at e_))) in
  p "arms that tie only after expansion keep both alternatives"
    (alts_eq (LL.resolve_charge ~blends:false ~expand tie_after) [ [ a; d_; e_ ]; [ a; b; e_ ] ]);
  (* gh-ocannl-2128: the alternatives a consumer joins its own loads with are every MAXIMAL path,
     the narrower arm included -- read beside [b] and [c_], [{a,d_}] becomes the wider one. *)
  let alts sc =
    LL.resolve_alternatives ~blends:false
      ~expand:(fun acc q -> [ Set.add acc q ])
      (LL.read_charge ~self:out sc)
  in
  p "a short-circuiting Where keeps its narrower arm as an alternative"
    (alts_eq (alts (where_ (at a) (add (at b) (at c_)) (at d_))) [ [ a; b; c_ ]; [ a; d_ ] ]);
  p "an arm contained in the other is dropped: it can never be the wider one"
    (alts_eq (alts (where_ (at a) (add (at b) (at c_)) (at b))) [ [ a; b; c_ ] ]);
  p
    "a consumer overlapping the wider arm is charged the narrower one: the widest join, not the \
     wider arm"
    (let p_ = mk "p" and w = mk "w" in
     let dep = alts (where_ (at a) (add (at b) (at c_)) (at d_)) in
     let expand acc q =
       if Tn.equal q p_ then List.map dep ~f:(Set.union acc) else [ Set.add acc q ]
     in
     let charge = LL.read_charge ~self:out (add (at p_) (add (at b) (add (at c_) (at w)))) in
     alts_eq (LL.resolve_alternatives ~blends:false ~expand charge) [ [ a; b; c_; d_; w ] ]);
  (* The growth bound: k successive selects with disjoint arms have 2^k paths, all of one width,
     none containing another. Past [max_fanin_alternatives] the beam merges its tail into a union,
     so it keeps at most that many, every exact path inside a kept one, and the result is a function
     of the sets: the same with every select's predicate inverted and arms swapped. *)
  let cap = LL.max_fanin_alternatives in
  let n_sel = Int.ceil_log2 cap + 1 in
  let selects =
    List.init n_sel ~f:(fun s ->
        let n = Printf.sprintf "%d" s in
        (mk ("cond" ^ n), mk ("then" ^ n), mk ("else" ^ n)))
  in
  let spelled ~inverted =
    List.map selects ~f:(fun (c, t, e) ->
        if inverted then where_ (not_ (at c)) (at e) (at t) else where_ (at c) (at t) (at e))
    |> List.reduce_exn ~f:add
  in
  let exact_paths =
    List.fold selects
      ~init:[ Set.empty (module Tn) ]
      ~f:(fun paths (c, t, e) ->
        List.concat_map paths ~f:(fun path ->
            [ Set.add (Set.add path c) t; Set.add (Set.add path c) e ]))
  in
  let plain = alts (spelled ~inverted:false) and inverted = alts (spelled ~inverted:true) in
  p "the selects' exact paths outnumber the cap, so the bound is exercised"
    (List.length exact_paths > cap);
  p "past the cap the alternatives number at most max_fanin_alternatives" (List.length plain <= cap);
  p_all "every exact path lies inside a kept alternative: the cap only over-charges" exact_paths
    ~f:(fun path -> List.exists plain ~f:(fun kept -> Set.is_subset path ~of_:kept));
  p_all "no kept alternative contains another" plain ~f:(fun s ->
      List.count plain ~f:(fun s' -> Set.is_subset s ~of_:s') = 1);
  p "the capped alternatives do not depend on how the selects are spelled"
    (List.equal Set.equal plain inverted);
  p "the setter's own reads and merge-buffer reads are no contributors"
    (set_eq (ev (add (at out) (LL.Get_merge_buffer (a, [| fixed 0 |])))) []);
  p "a gather charges its index and its table"
    (set_eq (ev (gather ~tn:b ~idcs:[| fixed 0 |] ~dyn_axis:0 ~dyn_value:(at a, single))) [ a; b ]);
  p "a projection charges only its rendered operand"
    (set_eq (ev (binop Ops.Arg2 (at a) (at b))) [ b ]);
  p "a stored computation's dead loop replays nothing"
    (set_eq
       (LL.evaluation_reads_of_proc ~blends:false ~self:out
          (seq (loop ~upto:(-1) i (put (at a))) (put (at b))))
       [ b ]);
  p "a stored computation's Tile_mma is opaque"
    (set_eq
       (LL.evaluation_reads_of_proc ~blends:false ~self:out
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

(* gh-ocannl-2213: [Emitted] visits each top-level scope of a rendering unit once, at its definition
   position and in its emission's order, with its node's events around its body, then the unit's
   operands with those scopes as leaves. The emission here is hand-written -- the order and the
   placements are its to give; [where_arm_scopes] checks the renderer's against the kernel text. *)
let () =
  let s1 = mk "emit_s1" and s2 = mk "emit_s2" and c_ = mk "emit_c" in
  let id1 = LL.get_scope s1 and id2 = LL.get_scope s2 in
  let def1 = scope ~id:id1 (LL.Set_local (id1, at a))
  and def2 = scope ~id:id2 (LL.Set_local (id2, at b)) in
  let code = put (add (at c_) (add def1 (add def2 def1))) in
  let order = [ (id2.scope_id, F.Unconditional); (id1.scope_id, F.Conditional 0) ] in
  let emission : F.emission = { order = (fun _ _ -> order) } in
  let events (local_scopes : F.scopes) code =
    let hooks =
      {
        (F.hooks ()) with
        scalar =
          (fun ctx acc sc ->
            match sc with
            | LL.Local_scope { id; _ } ->
                Continue (`Enter (id.scope_id, ctx.F.placement, ctx.gated) :: acc)
            | _ -> Continue acc);
        after_scalar =
          (fun _ acc sc ->
            match sc with
            | LL.Local_scope { id; _ } -> `Exit id.scope_id :: acc
            | LL.Get (tn, _) -> `Read tn.Tn.id :: acc
            | _ -> acc);
      }
    in
    List.rev (F.fold ~policy:{ policy with local_scopes } ~hooks ~init:[] code)
  in
  let emitted = events (Emitted emission) code in
  p "Emitted visits each definition once, in the emission's order, then the operands"
    (Poly.equal emitted
       [
         `Enter (id2.scope_id, Some F.Unconditional, false);
         `Read b.Tn.id;
         `Exit id2.scope_id;
         `Enter (id1.scope_id, Some (F.Conditional 0), true);
         `Read a.Tn.id;
         `Exit id1.scope_id;
         `Read c_.Tn.id;
       ]);
  p "the operand-order walk visits every occurrence where it stands"
    (Poly.equal (events Visit code)
       [
         `Read c_.Tn.id;
         `Enter (id1.scope_id, None, false);
         `Read a.Tn.id;
         `Exit id1.scope_id;
         `Enter (id2.scope_id, None, false);
         `Read b.Tn.id;
         `Exit id2.scope_id;
         `Enter (id1.scope_id, None, false);
         `Read a.Tn.id;
         `Exit id1.scope_id;
       ]);
  let refuses f = match f () with _ -> false | exception Invalid_argument _ -> true in
  p "Emitted refuses an emission order that omits one of the unit's scopes"
    (refuses (fun () ->
         events (Emitted { order = (fun _ _ -> [ (id2.scope_id, F.Unconditional) ]) }) code));
  p "Emitted refuses a structural visit of discarded operands"
    (refuses (fun () ->
         F.fold
           ~policy:{ policy with local_scopes = Emitted emission; discarded_operands = Visit }
           ~hooks:(F.hooks ()) ~init:() code));
  p "Emitted refuses a bare expression"
    (refuses (fun () ->
         F.fold_scalar
           ~policy:{ policy with local_scopes = Emitted emission }
           ~hooks:(F.hooks ()) ~init:() def1))

(* gh-ocannl-2213: [Scope_order]'s condition keys tell a float constant by its bits. [x / 0.0 > 0]
   and [x / -0.0 > 0] decide differently, so definitions under the two run under different
   conditions; under one condition spelled twice they share a group (Codex review of the
   gh-ocannl-2213 PR). *)
let () =
  let module O = LL.Scope_order in
  let i = sym () in
  let cond zero =
    LL.Binop
      ( Ops.Cmplt,
        (LL.Constant 0., single),
        (binop Ops.Div (add (embed i) (LL.Constant 1.)) (LL.Constant zero), single) )
  in
  let under zero scope : (unit, int list) O.entry =
    {
      scope;
      guard =
        [
          {
            key = Select { cond = cond zero; prec = single };
            holds = true;
            anywhere = false;
            payload = ();
          };
        ];
      def = Some [];
    }
  in
  let groups a b = List.length (O.plan ~reads:Fn.id [ under a 1; under b 2 ]).groups in
  p "conditions differing only in a zero's sign are two conditions" (groups 0. (-0.) = 2);
  p "one condition spelled twice is one condition" (groups (-0.) (-0.) = 1);
  (* A gate's test over the same first operand renders differently where that operand renders at an
     integer precision before the test widens it ([integer_op]) and where it renders at the test's
     own float precision. *)
  let gate_under first_prec scope : (unit, int list) O.entry =
    {
      scope;
      guard =
        [
          {
            key =
              Gate
                {
                  op = Ops.Relu_gate;
                  prec = Ops.double;
                  first = add (embed i) (LL.Constant 0.5);
                  first_prec;
                };
            holds = true;
            anywhere = true;
            payload = ();
          };
        ];
      def = Some [];
    }
  in
  let gate_groups a b =
    List.length (O.plan ~reads:Fn.id [ gate_under a 1; gate_under b 2 ]).groups
  in
  p "a gate whose first operand renders at another precision is another condition"
    (gate_groups Ops.int32 Ops.double = 2);
  p "a gate spelled twice is one condition" (gate_groups Ops.int32 Ops.int32 = 1)
