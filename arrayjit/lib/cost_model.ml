open Base
module Idx = Indexing
module Tn = Tnode

(* Analytic cost model, extraction half (gh-ocannl-491 task 1). See cost_model.mli for the
   approximation contract — every count here is an upper bound (exactness tracked per node) except
   under [opaque], the flagged under-counting escape hatch. *)

type node_footprint = {
  fp_read_bytes : int;
  fp_write_bytes : int;
  fp_rmw_bytes : int;
  fp_approx : bool;
}
[@@deriving sexp_of]

type summary = {
  per_node : (Tn.t * node_footprint) list;
  read_bytes : int;
  write_bytes : int;
  flops : int;
  flops_approx : bool;
  opaque : bool;
}
[@@deriving sexp_of]

(* Distinct cells one access can touch, with exactness. Interpretable maps: image cardinality =
   loop-box size / fiber size ({!Affine.fiber_cardinality}); an [`At_least] fiber (non-injective
   map) makes that an upper bound on the image. Uninterpretable components fall back to the whole
   node. Guarded accesses are counted guards-taken. All biases over-count. *)
let access_cells (a : Tn.t Affine.access) : int * bool =
  let node_cells = Tn.num_elems a.a_tn in
  let uninterpretable =
    a.a_dynamic
    || Array.exists a.a_map ~f:(function Idx.Sub_axis | Idx.Concat _ -> true | _ -> false)
  in
  if a.a_whole then (node_cells, a.a_guarded)
  else if uninterpretable then (node_cells, true)
  else
    let domain = List.map a.a_loops ~f:(fun (s, (lo, hi)) -> (s, hi - lo + 1)) in
    let box = List.fold domain ~init:1 ~f:(fun acc (_, w) -> acc * w) in
    let image, exact_image =
      match Affine.fiber_cardinality ~domain a.a_map with
      | `Exact f -> (box / max 1 f, true)
      | `At_least f -> (box / max 1 f, false)
    in
    if a.a_vec_last then
      (* Each map instance is the base of a run along the minor axis. When the base image is exact
         and the runs are provably pairwise disjoint ({!Affine.vec_runs_disjoint}), the product is
         the exact distinct-cell count (gh-ocannl-578); otherwise runs may overlap for strided
         bases, so it is an upper bound. *)
      let dims = Lazy.force a.a_tn.Tn.dims in
      let minor_dim = if Array.length dims = 0 then 0 else dims.(Array.length dims - 1) in
      let disjoint_runs = Affine.vec_runs_disjoint ~minor_dim a in
      ( min node_cells (image * max 1 a.a_vec_len),
        (not (exact_image && disjoint_runs)) || a.a_guarded )
    else (min node_cells image, (not exact_image) || a.a_guarded)

(* Same-node accesses whose images provably share no cell: the union of their images is then the sum
   of their cardinalities. *)
let rec pairwise_disjoint = function
  | [] -> true
  | a :: tl ->
      List.for_all tl ~f:(fun b -> not (Affine.may_touch_same_cell a b)) && pairwise_disjoint tl

(* The certainty pre-pass shared by both extractions (gh-ocannl-578): which nodes' reads (or any
   accesses) are not certain to execute as the access list says. Node-granular and therefore
   conservative in both directions of use — the floor zeroes an uncertain node's whole read
   contribution (only loosening the floor), and the upper extraction refuses exactness for it (only
   widening the approximation flag).

   - [gated_reads]: read by an operand the renderers evaluate conditionally
   ({!Ops.binop_conditionality} / {!Ops.ternop_conditionality}): a [Where] arm, of which at most one
   executes, or the gated right operand of [&&] / [||] / a gate; - [open_reads]: read by a producer
   statement of an open-placement node — an inline completion instantiates the producer only at
   surviving consumer sites and [cleanup_virtual_llc] drops the setter loop, its reads included (an
   open producer computing a larger domain than its consumers demand makes "recomputation only adds
   ops" false, so its whole effect attributes to the open placement); the upper extraction has no
   open placements and passes a vacuous [open_placement]; - [dead]: any access under a dead loop
   ([to_ < from_]) — the body never executes. *)
let access_uncertainty ~open_placement (code : Low_level.t) =
  let gated_reads = Hashtbl.create (module Tn) in
  let open_reads = Hashtbl.create (module Tn) in
  let dead = Hashtbl.create (module Tn) in
  let mark tbl tn = Hashtbl.set tbl ~key:tn ~data:() in
  (* [~through_scopes:false] stops at [Local_scope] bodies: the renderers hoist a scope's definition
     out of the statement's expression, so a body under a [Where] arm or a gated operand executes
     unconditionally and its reads are certain — only the arm's inline reads are gated
     (gh-ocannl-637). The dead-loop and open-producer markings descend (a dead body never runs,
     hoisted or not; an open producer's whole effect attributes to the open placement). *)
  let rec sc_reads ~into ~through_scopes (s : Low_level.scalar_t) =
    match s with
    | Low_level.Get (tn, _) -> mark into tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        mark into tn;
        sc_reads ~into ~through_scopes v
    | Get_merge_buffer _ | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Local_scope { body; _ } -> if through_scopes then code_reads ~into body
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        sc_reads ~into ~through_scopes a;
        sc_reads ~into ~through_scopes b;
        sc_reads ~into ~through_scopes c
    | Binop (op, (a, _), (b, _)) -> (
        (* A projection's discarded operand is never evaluated ([affine_accesses] omits it too), per
           {!Ops.binop_conditionality}. *)
        match Ops.binop_conditionality op with
        | Ops.Only_first -> sc_reads ~into ~through_scopes a
        | Ops.Only_second -> sc_reads ~into ~through_scopes b
        | Ops.Both_operands | Ops.Gated_second ->
            sc_reads ~into ~through_scopes a;
            sc_reads ~into ~through_scopes b)
    | Unop (_, (a, _)) -> sc_reads ~into ~through_scopes a
  and code_reads ~into (c : Low_level.t) =
    let sc_reads = sc_reads ~through_scopes:true in
    match c with
    | Low_level.Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ -> ()
    | Zero_out tn -> mark into tn
    | Seq (a, b) ->
        code_reads ~into a;
        code_reads ~into b
    | For_loop { body; _ } -> code_reads ~into body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> sc_reads ~into c.init);
        code_reads ~into body
    | If { cond = c0, _; body } ->
        sc_reads ~into c0;
        code_reads ~into body
    | Set { tn; llsc; _ } | Set_dynamic { tn; llsc; _ } -> (
        mark into tn;
        sc_reads ~into llsc;
        match c with Set_dynamic { dyn_value = v, _; _ } -> sc_reads ~into v | _ -> ())
    | Set_local (_, llsc) -> sc_reads ~into llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        mark into tn;
        sc_reads ~into a
    | Tile_mma { fallback; _ } -> code_reads ~into fallback
  in
  let gated_reads_of = sc_reads ~into:gated_reads ~through_scopes:false in
  let sc_reads = sc_reads ~through_scopes:true in
  let rec sc_walk (s : Low_level.scalar_t) =
    match s with
    | Low_level.Ternop (op, (a, _), (b, _), (c, _)) -> (
        match Ops.ternop_conditionality op with
        | Ops.All_three ->
            sc_walk a;
            sc_walk b;
            sc_walk c
        | Ops.Cond_and_one_arm ->
            sc_walk a;
            (* The arms' inline reads are conditional; their nested structure (hoisted scope bodies
               included) still walks in its own right. *)
            gated_reads_of b;
            gated_reads_of c;
            sc_walk b;
            sc_walk c)
    | Binop (op, (a, _), (b, _)) -> (
        match Ops.binop_conditionality op with
        | Ops.Only_first -> sc_walk a
        | Ops.Only_second -> sc_walk b
        | Ops.Gated_second ->
            (* Short-circuiting (&& / || and the gates' ?:): the right operand's inline reads are
               conditional. *)
            gated_reads_of b;
            sc_walk a;
            sc_walk b
        | Ops.Both_operands ->
            sc_walk a;
            sc_walk b)
    | Unop (_, (a, _)) -> sc_walk a
    | Get_dynamic { dyn_value = v, _; _ } -> sc_walk v
    | Local_scope { body; _ } -> walk body
    | Get _ | Get_merge_buffer _ | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  and walk (c : Low_level.t) =
    match c with
    | Low_level.Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _
    | Zero_out _ ->
        ()
    | Seq (a, b) ->
        walk a;
        walk b
    | For_loop { from_; to_; body; _ } ->
        if to_ < from_ then code_reads ~into:dead body else walk body
    | Scan_loop { from_; to_; carried; body; _ } ->
        List.iter carried ~f:(fun c -> sc_walk c.init);
        if to_ < from_ then code_reads ~into:dead body else walk body
    | If { cond = c0, _; body } ->
        sc_walk c0;
        walk body
    | Set { tn; llsc; _ } ->
        if open_placement tn then sc_reads ~into:open_reads llsc;
        sc_walk llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        if open_placement tn then (
          sc_reads ~into:open_reads llsc;
          sc_reads ~into:open_reads v);
        sc_walk llsc;
        sc_walk v
    | Set_local (_, llsc) -> sc_walk llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        if open_placement tn then sc_reads ~into:open_reads a;
        sc_walk a
    | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; fallback; _ } ->
        if open_placement d_tn then (
          mark open_reads a_tn;
          mark open_reads b_tn);
        walk fallback
  in
  walk code;
  let read_uncertain tn =
    Hashtbl.mem gated_reads tn || Hashtbl.mem open_reads tn || Hashtbl.mem dead tn
  in
  let any_uncertain tn = Hashtbl.mem dead tn in
  (read_uncertain, any_uncertain)

let footprints ~read_uncertain ~any_uncertain (accesses : Tn.t Affine.access list) :
    (Tn.t * node_footprint) list =
  (* Per node and direction: sum of per-access cell counts (a union upper bound, capped by the
     node's size); exact when every access in the direction is individually exact, the accesses are
     pairwise provably disjoint ({!Affine.may_touch_same_cell} — the union of disjoint images is
     their sum, gh-ocannl-578; subsumes the single-exact-access case), and the direction is certain
     to execute as listed ({!access_uncertainty}): a conditionally-evaluated read (a [Where] arm, a
     gated right operand) or any dead-loop access may touch fewer cells than its image, so its
     direction stays an upper bound. *)
  let tbl = Hashtbl.create (module Tn) in
  let order = ref [] in
  List.iter accesses ~f:(fun a ->
      let cell =
        Hashtbl.find_or_add tbl a.a_tn ~default:(fun () ->
            order := a.a_tn :: !order;
            ref [])
      in
      cell := a :: !cell);
  List.rev_map !order ~f:(fun tn ->
      let accs = List.rev !(Hashtbl.find_exn tbl tn) in
      let writes, reads = List.partition_tf accs ~f:(fun a -> a.Affine.a_write) in
      let direction ~certain accs =
        let counted = List.map accs ~f:(fun a -> (a, access_cells a)) in
        let cells = List.sum (module Int) counted ~f:(fun (_, (c, _)) -> c) in
        let exact =
          certain
          && List.for_all counted ~f:(fun (_, (_, approx)) -> not approx)
          && match counted with [] | [ _ ] -> true | _ -> pairwise_disjoint accs
        in
        (cells, not exact)
      in
      let read_cells, reads_approx = direction ~certain:(not (read_uncertain tn)) reads in
      let write_cells, writes_approx = direction ~certain:(not (any_uncertain tn)) writes in
      let rmw_cells =
        List.sum
          (module Int)
          writes
          ~f:(fun a -> if a.Affine.a_rmw then fst (access_cells a) else 0)
      in
      let width = Ops.prec_in_bytes (Lazy.force tn.Tn.storage_prec) in
      let node_bytes = Tn.num_elems tn * width in
      let cap n = min node_bytes (n * width) in
      ( tn,
        {
          fp_read_bytes = cap read_cells;
          fp_write_bytes = cap write_cells;
          fp_rmw_bytes = cap rmw_cells;
          fp_approx = reads_approx || writes_approx;
        } ))

let analyze (code : Low_level.t) : summary =
  let flops_approx = ref false and opaque = ref false in
  (* [scale] is the product of enclosing loop extents, [env] their (symbol, extent) bindings —
     needed by [Tile_mma], whose 2*m*n*k multiply-adds are cooperative across its [lane] loop, not
     repeated per lane. *)
  let rec go ~scale ~env (c : Low_level.t) : int =
    match c with
    | Low_level.Noop | Comment _ | Zero_out _ | Declare_local _ | Workgroup_barrier -> 0
    | Staged_compilation _ ->
        opaque := true;
        0
    | Seq (c1, c2) -> go ~scale ~env c1 + go ~scale ~env c2
    | For_loop { index; from_; to_; body; _ } ->
        let extent = max 0 (to_ - from_ + 1) in
        go ~scale:(scale * extent) ~env:((index, extent) :: env) body
    | Scan_loop { index; from_; to_; carried; body; _ } ->
        (* The inits run once; the body's work scales with the range like a serial loop's. *)
        let extent = max 0 (to_ - from_ + 1) in
        (scale * List.sum (module Int) carried ~f:(fun c -> sc_flops c.init))
        + go ~scale:(scale * extent) ~env:((index, extent) :: env) body
    | Set { llsc; _ } -> scale * sc_flops llsc
    | Set_dynamic { dyn_value = dv, _; llsc; _ } -> scale * (sc_flops dv + sc_flops llsc)
    | Set_from_vec { length; arg = a, _; _ } -> scale * (length + sc_flops a)
    | Set_local (_, llsc) -> scale * sc_flops llsc
    | If { cond = cnd, _; body } ->
        (* Guards-taken: the body is charged as if the guard always passes. *)
        flops_approx := true;
        (scale * sc_flops cnd) + go ~scale ~env body
    | Tile_mma { m; n; k; lane; _ } ->
        let lane_extent =
          List.Assoc.find env lane ~equal:Idx.equal_symbol |> Option.value ~default:1
        in
        scale / max 1 lane_extent * (2 * m * n * k)
  and sc_flops (sc : Low_level.scalar_t) : int =
    let inline, hoisted = sc_split sc in
    inline + hoisted
  (* The (inline, hoisted) components of a scalar's cost (gh-ocannl-637): [hoisted] is the work of
     the [Local_scope] bodies it contains, [inline] the operations of the expression itself. The
     distinction is what the renderers do with a scope: [C_syntax.pp_scalar] returns a scope's
     definition separately and the statement emits every definition BEFORE its expression, so a
     scope body under a [Where] arm or a gated operand executes unconditionally, while the arm's
     inline operations sit inside the [?:] / [&&] and execute only when selected. Charging both
     arms' hoisted bodies is therefore exact, and only inline arm work can make the count a
     guards-taken bound. *)
  and sc_split (sc : Low_level.scalar_t) : int * int =
    match sc with
    | Low_level.Local_scope { body; _ } -> (0, go ~scale:1 ~env:[] body)
    | Get_local _ | Get _ | Constant _ | Constant_bits _ | Embed_index _ -> (0, 0)
    | Get_merge_buffer _ ->
        (* Merge-buffer traffic is not represented by [affine_accesses] either: flag the
           under-count. *)
        opaque := true;
        (0, 0)
    | Get_dynamic { dyn_value = dv, _; _ } -> sc_split dv
    | Ternop (op, a1, a2, a3) ->
        (* FMA and Mul3 count as two arithmetic operations each — matching [peak_flops]'
           FMA-counted-as-two convention, so an FMA-form kernel scores the same compute leg as its
           mul+add form; the select is one.

           The upper walk charges every operand that is {e rendered}, which for a conditional one is
           more than {!Ops.ternop_conditionality} says can execute: both arms' hoisted scope bodies
           run, and both arms' inline operations are charged although only one arm's execute. Only
           an operand no renderer emits at all — a projection's discarded one, below — may be
           dropped. Charging both arms' inline work of a short-circuiting [?:] is an over-count
           whenever either arm has any — equal costs notwithstanding — so a nonzero inline arm flags
           the op count approximate (gh-ocannl-578); arms whose whole cost is hoisted keep it exact
           (gh-ocannl-637). The floor's [Int.min] over the inline parts stays sound under the same
           hoisting. *)
        let ops = match op with Ops.FMA | Ops.Mul3 -> 2 | Ops.Where -> 1 in
        let i1, h1 = arg a1 and i2, h2 = arg a2 and i3, h3 = arg a3 in
        (match Ops.ternop_conditionality op with
        | Ops.All_three -> ()
        | Ops.Cond_and_one_arm -> if i2 <> 0 || i3 <> 0 then flops_approx := true);
        (ops + i1 + i2 + i3, h1 + h2 + h3)
    | Binop (op, a1, a2) -> (
        match Ops.binop_conditionality op with
        (* A projection is not an operation: it renders as its selected operand alone, and the
           discarded one is not rendered at all, hoisted definitions included. *)
        | Ops.Only_first -> arg a1
        | Ops.Only_second -> arg a2
        | Ops.Gated_second ->
            (* Gates-taken: the right operand's inline work is charged as if the gate always passes
               — an over-count whenever there is any (gh-ocannl-578); its hoisted definitions run
               regardless. *)
            let i1, h1 = arg a1 and i2, h2 = arg a2 in
            if i2 <> 0 then flops_approx := true;
            (1 + i1 + i2, h1 + h2)
        | Ops.Both_operands ->
            let i1, h1 = arg a1 and i2, h2 = arg a2 in
            (1 + i1 + i2, h1 + h2))
    | Unop (Ops.Identity, a1) -> arg a1
    | Unop (_, a1) ->
        let i1, h1 = arg a1 in
        (1 + i1, h1)
  and arg (sc, _prec) = sc_split sc in
  let flops = go ~scale:1 ~env:[] code in
  let read_uncertain, any_uncertain = access_uncertainty ~open_placement:(fun _ -> false) code in
  let per_node = footprints ~read_uncertain ~any_uncertain (Low_level.affine_accesses code) in
  {
    per_node;
    read_bytes = List.fold per_node ~init:0 ~f:(fun acc (_, fp) -> acc + fp.fp_read_bytes);
    write_bytes = List.fold per_node ~init:0 ~f:(fun acc (_, fp) -> acc + fp.fp_write_bytes);
    flops;
    flops_approx = !flops_approx;
    opaque = !opaque;
  }

let total_bytes s = s.read_bytes + s.write_bytes
let arithmetic_intensity s = Float.of_int s.flops /. Float.of_int (max 1 (total_bytes s))

(* {2 The floor (dual) extraction — gh-ocannl-514 phase 3}

   Lower bounds where [analyze] gives upper bounds, for bounding every completion of a partial
   placement vector. The duality is exact: every approximation that biases the upper extraction UP
   flips direction here — guarded ([If]) work floors to zero (guards-never-taken), a
   conditionally-evaluated operand ({!Ops.binop_conditionality} / {!Ops.ternop_conditionality})
   counts at its cheapest instead of its dearest, so [Where] charges its condition plus the cheaper
   arm where the upper walk charges the dearer one and a gated right operand floors away, a node's
   multiple same-direction accesses sum only when the exact images are pairwise provably disjoint
   (both extractions then agree the union is the sum) and otherwise take their MAX (a union is at
   least its largest member) instead of the capped sum, non-exact images contribute zero, and opaque
   code ([Staged_compilation], merge-buffer reads) — the upper contract's one escape hatch — merely
   loosens a floor without breaking it. [fr_exact] is [false] when any flooring occurred: the floor
   is then sound but not tight. *)

type floor = { fr_flops : int; fr_bytes : int; fr_exact : bool } [@@deriving sexp_of]

(* An access's certain distinct-cell count: the exact image, or zero when only an upper bound is
   known. Reuses [access_cells]' exactness verdict. *)
let access_cells_floor a =
  let cells, approx = access_cells a in
  if approx then 0 else cells

let floor_flops ~open_placement (code : Low_level.t) : int * bool =
  let exact = ref true in
  let rec go ~scale ~env (c : Low_level.t) : int =
    match c with
    | Low_level.Noop | Comment _ | Zero_out _ | Declare_local _ | Workgroup_barrier -> 0
    | Staged_compilation _ ->
        exact := false;
        0
    | Seq (c1, c2) -> go ~scale ~env c1 + go ~scale ~env c2
    | For_loop { index; from_; to_; body; _ } ->
        let extent = max 0 (to_ - from_ + 1) in
        go ~scale:(scale * extent) ~env:((index, extent) :: env) body
    | Scan_loop { index; from_; to_; carried; body; _ } ->
        let extent = max 0 (to_ - from_ + 1) in
        (scale * List.sum (module Int) carried ~f:(fun c -> sc c.init))
        + go ~scale:(scale * extent) ~env:((index, extent) :: env) body
    | Set { tn; llsc; _ } -> if open_placement tn then set_open () else scale * sc llsc
    | Set_dynamic { tn; dyn_value = dv, _; llsc; _ } ->
        if open_placement tn then set_open () else scale * (sc dv + sc llsc)
    | Set_from_vec { tn; length; arg = a, _; _ } ->
        if open_placement tn then set_open () else scale * (length + sc a)
    | Set_local (_, llsc) -> scale * sc llsc
    | If { cond = cnd, _; body = _ } ->
        (* Guards-never-taken: the body\'s work is not certain, only the condition\'s is — the exact
           dual of the upper walk\'s guards-taken. *)
        exact := false;
        scale * sc cnd
    | Tile_mma { d = d_tn, _; m; n; k; lane; _ } -> (
        if
          (* Same lane-cooperative attribution as the upper walk — the count is exact when the lane
             binding is in scope; without it the certain floor is zero. An open accumulator is an
             open producer like the Set family: its work attributes to the open placement. *)
          open_placement d_tn
        then set_open ()
        else
          match List.Assoc.find env lane ~equal:Idx.equal_symbol with
          | Some lane_extent -> scale / max 1 lane_extent * (2 * m * n * k)
          | None ->
              exact := false;
              0)
  and set_open () =
    (* An open producer\'s work attributes to the open placement: an inline completion computes it
       only at surviving consumer sites (possibly fewer cells than the setter loop covers), a
       materialize completion as written. Zero is the certain floor across both. *)
    exact := false;
    0
  and sc (s : Low_level.scalar_t) : int =
    let inline, hoisted = split s in
    inline + hoisted
  (* (inline, hoisted) as in the upper walk's [sc_split]: hoisted scope bodies are certain work
     under every short-circuiting form (the renderers emit them before the statement), so the dual
     minimizes over the inline parts only (gh-ocannl-637). *)
  and split (s : Low_level.scalar_t) : int * int =
    match s with
    | Low_level.Local_scope { body; _ } -> (0, go ~scale:1 ~env:[] body)
    | Get_local _ | Get _ | Constant _ | Constant_bits _ | Embed_index _ -> (0, 0)
    | Get_merge_buffer _ ->
        exact := false;
        (0, 0)
    | Get_dynamic { dyn_value = dv, _; _ } -> split dv
    | Ternop (op, a1, a2, a3) -> (
        (* The per-op arithmetic count is the upper walk's; only which operands are charged flips,
           per {!Ops.ternop_conditionality}. *)
        let ops = match op with Ops.FMA | Ops.Mul3 -> 2 | Ops.Where -> 1 in
        let i1, h1 = arg a1 and i2, h2 = arg a2 and i3, h3 = arg a3 in
        match Ops.ternop_conditionality op with
        | Ops.All_three -> (ops + i1 + i2 + i3, h1 + h2 + h3)
        | Ops.Cond_and_one_arm ->
            (* Short-circuiting [?:] in every renderer: the condition and exactly one arm's inline
               work execute; both arms' hoisted bodies do. *)
            if i2 <> i3 then exact := false;
            (ops + i1 + Int.min i2 i3, h1 + h2 + h3))
    | Binop (op, a1, a2) -> (
        match Ops.binop_conditionality op with
        (* The projections render only the selected operand ([C_syntax.pp_scalar]); the discarded
           one's work cannot execute, hoisted definitions included. *)
        | Ops.Only_first -> arg a1
        | Ops.Only_second -> arg a2
        | Ops.Gated_second ->
            (* Short-circuiting renderings: && / || and the gates' ?: evaluate the right operand's
               inline work only when the left one passes; its hoisted definitions run regardless. *)
            let i1, h1 = arg a1 and i2, h2 = arg a2 in
            if i2 <> 0 then exact := false;
            (1 + i1, h1 + h2)
        | Ops.Both_operands ->
            let i1, h1 = arg a1 and i2, h2 = arg a2 in
            (1 + i1 + i2, h1 + h2))
    | Unop (Ops.Identity, a1) -> arg a1
    | Unop (_, a1) ->
        let i1, h1 = arg a1 in
        (1 + i1, h1)
  and arg (s, _prec) = split s in
  let flops = go ~scale:1 ~env:[] code in
  (flops, !exact)

(* An enclosing dead loop makes an access never execute; the per-access filter complements the
   node-granular [dead] set (which covers whole-node fallbacks whose loop context is coarser). *)
let under_dead_loop (a : Tn.t Affine.access) =
  List.exists a.a_loops ~f:(fun (_, (lo, hi)) -> hi < lo)

let completion_floor ?(open_placement = fun _ -> false) (code : Low_level.t) : floor =
  let flops, flops_exact = floor_flops ~open_placement code in
  let read_uncertain, any_uncertain = access_uncertainty ~open_placement code in
  (* Per node and direction, the certain traffic: the sum of the exact images when they are pairwise
     provably disjoint (a disjoint union attains its sum, gh-ocannl-578), otherwise the largest
     exact image (a union is at least its largest member). Nodes with an open placement level
     contribute zero: their fully-inlined completion moves no bytes for them, and the floor
     quantifies over every completion. Committing such a node to Materialize adds back
     {!node_floor_bytes} — the monotone refinement delta. Reads that are not certain in every
     completion (Where arms, open producers\' operands, dead code) floor to zero, as do guarded and
     non-exact accesses. *)
  let tbl = Hashtbl.create (module Tn) in
  let inexact = ref (not flops_exact) in
  List.iter (Low_level.affine_accesses code) ~f:(fun a ->
      if open_placement a.Affine.a_tn || under_dead_loop a || any_uncertain a.a_tn then
        inexact := true
      else
        let cell = Hashtbl.find_or_add tbl a.a_tn ~default:(fun () -> ref []) in
        cell := a :: !cell);
  let bytes =
    Hashtbl.fold tbl ~init:0 ~f:(fun ~key:tn ~data:accs acc ->
        let direction accs =
          let counted =
            List.map accs ~f:(fun a ->
                let cells = access_cells_floor a in
                let cells = if (not a.a_write) && read_uncertain a.a_tn then 0 else cells in
                if cells = 0 then inexact := true;
                (a, cells))
          in
          let nz = List.filter counted ~f:(fun (_, c) -> c > 0) in
          if pairwise_disjoint (List.map nz ~f:fst) then List.sum (module Int) nz ~f:snd
          else begin
            (* The union exceeds the retained maximum unless the images coincide — which is not
               proved, so the floor is loose. *)
            if List.length nz > 1 then inexact := true;
            List.fold nz ~init:0 ~f:(fun m (_, c) -> max m c)
          end
        in
        let writes, reads = List.partition_tf !accs ~f:(fun a -> a.Affine.a_write) in
        let width = Ops.prec_in_bytes (Lazy.force tn.Tn.storage_prec) in
        acc + ((direction reads + direction writes) * width))
  in
  { fr_flops = flops; fr_bytes = bytes; fr_exact = not !inexact }

let footprint_approximate s = List.exists s.per_node ~f:(fun (_, fp) -> fp.fp_approx)
let approximate s = s.flops_approx || footprint_approximate s

let roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops ~bytes () : float option =
  let legs =
    List.filter_opt
      [
        Option.map peak_flops ~f:(fun p -> Float.of_int flops /. p);
        Option.map peak_memory_bandwidth ~f:(fun p -> Float.of_int bytes /. p);
      ]
  in
  match legs with [] -> None | l -> Some (List.reduce_exn l ~f:Float.max)

(* {2 The cost of one inlined computation — gh-ocannl-637 Part 2}

   What a virtual node costs to recompute at one read site, priced by the extraction above rather
   than by the virtualizer's traced proxy (reduction extent × fan-in). Two sources, one per flip
   direction of {!Low_level.flip_candidate}: a node the policy left virtual has its stored templates
   ([optimize_ctx.computations]), a node a heuristic cap materialized has its setter nest in the
   final code. Both reduce to [analyze] over a rewritten statement — the query walks no IR of its
   own. *)

type recompute = { rc_flops : int; rc_bytes : int; rc_approx : bool; rc_opaque : bool }
[@@deriving sexp_of]

(* A stored template's loops binding a symbol of its index vector are the ones the ordinary point
   read substitutes away ([Low_level.inline_computation] binds them to the call-site indices and
   drops the loop); a reduction loop stays. Only a BARE iterator position binds unconditionally (the
   inliner's first pass); a symbol occurring inside an affine position may be bound by the
   structural match or left free by unit solving — with its loop kept and range-guarded — depending
   on the reader's index, which the query cannot see, so such a template prices with its loops
   intact and only as a bound. Collapsing a bound loop to a single iteration keeps the symbol bound,
   so [affine_accesses] still interprets every map, and gives the per-instantiation trip count. *)
let collapse_loops ~(bound : Idx.symbol -> bool) (code : Low_level.t) : Low_level.t =
  let rec go (c : Low_level.t) : Low_level.t =
    match c with
    | Low_level.For_loop ({ index; from_; body; _ } as f) when bound index ->
        For_loop { f with to_ = from_; body = go body }
    | For_loop ({ body; _ } as f) -> For_loop { f with body = go body }
    | Scan_loop ({ body; _ } as sc) -> Scan_loop { sc with body = go body }
    | Seq (a, b) -> Seq (go a, go b)
    | If ({ body; _ } as i) -> If { i with body = go body }
    | c -> c
  in
  go code

(* Whether the binding of an index vector's symbols depends on the reader: a symbol inside an affine
   position may be bound by the structural match or left free by unit solving (its loop kept,
   range-guarded), and a bare symbol repeated across positions ([t[i; i]], a diagonal producer)
   binds at its first occurrence and turns the later ones into call-site consistency guards — either
   way the work depends on the reader's arguments, which the price cannot see. *)
let substitution_dependent (at : Idx.axis_index array) =
  let repeated s =
    Array.count at ~f:(function Idx.Iterator s' -> Idx.equal_symbol s s' | _ -> false) > 1
  in
  Array.exists at ~f:(function
    | Idx.Affine { symbols; _ } -> not (List.is_empty symbols)
    | Idx.Concat _ | Idx.Sub_axis -> true
    | Idx.Iterator s -> repeated s
    | Idx.Fixed_idx _ -> false)

let collapse_bound_loops ~(at : Idx.axis_index array) (code : Low_level.t) : Low_level.t * bool =
  let bound s =
    Array.exists at ~f:(function Idx.Iterator s' -> Idx.equal_symbol s s' | _ -> false)
  in
  (collapse_loops ~bound code, substitution_dependent at)

(* Keep only [self]'s own setters — or, with [~nth], only its [nth] setter statement in program
   order: a shared-loop template carries sibling setters that instantiation filters out, and a
   routine's code carries everything else. A subtree without a kept setter becomes [Noop], so the
   loops that survive are exactly the ones enclosing the kept work. Returns the pruned code and how
   many setters of [self] the code carries. *)
let prune_to_setters ?nth ~(self : Tn.t) (code : Low_level.t) : Low_level.t * int =
  let seen = ref 0 in
  let keep () =
    let k = !seen in
    Int.incr seen;
    match nth with None -> true | Some n -> n = k
  in
  let rec go (c : Low_level.t) : Low_level.t =
    match c with
    | Low_level.Seq (a, b) -> (
        (* Left to right, so [nth] counts in program order. *)
        let a = go a in
        match (a, go b) with Noop, c | c, Noop -> c | a, b -> Seq (a, b))
    | For_loop ({ body; _ } as f) -> (
        match go body with Noop -> Noop | body -> For_loop { f with body })
    | Scan_loop ({ body; _ } as sc) -> (
        match go body with Noop -> Noop | body -> Scan_loop { sc with body })
    | If ({ body; _ } as i) -> ( match go body with Noop -> Noop | body -> If { i with body })
    | (Set { tn; _ } | Set_dynamic { tn; _ } | Set_from_vec { tn; _ } | Zero_out tn) as c ->
        if Tn.equal tn self && keep () then c else Noop
    | Tile_mma { d = d_tn, _; _ } as c -> if Tn.equal d_tn self && keep () then c else Noop
    | Set_local _ | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ ->
        Noop
  in
  let pruned = go code in
  (pruned, !seen)

(* A packed-uniform producer ([Set_from_vec]) inlines as the lane-extract scalar form, not as its
   vector store (gh-ocannl-509 task 4), so neither its template nor its setter nest is the code a
   read executes: such a node prices only as a bound. *)
let rec has_set_from_vec (c : Low_level.t) =
  match c with
  | Low_level.Set_from_vec _ -> true
  | Seq (a, b) -> has_set_from_vec a || has_set_from_vec b
  | For_loop { body; _ } | Scan_loop { body; _ } | If { body; _ } -> has_set_from_vec body
  | Tile_mma { fallback; _ } -> has_set_from_vec fallback
  | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _
  | Set _ | Set_dynamic _ | Set_local _ ->
      false

(* [analyze] over a rewritten statement, read as one computation of [self]: [self]'s own traffic is
   not traffic (inlined, the node is a scope local), so only the other nodes' reads count as bytes;
   the op count and its exactness are the extraction's own. *)
let cost_of_self ~(self : Tn.t) (s : summary) : recompute =
  let others = List.filter s.per_node ~f:(fun (tn, _) -> not (Tn.equal tn self)) in
  {
    rc_flops = s.flops;
    rc_bytes = List.sum (module Int) others ~f:(fun (_, fp) -> fp.fp_read_bytes);
    rc_approx = s.flops_approx || List.exists others ~f:(fun (_, fp) -> fp.fp_approx);
    rc_opaque = s.opaque;
  }

(* Whether the code reads a scope local it does not define: pruning a routine to one node's setters
   can separate a [Get_local] from the [Declare_local]/[Set_local] the cross-statement hoisting pass
   ([hoist_cross_statement_cse]) gave it, and the shared body it stands for is work a re-inlining
   executes — so such a price is only a bound. *)
let reads_undefined_local (code : Low_level.t) =
  let defined = Hash_set.create (module Low_level.Scope_id) in
  let used = Hash_set.create (module Low_level.Scope_id) in
  let rec sc (s : Low_level.scalar_t) =
    match s with
    | Low_level.Get_local id -> Hash_set.add used id
    | Local_scope { id; body; _ } ->
        Hash_set.add defined id;
        go body
    | Get_dynamic { dyn_value = v, _; _ } -> sc v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        sc a;
        sc b;
        sc c
    | Binop (_, (a, _), (b, _)) ->
        sc a;
        sc b
    | Unop (_, (a, _)) -> sc a
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  and go (c : Low_level.t) =
    match c with
    | Low_level.Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Zero_out _ -> ()
    | Declare_local { id; _ } -> Hash_set.add defined id
    | Set_local (id, v) ->
        Hash_set.add defined id;
        sc v
    | Seq (a, b) ->
        go a;
        go b
    | For_loop { body; _ } | If { body; _ } -> go body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun cr ->
            Hash_set.add defined cr.Low_level.prev;
            Hash_set.add defined cr.next;
            sc cr.init);
        go body
    | Set { llsc; _ } -> sc llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        sc v;
        sc llsc
    | Set_from_vec { arg = a, _; _ } -> sc a
    | Tile_mma { fallback; _ } -> go fallback
  in
  go code;
  Hash_set.exists used ~f:(fun id -> not (Hash_set.mem defined id))

(* The code one point read of the template executes, and whether that is only a bound: the
   sibling-free body with the bound loops collapsed, then the passes the emitted code also receives
   after virtualization — the simplifier ([simplify_llc]: a single-assignment scope under a [Where]
   arm collapses into the arm's expression, where it IS conditional; identities vanish) and the
   scalar CSE ([eliminate_common_subexpressions]: two alpha-equivalent scope bodies, a consumer
   computing [x + x] of a virtual [x], execute once) — a stored template predates both. *)
let instantiation ~static_indices ~self ?at body =
  let body, _ = prune_to_setters ~self body in
  let body, dependent =
    match at with None -> (body, false) | Some at -> collapse_bound_loops ~at body
  in
  (* The post-virtualization pipeline of [specialize_proc], in its order: simplify under the
     routine's interval environment, the one-hot reduction rewrite (a dense one-hot reduction
     inlines as an O(1) dynamic gather), then the scalar CSE. *)
  let body =
    Low_level.eliminate_common_subexpressions
      (Low_level.rewrite_one_hot_reductions ~static_indices
         (Low_level.simplify_llc static_indices body))
  in
  (body, dependent || has_set_from_vec body || reads_undefined_local body)

let template_cost ?(static_indices = []) ~(self : Tn.t) ?at (body : Low_level.t) : recompute =
  let body, bound_only = instantiation ~static_indices ~self ?at body in
  let r = cost_of_self ~self (analyze body) in
  { r with rc_approx = r.rc_approx || bound_only }

let add_recompute a b =
  {
    rc_flops = a.rc_flops + b.rc_flops;
    rc_bytes = a.rc_bytes + b.rc_bytes;
    rc_approx = a.rc_approx || b.rc_approx;
    rc_opaque = a.rc_opaque || b.rc_opaque;
  }

let zero_recompute = { rc_flops = 0; rc_bytes = 0; rc_approx = false; rc_opaque = false }

let recompute_cost ?(static_indices = []) (ctx : Low_level.optimize_ctx) : Tn.t -> recompute option
    =
  let memo = Hashtbl.create (module Tn) in
  let rec cost ~visiting (tn : Tn.t) : recompute option =
    match Hashtbl.find memo tn with
    | Some r -> r
    | None ->
        let r =
          match Hashtbl.find ctx.Low_level.computations tn with
          | None -> None
          | Some comps ->
              let visiting = Set.add visiting tn in
              (* The inliner guards every component of a multi-setter node with the range
                 comparisons and the [Where] that select it: work no template carries, so their sum
                 is a bound ([set_computation_count > 1] in [inline_computation]). *)
              let guarded_components = List.count comps ~f:(fun (at, _) -> Option.is_some at) > 1 in
              Option.map ~f:(fun r -> { r with rc_approx = r.rc_approx || guarded_components })
              @@ Some
                   (List.fold comps ~init:zero_recompute ~f:(fun acc (at, body) ->
                        let body, bound_only = instantiation ~static_indices ~self:tn ?at body in
                        let s = analyze body in
                        let own = cost_of_self ~self:tn s in
                        let own = { own with rc_approx = own.rc_approx || bound_only } in
                        (* Reads of producers that will themselves inline at instantiation add their
                           own recompute per enclosing iteration; their read cells are then not
                           traffic. A producer without a stored template is a leaf whatever its
                           placement — its read already counted as bytes. *)
                        let accs = Low_level.affine_accesses body in
                        let expanded, expanded_nodes =
                          List.fold accs
                            ~init:(zero_recompute, Set.empty (module Tn))
                            ~f:(fun (exp, nodes) a ->
                              let p = a.Affine.a_tn in
                              if a.a_write || Tn.equal p tn then (exp, nodes)
                              else if
                                Tn.Placements.known_non_virtual ctx.placements p
                                || Set.mem visiting p
                              then (exp, nodes)
                              else
                                match cost ~visiting p with
                                | None -> (exp, nodes)
                                | Some r ->
                                    let scale =
                                      List.fold a.a_loops ~init:1 ~f:(fun acc (_, (lo, hi)) ->
                                          acc * max 0 (hi - lo + 1))
                                    in
                                    ( add_recompute exp
                                        {
                                          rc_flops = scale * r.rc_flops;
                                          rc_bytes = scale * r.rc_bytes;
                                          rc_approx = r.rc_approx || a.a_guarded;
                                          rc_opaque = r.rc_opaque;
                                        },
                                      Set.add nodes p ))
                        in
                        (* An expanded producer's cells are not traffic, whatever the number of
                           sites reading it: its footprint leaves the leaf bytes once. *)
                        let leaf_bytes =
                          List.sum
                            (module Int)
                            s.per_node
                            ~f:(fun (p, fp) ->
                              if Set.mem expanded_nodes p then fp.fp_read_bytes else 0)
                        in
                        let own = { own with rc_bytes = own.rc_bytes - leaf_bytes } in
                        add_recompute acc (add_recompute own expanded)))
        in
        Hashtbl.set memo ~key:tn ~data:r;
        r
  in
  fun tn -> cost ~visiting:(Set.empty (module Tn)) tn

(* One setter statement of [self], priced as one instantiation: the loops its index vector mentions
   — the ones that enumerate the cells it writes — collapse to a single iteration, and what remains
   (reduction loops, the operands' reads at ONE cell) is one cell's work, read directly rather than
   averaged over the cells the nest writes (an operand read at a fixed position is read by every
   cell's computation, which an aggregate footprint divided by the cell count loses). Exact only
   when the write is injective (one cell per iteration of the collapsed loops,
   [Affine.fiber_cardinality] = 1); a whole-node write ([Zero_out]) is free. *)
let setter_cost ~(self : Tn.t) (pruned : Low_level.t) : recompute =
  let write =
    List.find (Low_level.affine_accesses pruned) ~f:(fun a ->
        a.Affine.a_write && Tn.equal a.a_tn self)
  in
  match write with
  | None | Some { Affine.a_whole = true; _ } -> zero_recompute
  | Some w ->
      let bound s = Array.exists w.a_map ~f:(Idx.axis_index_mentions_symbol s) in
      let one = collapse_loops ~bound pruned in
      (* Injective over the loops the map mentions — the ones that enumerate cells; a loop the map
         does not mention is a reduction replayed inside one instantiation, and its repeated write
         of the same cell is that instantiation's accumulation, not a second cell. *)
      let injective =
        (not w.a_dynamic) && (not w.a_vec_last)
        &&
        let domain =
          List.filter_map w.a_loops ~f:(fun (s, (lo, hi)) ->
              Option.some_if (bound s) (s, hi - lo + 1))
        in
        match Affine.fiber_cardinality ~domain w.a_map with `Exact 1 -> true | _ -> false
      in
      let r = cost_of_self ~self (analyze one) in
      {
        r with
        rc_approx =
          r.rc_approx || (not injective) || substitution_dependent w.a_map
          || has_set_from_vec pruned || reads_undefined_local pruned;
      }

(* Re-inlining a node with several setters (a block/concat node, one range-guarded component per
   setter) replays EVERY component at each read site — the guards select the value, the hoisted
   component bodies all execute — so the per-read cost is the SUM of the per-cell costs of the
   setters, not the node's total work averaged over the cells it writes. *)
(* Whether a pruned single-setter statement is the node's zero-initialization: not a component the
   inliner guards ([set_computation_count] counts the value-carrying computations only). *)
let rec zero_out_only (c : Low_level.t) =
  match c with
  | Low_level.Zero_out _ -> true
  | For_loop { body; _ } | Scan_loop { body; _ } | If { body; _ } -> zero_out_only body
  | Seq (a, b) -> zero_out_only a || zero_out_only b
  | _ -> false

let producer_cost ~(self : Tn.t) (code : Low_level.t) : recompute option =
  match prune_to_setters ~self code with
  | Low_level.Noop, _ -> None
  | _, n ->
      let r, components =
        List.fold (List.init n ~f:Fn.id) ~init:(zero_recompute, 0) ~f:(fun (acc, comps) k ->
            let pruned, _ = prune_to_setters ~nth:k ~self code in
            ( add_recompute acc (setter_cost ~self pruned),
              if zero_out_only pruned then comps else comps + 1 ))
      in
      (* [inline_computation] wraps each value-carrying component of a multi-setter node in the
         range guards and the [Where] that select it — work the setters do not carry, so the sum is
         a bound. *)
      Some { r with rc_approx = r.rc_approx || components > 1 }

let modeled_recompute_flops (ctx : Low_level.optimize_ctx) ~static_indices (llc : Low_level.t) :
    Tn.t -> [ `Materialize | `Inline ] -> int option =
  let by_template = recompute_cost ~static_indices ctx in
  fun tn flip ->
    let r =
      match flip with `Materialize -> by_template tn | `Inline -> producer_cost ~self:tn llc
    in
    match r with Some r when (not r.rc_approx) && not r.rc_opaque -> Some r.rc_flops | _ -> None

let () = Low_level.recompute_pricer := modeled_recompute_flops

module Calibration = struct
  (* The calibration TSV schema and the envelope fitter over it (gh-ocannl-514 phase 0). This module
     is the schema's single owner: rows are emitted through [to_line] (Autotune) and read back
     through [of_line] (tools/fit_envelope.exe), so writer and reader cannot drift apart. *)

  type row = {
    backend : string;
    digest : string;
    routine : string;
        (** The tuned computation's name (gh-ocannl-635): the routine name the candidate compiles
            derive from the comp's block comment, the same name its generated sources carry. It is
            what makes a row — and every fit witness quoting one — say WHICH kernel demonstrated a
            rate, rather than only which candidate of an unnamed computation. Empty for rows
            recorded before the column existed (the 11-column schema), which stay fittable. *)
    label : string;
    measured_ms : float;
    model_ms : float option;
    kernels : int;
    flops : int;
    bytes : int;
    flops_approx : bool;
    bytes_approx : bool;
    opaque : bool;
  }
  [@@deriving sexp_of]

  (* Milliseconds are serialized FLOORED at the 6th decimal (not rounded to nearest): a stored time
     never exceeds the true measurement, so constants fit from the file remain conservative with
     respect to the original in-process measurement — round-to-nearest could round a short kernel's
     time up by 0.5 ns, a 1e-4 relative error at 5 us, far above [report]'s 2e-6 bump. The floored
     decimal survives the round-trip: [d /. 1e6] for integral [d] is within ~1e-11 of the decimal,
     so ["%.6f"] prints [d] back exactly. *)
  let floor6 v = Float.round_down (v *. 1e6) /. 1e6

  (* The routine and label columns are the only free text in a row, hence the only place a stray tab
     or newline could split one line into fragments no reader can parse. A name carrying one loses
     the character, not the row. *)
  let cell = String.map ~f:(function '\t' | '\n' | '\r' -> ' ' | c -> c)

  let to_line r =
    Printf.sprintf "%s\t%s\t%s\t%s\t%.6f\t%s\t%d\t%d\t%d\t%b\t%b\t%b" r.backend r.digest
      (cell r.routine) (cell r.label) (floor6 r.measured_ms)
      (match r.model_ms with Some m -> Printf.sprintf "%.6f" (floor6 m) | None -> "")
      r.kernels r.flops r.bytes r.flops_approx r.bytes_approx r.opaque

  let of_line line =
    let build backend digest routine label measured model kernels flops bytes flops_approx
        bytes_approx opaque =
      try
        Some
          {
            backend;
            digest;
            routine;
            label;
            measured_ms = Float.of_string measured;
            model_ms = (if String.is_empty model then None else Some (Float.of_string model));
            kernels = Int.of_string kernels;
            flops = Int.of_string flops;
            bytes = Int.of_string bytes;
            flops_approx = Bool.of_string flops_approx;
            bytes_approx = Bool.of_string bytes_approx;
            opaque = Bool.of_string opaque;
          }
      with _ -> None
    in
    match String.split line ~on:'\t' with
    | [
     backend;
     digest;
     routine;
     label;
     measured;
     model;
     kernels;
     flops;
     bytes;
     flops_approx;
     bytes_approx;
     opaque;
    ] ->
        build backend digest routine label measured model kernels flops bytes flops_approx
          bytes_approx opaque
    (* Rows recorded before the routine column (gh-ocannl-635). Unlike the 9-column rows that
       predate the approx flags, these carry everything a leg needs to prove its counts exact, so
       they still fit — they just cannot name the computation they measured. *)
    | [
     backend;
     digest;
     label;
     measured;
     model;
     kernels;
     flops;
     bytes;
     flops_approx;
     bytes_approx;
     opaque;
    ] ->
        build backend digest "" label measured model kernels flops bytes flops_approx bytes_approx
          opaque
    | _ -> None

  (* How a row names itself in a report: the tuned computation, then the candidate within it. *)
  let qualified ~routine ~label = if String.is_empty routine then label else routine ^ "/" ^ label
  let row_name r = qualified ~routine:r.routine ~label:r.label

  type fit = {
    fit_backend : string;
    fit_rows : int;
    fit_opaque : int;
    fit_flops_approx : int;
    fit_bytes_approx : int;
    fit_multi_kernel : int;
    fit_violations : int;
    fit_fission_slack : (float * string) option;
    fit_peak_flops : (float * string) option;
    fit_peak_memory_bandwidth : (float * string) option;
  }
  [@@deriving sexp_of]

  let fit rows =
    let by_backend = Hashtbl.create (module String) in
    let order = ref [] in
    List.iter rows ~f:(fun r ->
        let cell =
          Hashtbl.find_or_add by_backend r.backend ~default:(fun () ->
              order := r.backend :: !order;
              ref [])
        in
        cell := r :: !cell);
    List.rev_map !order ~f:(fun backend ->
        let rows = List.rev !(Hashtbl.find_exn by_backend backend) in
        (* Each leg fits from the rows where THAT leg's counts are exact (and non-opaque, positively
           timed): guards-taken / union over-counting can "achieve" a counts/time ratio far above
           any hardware peak, and letting it drive a maximum would inflate the envelope machine-wide
           — but exactness is per leg (a multi-read footprint makes bytes approximate without
           touching an exact op count), so a row feeds whichever legs it can prove. The fitted
           constant per leg is the tightest sound one for those rows — [bound <= measured] needs
           [peak >= counts/measured] on every row — so it is the maximum achieved [counts/time]. *)
        let timed = List.filter rows ~f:(fun r -> (not r.opaque) && Float.(r.measured_ms > 0.)) in
        let leg ~exact count =
          List.fold timed ~init:None ~f:(fun acc r ->
              let c = count r in
              if (not (exact r)) || c <= 0 then acc
              else
                let rate = Float.of_int c /. (r.measured_ms *. 1e-3) in
                match acc with
                | Some (best, _) when Float.(best >= rate) -> acc
                | _ -> Some (rate, Printf.sprintf "%s (%s)" (row_name r) r.digest))
        in
        let peak_flops = leg ~exact:(fun r -> not r.flops_approx) (fun r -> r.flops) in
        let peak_bandwidth = leg ~exact:(fun r -> not r.bytes_approx) (fun r -> r.bytes) in
        (* Multi-kernel rows aggregate per-kernel counts, so the per-leg maxima above are necessary
           for them but not sufficient: [summaries_roofline] sums per-kernel max-of-legs, which can
           exceed the aggregate legs (a compute-bound + bandwidth-bound kernel mix approaches twice
           either). The aggregate SUFFICIENT condition is [flops/peak_flops + bytes/peak_bandwidth
           <= time] (max <= sum per kernel), so both legs are raised uniformly — preserving their
           ratio — by the smallest slack that enforces it on every multi-kernel row. Raising peaks
           is the safe direction: the bound stays a lower bound, only pruning weakens. With one leg
           absent the bound degenerates to the other leg's aggregate, which the necessary maximum
           already covers. *)
        let fission_slack =
          match (peak_flops, peak_bandwidth) with
          | Some (pf, _), Some (pb, _) ->
              List.fold timed ~init:None ~f:(fun acc r ->
                  (* Only fully-exact rows can force slack: an over-counted leg would inflate the
                     aggregate sum, and with it both fitted constants. *)
                  if r.kernels <= 1 || r.flops_approx || r.bytes_approx then acc
                  else
                    let t = r.measured_ms *. 1e-3 in
                    let sum = (Float.of_int r.flops /. pf) +. (Float.of_int r.bytes /. pb) in
                    let s = sum /. t in
                    match acc with
                    | Some (best, _) when Float.(best >= s) -> acc
                    | _ when Float.(s <= 1.) -> acc
                    | _ -> Some (s, Printf.sprintf "%s (%s)" (row_name r) r.digest))
          | _ -> None
        in
        let apply_slack =
          match fission_slack with
          | None -> fun p -> p
          | Some (s, _) -> Option.map ~f:(fun (v, w) -> (v *. s, w))
        in
        {
          fit_backend = backend;
          fit_rows = List.length timed;
          fit_opaque = List.count rows ~f:(fun r -> r.opaque);
          fit_flops_approx = List.count timed ~f:(fun r -> r.flops_approx);
          fit_bytes_approx = List.count timed ~f:(fun r -> r.bytes_approx);
          fit_multi_kernel = List.count timed ~f:(fun r -> r.kernels > 1);
          fit_violations =
            (* Fully-exact rows only, matching the runtime warning: on a row with an approx leg the
               exceedance may reflect over-counting, not an understated envelope, and this counter
               prompts a refit. *)
            List.count rows ~f:(fun r ->
                match r.model_ms with
                | Some m ->
                    Float.(m > r.measured_ms)
                    && (not r.flops_approx) && (not r.bytes_approx) && not r.opaque
                | None -> false);
          fit_fission_slack = fission_slack;
          fit_peak_flops = apply_slack peak_flops;
          fit_peak_memory_bandwidth = apply_slack peak_bandwidth;
        })

  let report f =
    let buf = Buffer.create 256 in
    let addf fmt = Printf.ksprintf (Buffer.add_string buf) fmt in
    addf
      "# backend %s: %d timed row%s (%d opaque excluded, %d approx-flops / %d approx-bytes \
       excluded per leg, %d multi-kernel), %d recorded bound violation%s\n"
      f.fit_backend f.fit_rows
      (if f.fit_rows = 1 then "" else "s")
      f.fit_opaque f.fit_flops_approx f.fit_bytes_approx f.fit_multi_kernel f.fit_violations
      (if f.fit_violations = 1 then "" else "s");
    Option.iter f.fit_fission_slack ~f:(fun (s, witness) ->
        addf "# fission slack %s applied to both legs, forced by multi-kernel row: %s\n"
          (Ndarray.concise_float ~prec:6 s) witness);
    let constant ~key ~leg = function
      | None -> addf "# no scoreable row constrains %s\n" key
      | Some (implied, witness) ->
          addf "# %s-leg binding row: %s\n" leg witness;
          (* [concise_float] truncates to [prec + 1] significant digits (relative error below
             [10^-prec] toward zero); the bump keeps the printed constant at or above the implied
             minimum, so the recorded rows satisfy the bound under the printed value too. *)
          addf "%s=%s\n" key (Ndarray.concise_float ~prec:6 (implied *. (1. +. 2e-6)))
    in
    constant ~key:"model_peak_flops" ~leg:"compute" f.fit_peak_flops;
    constant ~key:"model_peak_memory_bandwidth" ~leg:"memory" f.fit_peak_memory_bandwidth;
    Buffer.contents buf
end
