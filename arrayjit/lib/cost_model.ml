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
   loop-box size / fiber size ({!Affine.fiber_cardinality}, which reads a flattened [Sub_axis] run
   in its IR meaning, gh-ocannl-1162); an [`At_least] fiber (non-injective map) makes that an upper
   bound on the image. A dynamic access (gh-ocannl-1174) touches one cell per loop-box point, at a
   value of its dynamic coordinate the data picks (the view's unknown group: the axis, flattened
   over any [Sub_axis] run before it): the image of its map with the placeholder standing still
   counts the known coordinates' tuples, and each pairs with at most every value of that coordinate
   — an upper bound, capped by the box. A map of another rank than its node keeps the box/fiber
   bound but never claims exactness: the renderer still computes one cell (one run, for a vector
   store) per loop-box point from the symbols the map mentions, so the image is at most the box
   divided by the unmentioned symbols' widths, though which cells is unknown. A [Concat] coordinate
   falls back to the whole node: it is a choice of segment, each adding its cumulative offset, so
   its extents ADD along the axis and no product of loop widths bounds them (segments of widths 1
   and 3 name 4 cells where the box holds 3). Guarded accesses are counted guards-taken. All biases
   over-count.

   [fixed] (default: none) names enclosing loops whose symbols stand still: the count is then the
   cells ONE point of those loops touches — the box shrinks to the other loops, and a fixed symbol
   in the map reads as a constant (range 1). The ranking score's per-block count (gh-ocannl-1253)
   fixes the [Grid] loops. *)
let access_cells ?(fixed = fun (_ : Idx.symbol) -> false) (a : Tn.t Affine.access) : int * bool =
  let node_cells = Tn.num_elems a.a_tn in
  let dims = Lazy.force a.a_tn.Tn.dims in
  (* Viewed without [~dyn_axis]: the placeholder reads as the fixed index it is spelled as, so only
     the coordinates no data choice explains are unknown here. *)
  let uninterpretable =
    Array.exists (Affine.view ~dims a.a_map) ~f:(function
      | Affine.Unknown _ -> true
      | Affine.Known _ -> false)
  in
  let concat = Array.exists a.a_map ~f:(function Idx.Concat _ -> true | _ -> false) in
  if a.a_whole then (node_cells, a.a_guarded)
  else if concat then (node_cells, true)
  else
    let domain =
      List.filter_map a.a_loops ~f:(fun (s, (lo, hi)) ->
          if fixed s then None else Some (s, hi - lo + 1))
    in
    let box = List.fold domain ~init:1 ~f:(fun acc (_, w) -> acc * w) in
    let image, exact_image =
      match Affine.fiber_cardinality ~domain a.a_map with
      | `Exact f -> (box / max 1 f, not uninterpretable)
      | `At_least f -> (box / max 1 f, false)
    in
    match a.a_dyn_axis with
    | Some _ ->
        (* The data picks a value of the coordinate the view marks unknown — the dynamic axis with
           any [Sub_axis] run before it, which the component flattens over (gh-ocannl-1162) — not
           merely of the axis's own extent. *)
        let rows =
          Array.fold (Affine.view ?dyn_axis:a.a_dyn_axis ~dims a.a_map) ~init:1
            ~f:(fun n -> function
            | Affine.Unknown { size; _ } -> n * size
            | Affine.Known _ -> n)
        in
        (min node_cells (min box (image * rows)), true)
    | None when a.a_vec_last ->
        (* Each map instance is the base of a run along the minor axis. When the base image is exact
           and the runs are provably pairwise disjoint ({!Affine.vec_runs_disjoint}), the product is
           the exact distinct-cell count (gh-ocannl-578); otherwise runs may overlap for strided
           bases, so it is an upper bound. *)
        let disjoint_runs = Affine.vec_runs_disjoint ~dims a in
        ( min node_cells (image * max 1 a.a_vec_len),
          (not (exact_image && disjoint_runs)) || a.a_guarded )
    | None -> (min node_cells image, (not exact_image) || a.a_guarded)

(* Same-node accesses whose images provably share no cell: the union of their images is then the sum
   of their cardinalities. *)
let rec pairwise_disjoint = function
  | [] -> true
  | a :: tl ->
      List.for_all tl ~f:(fun b ->
          not (Affine.may_touch_same_cell ~dims:(Lazy.force a.Affine.a_tn.Tn.dims) a b))
      && pairwise_disjoint tl

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

(* {2 The ranking score — gh-ocannl-1253}

   [analyze] charges distinct cells under a perfect cache, which is what keeps it a lower bound —
   and what makes every untiled contraction score at its roofline floor. The ranking score charges
   what a plan requests once its reused data cannot stay cached: per block, the cells the block
   touches (its [Grid] loops standing still), times the blocks. The reuse factor of a tiling falls
   out of the per-block footprint; nothing here knows what a matmul is.

   gh-ocannl-1442: the score's rates are its own, not the bound's ceilings — an effective scalar
   issue rate and an achieved DRAM rate, each read as at most its ceiling — and under the per-block
   model the bytes of nodes that never leave the chip (workgroup tiles, thread-private scratch) are
   off the DRAM leg: the fitted issue rate carries their cost for the geometry it was fitted on. *)

type reuse = {
  ru_requested : (Tn.t * int) list;
  ru_working_set_bytes : int;
  ru_scratch : Tn.t list;
}
[@@deriving sexp_of]

let global_in (opt : Low_level.optimized) (tn : Tn.t) =
  not
    (Tn.Placements.is_local_peek opt.Low_level.optimize_ctx.Low_level.placements tn
    || Set.mem opt.Low_level.workgroup_shared tn)

(* Each loop symbol's hardware axis, from the statement-level loops. Symbols this walk does not
   reach — the loops of an inlined [Local_scope] body, a [Scan_loop] index — read as serial: they
   are iterated inside one work item. *)
let loop_axes (code : Low_level.t) =
  let tbl = Hashtbl.create (module Idx.Symbol) in
  let rec walk (c : Low_level.t) =
    match c with
    | Low_level.Seq (a, b) ->
        walk a;
        walk b
    | For_loop { index; axis; body; _ } ->
        Hashtbl.set tbl ~key:index ~data:axis;
        walk body
    | Scan_loop { body; _ } | If { body; _ } -> walk body
    | Tile_mma { fallback; _ } -> walk fallback
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_dynamic _ | Set_from_vec _
    | Set_local _ | Declare_local _ | Workgroup_barrier ->
        ()
  in
  walk code;
  fun s -> Hashtbl.find tbl s |> Option.value ~default:Low_level.Serial

let reuse_traffic ?max_resident_threads ~is_global (code : Low_level.t) : reuse =
  let axis = loop_axes code in
  let is_grid s = Low_level.equal_axis_type (axis s) Low_level.Grid in
  let width (lo, hi) = max 0 (hi - lo + 1) in
  (* Per node: requested cells (blocks × per-block cells, summed over its reads). Per launch nest —
     the accesses under one sequence of [Grid] loops: per-block bytes and the blocks. The workgroup
     is the kernel's one launch block ({!Low_level.launch_dims}, per-slot maxima over its nests),
     not a nest's own loop product: a smaller sibling nest still launches the full block, so it
     cannot have more blocks resident than the hardware runs. *)
  let threads = max 1 (Array.fold (Low_level.launch_dims code).Low_level.block ~init:1 ~f:( * )) in
  let requested = Hashtbl.create (module Tn) and order = ref [] in
  let nests = Hashtbl.create (module String) in
  let scratch = Hashtbl.create (module Tn) and scratch_order = ref [] in
  List.iter (Low_level.affine_accesses code) ~f:(fun (a : Tn.t Affine.access) ->
      if (not (is_global a.a_tn)) && not (Hashtbl.mem scratch a.a_tn) then (
        Hashtbl.set scratch ~key:a.a_tn ~data:();
        scratch_order := a.a_tn :: !scratch_order);
      if (not a.a_write) && is_global a.a_tn then (
        let grid = List.filter a.a_loops ~f:(fun (s, _) -> is_grid s) in
        let blocks = List.fold grid ~init:1 ~f:(fun acc (_, r) -> acc * width r) in
        let per_block = min (Tn.num_elems a.a_tn) (fst (access_cells ~fixed:is_grid a)) in
        let w = Ops.prec_in_bytes (Lazy.force a.a_tn.Tn.storage_prec) in
        let cell =
          Hashtbl.find_or_add requested a.a_tn ~default:(fun () ->
              order := a.a_tn :: !order;
              ref 0)
        in
        cell := !cell + (blocks * per_block * w);
        let key = String.concat ~sep:"," (List.map grid ~f:(fun (s, _) -> Idx.symbol_ident s)) in
        let bytes, nest_blocks = Hashtbl.find nests key |> Option.value ~default:(0, blocks) in
        Hashtbl.set nests ~key ~data:(bytes + (per_block * w), max nest_blocks blocks)));
  let working_set =
    Hashtbl.fold nests ~init:0 ~f:(fun ~key:_ ~data:(bytes, blocks) acc ->
        let resident =
          match max_resident_threads with
          | None -> blocks
          | Some t -> min blocks (max 1 (t / threads))
        in
        max acc (bytes * resident))
  in
  {
    ru_requested = List.rev_map !order ~f:(fun tn -> (tn, !(Hashtbl.find_exn requested tn)));
    ru_working_set_bytes = working_set;
    ru_scratch = List.rev !scratch_order;
  }

let requested_bytes (s : summary) (r : reuse) =
  (* Per node, the larger of the two charges: the score stays at or above [analyze]'s whatever the
     two extractions' approximations do, which keeps every bound below it a bound. *)
  let distinct tn =
    List.Assoc.find s.per_node tn ~equal:Tn.equal
    |> Option.value_map ~default:0 ~f:(fun fp -> fp.fp_read_bytes)
  in
  total_bytes s
  + List.sum (module Int) r.ru_requested ~f:(fun (tn, req) -> max 0 (req - distinct tn))

let spills ?cache_capacity_bytes (r : reuse) =
  match cache_capacity_bytes with
  | Some capacity -> r.ru_working_set_bytes > capacity
  | None -> false

(* A ranking rate is at most its ceiling — the envelope constant the bound divides by — so every leg
   priced at it is at least the bound's leg over the same counts; unset, it is the ceiling. *)
let ranking_rate ~ceiling rate = match rate with None -> ceiling | Some r -> Float.min r ceiling

let scratch_bytes (s : summary) (r : reuse) =
  List.sum
    (module Int)
    r.ru_scratch
    ~f:(fun tn ->
      List.Assoc.find s.per_node tn ~equal:Tn.equal
      |> Option.value_map ~default:0 ~f:(fun fp -> fp.fp_read_bytes + fp.fp_write_bytes))

let ranking_legs_of ?peak_flops ?peak_memory_bandwidth ?cache_capacity_bytes ?cache_bandwidth
    ?issue_rate ?dram_bandwidth ?(mma_flops = 0) (s : summary) (r : reuse option) :
    float option * float option =
  let total = Float.of_int (total_bytes s) in
  let memory =
    Option.map peak_memory_bandwidth ~f:(fun ceiling ->
        let dram = ranking_rate ~ceiling dram_bandwidth in
        let priced =
          match r with
          | None -> total /. dram
          | Some r -> (
              (* Under the per-block model, the chip-resident nodes' bytes leave the DRAM leg. *)
              let scratch = Float.of_int (scratch_bytes s r) in
              let global = Float.max 0. (total -. scratch) in
              let requested = Float.max 0. (Float.of_int (requested_bytes s r) -. scratch) in
              if spills ?cache_capacity_bytes r then requested /. dram
              else
                match cache_bandwidth with
                | None -> global /. dram
                | Some cache ->
                    (* A cache slower than DRAM would make raising the capacity raise the score. *)
                    Float.max (global /. dram) (requested /. Float.max cache dram))
        in
        (* Never below the bound's memory leg: all the distinct bytes at the ceiling. *)
        Float.max priced (total /. ceiling))
  in
  let compute =
    Option.map peak_flops ~f:(fun ceiling ->
        let issue = ranking_rate ~ceiling issue_rate in
        (* At the ceiling, the bound's own quotient: exact ties between candidates stay exact. *)
        if Float.(issue >= ceiling) then Float.of_int s.flops /. ceiling
        else
          (* Only tile-MMA work the backend accepts runs on its matrix units; the caller's count, at
             most the summary's operations. *)
          let mma = Float.of_int (min (max 0 mma_flops) s.flops) in
          let scalar = Float.of_int s.flops -. mma in
          (scalar /. issue) +. (mma /. ceiling))
  in
  (compute, memory)

let ranking_seconds_of ?peak_flops ?peak_memory_bandwidth ?cache_capacity_bytes ?cache_bandwidth
    ?issue_rate ?dram_bandwidth ?mma_flops (s : summary) (r : reuse option) : float option =
  let compute, memory =
    ranking_legs_of ?peak_flops ?peak_memory_bandwidth ?cache_capacity_bytes ?cache_bandwidth
      ?issue_rate ?dram_bandwidth ?mma_flops s r
  in
  match List.filter_opt [ compute; memory ] with
  | [] -> None
  | l -> Some (List.reduce_exn l ~f:Float.max)

let ranking_seconds ?peak_flops ?peak_memory_bandwidth ?cache_capacity_bytes ?cache_bandwidth
    ?issue_rate ?dram_bandwidth ?mma_flops ?max_resident_threads ~is_global (code : Low_level.t) :
    float option =
  let s = analyze code in
  let r =
    match (cache_capacity_bytes, cache_bandwidth) with
    | None, None -> None
    | _ -> Some (reuse_traffic ?max_resident_threads ~is_global code)
  in
  ranking_seconds_of ?peak_flops ?peak_memory_bandwidth ?cache_capacity_bytes ?cache_bandwidth
    ?issue_rate ?dram_bandwidth ?mma_flops s r

(* {2 Tile-MMA operand loads — gh-ocannl-1254}

   Not a priced term: a count the untimed choosers break exact ranking ties with. Twins that differ
   only in a staged tile's physical layout touch the same cells with the same operations, so every
   score above ties them; what differs is how many instructions a lane issues to bring the operand
   tiles into registers, which the backend's own load rule decides. *)

type operand_load =
  | Ldmatrix of { lane_bytes : int }
  | Fragment_load
  | Gather of { elt_bytes : int }
[@@deriving sexp_of, equal]

let operand_place (opt : Low_level.optimized) (tn : Tn.t) : Backend_intf.mma_operand_place =
  let space =
    if Set.mem opt.Low_level.workgroup_shared tn then `Shared
    else if Tn.Placements.is_local_peek opt.Low_level.optimize_ctx.Low_level.placements tn then
      `Thread
    else `Device
  in
  let layout =
    match Map.find opt.Low_level.swizzled tn with
    | Some Low_level.Swizzle_b128 -> `Swizzled_b128
    | Some Low_level.Swizzle_elem | None -> `Plain
  in
  (space, layout)

let operand_load_instructions ~lanes ~elements ~elt_bytes = function
  (* One matrix load delivers [lane_bytes] to each lane; a partial one still costs an
     instruction. *)
  | Ldmatrix { lane_bytes } ->
      ((elements * elt_bytes) + (lane_bytes * lanes) - 1) / (lane_bytes * lanes)
  (* One cooperative load per fragment ([nvcuda::wmma::load_matrix_sync]). *)
  | Fragment_load -> 1
  (* One load per element a lane gathers, whatever its width. *)
  | Gather _ -> (elements + lanes - 1) / lanes

(* One execution of an accepted statement, per lane: per intrinsic tile [(tm, tn, tk)] of the arm's
   own description, the emission loads [a] fragments ([tm x tk]) and [b] fragments ([tk x tn]): once
   per reduction step each, reused across the other's free axis, or -- where the arm reloads -- for
   every tile of that axis too. [a]'s [ldmatrix] form is [.x4] (16 bytes per lane), [b]'s [.x2] (8
   bytes per lane), in every inline-PTX arm; an arm whose fragments load whole ([mf_fragment_loads]:
   wmma, [simdgroup_load], rocWMMA) issues one load per fragment. Extents the tile does not divide
   (the arm declines them) have no count. *)
let statement_operand_loads (fragments : Backend_intf.mma_fragments) ~ldmatrix:(a_ldm, b_ldm)
    ~a_prec ~b_prec ~lanes ~m ~n ~k =
  let { Backend_intf.mf_tile = tm, tn, tk; mf_reloads; mf_fragment_loads } = fragments in
  if tm > 0 && tn > 0 && tk > 0 && m % tm = 0 && n % tn = 0 && k % tk = 0 then
    let mt = m / tm and nt = n / tn and kt = k / tk in
    let a_frags, b_frags =
      if mf_reloads then (mt * nt * kt, mt * nt * kt) else (mt * kt, nt * kt)
    in
    let loads prec ldm ~lane_bytes ~elements ~frags =
      let elt_bytes = Ops.prec_in_bytes prec in
      let form =
        if ldm then Ldmatrix { lane_bytes }
        else if mf_fragment_loads then Fragment_load
        else Gather { elt_bytes }
      in
      frags * operand_load_instructions ~lanes:(max 1 lanes) ~elements ~elt_bytes form
    in
    Some
      ( loads a_prec a_ldm ~lane_bytes:16 ~elements:(tm * tk) ~frags:a_frags,
        loads b_prec b_ldm ~lane_bytes:8 ~elements:(tk * tn) ~frags:b_frags )
  else None

(* Every [Tile_mma] statement the statement walk reaches, with its executions per lane group ([scale
   / lanes]), lanes and emission scope -- [None] where the renderer's layouts or the backend's hook
   decline it, so its scalar fallback renders. Shared by the operand-load count and the accepted
   tile-MMA work the ranking score prices at the matrix units' rate (gh-ocannl-1442). *)
type mma_site = {
  ms_stmt : Low_level.t;
  ms_scale : int;
  ms_lanes : int;
  ms_scope :
    (Backend_intf.mma_emission_scope
    * Backend_intf.mma_operand_place
    * Backend_intf.mma_operand_place)
    option;
}

let tile_mma_scopes ~operand_accepts (opt : Low_level.optimized) : mma_site list =
  let prec tn = Lazy.force tn.Tn.storage_prec in
  let layout tn ~ld ~idcs =
    Low_level.mma_operand_layout
      (Map.find opt.Low_level.swizzled tn)
      ~ld ~idcs ~dims:(Lazy.force tn.Tn.dims)
  in
  (* An operand as the hooks receive it: its leading dimension, space and layout — or nothing, where
     the renderer declines the layout before any hook (the scalar fallback renders). *)
  let operand tn ~ld ~idcs =
    match layout tn ~ld ~idcs with
    | `Decline _ -> None
    | (`Plain | `Swizzled_b128) as layout -> Some (ld, (fst (operand_place opt tn), layout))
  in
  (* The persistent-fragment regions the renderer recognizes ([Low_level.fragment_region], its own
     recognition), keyed by fragment: the backing target as the fragment scope addresses it, or
     [None] where the target's layout declines the scope (the ordinary rendering follows). *)
  let regions = Hashtbl.create (module Tn) in
  let rec find_regions (c : Low_level.t) =
    (match Low_level.fragment_region c with
    | Some r when Set.mem opt.Low_level.simdgroup_fragments r.Low_level.fr_fragment ->
        Hashtbl.set regions ~key:r.fr_fragment
          ~data:(operand r.fr_target ~ld:r.fr_target_ld ~idcs:r.fr_target_base)
    | Some _ | None -> ());
    match c with
    | Low_level.Seq (c1, c2) ->
        find_regions c1;
        find_regions c2
    | For_loop { body; _ } | Scan_loop { body; _ } | If { body; _ } -> find_regions body
    | Tile_mma _ | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_dynamic _
    | Set_from_vec _ | Set_local _ | Declare_local _ | Workgroup_barrier ->
        ()
  in
  find_regions opt.Low_level.llc;
  let sites = ref [] in
  let rec go ~scale ~env (c : Low_level.t) =
    match c with
    | Low_level.Seq (c1, c2) ->
        go ~scale ~env c1;
        go ~scale ~env c2
    | For_loop { index; from_; to_; body; _ } | Scan_loop { index; from_; to_; body; _ } ->
        let extent = max 0 (to_ - from_ + 1) in
        go ~scale:(scale * extent) ~env:((index, extent) :: env) body
    | If { body; _ } -> go ~scale ~env body
    | Tile_mma
        {
          d = d_tn, d_idcs;
          a = a_tn, a_idcs;
          b = b_tn, b_idcs;
          ta;
          tb;
          m;
          n;
          k;
          ldd;
          lda;
          ldb;
          lane;
          _;
        } ->
        let lanes =
          max 1 (List.Assoc.find env lane ~equal:Idx.equal_symbol |> Option.value ~default:1)
        in
        let a_prec = prec a_tn and b_prec = prec b_tn and d_prec = prec d_tn in
        (* The emission scope and acceptance, both the renderer's and the backend hook's own
           decisions, asked in the renderer's order. A recognized region of a marked fragment
           renders in the fragment scope over its backing target when the hook accepts it there,
           else as statements against the target; any other statement on its own, with its own
           destination. A statement whose layouts or hooks decline renders scalar or register-tiled
           code: nothing the arm description says applies to it. *)
        let scope =
          match
            ( operand a_tn ~ld:lda ~idcs:a_idcs,
              operand b_tn ~ld:ldb ~idcs:b_idcs,
              operand d_tn ~ld:ldd ~idcs:d_idcs )
          with
          | Some a, Some b, own -> (
              let accepts ~fragment_target ~d =
                operand_accepts ~d_prec ~a_prec ~b_prec ~ta ~tb ~m ~n ~k ~fragment_target ~d ~a ~b
              in
              let on_its_own =
                Option.bind own ~f:(fun own ->
                    Option.some_if
                      (accepts ~fragment_target:None ~d:own)
                      (Backend_intf.Mma_per_statement, snd a, snd b))
              in
              match Hashtbl.find regions d_tn with
              | Some (Some target) ->
                  let fragment_update =
                    match own with
                    | Some own -> accepts ~fragment_target:(Some target) ~d:own
                    | None -> false
                  in
                  if fragment_update then Some (Backend_intf.Mma_fragment_scope, snd a, snd b)
                  else
                    Option.some_if
                      (accepts ~fragment_target:None ~d:target)
                      (Backend_intf.Mma_per_statement, snd a, snd b)
              | Some None | None -> on_its_own)
          | _ -> None
        in
        sites := { ms_stmt = c; ms_scale = scale; ms_lanes = lanes; ms_scope = scope } :: !sites
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_dynamic _ | Set_from_vec _
    | Set_local _ | Declare_local _ | Workgroup_barrier ->
        ()
  in
  go ~scale:1 ~env:[] opt.Low_level.llc;
  List.rev !sites

let tile_mma_operand_loads ~operand_accepts ~operand_fragments ~operand_loads
    (opt : Low_level.optimized) =
  let prec tn = Lazy.force tn.Tn.storage_prec in
  List.map (tile_mma_scopes ~operand_accepts opt)
    ~f:(fun { ms_stmt; ms_scale = scale; ms_lanes = lanes; ms_scope = scope } ->
      match ms_stmt with
      | Low_level.Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; ta; tb; m; n; k; _ } -> (
          let a_prec = prec a_tn and b_prec = prec b_tn and d_prec = prec d_tn in
          let description =
            Option.map scope ~f:(fun (scope, a_place, b_place) ->
                ( operand_loads ~a_prec ~b_prec ~d_prec ~scope ~ta ~tb ~a:a_place ~b:b_place,
                  operand_fragments ~a_prec ~b_prec ~d_prec ~scope ))
          in
          match description with
          | Some (ldmatrix, Some fragments) ->
              Option.map
                (statement_operand_loads fragments ~ldmatrix ~a_prec ~b_prec ~lanes ~m ~n ~k)
                ~f:(fun (a_loads, b_loads) -> (scale / lanes, a_loads + b_loads))
          | Some (_, None) | None -> None)
      | _ -> None)

let accepted_tile_mma_flops ~operand_accepts (opt : Low_level.optimized) =
  List.sum
    (module Int)
    (tile_mma_scopes ~operand_accepts opt)
    ~f:(fun { ms_stmt; ms_scale; ms_lanes; ms_scope } ->
      match (ms_stmt, ms_scope) with
      | Low_level.Tile_mma { m; n; k; _ }, Some _ -> ms_scale / ms_lanes * (2 * m * n * k)
      | _ -> 0)

type load_count = No_tile_mma | Undescribed | Loads of int [@@deriving sexp_of, equal]

let combine_load_counts counts =
  if List.exists counts ~f:(equal_load_count Undescribed) then Undescribed
  else
    match
      List.filter_map counts ~f:(function Loads n -> Some n | No_tile_mma | Undescribed -> None)
    with
    | [] -> No_tile_mma
    | ns -> Loads (List.sum (module Int) ns ~f:Fn.id)

let operand_load_count ~operand_accepts ~operand_fragments ~operand_loads opt =
  match tile_mma_operand_loads ~operand_accepts ~operand_fragments ~operand_loads opt with
  | [] -> No_tile_mma
  | sites when List.exists sites ~f:Option.is_none -> Undescribed
  | sites ->
      Loads
        (List.sum
           (module Int)
           (List.filter_opt sites)
           ~f:(fun (executions, loads) -> executions * loads))

(* {2 The cost of one inlined computation — gh-ocannl-637 Part 2, gh-ocannl-1011}

   What a node costs to recompute at one read site, priced by the extraction above rather than by
   the virtualizer's traced proxy (reduction extent × fan-in). The code priced is the code a read
   executes: the inliner's own instantiation at a synthetic read site, after the pipeline the
   emitted code receives ({!Low_level.instantiate_at_synthetic_read}) — so the guards the inliner
   emits, its binding choices, the lane-extract form of a packed-uniform producer and every rewrite
   of the pipeline are in the code, and this section walks no IR of its own and restates no rule of
   the inliner's. *)

type recompute = {
  rc_flops : int;
  rc_bytes : int;
  rc_flops_approx : bool;
  rc_bytes_approx : bool;
  rc_opaque : bool;
}
[@@deriving sexp_of]

(* [analyze] over one read's code: [self]'s own traffic is not traffic (inlined, the node is a scope
   local), so only the other nodes' reads count as bytes; the counts and their per-leg exactness are
   the extraction's own. *)
let cost_of_self ~(self : Tn.t) (s : summary) : recompute =
  let others = List.filter s.per_node ~f:(fun (tn, _) -> not (Tn.equal tn self)) in
  {
    rc_flops = s.flops;
    rc_bytes = List.sum (module Int) others ~f:(fun (_, fp) -> fp.fp_read_bytes);
    rc_flops_approx = s.flops_approx;
    rc_bytes_approx = List.exists others ~f:(fun (_, fp) -> fp.fp_approx);
    rc_opaque = s.opaque;
  }

let instantiation_cost ?(static_indices = []) ?(placements = Tn.Placements.create ()) ~(self : Tn.t)
    (computations : (Idx.axis_index array option * Low_level.t) list) : recompute option =
  match Low_level.instantiate_at_synthetic_read ~placements ~static_indices self computations with
  | Error _ -> None
  | Ok code -> Some (cost_of_self ~self (analyze code))

let recompute_cost ?(static_indices = []) (ctx : Low_level.optimize_ctx) : Tn.t -> recompute option
    =
  let memo = Hashtbl.create (module Tn) in
  fun tn ->
    Hashtbl.find_or_add memo tn ~default:(fun () ->
        Option.bind (Hashtbl.find ctx.Low_level.computations tn) ~f:(fun computations ->
            instantiation_cost ~static_indices ~placements:ctx.placements ~self:tn computations))

let modeled_recompute_flops ~static_indices
    (world :
      Tn.t -> (Tn.Placements.t * (Idx.axis_index array option * Low_level.t) list, string) Result.t)
    : Tn.t -> int option =
 fun tn ->
  match world tn with
  | Error _ -> None
  | Ok (placements, computations) -> (
      (* The seam prices operations, so only the op count's exactness gates it: a guarded read's
         bytes are a bound (at most one arm's reads execute) while its op count can be exact. *)
      match instantiation_cost ~static_indices ~placements ~self:tn computations with
      | Some r when (not r.rc_flops_approx) && not r.rc_opaque -> Some r.rc_flops
      | _ -> None)

let () = Low_level.recompute_pricer := modeled_recompute_flops

(* The CPU default annotator's profitability gate (gh-ocannl-1458). A capped, saturating walk:
   counting stops mattering once the cap is reached, so no product can overflow. Unmodelled
   constructs raise [Unknown_work] — unknown work never passes a profitability gate. *)
exception Unknown_work

let bounded_work ~cap (llc : Low_level.t) : int option =
  let cap = max 0 cap in
  let add a b = if a >= cap - b then cap else a + b in
  let mul a b = if a = 0 || b = 0 then 0 else if a > cap / b then cap else min cap (a * b) in
  let extent from_ to_ =
    if to_ < from_ then 0
    else
      let diff = Int64.(of_int to_ - of_int from_) in
      if Int64.compare diff (Int64.of_int cap) >= 0 then cap else Int64.to_int_exn diff + 1
  in
  let rec scalar (s : Low_level.scalar_t) =
    match s with
    | Low_level.Local_scope { body; _ } -> stmt body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> 0
    | Get _ | Get_merge_buffer _ -> min cap 1
    | Get_dynamic { dyn_value = v, _; _ } -> add 1 (scalar v)
    | Unop (_, (a, _)) -> add 1 (scalar a)
    | Binop (_, (a, _), (b, _)) -> add 1 (add (scalar a) (scalar b))
    | Ternop (_, (a, _), (b, _), (c, _)) -> add 1 (add (scalar a) (add (scalar b) (scalar c)))
  and stmt (s : Low_level.t) =
    match s with
    | Noop | Comment _ | Declare_local _ -> 0
    | Seq (a, b) -> add (stmt a) (stmt b)
    | For_loop { from_; to_; body; _ } ->
        let e = extent from_ to_ in
        if e = 0 then 0 else mul e (stmt body)
    | If { cond = c, _; body } -> add (scalar c) (stmt body)
    | Set { llsc; _ } | Set_local (_, llsc) -> add 1 (scalar llsc)
    | Set_dynamic { dyn_value = v, _; llsc; _ } -> add 1 (add (scalar v) (scalar llsc))
    | Set_from_vec { length; arg = v, _; _ } -> add (min cap (max 0 length)) (scalar v)
    | Zero_out tn -> min cap (Tn.num_elems tn)
    | Staged_compilation _ | Workgroup_barrier | Scan_loop _ | Tile_mma _ -> raise Unknown_work
  in
  match stmt llc with w -> Some w | exception Unknown_work -> None

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
