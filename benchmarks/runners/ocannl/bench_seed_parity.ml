(* gh-ocannl-1490: the autotuner's per-segment seeds, censused and executed without a search.

   Tuner candidates are timed, not value-checked: an unsound seed that happens to be fast would be
   crowned silently. Two instruments over a training step's captured lowering, both taking the seeds
   from [Autotune.search_seeding] -- the enumeration [Autotune.tune] itself runs, so they see what a
   search seeds:

   - [census_lines]: the seeding census ([Autotune.seeding_census_lines]) of every lowering a search
   seeds, the whole routine and each keyed fission segment, with no search started (it used to take
   [BENCH_TUNE=1 autotune_budget_seconds=1 autotune_log_seeding=true]). - [parity]: each selected
   seed of each keyed segment compiled as the tuner compiles a per-segment sketch candidate (the
   search's segmentation replayed, that one segment sketched, the others on their [`Search]
   presets), run for one step, and compared with the untuned default pipeline's step from the same
   state. Every materialized node both routines write is compared, the parameters after the update
   and the loss included. Before each run the read-and-written nodes (the parameters) are restored,
   and every other node the routine writes is filled with NaN, so a seed that leaves cells unwritten
   cannot pass on values a previous routine left: the inputs discriminate. The reference itself is
   run twice, plain and poisoned, and a node the poisoned reference leaves non-finite where the
   plain one does not is not poisoned (it is read before it is written in a way the routine
   interface does not declare).

   A seed that computes a contraction on a tf32 arm (CUDA under [tf32_matmuls]) is compared with the
   default pipeline over a lowering whose accumulations of that contraction have their multiplicands
   rounded to tf32 (gh-ocannl-1602, [tf32_round]): against the f32 reference, a pool window's argmax
   flips and the routed gradient moves, so every such seed of a training step used to mismatch.
   [round:false] keeps the plain reference, the control that shows the rounding is load-bearing.
   What equal rounding leaves is the order of the f32 accumulation, which tensor cores do not keep,
   and a decision on a tie -- a relu gate on a pre-activation within ulps of 0, a pool window tied
   at 0 -- can still flip. A seed that mismatches with flipped decisions is judged against the
   reference replaying them ([decision_nodes], [replay_nodes]): the forward nodes the decisions are
   taken on (a selection's comparison operands, a gate's threshold operand) hold the seed's values
   and must be within the tolerance before any replay; everything else, the selections included, is
   recomputed from them and compared at the tolerance; and the flipped cells are bounded
   ([replay_bound]). [replay:false] turns that off. Nothing is masked: every cell of every compared
   node counts.

   [perturb] is the negative control: the seeded segment's code is perturbed after scheduling (every
   [Set] stores one more than it computes, a persistent fragment's transfers excepted:
   [perturb_code]), so each perturbed seed that compiles must be reported a mismatch, and must
   render its tensor-core statements as the seed compiled unperturbed does ([Rerendered] otherwise,
   gh-ocannl-1677) -- a control caught on the scalar fallback says nothing of the tensor-core path
   the seed takes. A point-in-time check of one step: it says nothing about seeds a later search
   proposes on different code, and nothing about timing. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode

type verdict =
  | Match of { err : float; node : string }
      (** Within tolerance; [err] is the worst normalized difference, at [node]. *)
  | Mismatch of { err : float; node : string }
  | Declined of string
      (** The candidate compile was refused, typed — a decline the search would have counted. *)
  | Not_applied  (** The seed never reached its segment: the replay did not find it. *)
  | Not_perturbed
      (** Negative control only: the seeded segment had no statement [perturb_code] perturbs. *)
  | Rerendered of string
      (** Negative control only (gh-ocannl-1677): the perturbed seed's tensor-core statements did
          not render as the unperturbed seed's (its mma census differs, or one of the two did not
          compile), so a mismatch would be caught on a path the seed does not take. A failure. *)
  | Unjudged of string
      (** gh-ocannl-1602: the seed computes on a tf32 arm, but which reference accumulations its
          tensor-core statements replaced could not be established, so no reference with its
          rounding exists. A failure, never a silent pass. *)
  | Failed of string  (** An untyped compile failure, or the run raised. *)

type seed_result = {
  label : string;  (** The candidate's label as the search prints it ([F_sketch[...]]). *)
  verdict : verdict;
  tensorized : bool;  (** The routine's mma census says it rendered tensorized. *)
  mma : Ir.C_syntax.mma_summary;
      (** The compiled candidate's mma census ([Context.routine.mma]); empty when it did not
          compile. *)
  compared : int;  (** Nodes compared. *)
  rounded : string list;
      (** gh-ocannl-1602: the nodes whose accumulated products the seed's reference rounds to tf32
          (the seed computes them on a tf32 arm); empty when the reference is the plain default
          pipeline. *)
  rounded_stmts : int;  (** The reference's accumulation statements so rounded. *)
  flipped : int;
      (** gh-ocannl-1602: the decision cells the seed decided otherwise than its reference; counted
          only for a seed that mismatched. *)
  replayed : bool;  (** The verdict is the replayed reference's ([flipped] cells replayed). *)
}

type reference = {
  ref_compared : int;  (** The nodes the reference writes, each a comparison candidate. *)
  ref_restored : int;  (** Read-and-written nodes restored before every run. *)
  ref_poisoned : int;  (** Written nodes filled with NaN before every run. *)
  ref_poison_unsafe : string list;
      (** Written nodes the poisoned reference left non-finite: never poisoned. *)
  ref_rounded_compiles : int;
      (** gh-ocannl-1602: tf32-rounded references compiled, one per distinct rounded node set in a
          row. *)
  ref_replay_compiles : int;  (** References compiled to replay a flipped seed's decisions. *)
  ref_decisions : string list;  (** The decision nodes ([decision_nodes]) a replay replays. *)
  ref_decision_cells : int;  (** Their cells: the replay bound's base. *)
  ref_unreplayable : string list;  (** Decisions the replay cannot reach. *)
}

(* The default bound on a seed's worst normalized difference, in every numerics regime: a seed that
   computes on a tf32 arm is compared with a reference rounded as it rounds (gh-ocannl-1602), so
   what is left is the order of the f32 accumulation, as for f32 seeds. The negative control's
   perturbation lands at 40 and above. *)
let default_tol () = 1e-3
let conv_seed (p : Autotune.sketch_params) = p.Autotune.sk_conv
let any_seed (_ : Autotune.sketch_params) = true

let target ctx =
  let backend = Context.backend_name ctx in
  (Ir.Schedule.backend_is_gpu backend, Ir.Schedule.backend_is_cpu backend)

let census_lines ctx ~static_indices opt =
  let is_gpu, is_cpu = target ctx in
  let limits = Context.hardware_limits ctx in
  let capabilities = Context.codegen_capabilities ctx in
  let lines = ref [] in
  let census ~where opt =
    lines :=
      List.rev_append
        (Autotune.seeding_census_lines ~where
           (Autotune.seeding_census ~coalesced:true ~capabilities ~is_gpu ~is_cpu ~limits opt))
        !lines
  in
  ignore
    (Autotune.search_seeding ~census ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices opt
      : Autotune.search_seeding);
  List.rev !lines

(* The negative control's perturbation: every [Set] stores one more than it computes, except a
   persistent fragment's transfers (gh-ocannl-1677). [Schedule.contract_tensorized_accumulator]
   brackets a tensorized reduction with lane 0's copies [fragment <- target] and [target <-
   fragment], and the renderer takes the region to the tensor cores only while both are plain copies
   ([LL.fragment_region]): perturbing them dissolved it, and the [Tile_mma] inside rendered as the
   lane-0 scalar fallback, so the control judged a path the seed does not take. The transfers are
   kept, and each store-back is followed by its own nest, lane 0 adding one to every target cell it
   stored -- the shape [Fuse_epilogue]'s epilogue takes after the region, which the renderer places
   after the hooks' visibility barrier. [fragments] are the routine's [simdgroup_fragments]; the
   count is of the statements perturbed. That the perturbed seed renders as the unperturbed one does
   is [parity]'s check, not this function's. *)
let perturb_code ~fragments (c : LL.t) : LL.t * int =
  let count = ref 0 in
  let fragment tn = Set.mem fragments tn in
  let store_back = function
    | LL.Set { tn; llsc = Get (src, _); _ } -> fragment src && not (fragment tn)
    | _ -> false
  in
  let rec stores_back (c : LL.t) =
    match c with
    | Set _ -> store_back c
    | Seq (a, b) -> stores_back a || stores_back b
    | For_loop { body; _ } | If { body; _ } -> stores_back body
    | _ -> false
  in
  (* The store-back's nest, its loops and guards kept, every store turned into [target += 1] and
     everything else dropped. *)
  let rec bump (c : LL.t) : LL.t =
    match c with
    | Set ({ tn; idcs; _ } as s) when store_back c ->
        Int.incr count;
        let prec = Lazy.force tn.Tn.storage_prec in
        Set { s with llsc = Binop (Ir.Ops.Add, (Get (tn, idcs), prec), (Constant 1., prec)) }
    | Seq (a, b) -> Seq (bump a, bump b)
    | For_loop f -> For_loop { f with body = bump f.body }
    | If i -> If { i with body = bump i.body }
    | _ -> Noop
  in
  let rec go ~bumped (c : LL.t) : LL.t =
    match c with
    | Set s when fragment s.tn || store_back c -> c
    | Set s ->
        Int.incr count;
        let prec = Lazy.force s.tn.Tn.storage_prec in
        Set { s with llsc = Binop (Ir.Ops.Add, (s.llsc, prec), (Constant 1., prec)) }
    | For_loop ({ axis = Workgroup; _ } as f) when (not bumped) && stores_back c ->
        Seq (For_loop { f with body = go ~bumped:true f.body }, bump c)
    | Seq (a, b) -> Seq (go ~bumped a, go ~bumped b)
    | For_loop f -> For_loop { f with body = go ~bumped f.body }
    | If i -> If { i with body = go ~bumped i.body }
    | Scan_loop s -> Scan_loop { s with body = go ~bumped s.body }
    | _ -> c
  in
  let c = go ~bumped:false c in
  (c, !count)

(* Two compiles' tensor-core statements rendered alike: the same renderings in emission order (the
   kernel names aside). *)
let same_renderings (a : Ir.C_syntax.mma_summary) (b : Ir.C_syntax.mma_summary) =
  List.equal
    (fun (_, r) (_, r') -> Ir.C_syntax.equal_mma_rendering r r')
    a.Ir.C_syntax.renderings b.Ir.C_syntax.renderings

let census_summary (m : Ir.C_syntax.mma_summary) =
  Printf.sprintf "%s, %d tensor-core statements, %d scalar fallbacks"
    (Ir.C_syntax.tensorization_name m.Ir.C_syntax.tensorization)
    m.Ir.C_syntax.statements m.Ir.C_syntax.scalar_fallbacks

(* gh-ocannl-1602: the reference's tf32 rounding. A tensorized f32 contraction on a tf32 arm reads
   both multiplicands rounded to tf32 (CUDA's [__float_to_tf32], [cvt.rna.tf32.f32]: an 11-bit
   significand, round to nearest, ties away from zero), multiplies them exactly (11 x 11 bits fit
   f32's 24) and accumulates in f32. The default pipeline never tensorizes a conv, so its products
   are f32's, and a pool window whose two candidates sit within tf32's error routes its gradient
   elsewhere than the seed's. The reference therefore rounds the multiplicands of exactly the
   accumulations the seed computes on a tf32 arm; what remains between the two is the order of the
   f32 accumulation.

   [tf32_round x] is that rounding from existing operations, there being no tf32 op and no bitcast:
   Veltkamp's split [c - (c - x)], [c = (2^s + 1) x], keeps x's top [24 - s] significand bits,
   rounded to nearest. [c] is an explicit [FMA (x, 2^s, x)] -- the product [2^s x] is exact, so
   fused or not it is [RN((2^s + 1) x)] -- and no bare multiplication is left for a contracting
   compiler (nvrtc's default [--fmad=true]) to fuse with a subtraction, which would collapse the
   split to the identity. Veltkamp's tie rule is not ties-away, so ties are recognized and resolved
   separately: [x] is a tie at 11 bits iff it is representable at 12 bits ([split 12 x = x]) and not
   at 11 ([h <> x]); its other neighbour is [x + (x - h)], exact, and the result is the neighbour of
   larger magnitude. Exact for finite normal [|x|] below about [4e34] (where [(2^13 + 1) x]
   overflows); subnormals round differently from [cvt.rna] and are outside this rounding's claim.
   [test/operations/tf32_reference_rounding] pins it bitwise against [cvt.rna]'s bit model, and
   against the hardware conversion where a tf32 arm exists. *)
let tf32_round ((x, prec) : LL.scalar_arg) : LL.scalar_t =
  let open LL in
  let a e = (e, prec) in
  let x' = (x, prec) in
  let split s =
    let c = Ternop (Ir.Ops.FMA, x', a (Constant (Float.of_int (1 lsl s))), x') in
    Binop (Ir.Ops.Sub, a c, a (Binop (Ir.Ops.Sub, a c, x')))
  in
  let h = split 13 in
  let other = Binop (Ir.Ops.Add, x', a (Binop (Ir.Ops.Sub, x', a h))) in
  let abs e = Binop (Ir.Ops.Max, a e, a (Unop (Ir.Ops.Neg, a e))) in
  let tie =
    Binop (Ir.Ops.And, a (Binop (Ir.Ops.Cmpeq, a (split 12), x')), a (Binop (Ir.Ops.Cmpne, a h, x')))
  in
  let away =
    Ternop (Ir.Ops.Where, a (Binop (Ir.Ops.Cmplt, a (abs h), a (abs other))), a other, a h)
  in
  Ternop (Ir.Ops.Where, a tie, a away, a h)

(* The multiplicands of an accumulation of a product into a node of [targets], in the three forms
   [Ir.Schedule.accumulated_product] recognizes (FMA, and the add of a product on either side) but
   over arbitrary operand expressions: [Some (rebuild, x, y)] where [rebuild x' y'] is the
   accumulation with its multiplicands replaced. *)
let accumulated_product ~is_acc (llsc : LL.scalar_t) =
  let open LL in
  match llsc with
  | Ternop (Ir.Ops.FMA, x, y, ((acc, _) as accp)) when is_acc acc ->
      Some ((fun x y -> Ternop (Ir.Ops.FMA, x, y, accp)), x, y)
  | Binop (Ir.Ops.Add, ((acc, _) as accp), (Binop (Ir.Ops.Mul, x, y), mp)) when is_acc acc ->
      Some ((fun x y -> Binop (Ir.Ops.Add, accp, (Binop (Ir.Ops.Mul, x, y), mp))), x, y)
  | Binop (Ir.Ops.Add, (Binop (Ir.Ops.Mul, x, y), mp), ((acc, _) as accp)) when is_acc acc ->
      Some ((fun x y -> Binop (Ir.Ops.Add, (Binop (Ir.Ops.Mul, x, y), mp), accp)), x, y)
  | _ -> None

let is_single = function Ir.Ops.Single_prec _ -> true | _ -> false

(* Whether [llsc] accumulates a product into [tn] at [idcs] (a [Set]'s own cell). *)
let accumulates_product ~tn ~idcs llsc =
  let is_acc = function
    | LL.Get (tn', idcs') -> Tn.equal tn tn' && Array.equal Ir.Indexing.equal_axis_index idcs idcs'
    | _ -> false
  in
  Option.is_some (accumulated_product ~is_acc llsc)

(* [code] with both multiplicands of every f32 product accumulated into a node of [targets] rounded
   to tf32 ([tf32_round]), and how many accumulation statements were rewritten. Statement-level
   [Set]s, and [Set_local]s of a scope over a target inside scalar expressions (a target inlined
   into its consumer), are both reached. *)
let round_products ~targets (code : LL.t) : LL.t * int =
  let count = ref 0 in
  let is_target tn = List.mem targets tn ~equal:Tn.equal in
  let round_arg ((_, prec) as arg : LL.scalar_arg) =
    if is_single prec then (tf32_round arg, prec) else arg
  in
  let rewrite ~is_acc llsc =
    match accumulated_product ~is_acc llsc with
    | Some (rebuild, x, y) ->
        Int.incr count;
        rebuild (round_arg x) (round_arg y)
    | None -> llsc
  in
  let rec stmt (c : LL.t) : LL.t =
    match c with
    | Set s when is_target s.tn ->
        let is_acc = function
          | LL.Get (tn, idcs) ->
              Tn.equal tn s.tn && Array.equal Ir.Indexing.equal_axis_index idcs s.idcs
          | _ -> false
        in
        Set { s with llsc = scalar (rewrite ~is_acc s.llsc) }
    | Set s -> Set { s with llsc = scalar s.llsc }
    | Set_local (id, llsc) when is_target id.LL.tn ->
        let is_acc = function LL.Get_local id' -> LL.equal_scope_id id id' | _ -> false in
        Set_local (id, scalar (rewrite ~is_acc llsc))
    | Set_local (id, llsc) -> Set_local (id, scalar llsc)
    | Seq (a, b) -> Seq (stmt a, stmt b)
    | For_loop f -> For_loop { f with body = stmt f.body }
    | If i -> If { i with body = stmt i.body }
    | Scan_loop s -> Scan_loop { s with body = stmt s.body }
    | _ -> c
  and scalar (e : LL.scalar_t) : LL.scalar_t =
    match e with
    | Local_scope s -> Local_scope { s with body = stmt s.body }
    | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
        Ternop (op, (scalar a, pa), (scalar b, pb), (scalar c, pc))
    | Binop (op, (a, pa), (b, pb)) -> Binop (op, (scalar a, pa), (scalar b, pb))
    | Unop (op, (a, pa)) -> Unop (op, (scalar a, pa))
    | _ -> e
  in
  let code = stmt code in
  (code, !count)

(* The nodes a segment writes with an accumulation of a product. *)
let product_targets (code : LL.t) =
  let found = ref [] in
  let rec stmt (c : LL.t) =
    match c with
    | Set { tn; idcs; llsc; _ } ->
        if accumulates_product ~tn ~idcs llsc && not (List.mem !found tn ~equal:Tn.equal) then
          found := tn :: !found
    | Seq (a, b) ->
        stmt a;
        stmt b
    | For_loop { body; _ } | If { body; _ } | Scan_loop { body; _ } -> stmt body
    | _ -> ()
  in
  stmt code;
  List.rev !found

(* The [Tile_mma] statements of a segment: (accumulator, a, b) nodes. *)
let tile_mmas (code : LL.t) =
  let found = ref [] in
  let rec stmt (c : LL.t) =
    match c with
    | Tile_mma { d = d, _; a = a, _; b = b, _; _ } -> found := (d, a, b) :: !found
    | Seq (x, y) ->
        stmt x;
        stmt y
    | For_loop { body; _ } | If { body; _ } | Scan_loop { body; _ } -> stmt body
    | _ -> ()
  in
  stmt code;
  List.rev !found

(* Whether the backend's arm for these storage precisions reads its operands as tf32: the first
   format triple a site with them resolves to ([Autotune.mma_format_triples], the seeder's own
   preference order) that the backend advertises has tf32 multiplicands. *)
let reads_tf32 (limits : Ir.Backend_intf.hardware_limits) ~a_prec ~b_prec ~d_prec =
  match limits.Ir.Backend_intf.mma with
  | None -> false
  | Some mma -> (
      match
        List.find (Autotune.mma_format_triples ~a_prec ~b_prec ~d_prec) ~f:(fun key ->
            List.Assoc.mem mma.Ir.Backend_intf.mma_format_tiles key
              ~equal:Ir.Backend_intf.equal_mma_format_triple)
      with
      | Some (a, b, _) ->
          Ir.Backend_intf.equal_mma_input_format a Ir.Backend_intf.Mma_tf32
          && Ir.Backend_intf.equal_mma_input_format b Ir.Backend_intf.Mma_tf32
      | None -> false)

(* [reads_tf32] for a uniform-f32 site, as the current numerics policy has it and as the tf32 policy
   would: a test gating a tf32 claim skips with the aggregation that lifts it. *)
let f32_reads_tf32 limits =
  reads_tf32 limits ~a_prec:Ir.Ops.single ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single

let f32_reads_tf32_under_tf32_policy limits =
  let saved = Ir.Numerics.get () in
  Exn.protect
    ~finally:(fun () -> Ir.Numerics.set_policy saved)
    ~f:(fun () ->
      Ir.Numerics.set_policy { saved with Ir.Numerics.tf32_matmuls = true };
      f32_reads_tf32 limits)

(* The worst normalized difference over the nodes [names]: per node, the largest absolute cell
   difference over the reference's largest magnitude. A non-finite cell matches only a non-finite
   reference cell of the same class. *)
let worst_difference ~reference ~candidate =
  List.fold reference ~init:(0., "", 0) ~f:(fun ((worst, at, n) as acc) (tn, want) ->
      match List.Assoc.find candidate ~equal:Tn.equal tn with
      | None -> acc
      | Some got ->
          let mag =
            Array.fold want ~init:0. ~f:(fun m v ->
                if Float.is_finite v then Float.max m (Float.abs v) else m)
          in
          let mag = if Float.(mag > 0.) then mag else 1. in
          let err =
            if Array.length got <> Array.length want then Float.infinity
            else
              Array.fold2_exn want got ~init:0. ~f:(fun e w g ->
                  if Float.is_finite w && Float.is_finite g then
                    Float.max e (Float.abs (g -. w) /. mag)
                  else if Float.is_nan w && Float.is_nan g then e
                  else if Float.equal w g then e
                  else Float.infinity)
          in
          if Float.(err > worst) || String.is_empty at then (err, Tn.debug_name tn, n + 1)
          else (worst, at, n + 1))

(* A mismatch's anatomy, for diagnosis: each compared node with cells over [tol] (normalized as in
   [worst_difference]), worst first, how many and the first few; then, at the first such cells of
   the worst node, every same-sized node's reference and candidate values -- for a selection node,
   the candidates it chose between. *)
let mismatch_detail ~tol ~reference ~candidate emit =
  let bad_cells want got =
    let mag =
      Array.fold want ~init:0. ~f:(fun m v ->
          if Float.is_finite v then Float.max m (Float.abs v) else m)
    in
    let mag = if Float.(mag > 0.) then mag else 1. in
    List.filter
      (List.init (Array.length want) ~f:Fn.id)
      ~f:(fun i ->
        let w = want.(i) and g = got.(i) in
        if Float.is_finite w && Float.is_finite g then Float.(abs (g -. w) /. mag > tol)
        else not (Float.equal w g || (Float.is_nan w && Float.is_nan g)))
  in
  let pairs =
    List.filter_map reference ~f:(fun (tn, want) ->
        match List.Assoc.find candidate ~equal:Tn.equal tn with
        | Some got when Array.length got = Array.length want -> Some (tn, want, got)
        | _ -> None)
  in
  let bad =
    List.filter_map pairs ~f:(fun (tn, want, got) ->
        match bad_cells want got with
        | [] -> None
        | cells ->
            Some
              ( worst_difference ~reference:[ (tn, want) ] ~candidate:[ (tn, got) ],
                (tn, want, got, cells) ))
    |> List.sort ~compare:(fun ((a, _, _), _) ((b, _, _), _) -> Float.compare b a)
    |> List.map ~f:snd
  in
  List.iteri bad ~f:(fun k (tn, want, got, cells) ->
      if k < 16 then
        emit
          (Printf.sprintf "  %s: %d of %d cells over tolerance; first: %s" (Tn.debug_name tn)
             (List.length cells) (Array.length want)
             (String.concat ~sep:" "
                (List.map (List.take cells 4) ~f:(fun i ->
                     Printf.sprintf "[%d] %.9g/%.9g" i want.(i) got.(i))))));
  match bad with
  | [] -> ()
  | (tn, want, _, cells) :: _ ->
      List.iter (List.take cells 3) ~f:(fun i ->
          emit
            (Printf.sprintf "  at %s[%d], same-sized nodes (reference/candidate): %s"
               (Tn.debug_name tn) i
               (String.concat ~sep:"; "
                  (List.filter_map pairs ~f:(fun (tn', w, g) ->
                       if Array.length w = Array.length want then
                         Some (Printf.sprintf "%s %.9g/%.9g" (Tn.debug_name tn') w.(i) g.(i))
                       else None)))))

(* An [@|] slice view has no host access of its own (its parent is read or written instead), so it
   is neither compared, restored nor poisoned; its parent, when the routine writes it, is. *)
let host_accessible tn = Option.is_none (Tn.slice_of tn)

(* Every other node read is a materialized node a routine writes (or reads and writes): a readback
   that raises propagates, so a transfer failure fails the seed rather than shrinking the
   comparison. *)
let read ctx nodes =
  List.filter_map nodes ~f:(fun tn ->
      if host_accessible tn then Some (tn, Context.get_values ctx tn) else None)

(* gh-ocannl-1602: the decisions of a lowering -- the discontinuities a value within rounding of a
   threshold can flip -- found structurally, with the forward nodes they are taken on: - [Selection
   tn]: a node written by a comparison ([cond = eq (t, v)], max-pooling's backward gate in
   [Operation.tropical]); its cells are the decisions. Never replayed: the reference recomputes it
   from the replayed operands, and it is compared. - [Operand tn]: a node a selection's comparison
   reads. - [Gate (op, tn)]: a node read as the threshold operand of a gate ([relu_gate (v1, g)],
   the relu's backward); its cells' side of the threshold are the decisions. Operands and gates are
   what a replay preloads. The second list names the decisions a replay cannot reach: a comparison
   written to a scope local (the node inlined into its consumer), or a comparison or gate whose
   operand is an expression rather than a node read. *)
type decision = Selection of Tn.t | Operand of Tn.t | Gate of Ir.Ops.binop * Tn.t

let decision_tn = function Selection tn | Operand tn | Gate (_, tn) -> tn
let is_replayed = function Selection _ -> false | Operand _ | Gate _ -> true

let equal_decision a b =
  match (a, b) with
  | Selection a, Selection b | Operand a, Operand b -> Tn.equal a b
  | Gate (o, a), Gate (o', b) -> Ir.Ops.equal_binop o o' && Tn.equal a b
  | _ -> false

let decision_nodes (code : LL.t) =
  let found = ref [] and unreachable = ref [] in
  let add d = if not (List.mem !found d ~equal:equal_decision) then found := d :: !found in
  let is_comparison = function
    | Ir.Ops.Cmpeq | Ir.Ops.Cmpne | Ir.Ops.Cmplt | Ir.Ops.Cmple -> true
    | _ -> false
  in
  let operand ~into = function
    | LL.Get (tn, _) -> add (Operand tn)
    | _ -> unreachable := ("an expression compared into " ^ Tn.debug_name into) :: !unreachable
  in
  let rec stmt (c : LL.t) =
    match c with
    | Set { tn; llsc; _ } ->
        (match llsc with
        | Binop (op, (a, _), (b, _)) when is_comparison op ->
            add (Selection tn);
            operand ~into:tn a;
            operand ~into:tn b
        | _ -> ());
        scalar llsc
    | Set_local (id, llsc) ->
        (match llsc with
        | Binop (op, _, _) when is_comparison op ->
            unreachable := ("comparison into the inlined " ^ Tn.debug_name id.LL.tn) :: !unreachable
        | _ -> ());
        scalar llsc
    | Seq (a, b) ->
        stmt a;
        stmt b
    | For_loop { body; _ } | If { body; _ } | Scan_loop { body; _ } -> stmt body
    | _ -> ()
  and scalar (e : LL.scalar_t) =
    match e with
    | Local_scope { body; _ } -> stmt body
    | Binop (((Ir.Ops.Relu_gate | Ir.Ops.Satur01_gate) as op), (a, _), (b, _)) ->
        (match a with
        | Get (tn, _) -> add (Gate (op, tn))
        | _ -> unreachable := (Ir.Ops.binop_cd_syntax op ^ " of an expression") :: !unreachable);
        scalar a;
        scalar b
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
    | _ -> ()
  in
  stmt code;
  (List.rev !found, List.dedup_and_sort !unreachable ~compare:String.compare)

(* The decision a cell takes: a selection's value, a gate's side of its threshold; an operand takes
   none of its own. *)
let decides d v =
  match d with Selection _ | Operand _ -> v | Gate (op, _) -> Ir.Ops.interpret_binop op v 1.

(* [code] with every write of a node in [replayed] made a self-copy, and its whole-node zeroings
   dropped: the node keeps the values it holds when the routine starts -- the candidate's, set
   before the run -- while every statement stays where it was, so the schedule, the segmentation and
   the routine's interface are those of the unreplayed reference. *)
let replay_nodes ~replayed (code : LL.t) : LL.t =
  let mem tn = List.mem replayed tn ~equal:Tn.equal in
  let rec stmt (c : LL.t) : LL.t =
    match c with
    | Set ({ tn; idcs; _ } as s) when mem tn -> Set { s with llsc = Get (tn, idcs) }
    | Zero_out tn when mem tn -> Noop
    | Seq (a, b) -> Seq (stmt a, stmt b)
    | For_loop f -> For_loop { f with body = stmt f.body }
    | If i -> If { i with body = stmt i.body }
    | Scan_loop s -> Scan_loop { s with body = stmt s.body }
    | _ -> c
  in
  stmt code

(* The most decision cells a seed may decide otherwise than its reference and still be judged by
   replay: [fraction] of the decision nodes' cells, at least one. A gate's threshold operand is a
   forward node, compared at the tolerance before any replay, and a selection's operands are forward
   nodes too: a flip is a decision the forward puts within the tolerance of its threshold, while a
   wrong forward shows in the forward nodes. The bound keeps the replay from absorbing a systematic
   disagreement. The default, one in a hundred thousand, is set from cifar_resnet's training step on
   CUDA tf32 (rog-nv-linux): against the rounded reference a seed flips 0 or 4 of 6.5M decision
   cells (6e-7), against the plain f32 reference 21 to 645 (3e-6 to 1e-4). *)
let default_replay_fraction = 1e-5

let replay_bound ?(fraction = default_replay_fraction) ~decision_cells () =
  Int.max 1 (Float.iround_down_exn (fraction *. Float.of_int decision_cells))

(* The default pipeline over [comp]'s lowering with the products accumulated into [targets] rounded
   to tf32: the transform rounds, then schedules exactly as a compile without a transform does
   ([Ir.Schedule.maybe_default_schedules] over the bindings' static symbols). With [replayed], those
   nodes hold the values set before the run ([replay_nodes]). Returns the context, the routine and
   how many accumulation statements were rounded. *)
let compile_rounded ?(replayed = []) ~targets ctx comp bindings =
  let backend_name = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let stmts = ref 0 in
  let transform (lowered : LL.optimized) =
    let llc, n = round_products ~targets lowered.LL.llc in
    let llc = if List.is_empty replayed then llc else replay_nodes ~replayed llc in
    stmts := n;
    Ir.Schedule.maybe_default_schedules ~backend_name ~limits
      ~static_indices:(Ir.Indexing.bound_symbols bindings)
      { lowered with LL.llc }
  in
  let rctx, r = Context.compile ~lowered_transform:transform ctx comp bindings in
  (rctx, r, !stmts)

let parity ?tol ?(perturb = false) ?(round = true) ?(replay = true) ?replay_fraction
    ?(select = conv_seed) ?(on_result = fun _ -> ()) ?detail ctx comp bindings ~bind ~static_indices
    opt =
  let tol = match tol with Some t -> t | None -> default_tol () in
  let is_gpu, is_cpu = target ctx in
  let limits = Context.hardware_limits ctx in
  let capabilities = Context.codegen_capabilities ctx in
  let accum_prec = capabilities.Ir.Backend_intf.accum_prec in
  let set_list s = Set.to_list s in
  (* The reference: the untuned default pipeline. *)
  let rctx, rroutine = Context.compile ctx comp bindings in
  bind rroutine;
  let rbw (r : Context.routine) = set_list (Set.inter r.inputs r.outputs) in
  let snapshot = read rctx (rbw rroutine) in
  let restore ctx (r : Context.routine) =
    List.fold (rbw r) ~init:ctx ~f:(fun ctx tn ->
        match List.Assoc.find snapshot ~equal:Tn.equal tn with
        | Some vs -> Context.set_values ctx tn vs
        | None -> ctx)
  in
  let written_only (r : Context.routine) = set_list (Set.diff r.outputs r.inputs) in
  let poison ~unsafe ctx (r : Context.routine) =
    List.fold (written_only r) ~init:ctx ~f:(fun ctx tn ->
        if List.mem unsafe tn ~equal:Tn.equal || not (host_accessible tn) then ctx
        else Context.set_values ctx tn (Array.create ~len:(Tn.num_elems tn) Float.nan))
  in
  let run ~unsafe ~poisoned ctx (r : Context.routine) =
    let ctx = restore ctx r in
    let ctx = if poisoned then poison ~unsafe ctx r else ctx in
    let ctx = Context.run ctx r in
    Context.sync ctx;
    read ctx (set_list r.outputs)
  in
  let plain = run ~unsafe:[] ~poisoned:false rctx rroutine in
  let poisoned = run ~unsafe:[] ~poisoned:true rctx rroutine in
  let unsafe =
    List.filter_map poisoned ~f:(fun (tn, got) ->
        match List.Assoc.find plain ~equal:Tn.equal tn with
        | Some want
          when Array.existsi got ~f:(fun i g ->
                   (not (Float.is_finite g)) && i < Array.length want && Float.is_finite want.(i))
          ->
            Some tn
        | Some _ | None -> None)
  in
  let ref_poisoned =
    List.count (written_only rroutine) ~f:(fun tn ->
        not (List.mem unsafe tn ~equal:Tn.equal || not (host_accessible tn)))
  in
  (* The reference's pools are freed before any candidate compiles: a routine that fits the device
     alone must not decline for sharing it with the reference. *)
  Context.release rctx;
  let ref_written tn = List.Assoc.mem plain ~equal:Tn.equal tn in
  (* gh-ocannl-1602: the decisions a flipped seed's reference replays -- on nodes the reference
     writes, the host reads and the routine does not restore -- and the ones it cannot reach. *)
  let decisions, unreplayable =
    let found, unreachable = decision_nodes opt.LL.llc in
    let restored tn = List.Assoc.mem snapshot ~equal:Tn.equal tn in
    let reachable, unreached =
      List.partition_tf found ~f:(fun d ->
          let tn = decision_tn d in
          ref_written tn && host_accessible tn && not (restored tn))
    in
    ( reachable,
      unreachable
      @ List.map unreached ~f:(fun d -> "decision node " ^ Tn.debug_name (decision_tn d)) )
  in
  let replayed_tns =
    List.dedup_and_sort ~compare:Tn.compare
      (List.filter_map decisions ~f:(fun d -> if is_replayed d then Some (decision_tn d) else None))
  in
  (* The decisions whose cells are counted: selections and gates (an operand decides nothing). *)
  let deciding = List.filter decisions ~f:(function Operand _ -> false | _ -> true) in
  let unreplayable_names = String.concat ~sep:"; " unreplayable in
  let decision_cells = List.sum (module Int) deciding ~f:(fun d -> Tn.num_elems (decision_tn d)) in
  (* gh-ocannl-1602: the reference of a seed that computes [targets] on a tf32 arm -- the default
     pipeline, scheduled exactly as the default compile schedules it, over the lowering with those
     targets' accumulated products rounded to tf32 ([round_products]). Compiled once per distinct
     target set in a row (a segment's seeds come together and share their site), and released before
     the next candidate compiles, as the plain reference is. *)
  let rounded_compiles = ref 0 in
  let last_rounded = ref None in
  let rounded_reference targets =
    let targets = List.sort targets ~compare:Tn.compare in
    match !last_rounded with
    | Some (key, values) when List.equal Tn.equal key targets -> values
    | _ ->
        (* Drop the previous set's values before this one's are read. *)
        last_rounded := None;
        let rctx, r, stmts = compile_rounded ~targets ctx comp bindings in
        bind r;
        let values =
          Exn.protect
            ~f:(fun () -> run ~unsafe:[] ~poisoned:false rctx r)
            ~finally:(fun () -> Context.release rctx)
        in
        Int.incr rounded_compiles;
        let result = (values, stmts) in
        last_rounded := Some (targets, result);
        result
  in
  (* gh-ocannl-1602: the reference that judges a seed whose decisions flipped -- the same pipeline
     (rounded at [targets]), the decisions' forward operands holding the candidate's values
     ([replay_nodes]), so everything the reference computes from them decides as the candidate's
     forward values decide. Compiled per flipped seed (flips are rare), and released before the next
     candidate compiles. *)
  let replay_compiles = ref 0 in
  let replayed_reference targets candidate =
    let rctx, r, _ = compile_rounded ~replayed:replayed_tns ~targets ctx comp bindings in
    Int.incr replay_compiles;
    bind r;
    Exn.protect
      ~finally:(fun () -> Context.release rctx)
      ~f:(fun () ->
        let rctx = restore rctx r in
        let rctx =
          List.fold replayed_tns ~init:rctx ~f:(fun rctx tn ->
              match List.Assoc.find candidate ~equal:Tn.equal tn with
              | Some vs -> Context.set_values rctx tn vs
              | None -> rctx)
        in
        let rctx = Context.run rctx r in
        Context.sync rctx;
        read rctx (set_list r.outputs))
  in
  let seeding =
    Autotune.search_seeding
      ~census:(fun ~where:_ _ -> ())
      ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices opt
  in
  let segments =
    List.map seeding.ss_coarse ~f:(fun e -> (false, seeding.ss_coarse_plan, e))
    @ List.map seeding.ss_fine_new ~f:(fun e -> (true, seeding.ss_fine_plan, e))
  in
  let prec tn = Lazy.force tn.Tn.storage_prec in
  (* gh-ocannl-1602: which reference accumulations a compiled seed computes on a tf32 arm. [Ok []]
     when none: no tensor-core statement reads tf32, or every one rendered as the scalar fallback
     (f32 arithmetic). A statement whose accumulator is a node the reference writes names it; one
     accumulating into a tile the schedule minted is matched to the only product accumulation of its
     pre-schedule segment the reference writes. Anything else -- some statements rendered and some
     fell back, or no single accumulation to match -- is [Error]: the arithmetic the seed performs
     is not established, and the seed is not judged against a guess. *)
  let tf32_targets (routine : Context.routine) segs =
    let per_seg =
      List.map segs ~f:(fun ((pre : LL.optimized), (post : LL.optimized)) ->
          ( pre,
            List.filter (tile_mmas post.llc) ~f:(fun (d, a, b) ->
                reads_tf32 limits ~a_prec:(prec a) ~b_prec:(prec b) ~d_prec:(prec d)) ))
    in
    if List.for_all per_seg ~f:(fun (_, mmas) -> List.is_empty mmas) then Ok []
    else
      let census = routine.Context.mma in
      match census.Ir.C_syntax.tensorization with
      | Ir.C_syntax.Not_requested | Ir.C_syntax.Scalar_fallback -> Ok []
      | Ir.C_syntax.Tensorized when census.Ir.C_syntax.scalar_fallbacks > 0 ->
          Error
            (Printf.sprintf
               "%d of %d tensor-core statements rendered as the scalar fallback: which products \
                read tf32 is not established"
               census.Ir.C_syntax.scalar_fallbacks census.Ir.C_syntax.statements)
      | Ir.C_syntax.Tensorized ->
          List.fold_result per_seg ~init:[] ~f:(fun acc ((pre : LL.optimized), mmas) ->
              List.fold_result mmas ~init:acc ~f:(fun acc (d, _, _) ->
                  let add tn = if List.mem acc tn ~equal:Tn.equal then acc else tn :: acc in
                  if ref_written d then Ok (add d)
                  else
                    match List.filter (product_targets pre.llc) ~f:ref_written with
                    | [ tn ] -> Ok (add tn)
                    | cands ->
                        Error
                          (Printf.sprintf
                             "the tensor-core statement accumulating into %s replaced one of %d \
                              product accumulations the reference writes (%s)"
                             (Tn.debug_name d) (List.length cands)
                             (String.concat ~sep:" " (List.map cands ~f:Tn.debug_name)))))
  in
  let one ~fine ~plan key p =
    let label = Autotune.fiss_sketch_label ~fine [ (key, p) ] in
    let seeded = ref [] and perturbed = ref true and segs = ref [] in
    let transform ~perturb fresh =
      seeded := [];
      perturbed := true;
      segs := [];
      let plan, schedule =
        Autotune.sketch_replay ~static_indices
          ~sketch:(fun p seg -> Autotune.sketch_schedule ~accum_prec ~p seg)
          plan
          [ (key, p) ]
      in
      let schedule i seg =
        let s = schedule i seg in
        if Option.is_some s then seeded := i :: !seeded;
        s
      in
      let _, tuples =
        Autotune.tuner_fission ~is_gpu ~limits ~replay:(plan, schedule)
          ~preset:(Autotune.segment_preset ~presets:`Search ~is_gpu ~is_cpu ~limits)
          ~static_indices fresh
      in
      List.mapi tuples ~f:(fun i (_, (pre : LL.optimized), _, (post : LL.optimized)) ->
          if List.mem !seeded i ~equal:Int.equal then (
            segs := (pre, post) :: !segs;
            if perturb then (
              let llc, count = perturb_code ~fragments:post.simdgroup_fragments post.llc in
              if count = 0 then perturbed := false;
              { post with llc })
            else post)
          else post)
    in
    let compile ~perturb =
      Context.compile_outcome ~lowered_transform:(transform ~perturb)
        ~provenance:Ir.Schedule_outcome.Candidate ~candidate:label ctx comp bindings
    in
    (* gh-ocannl-1677: under the negative control, the seed compiled unperturbed first, for its mma
       census only: a perturbation that changes how the seed's tensor-core statements render (or
       whether it compiles) is caught on a path the seed does not take, which proves nothing. *)
    let unperturbed =
      if not perturb then None
      else
        Some
          (match compile ~perturb:false with
          | Ok (cctx, routine) ->
              Context.release cctx;
              Ok routine.Context.mma
          | Error (Classified c) ->
              Error ("declined: " ^ Ir.Schedule_outcome.detail_of_cause c.Ir.Schedule_outcome.cause)
          | Error (Fatal f) -> Error ("failed: " ^ Exn.to_string f.exn))
    in
    let census = ref Ir.C_syntax.empty_mma_summary in
    let result ?(rounded = []) ?(rounded_stmts = 0) ?(flipped = 0) ?(replayed = false) verdict
        ~tensorized ~compared =
      {
        label;
        verdict;
        tensorized;
        mma = !census;
        compared;
        rounded = List.map rounded ~f:Tn.debug_name;
        rounded_stmts;
        flipped;
        replayed;
      }
    in
    match compile ~perturb with
    | Error (Classified c) -> (
        let detail = Ir.Schedule_outcome.detail_of_cause c.Ir.Schedule_outcome.cause in
        match unperturbed with
        | Some (Ok mma) ->
            result
              (Rerendered
                 (Printf.sprintf "the perturbed seed was declined (%s); unperturbed it compiles, %s"
                    detail (census_summary mma)))
              ~tensorized:false ~compared:0
        | None | Some (Error _) -> result (Declined detail) ~tensorized:false ~compared:0)
    | Error (Fatal f) -> result (Failed (Exn.to_string f.exn)) ~tensorized:false ~compared:0
    | Ok (cctx, routine) -> (
        census := routine.Context.mma;
        let tensorized =
          Ir.C_syntax.equal_tensorization routine.Context.mma.Ir.C_syntax.tensorization
            Ir.C_syntax.Tensorized
        in
        let rerendered =
          match unperturbed with
          | None -> None
          | Some (Error why) -> Some ("unperturbed, the seed " ^ why)
          | Some (Ok mma) ->
              if same_renderings mma routine.Context.mma then None
              else
                Some
                  (Printf.sprintf "perturbed %s; unperturbed %s"
                     (census_summary routine.Context.mma)
                     (census_summary mma))
        in
        let targets = if round then tf32_targets routine (List.rev !segs) else Ok [] in
        let executed =
          if List.is_empty !seeded then `Done (result Not_applied ~tensorized ~compared:0)
          else if not !perturbed then `Done (result Not_perturbed ~tensorized ~compared:0)
          else
            match (rerendered, targets) with
            | Some why, _ -> `Done (result (Rerendered why) ~tensorized ~compared:0)
            | None, Error why -> `Done (result (Unjudged why) ~tensorized ~compared:0)
            | None, Ok targets -> (
                match
                  bind routine;
                  run ~unsafe ~poisoned:true cctx routine
                with
                | candidate -> `Ran (targets, candidate)
                | exception exn ->
                    `Done (result (Failed (Exn.to_string exn)) ~tensorized ~compared:0))
        in
        (* The candidate's pools are freed before a rounded reference compiles. *)
        Context.release cctx;
        match executed with
        | `Done r -> r
        | `Ran (targets, candidate) -> (
            match if List.is_empty targets then (plain, 0) else rounded_reference targets with
            | exception exn ->
                result ~rounded:targets
                  (Failed ("the tf32-rounded reference: " ^ Exn.to_string exn))
                  ~tensorized ~compared:0
            | _, 0 when not (List.is_empty targets) ->
                result ~rounded:targets
                  (Unjudged
                     "the reference has no accumulation of a product into the nodes the seed \
                      computes on a tf32 arm, so nothing was rounded")
                  ~tensorized ~compared:0
            | reference, rounded_stmts -> (
                let rounded = targets in
                let err, node, compared = worst_difference ~reference ~candidate in
                let explain verdict ~reference =
                  match (verdict, detail) with
                  | Mismatch _, Some emit ->
                      List.iter (List.rev !segs)
                        ~f:(fun ((pre : LL.optimized), (post : LL.optimized)) ->
                          let names l = String.concat ~sep:" " (List.map l ~f:Tn.debug_name) in
                          emit
                            (Printf.sprintf
                               "  seeded segment: product accumulations into %s; tensor-core \
                                statements (d a b): %s"
                               (names (product_targets pre.llc))
                               (String.concat ~sep:", "
                                  (List.map (tile_mmas post.llc) ~f:(fun (d, a, b) ->
                                       names [ d; a; b ])))));
                      mismatch_detail ~tol ~reference ~candidate emit
                  | _ -> ()
                in
                if compared = 0 then
                  result ~rounded ~rounded_stmts (Failed "no node both routines write was compared")
                    ~tensorized ~compared
                else if Float.(err <= tol) then
                  result ~rounded ~rounded_stmts (Match { err; node }) ~tensorized ~compared
                else
                  (* gh-ocannl-1602: the decision cells the two runs decided differently, and
                     whether every gate's threshold operand -- a forward node -- is within the
                     tolerance (a selection's own cells are 0/1; its operands are forward nodes
                     compared in their own right). *)
                  let pair tn =
                    match
                      ( List.Assoc.find reference ~equal:Tn.equal tn,
                        List.Assoc.find candidate ~equal:Tn.equal tn )
                    with
                    | Some w, Some g when Array.length w = Array.length g -> Some (w, g)
                    | _ -> None
                  in
                  let flips =
                    List.sum
                      (module Int)
                      deciding
                      ~f:(fun d ->
                        match pair (decision_tn d) with
                        | Some (w, g) ->
                            Array.count (Array.zip_exn w g) ~f:(fun (w, g) ->
                                not (Float.equal (decides d w) (decides d g)))
                        | None -> 0)
                  in
                  let gates_within =
                    List.for_all replayed_tns ~f:(fun tn ->
                        match pair tn with
                        | Some (w, g) ->
                            let e, _, _ =
                              worst_difference ~reference:[ (tn, w) ] ~candidate:[ (tn, g) ]
                            in
                            Float.(e <= tol)
                        | None -> false)
                  in
                  let mismatch () =
                    let verdict = Mismatch { err; node } in
                    explain verdict ~reference;
                    result ~rounded ~rounded_stmts ~flipped:flips verdict ~tensorized ~compared
                  in
                  if flips = 0 || (not replay) || not gates_within then
                    if (not replay) || List.is_empty unreplayable then mismatch ()
                    else
                      result ~rounded ~rounded_stmts
                        (Unjudged
                           ("the seed mismatches, and decisions the replay cannot reach ("
                          ^ unreplayable_names ^ ") may have flipped"))
                        ~tensorized ~compared
                  else if not (List.is_empty unreplayable) then
                    result ~rounded ~rounded_stmts ~flipped:flips
                      (Unjudged
                         ("decisions flipped, and some the replay cannot reach ("
                        ^ unreplayable_names ^ ")"))
                      ~tensorized ~compared
                  else if flips > replay_bound ?fraction:replay_fraction ~decision_cells () then
                    mismatch ()
                  else
                    match replayed_reference targets candidate with
                    | exception exn ->
                        result ~rounded ~rounded_stmts ~flipped:flips
                          (Failed ("the replayed reference: " ^ Exn.to_string exn))
                          ~tensorized ~compared:0
                    | replayed_values ->
                        (* Every node but the replayed operands, which hold the candidate's own: the
                           selections included, recomputed from them. *)
                        let reference =
                          List.filter replayed_values ~f:(fun (tn, _) ->
                              not (List.mem replayed_tns tn ~equal:Tn.equal))
                        in
                        let err, node, compared = worst_difference ~reference ~candidate in
                        let verdict =
                          if compared = 0 then Failed "no node but the decisions was compared"
                          else if Float.(err <= tol) then Match { err; node }
                          else Mismatch { err; node }
                        in
                        explain verdict ~reference;
                        result ~rounded ~rounded_stmts ~flipped:flips ~replayed:true verdict
                          ~tensorized ~compared)))
  in
  let results =
    List.concat_map segments ~f:(fun (fine, plan, (key, _pre, params)) ->
        List.filter_map params ~f:(fun p ->
            if select p then (
              let r = one ~fine ~plan key p in
              on_result r;
              Some r)
            else None))
  in
  (* Every run restored the read-and-written nodes (the parameters) and then updated them on the
     device: leave the caller's context holding the values it came in with, so a second [parity]
     over it starts from the same state as the first. *)
  ignore
    (List.fold snapshot ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) : Context.t);
  let reference =
    {
      ref_compared = List.length plain;
      ref_restored = List.length snapshot;
      ref_poisoned;
      ref_poison_unsafe = List.map unsafe ~f:Tn.debug_name;
      ref_rounded_compiles = !rounded_compiles;
      ref_replay_compiles = !replay_compiles;
      ref_decisions = List.map replayed_tns ~f:Tn.debug_name;
      ref_decision_cells = decision_cells;
      ref_unreplayable = unreplayable;
    }
  in
  (reference, results)

(* Whether a seed's result passes: a declined seed is the search's business, not a parity defect.
   Under [perturb] a seed passes when its perturbation was detected. *)
let passed ~perturb r =
  match (perturb, r.verdict) with
  | false, (Match _ | Declined _) -> true
  | true, (Mismatch _ | Declined _) -> true
  | false, (Mismatch _ | Not_applied | Not_perturbed | Rerendered _ | Unjudged _ | Failed _)
  | true, (Match _ | Not_applied | Not_perturbed | Rerendered _ | Unjudged _ | Failed _) ->
      false

let verdict_line r =
  let v =
    match r.verdict with
    | Match { err; node } -> Printf.sprintf "MATCH worst %.3g at %s" err node
    | Mismatch { err; node } -> Printf.sprintf "MISMATCH worst %.3g at %s" err node
    | Declined d -> "DECLINED: " ^ d
    | Not_applied -> "NOT APPLIED: the seed never reached its segment"
    | Not_perturbed -> "NOT PERTURBED: the seeded segment has nothing to perturb"
    | Rerendered d -> "RERENDERED: the perturbation changed the tensor-core rendering: " ^ d
    | Unjudged d -> "UNJUDGED: " ^ d
    | Failed d -> "FAILED: " ^ d
  in
  Printf.sprintf "%s: %s (compared %d nodes%s%s)" r.label v r.compared
    (if r.tensorized then ", tensorized" else "")
    (match r.rounded with
    | [] -> ""
    | l ->
        Printf.sprintf "; reference rounded to tf32 at %s, %d statements" (String.concat ~sep:" " l)
          r.rounded_stmts)
  ^
  if r.flipped = 0 then ""
  else
    Printf.sprintf "; %d decision cells flipped, %s" r.flipped
      (if r.replayed then "replayed" else "not replayed")
