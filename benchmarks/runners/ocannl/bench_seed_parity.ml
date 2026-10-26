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
   Nothing is masked: every cell of every compared node counts.

   [perturb] is the negative control: the seeded segment's code is perturbed after scheduling (every
   [Set] stores one more than it computes), so each perturbed seed that compiles must be reported a
   mismatch. A point-in-time check of one step: it says nothing about seeds a later search proposes
   on different code, and nothing about timing. *)

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
  | Not_perturbed  (** Negative control only: the seeded segment had no [Set] to perturb. *)
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

(* The negative control's perturbation: every [Set] stores one more than it computes. *)
let rec perturb_code (c : LL.t) : LL.t =
  match c with
  | Set s ->
      let prec = Lazy.force s.tn.Tn.storage_prec in
      Set { s with llsc = Binop (Ir.Ops.Add, (s.llsc, prec), (Constant 1., prec)) }
  | Seq (a, b) -> Seq (perturb_code a, perturb_code b)
  | For_loop f -> For_loop { f with body = perturb_code f.body }
  | If i -> If { i with body = perturb_code i.body }
  | Scan_loop s -> Scan_loop { s with body = perturb_code s.body }
  | _ -> c

let rec has_set (c : LL.t) =
  match c with
  | Set _ -> true
  | Seq (a, b) -> has_set a || has_set b
  | For_loop { body; _ } | If { body; _ } | Scan_loop { body; _ } -> has_set body
  | _ -> false

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

(* An [@|] slice view has no host access of its own (its parent is read or written instead), so it
   is neither compared, restored nor poisoned; its parent, when the routine writes it, is. *)
let host_accessible tn = Option.is_none (Tn.slice_of tn)

(* Every other node read is a materialized node a routine writes (or reads and writes): a readback
   that raises propagates, so a transfer failure fails the seed rather than shrinking the
   comparison. *)
let read ctx nodes =
  List.filter_map nodes ~f:(fun tn ->
      if host_accessible tn then Some (tn, Context.get_values ctx tn) else None)

(* The default pipeline over [comp]'s lowering with the products accumulated into [targets] rounded
   to tf32: the transform rounds, then schedules exactly as a compile without a transform does
   ([Ir.Schedule.maybe_default_schedules] over the bindings' static symbols). Returns the context,
   the routine and how many accumulation statements were rounded. *)
let compile_rounded ~targets ctx comp bindings =
  let backend_name = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let stmts = ref 0 in
  let transform (lowered : LL.optimized) =
    let llc, n = round_products ~targets lowered.LL.llc in
    stmts := n;
    Ir.Schedule.maybe_default_schedules ~backend_name ~limits
      ~static_indices:(Ir.Indexing.bound_symbols bindings)
      { lowered with LL.llc }
  in
  let rctx, r = Context.compile ~lowered_transform:transform ctx comp bindings in
  (rctx, r, !stmts)

let parity ?tol ?(perturb = false) ?(round = true) ?(select = conv_seed) ?(on_result = fun _ -> ())
    ctx comp bindings ~bind ~static_indices opt =
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
    let transform fresh =
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
              if not (has_set post.llc) then perturbed := false;
              { post with llc = perturb_code post.llc })
            else post)
          else post)
    in
    let census = ref Ir.C_syntax.empty_mma_summary in
    let result ?(rounded = []) ?(rounded_stmts = 0) verdict ~tensorized ~compared =
      {
        mma = !census;
        label;
        verdict;
        tensorized;
        compared;
        rounded = List.map rounded ~f:Tn.debug_name;
        rounded_stmts;
      }
    in
    match
      Context.compile_outcome ~lowered_transform:transform ~provenance:Ir.Schedule_outcome.Candidate
        ~candidate:label ctx comp bindings
    with
    | Error (Classified c) ->
        result
          (Declined (Ir.Schedule_outcome.detail_of_cause c.Ir.Schedule_outcome.cause))
          ~tensorized:false ~compared:0
    | Error (Fatal f) -> result (Failed (Exn.to_string f.exn)) ~tensorized:false ~compared:0
    | Ok (cctx, routine) -> (
        census := routine.Context.mma;
        let tensorized =
          Ir.C_syntax.equal_tensorization routine.Context.mma.Ir.C_syntax.tensorization
            Ir.C_syntax.Tensorized
        in
        let targets = if round then tf32_targets routine (List.rev !segs) else Ok [] in
        let executed =
          if List.is_empty !seeded then `Done (result Not_applied ~tensorized ~compared:0)
          else if not !perturbed then `Done (result Not_perturbed ~tensorized ~compared:0)
          else
            match targets with
            | Error why -> `Done (result (Unjudged why) ~tensorized ~compared:0)
            | Ok targets -> (
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
            | reference, rounded_stmts ->
                let rounded = targets in
                let err, node, compared = worst_difference ~reference ~candidate in
                if compared = 0 then
                  result ~rounded ~rounded_stmts (Failed "no node both routines write was compared")
                    ~tensorized ~compared
                else
                  let verdict =
                    if Float.(err <= tol) then Match { err; node } else Mismatch { err; node }
                  in
                  result ~rounded ~rounded_stmts verdict ~tensorized ~compared))
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
  let reference =
    {
      ref_compared = List.length plain;
      ref_restored = List.length snapshot;
      ref_poisoned;
      ref_poison_unsafe = List.map unsafe ~f:Tn.debug_name;
      ref_rounded_compiles = !rounded_compiles;
    }
  in
  (reference, results)

(* Whether a seed's result passes: a declined seed is the search's business, not a parity defect.
   Under [perturb] a seed passes when its perturbation was detected. *)
let passed ~perturb r =
  match (perturb, r.verdict) with
  | false, (Match _ | Declined _) -> true
  | true, (Mismatch _ | Declined _) -> true
  | false, (Mismatch _ | Not_applied | Not_perturbed | Unjudged _ | Failed _)
  | true, (Match _ | Not_applied | Not_perturbed | Unjudged _ | Failed _) ->
      false

let verdict_line r =
  let v =
    match r.verdict with
    | Match { err; node } -> Printf.sprintf "MATCH worst %.3g at %s" err node
    | Mismatch { err; node } -> Printf.sprintf "MISMATCH worst %.3g at %s" err node
    | Declined d -> "DECLINED: " ^ d
    | Not_applied -> "NOT APPLIED: the seed never reached its segment"
    | Not_perturbed -> "NOT PERTURBED: the seeded segment has no Set"
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
