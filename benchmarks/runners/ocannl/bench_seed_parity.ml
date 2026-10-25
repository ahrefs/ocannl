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
  | Failed of string  (** An untyped compile failure, or the run raised. *)

type seed_result = {
  label : string;  (** The candidate's label as the search prints it ([F_sketch[...]]). *)
  verdict : verdict;
  tensorized : bool;  (** The routine's mma census says it rendered tensorized. *)
  compared : int;  (** Nodes compared. *)
}

type reference = {
  ref_compared : int;  (** The nodes the reference writes, each a comparison candidate. *)
  ref_restored : int;  (** Read-and-written nodes restored before every run. *)
  ref_poisoned : int;  (** Written nodes filled with NaN before every run. *)
  ref_poison_unsafe : string list;
      (** Written nodes the poisoned reference left non-finite: never poisoned. *)
}

(* The default bound on a seed's worst normalized difference: 1e-3 for f32 arithmetic, 1e-2 where
   the numerics policy lets tensor-core matmuls read f32 operands as tf32 (a 10-bit mantissa; on
   CUDA the in-memory smoke's conv seeds reach 4e-4 there). The negative control's perturbation
   lands at 40 and above. *)
let default_tol () = if (Ir.Numerics.get ()).Ir.Numerics.tf32_matmuls then 1e-2 else 1e-3
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

let parity ?tol ?(perturb = false) ?(select = conv_seed) ?(on_result = fun _ -> ()) ctx comp
    bindings ~bind ~static_indices opt =
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
  let reference =
    {
      ref_compared = List.length plain;
      ref_restored = List.length snapshot;
      ref_poisoned =
        List.count (written_only rroutine) ~f:(fun tn ->
            not (List.mem unsafe tn ~equal:Tn.equal || not (host_accessible tn)));
      ref_poison_unsafe = List.map unsafe ~f:Tn.debug_name;
    }
  in
  (* The reference's pools are freed before any candidate compiles: a routine that fits the device
     alone must not decline for sharing it with the reference. *)
  Context.release rctx;
  let seeding =
    Autotune.search_seeding
      ~census:(fun ~where:_ _ -> ())
      ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices opt
  in
  let segments =
    List.map seeding.ss_coarse ~f:(fun e -> (false, seeding.ss_coarse_plan, e))
    @ List.map seeding.ss_fine_new ~f:(fun e -> (true, seeding.ss_fine_plan, e))
  in
  let one ~fine ~plan key p =
    let label = Autotune.fiss_sketch_label ~fine [ (key, p) ] in
    let seeded = ref [] and perturbed = ref true in
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
      List.mapi tuples ~f:(fun i (_, _, _, (post : LL.optimized)) ->
          if perturb && List.mem !seeded i ~equal:Int.equal then (
            if not (has_set post.llc) then perturbed := false;
            { post with llc = perturb_code post.llc })
          else post)
    in
    let result verdict ~tensorized ~compared = { label; verdict; tensorized; compared } in
    match
      Context.compile_outcome ~lowered_transform:transform ~provenance:Ir.Schedule_outcome.Candidate
        ~candidate:label ctx comp bindings
    with
    | Error (Classified c) ->
        result
          (Declined (Ir.Schedule_outcome.detail_of_cause c.Ir.Schedule_outcome.cause))
          ~tensorized:false ~compared:0
    | Error (Fatal f) -> result (Failed (Exn.to_string f.exn)) ~tensorized:false ~compared:0
    | Ok (cctx, routine) ->
        let tensorized =
          Ir.C_syntax.equal_tensorization routine.Context.mma.Ir.C_syntax.tensorization
            Ir.C_syntax.Tensorized
        in
        let outcome =
          if List.is_empty !seeded then result Not_applied ~tensorized ~compared:0
          else if not !perturbed then result Not_perturbed ~tensorized ~compared:0
          else
            match
              bind routine;
              run ~unsafe ~poisoned:true cctx routine
            with
            | candidate ->
                let err, node, compared = worst_difference ~reference:plain ~candidate in
                if compared = 0 then
                  result (Failed "no node both routines write was compared") ~tensorized ~compared
                else
                  let verdict =
                    if Float.(err <= tol) then Match { err; node } else Mismatch { err; node }
                  in
                  result verdict ~tensorized ~compared
            | exception exn -> result (Failed (Exn.to_string exn)) ~tensorized ~compared:0
        in
        Context.release cctx;
        outcome
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
  (reference, results)

(* Whether a seed's result passes: a declined seed is the search's business, not a parity defect.
   Under [perturb] a seed passes when its perturbation was detected. *)
let passed ~perturb r =
  match (perturb, r.verdict) with
  | false, (Match _ | Declined _) -> true
  | true, (Mismatch _ | Declined _) -> true
  | false, (Mismatch _ | Not_applied | Not_perturbed | Failed _)
  | true, (Match _ | Not_applied | Not_perturbed | Failed _) ->
      false

let verdict_line r =
  let v =
    match r.verdict with
    | Match { err; node } -> Printf.sprintf "MATCH worst %.3g at %s" err node
    | Mismatch { err; node } -> Printf.sprintf "MISMATCH worst %.3g at %s" err node
    | Declined d -> "DECLINED: " ^ d
    | Not_applied -> "NOT APPLIED: the seed never reached its segment"
    | Not_perturbed -> "NOT PERTURBED: the seeded segment has no Set"
    | Failed d -> "FAILED: " ^ d
  in
  Printf.sprintf "%s: %s (compared %d nodes%s)" r.label v r.compared
    (if r.tensorized then ", tensorized" else "")
