(* gh-ocannl-498 rematerialization: the budget-driven recompute-vs-store planner. The contract is in
   memory_budget.mli; this file holds the reasoning behind each step of the selection. *)

open Base
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Backends = Context.Backends

type t = Bytes of int | Minimize [@@deriving sexp_of]

type plan = {
  bp_baseline : LL.footprint;
  bp_final : LL.footprint;
  bp_flips : (Tn.t * int * int) list;
  bp_considered : int;
  bp_dropped : int;
  bp_within_budget : bool;
}
[@@deriving sexp_of]

(* gh-ocannl-498: compare the rationals [ra/ca] and [rb/cb] EXACTLY, for ranking candidates by
   footprint relief per unit of recompute cost. [ca] and [cb] must be positive; the numerators are
   byte counts and may be negative, since inlining a node can lengthen other nodes' spans and cost
   footprint rather than free it.

   Cross-multiplying would be the obvious comparison and is wrong: both factors are legitimately
   large (bytes against reduction extent x read multiplicity), so the products can wrap and silently
   invert the order. The Euclidean/continued-fraction descent uses only division and remainder, so
   it cannot overflow, and stays bit-reproducible unlike a float ratio -- but it assumes
   NON-NEGATIVE numerators: OCaml's division truncates toward zero, so a negative numerator inverts
   the very comparison the descent is making ([-1/10] would rank above [1/10], and [0/1] above
   [-1/5]). The sign is therefore settled first, and two negatives are compared by reversed
   magnitude. *)
let compare_relief_ratio ra ca rb cb =
  let rec nonneg ra ca rb cb =
    (* Both numerators non-negative here; denominators positive. *)
    let qa = ra / ca and qb = rb / cb in
    if qa <> qb then Int.compare qa qb
    else
      let ma = ra - (qa * ca) and mb = rb - (qb * cb) in
      if ma = 0 then if mb = 0 then 0 else -1
      else if mb = 0 then 1
      else (* both fractional parts nonzero: compare ca/ma with cb/mb, inverted. *)
        nonneg cb mb ca ma
  in
  match (ra >= 0, rb >= 0) with
  | true, true -> nonneg ra ca rb cb
  | true, false -> 1
  | false, true -> -1
  (* Both negative: |ra|/ca vs |rb|/cb with the order reversed. *)
  | false, false -> nonneg (-rb) cb (-ra) ca

let log_memory_budget () = Utils.get_global_flag ~default:false ~arg_name:"log_memory_budget"

(* One hermetic analysis of [comp] from [ctx]'s lineage with [inline] additionally preferred inline:
   the footprint the resulting placement vector implies, plus the decision surface it reports.
   {!Context.lowered_for_decisions} forks the lineage state, so nothing here reaches [ctx] -- and
   the gh-560 analysis cache makes every call after the first one a specialization replay. *)
let analyze_footprint ?name ~(inline : Tn.t list) ~(footprint : Tn.t list) ctx comp bindings :
    LL.footprint * LL.flip_candidate list =
  let lowered = Context.lowered_for_decisions ?name ~inline ~footprint ctx comp bindings in
  ( Backends.score_footprint ~backend_name:(Context.backend_name ctx)
      ~limits:(Context.hardware_limits ctx) ~static_indices:(Idx.bound_symbols bindings) lowered,
    lowered.LL.flip_candidates )

let footprint ?name ctx comp bindings =
  fst (analyze_footprint ?name ~inline:[] ~footprint:[] ctx comp bindings)

(* A decision is a node with its direction (gh-ocannl-616): [`Inline] recomputes at use,
   [`Footprint] confines the node to a sub-image scratch filled at the producer; both relieve
   footprint, differently — an inlined reading keeps the template's leaves live up to the late
   consumer, a footprint-scoped one only the scratch. *)
type direction = [ `Inline | `Footprint ]

let direction_of (fc : LL.flip_candidate) : direction =
  match fc.LL.fc_flip with `Footprint -> `Footprint | `Inline | `Materialize -> `Inline

let direction_name (d : direction) = match d with `Inline -> "inline" | `Footprint -> "footprint"

let split (decs : (Tn.t * direction) list) =
  let pick d = List.filter_map decs ~f:(fun (tn, d') -> Option.some_if (Poly.equal d d') tn) in
  (pick `Inline, pick `Footprint)

let fit ?name ?max_candidates ~budget ctx comp bindings =
  (* [Minimize] promises every flip that still relieves footprint, so it must not silently stop at a
     default cut -- and a config-only user (memory_budget=minimize) has no way to raise one. It
     therefore defaults to unbounded, paying two lowerings per candidate; a caller who wants that
     bounded passes [max_candidates] explicitly, which applies to both budget kinds. A byte budget
     stops as soon as it is met, so its default cut is a cost guard, not a semantic one. *)
  let max_candidates =
    match (max_candidates, budget) with
    | Some n, _ -> n
    | None, Minimize -> Int.max_value
    | None, Bytes _ -> 32
  in
  if not (Utils.get_global_flag ~default:false ~arg_name:"buffer_aliasing") then
    raise
    @@ Utils.User_error
         "Memory_budget.fit: a memory budget needs the liveness memory planner (config \
          buffer_aliasing=true) -- without it every node is always-live, the footprint score \
          degenerates to bump packing, and the relief of demoting an intermediate is unrelated to \
          what the allocator would do"
  else begin
    let logf fmt =
      Stdlib.Printf.ksprintf
        (fun s -> if log_memory_budget () then Stdio.eprintf "memory budget: %s\n%!" s)
        fmt
    in
    let score (decs : (Tn.t * direction) list) =
      let inline, footprint = split decs in
      fst (analyze_footprint ?name ~inline ~footprint ctx comp bindings)
    in
    let bp_baseline, surface = analyze_footprint ?name ~inline:[] ~footprint:[] ctx comp bindings in
    (* The acceptance-stopping predicate: [Minimize] is never satisfied, so it keeps taking flips
       that still help. [within] is the reported outcome, where a target-less [Minimize] trivially
       holds -- there is no budget for it to miss. *)
    let met (fp : LL.footprint) =
      match budget with Minimize -> false | Bytes b -> fp.LL.fp_total <= b
    in
    let within (fp : LL.footprint) =
      match budget with Minimize -> true | Bytes b -> fp.LL.fp_total <= b
    in
    let done_ () =
      {
        bp_baseline;
        bp_final = bp_baseline;
        bp_flips = [];
        bp_considered = 0;
        bp_dropped = 0;
        bp_within_budget = within bp_baseline;
      }
    in
    if met bp_baseline then (
      logf "baseline %d bytes is already within budget; no flips" bp_baseline.LL.fp_total;
      (ctx, done_ ()))
    else
      (* The [`Inline] and [`Footprint] directions (gh-ocannl-616): demoting a materialized
         intermediate to recompute-at-use relieves footprint, and so does confining it to a
         sub-image scratch; a node carrying both records is scored in each direction, and the first
         that pays takes the node (its other direction is then skipped). Ranked
         CHEAPEST-recompute-first for the pre-filter (the surface's own order is
         most-expensive-first, which the [Materialize]-direction search wants), so a
         [max_candidates] cut keeps the flips a budget would most want to pay for. *)
      let all =
        List.fold surface ~init:[] ~f:(fun acc fc ->
            match fc.LL.fc_flip with
            | `Materialize -> acc
            | `Inline | `Footprint ->
                if
                  List.exists acc ~f:(fun c ->
                      Tn.equal c.LL.fc_tn fc.LL.fc_tn && Poly.equal c.LL.fc_flip fc.LL.fc_flip)
                then acc
                else fc :: acc)
        |> List.sort ~compare:(fun a b ->
            match Int.compare a.LL.fc_recompute_cost b.LL.fc_recompute_cost with
            | 0 -> Tn.compare a.LL.fc_tn b.LL.fc_tn
            | c -> c)
      in
      let considered = List.take all max_candidates in
      let bp_dropped = List.length all - List.length considered in
      if bp_dropped > 0 then
        logf "%d of %d inline candidates dropped by max_candidates=%d (cheapest recompute kept)"
          bp_dropped (List.length all) max_candidates;
      (* Round 1: each candidate's relief against the ACTUAL baseline layout. A node whose span was
         already shared relieves nothing on its own (the gh-ocannl-558 lesson in reverse: relief is
         not a function of the node's own size). Solo relief only RANKS here -- a zero-relief
         candidate is kept, at the back, because relief is not additive in either direction: two
         nodes pinning the same arena peak each free nothing alone and the whole range together, so
         dropping them outright would report an otherwise reachable budget unreachable. Round 2
         picks those up jointly. *)
      let scored =
        List.map considered ~f:(fun fc ->
            let fp = score [ (fc.LL.fc_tn, direction_of fc) ] in
            let relief = bp_baseline.LL.fp_total - fp.LL.fp_total in
            logf "candidate %s (%s): recompute cost %d, solo relief %d bytes"
              (Tn.debug_name fc.LL.fc_tn)
              (direction_name (direction_of fc))
              fc.LL.fc_recompute_cost relief;
            (fc, relief))
      in
      let ranked =
        List.sort scored ~compare:(fun (a, ra) (b, rb) ->
            let ca = max 1 a.LL.fc_recompute_cost and cb = max 1 b.LL.fc_recompute_cost in
            (* Descending by ratio, so [b] against [a]. *)
            match compare_relief_ratio rb cb ra ca with
            | 0 -> ( match Int.compare rb ra with 0 -> Tn.compare a.LL.fc_tn b.LL.fc_tn | c -> c)
            | c -> c)
      in
      (* Round 2: accept a prefix, re-scoring the CUMULATIVE vector each time. Inlining one node
         moves the others' live spans, so a candidate's solo relief is not what it is worth here. A
         candidate that adds nothing is held SPECULATIVELY rather than dropped: if a later one then
         relieves bytes on top of it, the whole speculative group is committed together (the
         two-nodes-at-one-peak case).

         Every candidate is therefore scored BOTH ways, with and without the held group, and the
         three outcomes are treated differently. Held flips are not merely unpaid, they can be
         actively HARMFUL — a flip whose marginal was negative moved someone's span the wrong way —
         and judging every later candidate only in their company would let one bad hold mask a
         candidate that pays on its own, losing it and, with it, a reachable budget.

         - joint strictly better: the group is load-bearing, so commit it with the candidate. -
         joint strictly worse: the group is harmful here, so commit the candidate alone and DISCARD
         the group (no group is reconsidered once discarded — this is a bounded planner, not a
         search over subsets). - equal: the group is merely neutral. Commit the candidate alone but
         KEEP holding it: committing it would pay recompute for zero bytes, and discarding it would
         throw away a flip that may still be half of a later pair. Dropping neutral holds eagerly
         measurably costs relief (on test/operations/memory_budget_planner's step, 1196164 ->
         1228932 bytes).

         Speculatives never joined by a paying flip are discarded at the end, so no recompute is
         ever paid for zero bytes. The relief of a joint commit is reported on the flip that closed
         it, and the sum over [bp_flips] is exactly [bp_baseline - bp_final]. *)
      let accepted = ref [] and flips = ref [] and cur = ref bp_baseline in
      (* Held (node, recompute cost) pairs, most recently held first. *)
      let speculative = ref [] in
      let names l =
        String.concat ~sep:", " (List.map l ~f:(fun ((tn, _), _) -> Tn.debug_name tn))
      in
      List.iter ranked ~f:(fun (fc, solo) ->
          let tn = fc.LL.fc_tn and cost = fc.LL.fc_recompute_cost in
          let dec = (tn, direction_of fc) in
          let taken =
            List.exists (List.map !speculative ~f:fst @ !accepted) ~f:(fun (t, _) -> Tn.equal t tn)
          in
          if taken then
            logf "skip %s (%s): the node's other direction is accepted or held" (Tn.debug_name tn)
              (direction_name (snd dec))
          else if not (met !cur) then begin
            let held = !speculative in
            let cand_alone = dec :: !accepted in
            let fp_alone = score cand_alone in
            let cand_joint, fp_joint =
              if List.is_empty held then (cand_alone, fp_alone)
              else
                let c = (dec :: List.map held ~f:fst) @ !accepted in
                (c, score c)
            in
            let verdict =
              match compare_int fp_joint.LL.fp_total fp_alone.LL.fp_total with
              | c when c < 0 -> `Load_bearing
              | 0 -> `Neutral
              | _ -> `Harmful
            in
            let cand = match verdict with `Load_bearing -> cand_joint | _ -> cand_alone in
            let fp = match verdict with `Load_bearing -> fp_joint | _ -> fp_alone in
            let marginal = !cur.LL.fp_total - fp.LL.fp_total in
            if marginal > 0 then (
              logf "accept %s (%s): %d bytes (solo %d), cost %d%s, footprint now %d"
                (Tn.debug_name tn)
                (direction_name (snd dec))
                marginal solo cost
                (match (verdict, held) with
                | _, [] -> ""
                | `Load_bearing, _ -> Printf.sprintf " jointly with %s" (names held)
                | `Neutral, _ -> Printf.sprintf " alone, still holding %s" (names held)
                | `Harmful, _ -> Printf.sprintf " alone, dropping harmful held %s" (names held))
                fp.LL.fp_total;
              (* [flips] is reverse-chronological until the final [List.rev]. A joint commit's held
                 flips carry 0 and the group's relief lands on the flip that made it pay. *)
              flips :=
                (tn, marginal, cost)
                ::
                (match verdict with
                | `Load_bearing -> List.map held ~f:(fun ((h, _), c) -> (h, 0, c))
                | `Neutral | `Harmful -> [])
                @ !flips;
              accepted := cand;
              (* A neutral group stays held: committing it would pay recompute for zero bytes, and
                 dropping it would discard a flip that may still be half of a later pair. *)
              (speculative := match verdict with `Neutral -> held | _ -> []);
              cur := fp)
            else (
              logf "hold %s: no marginal relief yet (solo was %d); speculative" (Tn.debug_name tn)
                solo;
              speculative := (dec, cost) :: !speculative)
          end);
      (match !speculative with
      | [] -> ()
      | held ->
          logf "dropping %d speculative flip(s) that never paid: %s" (List.length held) (names held));
      let bp_final = !cur in
      let bp_within_budget = within bp_final in
      (match budget with
      | Minimize ->
          logf "minimized: %d -> %d bytes with %d flip(s)" bp_baseline.LL.fp_total
            bp_final.LL.fp_total (List.length !flips)
      | Bytes b ->
          logf "budget %d bytes: %d -> %d bytes with %d flip(s), %s" b bp_baseline.LL.fp_total
            bp_final.LL.fp_total (List.length !flips)
            (if bp_within_budget then "within budget" else "STILL OVER BUDGET"));
      let inline, footprint = split !accepted in
      let ctx = if List.is_empty inline then ctx else Context.decide_inline ctx inline in
      let ctx = if List.is_empty footprint then ctx else Context.decide_footprint ctx footprint in
      ( ctx,
        {
          bp_baseline;
          bp_final;
          bp_flips = List.rev !flips;
          bp_considered = List.length considered;
          bp_dropped;
          bp_within_budget;
        } )
  end
