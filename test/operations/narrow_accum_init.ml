(* gh-ocannl-2076: an accumulator opened by a narrowing read rounds that read at its own precision,
   whichever placement it gets, on every backend and under either [narrow_compute_f32].

   [s =: src; s =+ v ++ "ab => a"] over a narrow [s] and a wider [src] lowers, virtual, to one
   scope: an init reading [src], then the updates. The updates make it an accumulator, so the scope
   resides at [accum_prec] and is not one of gh-ocannl-1956's all-read scopes pinned at storage
   precision. Its init used to render at [compute_prec], so wherever that is wider than the node
   (cc's default for half and bfloat16, every backend for fp8) a virtual [s] started from the
   unrounded source while a materialized one started from its store's rounding. The contract
   (docs/precision_inference.md, section "Storage, compute and placement") says a narrowing read
   rounds in either placement; only the accumulation after it may stay wide.

   Every expectation is exact by construction, so the claims are equalities and their digits go to
   stderr. With [u] the narrow ulp at 1, the updates add [u] then [-u]: every partial sum stays on
   the narrow grid, so the accumulation's residency cannot move the value; only the init's rounding
   can. Two sources: - an f32 [1 + 3/8 u]: rounded, [1]; unrounded it reads [1 + 3/8 u], and three
   of it rounds up to [3 + 2 u] at a store of the node's own precision, where a single factor would
   be rounded away; - an f64 just above the tie between [1] and [1 + u], close enough that f32
   rounds onto the tie (gh-ocannl-1956's [f64_master]): one step gives [1 + u], and three of it is a
   tie going to [3 + 4 u]; through f32 it would give [1] (and [3]), unrounded [1 + u/2] (and [3 + 2
   u]). Every f64 narrowing rounds once on every backend with an f64 (gh-ocannl-2075); Metal has
   none. *)

open! Base
open Ocannl.Operation.DSL_modules
module Train = Ocannl.Train
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* The init is a plain assignment of the source; the reduction accumulates onto it. *)
let%cd init_then_sum_asn ~t ~t1 ~t2 ~projections:_ =
  t =: id t1 ~logic:".";
  t =+ id t2 ~logic:"ab => a"

let no_grad2 ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp

let init_then_sum =
  Tensor.binop
    ~compose_op:(Shape.Einsum ("a; ab => a", []))
    ~op_asn:init_then_sum_asn ~grad_asn:no_grad2 ~op_label:"init_sum"
    ~grad_spec:Tensor.Prohibit_grad

type placement = Virtual | Materialized | Auto

let placements = [ ("materialized", Materialized); ("virtual", Virtual); ("auto", Auto) ]
let ulp prec = 2. **. Float.of_int (1 - Ops.significand_bits prec)

let accumulator ~prec ~src_prec ~init placement =
  let u = ulp prec in
  let src = NTDSL.init ~l:"src" ~prec:src_prec ~o:[ 2 ] ~f:(fun _ -> init) () in
  let updates = [| u; -.u |] in
  let v = NTDSL.init ~l:"v" ~prec ~o:[ 2; 2 ] ~f:(fun idcs -> updates.(idcs.(1))) () in
  let s = init_then_sum src v () in
  Tn.update_prec s.Tensor.value prec;
  (match placement with
  | Virtual -> Train.set_virtual s.Tensor.value
  | Materialized -> Train.set_materialized s.Tensor.value
  | Auto -> ());
  s

let forward y ~out_prec =
  Tn.update_prec y.Tensor.value out_prec;
  Train.set_materialized y.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) y in
  ((Context.get_values ctx y.Tensor.value).(0), ctx)

(* The placement the compile resolved, from the context that ran it: a declared intent alone would
   let a case compare a form with itself. [Auto] is the heuristics' call, either way. *)
let placed ~label ctx (s : Tensor.t) placement =
  let plc = Context.placements ctx and tn = s.Tensor.value in
  Stdio.eprintf "%s: accumulator placed %s (not part of the golden)\n" label
    (Tn.debug_memory_mode (Tn.Placements.get plc tn));
  match placement with
  | Virtual -> Tn.Placements.known_virtual plc tn
  | Materialized -> Tn.Placements.known_non_virtual plc tn
  | Auto -> Tn.Placements.known_virtual plc tn || Tn.Placements.known_non_virtual plc tn

(* The two slots a read can sit in: an f32 one shows the accumulator's value; one of its own
   precision stores the read, so the read is multiplied by three to carry an omitted rounding past
   that store. *)
let slots prec =
  [
    ( "read in an f32 slot",
      fun s ->
        let one = NTDSL.init ~l:"one" ~prec:Ops.single ~o:[ 2 ] ~f:(fun _ -> 1.) () in
        let%op y = s *. one in
        forward y ~out_prec:Ops.single );
    ( "read in a slot of its own precision",
      fun s ->
        let three = NTDSL.init ~l:"three" ~prec ~o:[ 2 ] ~f:(fun _ -> 3.) () in
        let%op y = s *. three in
        forward y ~out_prec:prec );
  ]

(* Every placement's run where [when_] holds, ahead of the claims: a leg's evaluation makes exactly
   one claim. *)
let case ~when_ name ~prec ~src_prec ~init ~expected =
  let leg = Verdict.leg ~when_ ~on:backend_name () in
  List.iter2_exn (slots prec) expected ~f:(fun (slot, read) expected ->
      let name = name ^ ", " ^ slot in
      Stdio.printf "%s: every placement gives %h\n" name expected;
      let outcomes =
        if not when_ then []
        else
          List.map placements ~f:(fun (where, placement) ->
              let label = name ^ ", " ^ where in
              Tensor.unsafe_reinitialize ();
              let s = accumulator ~prec ~src_prec ~init placement in
              let v, ctx = read s in
              Stdio.eprintf "%s: reads %h (not part of the golden)\n" label v;
              (where, (placed ~label ctx s placement, v)))
      in
      List.iter placements ~f:(fun (where, _) ->
          let label = name ^ ", " ^ where in
          let outcome () = List.Assoc.find_exn outcomes ~equal:String.equal where in
          in_legf ~leg "%s: placed as asked" label (fun claim -> p claim (fst (outcome ())));
          in_legf ~leg "%s: the init's rounding reaches the value" label (fun claim ->
              p claim Float.(snd (outcome ()) = expected))))

let accumulators prec =
  let name = Ops.prec_string prec in
  let u = ulp prec in
  let bits = Ops.significand_bits prec in
  case ~when_:true
    (name ^ " accumulator opened by an f32 read")
    ~prec ~src_prec:Ops.single
    ~init:(1. +. (3. /. 8. *. u))
    ~expected:[ 1.; 3. ];
  case
    ~when_:(Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.supports_f64
    (name ^ " accumulator opened by an f64 read")
    ~prec ~src_prec:Ops.double
    ~init:(1. +. (2. **. Float.of_int (-bits)) +. (2. **. -40.))
    ~expected:[ 1. +. u; 3. +. (4. *. u) ]

let () =
  let saved = Ir.Numerics.get () in
  let policy name policy =
    Ir.Numerics.set_policy policy;
    Verdict.case name (fun () -> List.iter [ Ops.half; Ops.bfloat16; Ops.fp8 ] ~f:accumulators)
  in
  policy "narrow floats computed in single" { saved with narrow_compute_f32 = true };
  policy "narrow floats computed at storage width" { saved with narrow_compute_f32 = false };
  Ir.Numerics.set_policy saved
