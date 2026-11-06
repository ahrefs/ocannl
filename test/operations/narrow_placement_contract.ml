(* gh-ocannl-1965: what placement may change at a narrow precision. The contract is in
   docs/precision_inference.md: a store rounds, a narrowing read rounds, and virtual narrow
   arithmetic, a virtual narrow constant and a virtual accumulation have no store, so they stay at
   the precision they render at. So placement decides a value exactly where that precision is wider
   than the node's storage (gh-ocannl-517's policy). This pins both directions, on every backend and
   under each narrow-compute policy: the values agree where the precisions do, and where they do not
   the virtual form holds exactly the unrounded value.

   Whether a precision is wider is the backend's own answer ([Context.codegen_capabilities]), the
   same functions code generation reads, so a claim here predicts from the renderer's precisions and
   checks the executed value against them; nothing restates a per-backend table.

   Every input lies on the narrow grid and every expectation is exact by construction, so the claims
   are equalities and their digits go to stderr. With [u] the narrow ulp at 1 and [e = 3/8 u], [1 +
   e] rounds to [1], and [3 (1 + e)] rounds up to [3 + 2 u], the grid point above 3: that is what
   carries an omitted rounding past the store of a slot of the node's own precision, where a single
   factor would be rounded away. All three fit f32 exactly. *)

open! Base
open Ocannl.Operation.DSL_modules
module Train = Ocannl.Train
module Tn = Ir.Tnode
module Ops = Ir.Ops
open Verdict.Claims

type form = Virtual | Materialized

let forms = [ ("materialized", Materialized); ("virtual", Virtual) ]

let place form (t : Tensor.t) =
  match form with
  | Virtual -> Train.set_virtual t.value
  | Materialized -> Train.set_materialized t.value

(* The placement the compile resolved, from the context that ran it: a declared intent alone would
   let a case compare a form with itself. *)
let placed ~label ctx (t : Tensor.t) form =
  let plc = Context.placements ctx in
  Stdio.eprintf "%s: node placed %s (not part of the golden)\n" label
    (Tn.debug_memory_mode (Tn.Placements.get plc t.value));
  match form with
  | Virtual -> Tn.Placements.known_virtual plc t.value
  | Materialized -> Tn.Placements.known_non_virtual plc t.value

let ulp prec = 2. **. Float.of_int (1 - Ops.significand_bits prec)
let excess prec = 3. /. 8. *. ulp prec
let leaf ~prec l v = NTDSL.init ~l ~prec ~o:[ 1 ] ~f:(fun _ -> v) ()

let forward y ~out_prec =
  Tn.update_prec y.Tensor.value out_prec;
  Train.set_materialized y.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) y in
  ((Context.get_values ctx y.Tensor.value).(0), ctx)

(* [node] makes the node under test, which [read] consumes. The predictions are the value of each
   form, already resolved from the backend's precisions by the caller. *)
let check name ~node ~read ~materialized ~virtual_ =
  List.iter forms ~f:(fun (where, form) ->
      let label = name ^ ", " ^ where in
      Tensor.unsafe_reinitialize ();
      let t = node () in
      place form t;
      let v, ctx = read t in
      let expected = match form with Virtual -> virtual_ | Materialized -> materialized in
      Stdio.eprintf "%s: reads %h, predicted %h (not part of the golden)\n" label v expected;
      pf "%s: placed as asked" label (placed ~label ctx t form);
      pf "%s: gives what the precisions predict" label Float.(v = expected))

(* The two slots a read can sit in. The first shows the node's value directly. The second is a store
   at the node's own precision, whose rounding would hide a single factor. *)
let in_slots name ~prec ~node ~wider ~rounded ~unrounded =
  let u = ulp prec in
  pf "%s: the rounded and the unrounded value differ" name Float.(rounded <> unrounded);
  check (name ^ ", read in an f32 slot") ~node
    ~read:(fun t ->
      let one = leaf ~prec:Ops.single "one" 1. in
      let%op y = t *. one in
      forward y ~out_prec:Ops.single)
    ~materialized:rounded
    ~virtual_:(if wider then unrounded else rounded);
  check
    (name ^ ", read in a slot of its own precision")
    ~node
    ~read:(fun t ->
      let three = leaf ~prec "three" 3. in
      let%op y = t *. three in
      forward y ~out_prec:prec)
    ~materialized:3.
    ~virtual_:(if wider then 3. +. (2. *. u) else 3.)

let sum_of ~prec ~a ~e =
  let%op y = a + e in
  Tn.update_prec y.Tensor.value prec;
  y

let arithmetic prec ~wider =
  let name = Ops.prec_string prec in
  let e = excess prec in
  in_slots (name ^ " arithmetic") ~prec ~wider ~rounded:1. ~unrounded:(1. +. e) ~node:(fun () ->
      sum_of ~prec ~a:(leaf ~prec "a" 1.) ~e:(leaf ~prec "e" e))

(* A literal has no operator, and is no read of stored data either: its value is off the grid. *)
let constant prec ~wider =
  let name = Ops.prec_string prec in
  let c = 1. +. excess prec in
  in_slots (name ^ " constant") ~prec ~wider ~rounded:1. ~unrounded:c ~node:(fun () ->
      let t = NTDSL.number c in
      Tn.update_prec t.Tensor.value prec;
      t)

(* A cast to the precision its argument already has is the identity: nothing narrows, so it is no
   way to ask for a rounding. It behaves as any other node, here over a virtual argument. *)
let same_precision_cast prec ~wider =
  let name = Ops.prec_string prec in
  let e = excess prec in
  in_slots (name ^ " cast of virtual arithmetic to its own precision")
    ~prec ~wider ~rounded:1. ~unrounded:(1. +. e) ~node:(fun () ->
      let y = sum_of ~prec ~a:(leaf ~prec "a" 1.) ~e:(leaf ~prec "e" e) in
      Train.set_virtual y.Tensor.value;
      let t = Ocannl.Operation.cast y () in
      Tn.update_prec t.Tensor.value prec;
      t)

(* The way to ask for one: compute wider and cast. The narrowing read rounds in either placement,
   whatever the backend computes in, though its source is virtual arithmetic. *)
let narrowing_cast prec =
  let name = Ops.prec_string prec in
  let e = excess prec in
  in_slots (name ^ " narrowing cast of virtual f32 arithmetic")
    ~prec ~wider:false ~rounded:1. ~unrounded:(1. +. e) ~node:(fun () ->
      let y =
        sum_of ~prec:Ops.single ~a:(leaf ~prec:Ops.single "a" 1.) ~e:(leaf ~prec:Ops.single "e" e)
      in
      Train.set_virtual y.Tensor.value;
      let t = Ocannl.Operation.cast y () in
      Tn.update_prec t.Tensor.value prec;
      t)

(* An accumulation resides at [accum_prec], which is not [compute_prec]: a backend can compute a
   format at storage width and accumulate it wider. [1, e, e] tells the three outcomes apart:
   rounded at every update it stays [1]; held wide it reaches [1 + 2 e = 1 + 3/4 u], which the store
   of a materialized node rounds up to [1 + u] and a virtual read in an f32 slot sees as it is. *)
let reduction prec ~accum_wider =
  let name = Ops.prec_string prec in
  let u = ulp prec and e = excess prec in
  let cells = [| 1.; e; e |] in
  pf "%s reduction: the three outcomes differ" name
    Float.(1. <> 1. +. u && 1. <> 1. +. (2. *. e) && 1. +. u <> 1. +. (2. *. e));
  check
    (name ^ " reduction, read in an f32 slot")
    ~node:(fun () ->
      let v = NTDSL.init ~l:"v" ~prec ~o:[ 2; 3 ] ~f:(fun idcs -> cells.(idcs.(1))) () in
      let%op s = v ++ "ab => a" in
      Tn.update_prec s.Tensor.value prec;
      s)
    ~read:(fun t ->
      let ones = NTDSL.init ~l:"ones" ~prec:Ops.single ~o:[ 2 ] ~f:(fun _ -> 1.) () in
      let%op y = t *. ones in
      forward y ~out_prec:Ops.single)
    ~materialized:(if accum_wider then 1. +. u else 1.)
    ~virtual_:(if accum_wider then 1. +. (2. *. e) else 1.)

let contract prec =
  let caps = Context.codegen_capabilities (Context.auto ()) in
  let wider = not (Ops.equal_prec (caps.compute_prec prec) prec) in
  let accum_wider = not (Ops.equal_prec (caps.accum_prec prec) prec) in
  Stdio.eprintf "%s: compute %s, accumulator %s (not part of the golden)\n" (Ops.prec_string prec)
    (Ops.prec_string (caps.compute_prec prec))
    (Ops.prec_string (caps.accum_prec prec));
  arithmetic prec ~wider;
  constant prec ~wider;
  same_precision_cast prec ~wider;
  narrowing_cast prec;
  reduction prec ~accum_wider

let () =
  let saved = Ir.Numerics.get () in
  let policy name policy precs =
    Ir.Numerics.set_policy policy;
    Verdict.case name (fun () -> List.iter precs ~f:contract)
  in
  let narrow = [ Ops.half; Ops.bfloat16; Ops.fp8 ] in
  policy "narrow floats computed in single" { saved with narrow_compute_f32 = true } narrow;
  policy "narrow floats computed at storage width" { saved with narrow_compute_f32 = false } narrow;
  (* Where the target's arithmetic is not genuinely 16-bit the policy is ignored, and the prediction
     follows the backend's answer. *)
  policy "f16 computed in f16 where the target can"
    { saved with narrow_compute_f32 = true; fp16_arithmetic = Fp16_narrow }
    [ Ops.half ];
  Ir.Numerics.set_policy saved
