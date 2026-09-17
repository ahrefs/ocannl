(* Regression test for gh-ocannl-599: host access to a node this lineage placed [Local] is refused,
   in both directions.

   [Local] is the unobservable placement class — routine-scoped scratch the backend may keep in
   registers or on the stack, with no context buffer any routine writes back. The docs said so
   (context.mli, "On-demand host access"), but [Context.to_host] / [from_host] did not enforce it,
   and the resulting silence is the dangerous kind: [set_values] allocates a context buffer through
   the [init_from_host] fallback, [run] computes into the routine's local storage, and [get_values]
   then hands back exactly the bytes that were uploaded. Nothing anywhere said the read was
   meaningless — so an executed-parity test seeded with a sentinel reads plausible numbers no kernel
   wrote, and passes outright whenever the seeded value happens to coincide with the expected one
   (found while giving the hand-built-IR virtualization tests executed legs, gh-ocannl-589).

   The intermediate here is placed [Local] by the ordinary pipeline, not by hand: its reduction
   extent (32) exceeds [virtualize_max_inline_reduction] (16), so the recompute-cost guard forces it
   non-virtual, and being unobservable and small it resolves to [Local] rather than [On_device].

   Three legs: the refusal itself; the positive control that the very same program reads back real
   computed values once the node is materialized (so the guard fires on the placement, not on
   everything); and the precision control that a node this lineage never decided keeps the ordinary
   "not present in context" refusal. *)

open Base
open Stdio
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* [a] is [4 x 32] with row [i] constant at [i + 1], and [x] is [0 .. 31], so [h.(i) = (i+1) * 496]:
   distinct per row, dependent on every term of the reduction, and nowhere near the seeded
   sentinel. *)
let rows = 4
let inner = 32
let a_values = Array.init (rows * inner) ~f:(fun n -> Float.of_int ((n / inner) + 1))
let x_values = Array.init inner ~f:Float.of_int
let sum_x = Float.of_int (inner * (inner - 1) / 2)
let h_expected = Array.init rows ~f:(fun i -> Float.of_int (i + 1) *. sum_x)
let sentinel = Array.create ~len:rows (-1.)

(* Non-empty by construction (gh-ocannl-746): [Array.for_all2_exn] answers [true] on two empty
   arrays, and a readback and the reference it is compared against go empty TOGETHER -- the
   reference went through the same path. {!Verdict.p_all2} is the claim-shaped form; the sites
   reached through this helper keep a boolean, so the guard lives here. *)
let close got expected =
  (not (Array.is_empty got))
  && Array.for_all2_exn got expected ~f:(fun v w -> Float.(abs (v -. w) <= 1e-3))

(* The program: [h = a * x] — the contraction whose extent trips the cost guard — consumed by an
   observable output. [~materialize_h] is the one difference between the two arms. *)
let build ~materialize_h =
  let x = TDSL.ndarray x_values ~label:[ "lha_x" ] ~output_dims:[ inner ] () in
  let a = TDSL.ndarray a_values ~label:[ "lha_a" ] ~input_dims:[ inner ] ~output_dims:[ rows ] () in
  let%op h = a * x in
  let%op y = h *. h in
  if materialize_h then Train.set_materialized h.Tensor.value;
  let ctx = Train.forward_once (Context.cpu ()) y in
  (ctx, h)

(* The cap leg's program (gh-ocannl-609 review round 1): the SAME contraction, read once, so the
   recompute-cost guard is the only thing standing between it and inlining -- [y = h *. h] above
   reads [h] twice and would trip the visit cap the moment the reduction cap stops firing, which is
   exactly the "another cap can still force materialization" the refusal now hedges about. *)
let build_single_read () =
  let x = TDSL.ndarray x_values ~label:[ "lha1_x" ] ~output_dims:[ inner ] () in
  let a =
    TDSL.ndarray a_values ~label:[ "lha1_a" ] ~input_dims:[ inner ] ~output_dims:[ rows ] ()
  in
  let%op h = a * x in
  let%op y = h *. 2 in
  let ctx = Train.forward_once (Context.cpu ()) y in
  (ctx, h)

(* [virtualize_settings] is a process-wide mutable record (the established way to exercise a cap --
   see [virtual_chain_fanin]); restore it however the body exits. *)
let with_max_inline_reduction cap f =
  let s = Ir.Low_level.virtualize_settings in
  let saved = s.Ir.Low_level.max_inline_reduction in
  s.Ir.Low_level.max_inline_reduction <- cap;
  Exn.protect ~f ~finally:(fun () -> s.Ir.Low_level.max_inline_reduction <- saved)

let refused_as_local f =
  try
    ignore (f ());
    false
  with Utils.User_error msg -> String.is_substring msg ~substring:"placed Local"

let refusal_message f =
  try
    ignore (f ());
    None
  with Utils.User_error msg -> Some msg

let () =
  (* --- The refusal --- *)
  let ctx, h = build ~materialize_h:false in
  let hv = h.Tensor.value in
  p "the pipeline placed the intermediate Local"
    (match Ir.Tnode.Placements.get (Context.placements ctx) hv with
    | Some (Ir.Tnode.Local, _) -> true
    | _ -> false);
  p "get_values on a Local node is refused" (refused_as_local (fun () -> Context.get_values ctx hv));
  (* Seeding a [Local] node is a no-op from the routine's point of view — it computes into its own
     local storage — so the upload is silently lost either way. Refusing here is also what keeps the
     context from acquiring a buffer for a node that has none, which is what made the read above
     look legitimate in the first place. *)
  p "set_values on a Local node is refused"
    (refused_as_local (fun () -> Context.set_values ctx hv sentinel));
  p "get_value on a Local node is refused"
    (refused_as_local (fun () -> Context.get_value ctx hv [| 0 |]));
  p "set_value on a Local node is refused"
    (refused_as_local (fun () -> Context.set_value ctx hv [| 0 |] 0.));

  (* gh-ocannl-609: WHY the node is Local is the only thing that distinguishes the two remedies.
     Here the recompute-cost guard forced it, so raising [virtualize_max_inline_reduction] keeps it
     virtual (and observable) — advice the message cannot give from the placement alone. Both legs
     read the same message: the tag it carries, and the option that tag admits. *)
  let msg =
    Option.value_exn ~here:[%here] (refusal_message (fun () -> Context.get_values ctx hv))
  in
  p "the refusal names the decision that placed the node Local"
    (String.is_substring msg ~substring:"39:inline-reduction-cap");
  p "the refusal offers raising that cap as the alternative to materializing"
    (String.is_substring msg ~substring:"virtualize_max_inline_reduction");

  (* And the advice is tested, not merely spelled: the same contraction read once is placed Local at
     the default cap and stays VIRTUAL with the cap raised above its extent (32). Both arms are the
     same program, so the only thing that moved is the setting the message names. *)
  let capped_ctx, capped_h = with_max_inline_reduction 16 build_single_read in
  p "the cap leg is placed Local at the default cap"
    (match Ir.Tnode.Placements.get (Context.placements capped_ctx) capped_h.Tensor.value with
    | Some (Ir.Tnode.Local, prov) ->
        String.equal (Ir.Tnode.leading_provenance prov) "39:inline-reduction-cap"
    | _ -> false);
  let raised_ctx, raised_h = with_max_inline_reduction 64 build_single_read in
  p "raising the setting the refusal names leaves that node virtual"
    (match Ir.Tnode.Placements.get (Context.placements raised_ctx) raised_h.Tensor.value with
    | Some (Ir.Tnode.Virtual, _) -> true
    | _ -> false);

  (* --- Positive control: materialized, the same program reads back what the kernel computed ---
     This is the leg that makes the refusal a placement check rather than a blanket one, and it
     supplies the oracle the Local arm cannot have: the contraction's values, not the sentinel's. *)
  let mat_ctx, mat_h = build ~materialize_h:true in
  let mat_hv = mat_h.Tensor.value in
  p "materialized: the same node is On_device"
    (match Ir.Tnode.Placements.get (Context.placements mat_ctx) mat_hv with
    | Some (Ir.Tnode.On_device, _) -> true
    | _ -> false);
  p "materialized: get_values returns the computed contraction"
    (close (Context.get_values mat_ctx mat_hv) h_expected);
  let mat_ctx = Context.set_values mat_ctx mat_hv sentinel in
  p "materialized: set_values/get_values round-trip"
    (close (Context.get_values mat_ctx mat_hv) sentinel);

  (* --- Precision control: an undecided node keeps the ordinary refusal --- [fresh] belongs to no
     routine of this lineage, so no placement was decided for it; the guard must not claim it is
     Local. *)
  let fresh = TDSL.param ~value:0. "lha_fresh" () in
  let fresh_verdict =
    try
      ignore (Context.get_values ctx fresh.Tensor.value);
      "read"
    with Utils.User_error msg ->
      if String.is_substring msg ~substring:"placed Local" then "refused as Local"
      else if String.is_substring msg ~substring:"not present in context" then "not present"
      else "other"
  in
  printf "an undecided node: %s\n" fresh_verdict
