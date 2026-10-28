(* [diff.zero_grads] is an [Assignments.comp] whose [embedded_nodes] is derived where gradients are
   minted (gh-ocannl-771): a routine built out of the zeroing code alone -- which is exactly what
   {!Train.zero_params_grads} does -- must be able to ALLOCATE the gradients it zeroes rather than
   demand them of a prior context.

   The two directions of that derivation fail very differently. A MISSING node fails loudly: linking
   the routine refuses a node that is neither embedded nor already in the context. An EXTRA node
   fails nothing at all -- the routine allocates a buffer nobody asked for, [zero_params_grads]
   quietly claims a gradient it does not zero, and every existing test stays green. So the invariant
   worth pinning is EXACTNESS, in both directions, and this test pins it at both ends of the
   recursion:

   - the base case, {!Tensor.force_param_diff}: a [Tensor.param] mints one fresh gradient and zeroes
   only it, so its [embedded_nodes] must be exactly the singleton [{ p.diff.grad }]. That equality
   is what makes {!Train.zero_params_grads} -- which sequences one parameter's [zero_grads] per
   trained parameter -- mean what its name says; - the inductive case, {!Tensor.op}: a composite
   loss sequences its operands' [zero_grads] and appends the zeroing of its own gradient, so its
   [embedded_nodes] must equal the set of nodes its zeroing [Fetch]es actually write. That set is
   DERIVED from the tree through [Asgns.collect_written] rather than written down here: a
   restatement of the expected gradients would pin this test's idea of the graph, not the
   relationship between the two fields.

   Field agreement is not by itself enough for the inductive case, because both fields come off the
   SAME tree: an operand's [zero_grads] that stopped being folded in shrinks the two together, and
   their equality survives (Codex round 1, P2) while the gradient goes unzeroed across backprop runs
   -- the "missing zero_grads" bug {!Tensor.diff} warns about. Recursion DEPTH therefore gets a
   witness derived from somewhere else: [diff.backprop], an independently built tree over the same
   graph, whose written set is the gradients backprop accumulates into. Every one of them must be
   zeroed, and [zero_grads] must zero nothing beyond them but the loss's own seed gradient. Both
   claims are relationships between two fields; no expected gradient is named anywhere below. *)

open Base
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

let diff_of t = Option.value_exn ~here:[%here] t.Tensor.diff
let grad_of t = (diff_of t).Tensor.grad
let zero_grads_of t = (diff_of t).Tensor.zero_grads
let names s = String.concat ~sep:", " (List.map (Set.to_list s) ~f:Tn.label_name)

(* Set equality of the two fields, as one predicate. The positive claims below spell it as its two
   inclusions instead (each guarded against a vacuously empty side); this is the form the negative
   control perturbs, where a single boolean over a deliberately broken [comp] is what is wanted. *)
let embeds_exactly_what_it_writes (c : Asgns.comp) =
  Set.equal c.Asgns.embedded_nodes (Asgns.collect_written c.Asgns.asgns)

(* ----- The base case: [Tensor.param] via [Tensor.force_param_diff] ----- *)

let param_case () =
  Tensor.unsafe_reinitialize ();
  let p = TDSL.param ~values:[| 1.; 2.; 3. |] "p" () in
  let z = zero_grads_of p in
  Stdio.eprintf "param zero_grads embeds: %s\n" (names z.Asgns.embedded_nodes);
  Verdict.p "a parameter's zero_grads embeds exactly its own gradient"
    (Set.equal z.Asgns.embedded_nodes (Set.singleton (module Tn) (grad_of p)));
  (* The other half of the same fact: the code really does zero that one gradient and nothing else,
     so the singleton above is not an over-claim that happens to typecheck. *)
  Verdict.p "a parameter's zero_grads writes exactly its own gradient"
    (Set.equal (Asgns.collect_written z.Asgns.asgns) (Set.singleton (module Tn) (grad_of p)))

(* ----- The inductive case: a composite loss ----- *)

let composite_case () =
  Tensor.unsafe_reinitialize ();
  let x =
    NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~o:[ 3 ]
      ~f:(function [| d |] -> Float.of_int (1 + d) | _ -> assert false)
      ()
  in
  let w = TDSL.param ~values:[| 0.5; -0.5; 2.0 |] "w" () in
  let b = TDSL.param ~values:[| 0.25; 0.0; -0.25 |] "b" () in
  let%op y = (w *. x) + b in
  let%op loss = y *. y in
  let z = zero_grads_of loss in
  let embedded = z.Asgns.embedded_nodes in
  let written = Asgns.collect_written z.Asgns.asgns in
  Stdio.eprintf "composite zero_grads embeds: %s\n" (names embedded);
  Stdio.eprintf "composite zero_grads writes: %s\n" (names written);
  (* Set equality, split into its two inclusions so that each is guarded against a collection that
     went empty: [Set.equal] of two empty sets is true, and in a golden that line is
     indistinguishable from one a real graph passed. The floor of three says this is a composite --
     a graph that collapsed to a single gradient would satisfy both inclusions honestly and still
     have stopped testing the induction. *)
  Verdict.p_all ~min:3 "every gradient the composite loss's zero_grads writes is embedded in it"
    (Set.to_list written) ~f:(Set.mem embedded);
  Verdict.p_all ~min:3 "every node the composite loss's zero_grads embeds is one it writes"
    (Set.to_list embedded) ~f:(Set.mem written);
  (* Field agreement is not the whole invariant, and on its own it cannot see the regression that
     matters most here (Codex round 1, P2): if [Tensor.op] stopped folding one operand's
     [zero_grads] into its own, [embedded] and [written] would shrink TOGETHER -- both are derived
     from the same tree -- so their equality would still hold while a gradient the backprop
     accumulates into went unzeroed across runs. That is the "missing zero_grads" bug {!Tensor.diff}
     warns about, and recursion DEPTH is a different fact from field agreement.

     It needs a witness derived from somewhere else, and [diff.backprop] is exactly that: an
     independently built tree over the same graph, whose written set is the gradients backprop
     accumulates into. The relationship between the two fields is what gets pinned -- no list of
     expected gradients appears here either. *)
  let bprop_written = Asgns.collect_written (diff_of loss).Tensor.backprop.Asgns.asgns in
  Stdio.eprintf "composite backprop writes: %s\n" (names bprop_written);
  Verdict.p_all "every gradient the composite loss's backprop accumulates into is zeroed"
    (Set.to_list bprop_written) ~f:(Set.mem written);
  (* And nothing beyond them: the loss's own gradient (the backprop seed, which backprop reads
     rather than writes) is the single documented extra, so the two fields are characterized exactly
     against each other rather than bounded from one side. *)
  Verdict.p "the composite loss's zero_grads zeroes the backprop's gradients plus its own seed"
    (Set.equal written (Set.add bprop_written (grad_of loss)));
  (* The parameters' gradients are among them: that is why a routine compiled from the loss's
     zeroing code alone can allocate what the backprop later accumulates into. The population comes
     from [loss.params] rather than from the two names bound above, which would be the same
     restatement the derived sets exist to avoid. *)
  let param_grads =
    Set.of_list (module Tn) (List.map (Set.to_list loss.Tensor.params) ~f:grad_of)
  in
  Verdict.p_all "the composite loss's zero_grads embeds each parameter's gradient"
    (Set.to_list param_grads) ~f:(Set.mem embedded);
  (* The lost-recursion-step control, which is the reason the backprop witness is here. Shrinking
     both sets by the same gradient is precisely what a dropped operand recursion does to them; the
     first claim records that field agreement survives it (so the inclusions above are genuinely
     blind to it), the second that backprop coverage does not.

     The victims must be INTERIOR gradients, and they are derived as such (Codex round 2, P2):
     [Tnode.compare] orders by monotonically assigned uid, so picking the minimum of [bprop_written]
     deterministically lands on the earliest-created node -- a PARAMETER gradient, already covered
     by the claim just above. A control that only re-drops a parameter gradient never shows what it
     is here to show: that the witness catches an omitted intermediate while both parameter
     endpoints stay present. Excluding [param_grads] leaves exactly the interior ones, and
     quantifying over all of them beats picking one -- the guard then also asserts this graph HAS an
     interior gradient to drop, which is what makes the control meaningful at all. *)
  let interior_grads = Set.to_list (Set.diff bprop_written param_grads) in
  Stdio.eprintf "composite interior gradients: %s\n" (names (Set.diff bprop_written param_grads));
  Verdict.p_all "field agreement alone cannot see any lost interior recursion step" interior_grads
    ~f:(fun g -> Set.equal (Set.remove embedded g) (Set.remove written g));
  Verdict.p_all "backprop coverage sees every lost interior recursion step" interior_grads
    ~f:(fun g -> not (Set.for_all bprop_written ~f:(Set.mem (Set.remove written g))));
  (* The field-agreement control: both directions of that check must be able to fail on this very
     tree, or the two inclusions above could be passing for a reason that has nothing to do with the
     derivation. *)
  let stranger = x.Tensor.value in
  Verdict.p "the exactness check rejects an extra embedded node"
    (not
       (embeds_exactly_what_it_writes { z with Asgns.embedded_nodes = Set.add embedded stranger }));
  Verdict.p "the exactness check rejects a dropped embedded node"
    (not
       (embeds_exactly_what_it_writes
          { z with Asgns.embedded_nodes = Set.remove embedded (grad_of loss) }))

let () =
  param_case ();
  composite_case ()
