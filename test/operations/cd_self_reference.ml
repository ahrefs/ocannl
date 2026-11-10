(* gh-ocannl-2056: a [%cd] inline declaration whose right-hand side reads the declared tensor
   itself, [{ x } =: x + 1].

   The declared tensor's label is enriched from one right-hand-side tensor (the label hint), and the
   declaration is a non-recursive [let x = NTDSL.term ~label:("x" :: <hint>) ()] around the
   assignment. Where the only punnable operand was [x] itself, the hint read [x] in its own binding
   and the expansion failed with "Unbound value x"; where [x] was the preferred operand of two, as
   in [{ y } =: y + t], it did too, though [t] could have given the label. The declared name is now
   never its own hint, so self-reference compiles, executes as the update it reads as, and labels
   from the other operand when there is one. Each case runs its routine twice: the second run reads
   what the first wrote, so the values discriminate a read of the declared tensor from a read of
   anything else. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode

(* The declaration's scope is the extension point, so the declared node is found as the routine's
   one output. *)
let run_twice ~name comp =
  let ctx, routine = Context.compile ~name (Context.auto ()) comp Ir.Indexing.Empty in
  let declared = Set.choose_exn routine.outputs in
  let ctx = Context.run ctx routine in
  let first = Array.to_list (Context.get_values ctx declared) in
  let ctx = Context.run ctx routine in
  let second = Array.to_list (Context.get_values ctx declared) in
  Context.release ctx;
  Stdio.printf "%s: label %s, %d output(s)\n" name
    (String.concat ~sep:"," declared.Tn.label)
    (Set.length routine.outputs);
  (declared, first, second)

let () =
  Tensor.unsafe_reinitialize ();
  (* The only right-hand tensor is the declared one: no hint at all. *)
  let declared, first, second =
    run_twice ~name:"csr_ctr" [%cd { csr_ctr; o = [ 2 ]; zero_valid = true } =: csr_ctr + 1]
  in
  p "{ x } =: x + 1 labels x by its name alone"
    (List.equal String.equal declared.label [ "csr_ctr" ]);
  p_all "{ x } =: x + 1 starts from the declared zero" first ~f:(Float.equal 1.);
  p_all "and a second run reads what the first wrote" second ~f:(Float.equal 2.)

let csr_t = Tensor.term_init [| 3.; 5. |] ~grad_spec:Tensor.Prohibit_grad ~label:[ "csr_t" ] ()

let () =
  (* The declared tensor is the preferred hint of two: the other operand labels it. *)
  let declared, first, second =
    run_twice ~name:"csr_acc" [%cd { csr_acc; zero_valid = true } =: csr_acc + csr_t]
  in
  p "{ y } =: y + t takes its label hint from t"
    (List.equal String.equal declared.label ("csr_acc" :: csr_t.value.label));
  p "{ y } =: y + t computes t on the first run" (List.equal Float.equal first [ 3.; 5. ]);
  p "and y + t on the second" (List.equal Float.equal second [ 6.; 10. ])

let () =
  (* Self-reference through the value accessor, the only right-hand tensor again. *)
  let declared, first, second =
    run_twice ~name:"csr_node"
      [%cd { csr_node; o = [ 2 ]; zero_valid = true } =: csr_node.value + 2]
  in
  p "{ z } =: z.value + 2 labels z by its name alone"
    (List.equal String.equal declared.label [ "csr_node" ]);
  p_all "{ z } =: z.value + 2 starts from the declared zero" first ~f:(Float.equal 2.);
  p_all "and a second run reads what the first wrote" second ~f:(Float.equal 4.)
