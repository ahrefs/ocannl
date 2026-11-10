open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Row = Ocannl.Row

(* gh-ocannl-2036: one tensor in both operand slots of [...|h,w,..c..; ...|..c.. => ...|h]. Einsum
   equates each operand with its slot, without broadcasting, so the tensor's output row would be
   both [h; w].<c> and <c>: no finite row is both, and the occurs check refuses it. Its message says
   why and names the program meant instead, a contraction with the tensor's own broadcast, which the
   last case executes. The row-level pins, the satisfiable inequality twin included, are in
   test/einsum/test_row_self_reference.ml. *)

let spec_refusal build =
  Tensor.unsafe_reinitialize ();
  match Train.forward_once (Context.auto ()) (build ()) with
  | _ -> None
  | exception Row.Shape_error (msg, _) -> Some msg

let by_self_reference =
  Option.value_map ~default:false
    ~f:(String.is_prefix ~prefix:"Infinite number of axes by self-reference")

let () =
  Verdict.case "same tensor in both slots" @@ fun () ->
  let msg =
    spec_refusal (fun () ->
        let st = TDSL.uniform () () in
        let%op y = st +* "...|h,w,..c..; ...|..c.. => ...|h" st in
        y)
  in
  p "the same tensor in both slots is refused by the occurs check" (by_self_reference msg);
  p "the refusal says einsum does not broadcast its operands"
    (Option.exists msg ~f:(String.is_substring ~substring:"does not broadcast"))

(* Broadcasting the narrow slot does not help: the [..c..] operand would need more axes than [st],
   the result it broadcasts from. *)
let () =
  Verdict.case "broadcast into the narrow slot" @@ fun () ->
  p "a pointwise result in the narrow slot is refused by the occurs check"
    (by_self_reference
       (spec_refusal (fun () ->
            let st = TDSL.uniform () () in
            let%op y = st +* "...|h,w,..c..; ...|..c.. => ...|h" (st *. !.2.) in
            y)))

(* The hinted program: [st *. x] broadcasts [st] over [x]'s [3; 4; 5], so [c = [5]], and [y] sums
   [st[c] * x[h, w, c] * st[c]] over [w] and [c]. *)
let () =
  Verdict.case "contraction with the tensor's own broadcast" @@ fun () ->
  Tensor.unsafe_reinitialize ();
  let st = TDSL.uniform () () in
  let x = TDSL.range_of_shape ~label:[ "x" ] ~output_dims:[ 3; 4; 5 ] () in
  let%op y = (st *. x) +* "...|h,w,..c..; ...|..c.. => ...|h" st in
  Train.set_materialized st.Tensor.value;
  let ctx = Context.auto () in
  Stdio.eprintf "(not part of the golden) backend: %s\n%!" (Context.backend_name ctx);
  let ctx = Train.forward_once ctx y in
  let dims t = Array.to_list (Shape.to_dims t.Tensor.shape) in
  p "the statistic takes x's last axis: [5]" (List.equal Int.equal (dims st) [ 5 ]);
  p "the result is over h: [3]" (List.equal Int.equal (dims y) [ 3 ]);
  let s = Context.get_values ctx st.Tensor.value and got = Context.get_values ctx y.Tensor.value in
  p "the statistic's draw is distinct per cell"
    (List.length (List.dedup_and_sort ~compare:Float.compare (Array.to_list s)) = 5);
  let want =
    Array.init 3 ~f:(fun h ->
        let acc = ref 0. in
        for w = 0 to 3 do
          for c = 0 to 4 do
            acc := !acc +. (s.(c) *. Float.of_int ((h * 20) + (w * 5) + c) *. s.(c))
          done
        done;
        !acc)
  in
  p_close "y against the host contraction" got want ~tol:(fun w ->
      1e-4 *. Float.max 1. (Float.abs w))
