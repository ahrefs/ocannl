(* gh-ocannl-1982: a padded convolution on a leaf built by [init], whose array nobody else holds,
   gets its margins the way a [range_of_shape] leaf does. The leaf used to commit an unpadded layout
   at creation, so the convolution was refused at lowering, in a program that had compiled nothing,
   by a message advising it to compose the consumer before the operand's first compilation.

   The two bitwise comparisons rest on a premise derived from the operands themselves, beside the
   cases that make them; the values discriminate: no two cells of [x] are equal, and every edge
   window reaches a margin with a nonzero kernel weight. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Asgns = Ir.Assignments

let n = 5
let kernel_vals = [| 1.; 2.; 4. |]
let x_vals = Array.init n ~f:(fun i -> Float.of_int ((3 * i) + 1))
let w_vals = [| 1.; 3.; 5.; 7.; 9. |]
let flat f idcs = f idcs.(Array.length idcs - 1)
let at vals i = vals.(i)

let init_leaf ?grad_spec ~l ~o vals =
  let grad_spec = Option.value grad_spec ~default:Tensor.Prohibit_grad in
  Operation.init ~grad_spec ~l ~prec:Ir.Ops.single ~o ~f:(flat (at vals)) ()

(* The kernel of 3 sits at [left = 1]: [out[o] = sum_k in[o + k - 1] * kernel[k]], zero outside. *)
let left = 1

let conv input =
  Array.init (Array.length input) ~f:(fun o ->
      Array.foldi kernel_vals ~init:0. ~f:(fun k acc kv ->
          let c = o + k - left in
          if c >= 0 && c < Array.length input then acc +. (input.(c) *. kv) else acc))

(* The input gradient of [sum (conv x *. w)]. *)
let conv_input_grad w =
  Array.init n ~f:(fun c ->
      Array.foldi w ~init:0. ~f:(fun o acc wo ->
          Array.foldi kernel_vals ~init:acc ~f:(fun k acc kv ->
              if o + k - left = c then acc +. (wo *. kv) else acc)))

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(* [i] where it is a cell of an [n]-cell operand, else the zero cell [Exact_sums.with_zero]
   appends. *)
let inside i = if i >= 0 && i < n then i else n

let () =
  (* The bitwise premise, from the operands' own data: the convolution's cells sum [x.(o + k - left)
     * kernel.(k)] over the kernel, and its input gradient's sum [w.(o) * kernel.(k)] over the taps
     [(o, k)] reading the cell. Every product is a small positive integer, so every partial sum of
     every cell is exact in single precision, in any order; each case below compares these host sums
     with the values it executed. *)
  let x_r = Exact_sums.with_zero x_vals
  and w_r = Exact_sums.with_zero w_vals
  and k_r = Exact_sums.with_zero kernel_vals in
  let forward =
    Ll_test.reduction_cells ~out:[| 1; n |] ~over:[| 3 |] (fun o r ->
        x_r [| inside (o.(1) + r.(0) - left) |] *. k_r [| r.(0) |])
  in
  let gradient =
    Ll_test.reduction_cells ~out:[| 1; n |] ~over:[| n; 3 |] (fun o r ->
        w_r [| (if r.(0) + r.(1) - left = o.(1) then r.(0) else n) |] *. k_r [| r.(1) |])
  in
  let sig_bits = Exact_sums.sig_bits in
  p_all "every partial sum of every convolution cell is exact in f32" forward
    ~f:(Ll_test.exact_partials ~sig_bits);
  p_all "every partial sum of every input-gradient cell is exact in f32" gradient
    ~f:(Ll_test.exact_partials ~sig_bits);
  Verdict.case "a padded convolution on an init leaf, in a fresh session" (fun () ->
      Tensor.unsafe_reinitialize ();
      let x = init_leaf ~l:"x" ~o:[ 1; n ] x_vals in
      let kernel = init_leaf ~l:"kernel" ~o:[ 3 ] kernel_vals in
      let%op y = x +* "s, o=+k; k => s, o" kernel in
      Train.set_materialized x.Tensor.value;
      Train.set_materialized y.Tensor.value;
      let ctx = Train.forward_once (Context.auto ()) y in
      Exact_sums.check_sums "the convolution's host sums are its executed values" forward
        (Context.get_values ctx y.Tensor.value);
      p "the leaf took margins from its consumer"
        (match Ir.Tnode.get_padding x.Tensor.value with
        | Some (pads, value) -> pads.(1).Ir.Ops.left = left && Float.equal value 0.
        | None -> false);
      p "the padded init leaf is host-constant" (Ir.Tnode.known_host_constant x.Tensor.value);
      pf_all2 "the convolution is the zero-padded correlation of the leaf's data"
        (Context.get_values ctx y.Tensor.value)
        (conv x_vals) ~f:Float.equal;
      pf_all2 "the leaf reads back as its data, margins not included"
        (Context.get_values ctx x.Tensor.value)
        x_vals ~f:Float.equal;
      Context.release ctx);
  Verdict.case "a training step through a padded convolution on an init leaf" (fun () ->
      Tensor.unsafe_reinitialize ();
      let x = init_leaf ~grad_spec:Tensor.Require_grad ~l:"x" ~o:[ 1; n ] x_vals in
      let kernel = init_leaf ~l:"kernel" ~o:[ 3 ] kernel_vals in
      let w = init_leaf ~l:"w" ~o:[ 1; n ] w_vals in
      let%op y = x +* "s, o=+k; k => s, o" kernel in
      let%op loss = (y *. w) ++ "...|... => |->0" in
      let gx = grad_of x in
      Train.set_materialized gx;
      let ctx = Train.update_once (Context.auto ()) loss in
      Exact_sums.check_sums "the input gradient's host sums are its executed values" gradient
        (Context.get_values ctx gx);
      pf_all2 "the leaf's gradient is the transposed convolution of w" (Context.get_values ctx gx)
        (conv_input_grad w_vals) ~f:Float.equal;
      Context.release ctx);
  (* The refusals are printed whole (gh-ocannl-2041): each names the operand it refuses and gives
     only the remedies for how that operand's layout was committed. *)
  Verdict.case "a [wrap] leaf keeps the array it was given, so the same convolution is refused"
    (fun () ->
      Tensor.unsafe_reinitialize ();
      let wrapped ~l ~o vals =
        let data =
          Ir.Ndarray.init_array ~debug:l Ir.Ops.single ~dims:(Array.of_list o) ~padding:None
            ~f:(flat (at vals))
        in
        TDSL.wrap ~l ~b:[] ~o data ()
      in
      (* Both operands keep their arrays; only [x] is read through the padded window. *)
      let x = wrapped ~l:"x" ~o:[ 1; n ] x_vals in
      let kernel = wrapped ~l:"kernel" ~o:[ 3 ] kernel_vals in
      let%op y = x +* "s, o=+k; k => s, o" kernel in
      match Train.forward_once (Context.auto ()) y with
      | exception Row.Shape_error (msg, _) -> Stdio.printf "refusal: %s\n" msg
      | (_ : Context.t) -> p "the convolution was refused" false);
  Verdict.case "an init leaf an earlier routine compiled unpadded refuses the convolution"
    (fun () ->
      Tensor.unsafe_reinitialize ();
      let x = init_leaf ~l:"x" ~o:[ 1; n ] x_vals in
      let kernel = init_leaf ~l:"kernel" ~o:[ 3 ] kernel_vals in
      let%op doubled = x *. !.2. in
      let ctx = Train.forward_once (Context.auto ()) doubled in
      let%op y = x +* "s, o=+k; k => s, o" kernel in
      match Train.forward_once ctx y with
      | exception Row.Shape_error (msg, _) -> Stdio.printf "refusal: %s\n" msg
      | (_ : Context.t) -> p "the convolution was refused" false)
