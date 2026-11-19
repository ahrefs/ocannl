(* gh-ocannl-2334: the bitwise-premise helpers the padded-window and transposed-convolution tests
   share, once. [Premise_scan] reads this source across files (its [support_sources]), so a test
   whose premise is checked through these definitions is covered the way a local copy was.

   A padded window's or a transposed convolution's cell is a [Ll_test.reduction_cells] cell over
   every tap or pair, its operands read through [with_zero], the coordinate function sending a
   margin tap, or a pair that misses the cell, to the appended zero. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let sig_bits = Ir.Ops.significand_bits Ir.Ops.single

(* [vals] as an operand reader with one zero cell appended: the index [Array.length vals] stands for
   a tap that reads a margin, or a term no tap reaches. *)
let with_zero vals =
  let len = Array.length vals in
  Ll_test.operand ~b:[] ~o:[ len + 1 ] ~i:[] (fun ix -> if ix.(0) < len then vals.(ix.(0)) else 0.)

(* The premise's host sums, compared bitwise with the executed values [got]. *)
let check_sums label products got =
  p_all2 label got
    (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
    ~f:Float.equal

(* The label [exact_sum]'s comparison of host sums and executed values carries. *)
let premise_label label = label ^ ": the premise's host sums are the executed values"

let matches label ~dims ~want (got_dims, got) =
  p (label ^ ": extent") (Array.equal Int.equal got_dims dims);
  pf_all2 "%s: values" label got want ~f:Float.equal

(* A form that sums, with its bitwise premise derived from its operands: each result cell [o] sums
   [x.(x_at o r) * w.(w_at o r)] over every index [r] of [over], an index past an operand's data
   reading its appended zero. The products are small positive integers, so every partial sum of
   every cell is exact in single precision, in any order, which the [p_all] below checks rather than
   this sentence asserts; and the executed values are these host sums as well as the oracle [want].
   [build] makes the tensor in a fresh session, from operands with no parameters. *)
let exact_sum label ~dims ~want ~over ~x ~x_at ~w ~w_at build =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let x_r = with_zero x and w_r = with_zero w in
      let products =
        Ll_test.reduction_cells ~out:dims ~over (fun o r ->
            x_r [| x_at o r |] *. w_r [| w_at o r |])
      in
      p_all
        (label ^ ": every partial sum of every cell is exact in f32")
        products
        ~f:(Ll_test.exact_partials ~sig_bits);
      let t = build () in
      let ctx = Train.forward_once (Context.auto ()) t in
      let got = Context.get_values ctx t.Tensor.value in
      matches label ~dims ~want (Ir.Tnode.dims_without_padding t.Tensor.value, got);
      check_sums (premise_label label) products got)
