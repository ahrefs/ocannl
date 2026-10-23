(* gh-ocannl-1496: a padded window's transpose -- the input gradient of a padded convolution --
   leaves its gradient's margins holding the padding value once its loop nest is done.

   The backward of [y = x +* "s, o=+k; k => s, o" kernel] accumulates [x.grad[s, o + k - left] +=
   y.grad[s, o] * kernel[k]]. The positions [o + k - left] outside [x]'s extent are the padding
   positions; they land in [x.grad]'s margins (a gradient shares its value's padding), and the store
   is deliberately unguarded so the hot loop carries no per-element test. Those contributions used
   to stay there, after the margin reset the writer emits first.

   Reachability: a user program observes the margins as soon as it reads the gradient through a
   padded window of its own -- here a [%cd] assignment smoothing the gradient with the same padded
   convolution, as a custom optimizer would. Shape inference accepts it (the gradient already
   carries the margins the window demands) and the read is unguarded too (covered margins hold the
   neutral element, by contract), so the smoothed value at each edge picked up the margin's
   contribution: on the base commit both edge cells of [sm] differed from the zero-padded oracle,
   and the margins held the padding positions' contributions. A forward result-side padded window
   ([=> o=+k]), the other way the issue imagined margins being written and read, does not reach
   lowering: its projections are not inferred.

   Every product below is a small integer, so every sum is exact in single precision on every
   backend, in any order; the values discriminate: each padding position's contribution is
   nonzero. *)

open Base
open Ocannl
open Operation.DSL_modules
module Tn = Ir.Tnode
open Verdict.Claims

let n = 4
let kernel_vals = [| 1.; 2.; 4. |]
let src_vals = Array.init n ~f:(fun i -> Float.of_int (i + 1))
let w_vals = [| 1.; 3.; 5.; 7. |]
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let ndarray ~grad dims values =
  Tensor.ndarray
    ~grad_spec:(if grad then Tensor.Require_grad else Tensor.Prohibit_grad)
    values ~batch_dims:[] ~input_dims:[] ~output_dims:dims ()

(* The zero-padded correlation [out[o] = sum_k in[o + k - left] * kernel[k]]. *)
let conv ~left input =
  Array.init (Array.length input) ~f:(fun o ->
      Array.foldi kernel_vals ~init:0. ~f:(fun k acc kv ->
          let c = o + k - left in
          if c >= 0 && c < Array.length input then acc +. (input.(c) *. kv) else acc))

(* Its transpose, the input gradient of [sum (out *. w)]: [g[c] = sum over (o, k) with o + k - left
   = c of w[o] * kernel[k]]. *)
let conv_input_grad ~left w =
  Array.init n ~f:(fun c ->
      Array.foldi w ~init:0. ~f:(fun o acc wo ->
          Array.foldi kernel_vals ~init:acc ~f:(fun k acc kv ->
              if o + k - left = c then acc +. (wo *. kv) else acc)))

let () =
  Tensor.unsafe_reinitialize ();
  let src = ndarray ~grad:true [ 1; n ] src_vals in
  let kernel = ndarray ~grad:false [ 3 ] kernel_vals in
  let w = ndarray ~grad:false [ 1; n ] w_vals in
  (* [x] is computed, so its layout is not committed at creation and the padded convolution can
     grant it margins (padding_lifecycle). *)
  let%op x = src ++ "s, c => s, c" in
  let%op y = x +* "s, o=+k; k => s, o" kernel in
  let%op loss = (y *. w) ++ "...|... => |->0" in
  let gx = grad_of x in
  let%cd smooth = { sm } =:+ x.grad * kernel ~logic:"s, o=+k; k => s, o" in
  Train.set_materialized gx;
  Train.set_materialized sm.Tensor.value;
  let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
  let ctx, routine = Context.compile ~name:"smooth" ctx smooth Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let left, right =
    match Tn.get_padding gx with
    | Some (pads, value) ->
        let Ir.Ops.{ left; right } = pads.(1) in
        p "x's gradient is padded along the convolved axis only, committing 0"
          (left > 0 && right > 0
          && pads.(0).Ir.Ops.left + pads.(0).Ir.Ops.right = 0
          && Float.equal value 0.);
        (left, right)
    | None ->
        p "x's gradient is padded along the convolved axis only, committing 0" false;
        (0, 0)
  in
  let g = conv_input_grad ~left w_vals in
  (* The discrimination premise: some padding position receives a nonzero contribution, so a margin
     that kept it differs from the padding value. *)
  p "a padding position of x's gradient receives a nonzero contribution"
    (left > 0
    && Array.existsi w_vals ~f:(fun o wo ->
        Array.existsi kernel_vals ~f:(fun k kv ->
            let c = o + k - left in
            (c < 0 || c >= n) && Float.(wo *. kv <> 0.))));
  pf_all2 "x's gradient holds the transposed convolution of w" (Context.get_values ctx gx) g
    ~f:Float.equal;
  let physical = Ir.Ndarray.retrieve_flat_values (Context.to_host ctx gx) in
  p "x's gradient's physical buffer is its margins plus its extent"
    (Array.length physical = left + n + right);
  p_all "x's gradient's margins hold the padding value after the backward pass"
    (Array.to_list (Array.filteri physical ~f:(fun j _ -> j < left || j >= left + n)))
    ~f:(fun v -> Float.equal v 0.);
  pf_all2 "a padded window reading x's gradient sees the zero-padded gradient"
    (Context.get_values ctx sm.Tensor.value)
    (conv ~left g) ~f:Float.equal
