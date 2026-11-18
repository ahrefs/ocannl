(* gh-ocannl-1618: the execute-audit of result-side affine index forms. An einsum spec whose RESULT
   slot carries an affine axis (after [=>]) writes through that index: a stride-only axis [2*i]
   scatters with gaps, a stride and offset [2*i+1] interleaves, and an unpadded window [i<+k] is a
   transposed convolution. Each accepted form below runs and is compared against an oracle computed
   here in plain OCaml, extent included; the values are small integers, and each form that sums
   derives its bitwise premise from its operands ([exact_sum], and the gradients' own).

   The forms: stride-only and stride-and-offset axes (unary, binary, and the [interleave] operation,
   which writes them from [%cd ~logic]); unpadded windows at stride 1, strided, dilated, and both; a
   multi-term index whose window label is bound by the same operand ([i<+3*j]); a window whose
   kernel label is captured and set, and one left unbound (it closes to extent 1: the identity); the
   ternary route; two windows in one result (a 2-D transposed convolution); a window on a batch and
   on an input axis; the legacy [+] under [use_padding = false]; a [%cd ~logic] assignment; and the
   gradients of a strided, dilated transposed convolution and of a stride-and-offset scatter. The
   padded result-side window ([=+]) is refused by [Shape.refuse_result_side_padded_window]
   (gh-ocannl-1550, test [result_side_padded_window]); the recipe its message gives for that layer,
   the full transposed convolution followed by a crop, runs here against PyTorch's [ConvTranspose1d]
   semantics with [padding] and [output_padding], the crop both as a slice of a concatenated axis
   (gh-ocannl-1656) and as a one-hot window (gh-ocannl-1619).

   The audit found the offset forms wrong: shape inference sizes an affine axis without its offset,
   so an offset at or past the stride ([2*i+2], [1*i+1]) or any offset on an unpadded window
   ([2*i<+1+k]) wrote past the result's extent and those cells were silently dropped (or, at stride
   1, died with an internal projection error); on an operand the same forms read past the extent.
   They are now refused while the tensor is built ([Shape.refuse_out_of_extent_offset]), on every
   route and in either slot, and the refusals are pinned below. The [%op]/[%cd] renderer also wrote
   the offset after the window ([2*o<+k+1]) and dropped a stride of 1 ([i+1]), so such specs failed
   with a parse error of a spec the user never wrote; a padded operand window with an offset now
   reaches the same routine under [%op] as from a plain string, which the last case pins. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let n = 4
let a_vals = Array.init n ~f:(fun i -> Float.of_int (i + 1))
let b_vals = Array.init n ~f:(fun i -> Float.of_int (10 * (i + 1)))
let kernel_vals = [| 1.; 2.; 4. |]
let kernel2_vals = [| 3.; 5.; 7. |]
let kk = Array.length kernel_vals
let pg = Tensor.Prohibit_grad

let ndarray ?(grad_spec = pg) ?(batch_dims = []) ?(input_dims = []) output_dims values =
  Tensor.ndarray ~grad_spec values ~batch_dims ~input_dims ~output_dims ()

let operand ?grad_spec () = ndarray ?grad_spec [ 1; n ] a_vals
let kernel ?grad_spec () = ndarray ?grad_spec [ kk ] kernel_vals

(* {2 Oracles} *)

let scatter ~len writes =
  let res = Array.create ~len 0. in
  List.iter writes ~f:(fun (o, v) -> res.(o) <- res.(o) +. v);
  res

(* [res[stride * i + dilation * k] += a[i] * w[k]], extent [stride * (n - 1) + dilation * (K - 1) +
   1]: the full transposed convolution. *)
let transposed ?(stride = 1) ?(dilation = 1) a w =
  let len = (stride * (Array.length a - 1)) + (dilation * (Array.length w - 1)) + 1 in
  scatter ~len
    (List.concat_mapi (Array.to_list a) ~f:(fun i av ->
         List.mapi (Array.to_list w) ~f:(fun k wv -> ((stride * i) + (dilation * k), av *. wv))))

(* PyTorch's [ConvTranspose1d] (no bias, one channel), straight from its definition: [res[stride * i
   + dilation * k - padding] += a[i] * w[k]] for the result positions in [[0, (n - 1) * stride - 2 *
   padding + dilation * (K - 1) + output_padding + 1)]. Independent of [transposed]: the recipe
   below crops the full transposed convolution, this oracle never builds it. *)
let conv_transpose ~stride ~dilation ~padding ~output_padding a w =
  let len =
    ((Array.length a - 1) * stride)
    - (2 * padding)
    + (dilation * (Array.length w - 1))
    + output_padding + 1
  in
  let res = Array.create ~len 0. in
  Array.iteri a ~f:(fun i av ->
      Array.iteri w ~f:(fun k wv ->
          let o = (stride * i) + (dilation * k) - padding in
          if o >= 0 && o < len then res.(o) <- res.(o) +. (av *. wv)));
  res

(* [res[stride * i + offset] = a[i]], extent [stride * n]. *)
let strided ~stride ~offset a =
  scatter
    ~len:(stride * Array.length a)
    (List.mapi (Array.to_list a) ~f:(fun i av -> ((stride * i) + offset, av)))

(* {2 Executed forms} *)

let exn_message = function Row.Shape_error (msg, _) -> msg | e -> Stdlib.Printexc.to_string e

let matches label ~dims ~want (got_dims, got) =
  p (label ^ ": extent") (Array.equal Int.equal got_dims dims);
  pf_all2 "%s: values" label got want ~f:Float.equal

let read ctx t =
  (Ir.Tnode.dims_without_padding t.Tensor.value, Context.get_values ctx t.Tensor.value)

let executes label ~dims ~want build =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      match
        let t = build () in
        read (Train.forward_once (Context.auto ()) t) t
      with
      | exception e ->
          Stdio.eprintf "%s: raised %s\n%!" label (exn_message e);
          pf "%s: executes" label false
      | got -> matches label ~dims ~want got)

(* [vals] as an operand reader with one zero cell appended: the index [Array.length vals] stands for
   a term no tap reaches. *)
let with_zero vals =
  let len = Array.length vals in
  Ll_test.operand ~b:[] ~o:[ len + 1 ] ~i:[] (fun ix -> if ix.(0) < len then vals.(ix.(0)) else 0.)

(* The pair [r = (i, k)] of a transposed convolution [res[stride * i + dilation * k] += a[i] * w[k]]
   reaches the cell at [pos]: the operand index [i] where it does, else [len], the appended zero. *)
let tap ?(stride = 1) ?(dilation = 1) ~len pos r =
  if (stride * r.(0)) + (dilation * r.(1)) = pos then r.(0) else len

(* The compile and run [Train.forward_once] and [Train.update_once] perform for a program with no
   parameters, returning the run's context: a premise's executed reference is read from it. *)
let run comp =
  let ctx, routine = Train.to_routine (Context.auto ()) Ir.Indexing.Empty comp in
  Context.run ctx routine

let sig_bits = Ir.Ops.significand_bits Ir.Ops.single

let check_sums label products got =
  p_all2
    (label ^ ": the premise's host sums are the executed values")
    got
    (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
    ~f:Float.equal

(* A form that sums, with its bitwise premise derived from its operands: each result cell [o] sums
   [x.(x_at o r) * w.(w_at o r)] over every index [r] of [over], an index past an operand's data
   reading its appended zero (a term no tap reaches the cell with). The products are small positive
   integers, so every partial sum of every cell is exact in single precision, in any order, which
   the [p_all] below checks rather than this sentence asserts; and the executed values are these
   host sums as well as the oracle [want]. *)
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
      Train.set_materialized t.Tensor.value;
      let ctx = run (Train.forward t) in
      let got = Context.get_values ctx t.Tensor.value in
      matches label ~dims ~want (Ir.Tnode.dims_without_padding t.Tensor.value, got);
      check_sums label products got)

let run_forms () =
  executes "unary stride-only 2*i"
    ~dims:[| 1; 2 * n |]
    ~want:(strided ~stride:2 ~offset:0 a_vals)
    (fun () ->
      let a = operand () in
      [%op a ++ "s, i => s, 2*i"]);
  executes "unary stride and offset 2*i+1"
    ~dims:[| 1; 2 * n |]
    ~want:(strided ~stride:2 ~offset:1 a_vals)
    (fun () ->
      let a = operand () in
      [%op a ++ "s, i => s, 2*i+1"]);
  executes "unary stride and offset 3*i+2"
    ~dims:[| 1; 3 * n |]
    ~want:(strided ~stride:3 ~offset:2 a_vals)
    (fun () ->
      let a = operand () in
      [%op a ++ "s, i => s, 3*i+2"]);
  executes "binary stride and offset 2*i+1"
    ~dims:[| 1; 2 * n |]
    ~want:(strided ~stride:2 ~offset:1 (Array.map2_exn a_vals b_vals ~f:( *. )))
    (fun () ->
      let a = operand () in
      let b = ndarray [ 1; n ] b_vals in
      [%op a +* "s, i; s, i => s, 2*i+1" b]);
  executes "interleave (%cd ~logic 2*i and 2*i+1)"
    ~dims:[| 2 * n |]
    ~want:(Array.init (n * 2) ~f:(fun j -> if j % 2 = 0 then a_vals.(j / 2) else b_vals.(j / 2)))
    (fun () ->
      let a = ndarray [ n ] a_vals in
      let b = ndarray [ n ] b_vals in
      [%op interleave a b]);
  exact_sum "binary window i<+k" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~len:n o.(1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, i<+k" w]);
  exact_sum "binary strided window 2*i<+k" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:2 ~len:n o.(1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; (2 * (n - 1)) + kk |]
    ~want:(transposed ~stride:2 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+k" w]);
  exact_sum "binary dilated window i<+2*k" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~dilation:2 ~len:n o.(1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; n + (2 * (kk - 1)) |]
    ~want:(transposed ~dilation:2 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, i<+2*k" w]);
  exact_sum "binary strided dilated window 2*i<+3*k" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:2 ~dilation:3 ~len:n o.(1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; (2 * (n - 1)) + (3 * (kk - 1)) + 1 |]
    ~want:(transposed ~stride:2 ~dilation:3 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+3*k" w]);
  (* gh-ocannl-1619: the padded transposed convolution (PyTorch's [ConvTranspose1d] with [padding])
     has no single spec -- a result-side [=+] is refused -- and is built in two steps, the recipe
     [Shape.refuse_result_side_padded_window] names and [docs/syntax_extensions.md] documents: the
     full transposed convolution ([<+]), then a crop of [padding] cells at the start and [padding -
     output_padding] at the end, written as an operand-side valid window over a one-hot kernel: an
     ndarray literal of extent [2 * padding - output_padding + 1] with its 1 at [padding] reads
     [full[o + padding]]. The cases: stride 2, dilation 2, kernel 3, padding 2 (extent 7 of the full
     11); and the decoder's 2x upsampling, stride 2, kernel 3, padding 1, output_padding 1 (extent
     2n). *)
  exact_sum "padded transposed conv recipe: 2*i<+2*k, then a one-hot crop, padding 2"
    ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:2 ~dilation:2 ~len:n (o.(1) + 2) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; ((n - 1) * 2) - 4 + 4 + 1 |]
    ~want:(conv_transpose ~stride:2 ~dilation:2 ~padding:2 ~output_padding:0 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+2*k" w +* "s, o<+c; c => s, o" [ 0.; 0.; 1.; 0.; 0. ]]);
  exact_sum
    "padded transposed conv recipe: 2*i<+k, then a one-hot crop, padding 1, output_padding 1"
    ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:2 ~len:n (o.(1) + 1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; 2 * n |]
    ~want:(conv_transpose ~stride:2 ~dilation:1 ~padding:1 ~output_padding:1 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+k" w +* "s, o<+c; c => s, o" [ 0.; 1. ]]);
  (* [output_padding > padding] grows the end instead: crop only the start, then append
     [output_padding - padding] zero cells by concatenation. Stride 3, kernel 3, padding 1,
     output_padding 2: the full 12 lose 1 cell at the start and gain 1 at the end. *)
  exact_sum "padded transposed conv recipe: 3*i<+k, crop the start, append a zero, output_padding 2"
    ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:3 ~len:n (o.(1) + 1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; ((n - 1) * 3) - 2 + 2 + 2 + 1 |]
    ~want:(conv_transpose ~stride:3 ~dilation:1 ~padding:1 ~output_padding:2 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      let zeros = ndarray [ 1; 1 ] [| 0. |] in
      let%op cropped = a +* "s, i; k => s, 3*i<+k" w +* "s, o<+c; c => s, o" [ 0.; 1. ] in
      [%op (cropped, zeros) ++^ "s, a; s, b => s, a^b"]);
  (* gh-ocannl-1656: the same three configurations with the crop as a true slice of the concatenated
     axis -- the recipe the refusal message and the docs now give -- against the same oracle; the
     one-hot crops above stay as an independent control. Integer components where the sizes are
     literals, captured and set components where they are values, and the [output_padding > padding]
     end grown by a replacement form instead of a concatenation. *)
  exact_sum "padded transposed conv recipe: 2*i<+2*k, then a slice, padding 2" ~over:[| n; kk |]
    ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:2 ~dilation:2 ~len:n (o.(1) + 2) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; ((n - 1) * 2) - 4 + 4 + 1 |]
    ~want:(conv_transpose ~stride:2 ~dilation:2 ~padding:2 ~output_padding:0 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+2*k" w ++ "s, 2^o^2 => s, o"]);
  exact_sum
    "padded transposed conv recipe: 2*i<+k, then a captured slice, padding 1, output_padding 1"
    ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:2 ~len:n (o.(1) + 1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; 2 * n |]
    ~want:(conv_transpose ~stride:2 ~dilation:1 ~padding:1 ~output_padding:1 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      let padding = 1 and output_padding = 1 in
      let%op full = a +* "s, i; k => s, 2*i<+k" w in
      let%op cropped = full ++ "s, p^o^q => s, o" [ "p"; "q" ] in
      Shape.set_dim p padding;
      Shape.set_dim q (padding - output_padding);
      cropped);
  exact_sum "padded transposed conv recipe: 3*i<+k, slice the start, pad the end, output_padding 2"
    ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~stride:3 ~len:n (o.(1) + 1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; ((n - 1) * 3) - 2 + 2 + 2 + 1 |]
    ~want:(conv_transpose ~stride:3 ~dilation:1 ~padding:1 ~output_padding:2 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 3*i<+k" w ++ "s, 1^o => s, o" ++ "s, a => s, a^1"]);
  (* Both terms of the index come from the operand: [res[i + 3j] = a[i, j]], injective for i < 3. *)
  let ij_vals = Array.init (2 * n) ~f:(fun x -> Float.of_int ((10 * (x / n)) + (x % n) + 1)) in
  executes "unary multi-term i<+3*j"
    ~dims:[| 1 + (3 * (n - 1)) + 1 |]
    ~want:
      (scatter
         ~len:(1 + (3 * (n - 1)) + 1)
         (List.init (2 * n) ~f:(fun x -> ((x / n) + (3 * (x % n)), ij_vals.(x)))))
    (fun () ->
      let a = ndarray [ 2; n ] ij_vals in
      [%op a ++ "i, j => i<+3*j"]);
  exact_sum "unary window, kernel captured and set" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~len:n o.(1) r)
    ~w:(Array.create ~len:kk 1.)
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals (Array.create ~len:kk 1.))
    (fun () ->
      let a = operand () in
      let%op r = a ++ "s, i => s, i<+k" [ "k" ] in
      Shape.set_dim k kk;
      r);
  executes "unary window, kernel unbound (closes to 1)" ~dims:[| 1; n |] ~want:a_vals (fun () ->
      let a = operand () in
      [%op a ++ "s, i => s, i<+k"]);
  exact_sum "ternary window i<+k" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~len:n o.(1) r)
    ~w:(Array.map2_exn kernel_vals kernel2_vals ~f:( *. ))
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals (Array.map2_exn kernel_vals kernel2_vals ~f:( *. )))
    (fun () ->
      Operation.einsum3 "s, i; k; k => s, i<+k" ~grad_spec:pg (operand ()) (kernel ())
        (ndarray [ kk ] kernel2_vals) ());
  exact_sum "binary two windows i<+k, j<+l" ~over:[| n; 2; kk; kk |]
    ~x:(Array.init (n * 2) ~f:(fun x -> Float.of_int ((10 * (x / 2)) + (x % 2) + 1)))
    ~x_at:(fun o r ->
      if r.(0) + r.(2) = o.(0) && r.(1) + r.(3) = o.(1) then (r.(0) * 2) + r.(1) else n * 2)
    ~w:(Array.init (kk * kk) ~f:(fun x -> kernel_vals.(x / kk) *. kernel2_vals.(x % kk)))
    ~w_at:(fun _ r -> (r.(2) * kk) + r.(3))
    ~dims:[| n + kk - 1; 2 + kk - 1 |]
    ~want:
      (let cols = 2 + kk - 1 in
       scatter
         ~len:((n + kk - 1) * cols)
         (List.concat
         @@ List.init (n * 2) ~f:(fun x ->
             let i = x / 2 and j = x % 2 in
             List.concat
             @@ List.init kk ~f:(fun k ->
                 List.init kk ~f:(fun l ->
                     ( ((i + k) * cols) + j + l,
                       Float.of_int ((10 * i) + j + 1) *. kernel_vals.(k) *. kernel2_vals.(l) ))))))
    (fun () ->
      let a =
        ndarray [ n; 2 ]
          (Array.init (n * 2) ~f:(fun x -> Float.of_int ((10 * (x / 2)) + (x % 2) + 1)))
      in
      let w =
        ndarray [ kk; kk ]
          (Array.init (kk * kk) ~f:(fun x -> kernel_vals.(x / kk) *. kernel2_vals.(x % kk)))
      in
      [%op a +* "i, j; k, l => i<+k, j<+l" w]);
  exact_sum "batch-axis window b<+k" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~len:n o.(0) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| n + kk - 1; 1 |]
    ~want:(transposed a_vals kernel_vals)
    (fun () ->
      let a = ndarray ~batch_dims:[ n ] [ 1 ] a_vals in
      let w = kernel () in
      [%op a +* "b|s; k => b<+k|s" w]);
  exact_sum "input-axis window i<+k" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~len:n o.(1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals kernel_vals)
    (fun () ->
      let a = ndarray ~input_dims:[ n ] [ 1 ] a_vals in
      let w = kernel () in
      [%op a +* "i->s; k => i<+k->s" w]);
  exact_sum "legacy + with use_padding = false" ~over:[| n; kk |] ~x:a_vals
    ~x_at:(fun o r -> tap ~len:n o.(1) r)
    ~w:kernel_vals
    ~w_at:(fun _ r -> r.(1))
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals kernel_vals)
    (fun () ->
      let use_padding = false in
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, i+k" w]);
  case "%cd ~logic window 2*i<+k" (fun () ->
      Tensor.unsafe_reinitialize ();
      let label = "%cd ~logic window 2*i<+k" in
      let dims = [| 1; (2 * (n - 1)) + kk |] in
      (* The premise as [exact_sum] derives it, for the assignment's own compile. *)
      let a_r = with_zero a_vals and w_r = with_zero kernel_vals in
      let products =
        Ll_test.reduction_cells ~out:dims ~over:[| n; kk |] (fun o r ->
            a_r [| tap ~stride:2 ~len:n o.(1) r |] *. w_r [| r.(1) |])
      in
      p_all
        (label ^ ": every partial sum of every cell is exact in f32")
        products
        ~f:(Ll_test.exact_partials ~sig_bits);
      let a = operand () in
      let w = kernel () in
      let%cd asn = { sm } =:+ a * w ~logic:"s, i; k => s, 2*i<+k" in
      Train.set_materialized sm.Tensor.value;
      let ctx = Context.auto () in
      let ctx, routine = Context.compile ~name:"cd_window" ctx asn Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      let got = Context.get_values ctx sm.Tensor.value in
      matches label ~dims
        ~want:(transposed ~stride:2 a_vals kernel_vals)
        (Ir.Tnode.dims_without_padding sm.Tensor.value, got);
      check_sums label products got)

(* {2 Gradients}

   The backward of a result-side form reads the result's gradient through the same index. With the
   loss [sum_o c[o] * r[o]] and distinct weights [c[o] = o + 1], the result's gradient is [c], so a
   strided, dilated transposed convolution [r[2i + 3k] += a[i] * w[k]] has [da[i] = sum_k c[2i + 3k]
   * w[k]] and [dw[k] = sum_i c[2i + 3k] * a[i]], and a scatter [r[2i + 1] = a[i]] has [da[i] = c[2i
   + 1]]. *)

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let weights m = Array.init m ~f:(fun o -> Float.of_int (o + 1))

let run_gradients () =
  case "gradient of 2*i<+3*k" (fun () ->
      Tensor.unsafe_reinitialize ();
      let m = (2 * (n - 1)) + (3 * (kk - 1)) + 1 in
      let c_vals = weights m in
      let a = operand ~grad_spec:Tensor.Require_grad () in
      let w = kernel ~grad_spec:Tensor.Require_grad () in
      let c = ndarray [ 1; m ] c_vals in
      let%op r = a +* "s, i; k => s, 2*i<+3*k" w in
      let%op loss = (r *. c) ++ "s, o => 0" in
      Train.set_materialized (grad_of a);
      Train.set_materialized (grad_of w);
      (* The bitwise premise, from the operands' own data: [da]'s cells sum [c.(2i + 3k) * w.(k)]
         over the kernel, [dw]'s [c.(2i + 3k) * a.(i)] over the operand; every product is a small
         positive integer, so every partial sum of every cell is exact in f32, in any order. *)
      let c_r = with_zero c_vals and a_r = with_zero a_vals and w_r = with_zero kernel_vals in
      let da_products =
        Ll_test.reduction_cells ~out:[| 1; n |] ~over:[| kk |] (fun o r ->
            c_r [| (2 * o.(1)) + (3 * r.(0)) |] *. w_r [| r.(0) |])
      in
      let dw_products =
        Ll_test.reduction_cells ~out:[| kk |] ~over:[| n |] (fun o r ->
            c_r [| (2 * r.(0)) + (3 * o.(0)) |] *. a_r [| r.(0) |])
      in
      p_all "d/da of the strided dilated window: every partial sum of every cell is exact in f32"
        da_products
        ~f:(Ll_test.exact_partials ~sig_bits);
      p_all "d/dw of the strided dilated window: every partial sum of every cell is exact in f32"
        dw_products
        ~f:(Ll_test.exact_partials ~sig_bits);
      let ctx = run (Train.grad_update loss) in
      check_sums "d/da of the strided dilated window" da_products
        (Context.get_values ctx (grad_of a));
      check_sums "d/dw of the strided dilated window" dw_products
        (Context.get_values ctx (grad_of w));
      let sum l f = List.sum (module Float) l ~f in
      pf_all2 "d/da of the strided dilated window"
        (Context.get_values ctx (grad_of a))
        (Array.init n ~f:(fun i ->
             sum (List.init kk ~f:Fn.id) (fun k -> c_vals.((2 * i) + (3 * k)) *. kernel_vals.(k))))
        ~f:Float.equal;
      pf_all2 "d/dw of the strided dilated window"
        (Context.get_values ctx (grad_of w))
        (Array.init kk ~f:(fun k ->
             sum (List.init n ~f:Fn.id) (fun i -> c_vals.((2 * i) + (3 * k)) *. a_vals.(i))))
        ~f:Float.equal);
  case "gradient of 2*i+1" (fun () ->
      Tensor.unsafe_reinitialize ();
      let c_vals = weights (2 * n) in
      let a = operand ~grad_spec:Tensor.Require_grad () in
      let c = ndarray [ 1; 2 * n ] c_vals in
      let%op r = a ++ "s, i => s, 2*i+1" in
      let%op loss = (r *. c) ++ "s, o => 0" in
      Train.set_materialized (grad_of a);
      let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
      pf_all2 "d/da of the stride-and-offset scatter"
        (Context.get_values ctx (grad_of a))
        (Array.init n ~f:(fun i -> c_vals.((2 * i) + 1)))
        ~f:Float.equal)

(* {2 Refused offset forms} *)

type outcome = Refused_at_construction of string | Refused_later of string | Ran

(* [build] constructs the operation and returns the thunk that compiles and runs it. A failed case
   leaves the session tainted (gh-ocannl-1301); it is rewound, so the next case starts from the last
   closure. *)
let attempt build =
  let outcome =
    match build () with
    | exception Row.Shape_error (msg, _) -> Refused_at_construction msg
    | run -> ( match run () with () -> Ran | exception e -> Refused_later (exn_message e))
  in
  if Option.is_some (Shape.taint_provenance ()) then Tensor.rewind_to_last_closure ();
  outcome

let forward t () = ignore (Train.forward_once (Context.auto ()) t : Context.t)

let refused ?(print_message = false) label ~axis ~spec build =
  case label (fun () ->
      let claim_label =
        Printf.sprintf "%s: refused at construction, naming the affine axis %s" label axis
      in
      match attempt build with
      | Refused_at_construction msg ->
          if print_message then Stdio.printf "%s: %s\n" label msg;
          let names_it =
            String.is_substring msg ~substring:("affine axis `" ^ axis ^ "`")
            && String.is_substring msg ~substring:("einsum spec \"" ^ spec ^ "\"")
          in
          if not names_it then Stdio.eprintf "%s: refused with another message: %s\n" label msg;
          pf "%s" claim_label names_it
      | Refused_later msg ->
          Stdio.eprintf "%s: refused only after construction: %s\n" label msg;
          pf "%s" claim_label false
      | Ran ->
          Stdio.eprintf "%s: was not refused\n" label;
          pf "%s" claim_label false)

let a8 () = ndarray [ 1; 8 ] (Array.init 8 ~f:(fun i -> Float.of_int (i + 1)))
let a9 () = ndarray [ 1; 9 ] (Array.init 9 ~f:(fun i -> Float.of_int (i + 1)))

let run_refusals () =
  Tensor.unsafe_reinitialize ();
  refused ~print_message:true "result offset equal to the stride" ~axis:"2*i+2"
    ~spec:"s, i => s, 2*i+2" (fun () ->
      let a = operand () in
      let%op r = a ++ "s, i => s, 2*i+2" in
      forward r);
  refused "result offset past the stride" ~axis:"2*i+5" ~spec:"s, i => s, 2*i+5" (fun () ->
      forward (Operation.einsum1 "s, i => s, 2*i+5" ~grad_spec:pg (operand ()) ()));
  refused "result offset at stride 1, %op" ~axis:"1*i+1" ~spec:"s, i => s, 1*i+1" (fun () ->
      let a = operand () in
      let%op r = a ++ "s, i => s, 1*i+1" in
      forward r);
  refused "result offset at stride 1, binary" ~axis:"1*i+1" ~spec:"s, i; s, i => s, 1*i+1"
    (fun () ->
      forward
        (Operation.einsum "s, i; s, i => s, 1*i+1" ~grad_spec:pg (operand ())
           (ndarray [ 1; n ] b_vals)
           ()));
  refused ~print_message:true "result unpadded window with an offset" ~axis:"2*i<+1+k"
    ~spec:"s, i; k => s, 2*i<+1+k" (fun () ->
      let a = operand () in
      let w = kernel () in
      let%op r = a +* "s, i; k => s, 2*i<+1+k" w in
      forward r);
  refused "result unpadded stride-1 window with an offset" ~axis:"i<+1+k"
    ~spec:"s, i; k => s, i<+1+k" (fun () ->
      forward (Operation.einsum "s, i; k => s, i<+1+k" ~grad_spec:pg (operand ()) (kernel ()) ()));
  refused "result window with an offset, ternary" ~axis:"i<+2+k" ~spec:"s, i; k; k => s, i<+2+k"
    (fun () ->
      forward
        (Operation.einsum3 "s, i; k; k => s, i<+2+k" ~grad_spec:pg (operand ()) (kernel ())
           (kernel ()) ()));
  refused "result offset, %cd ~logic" ~axis:"2*i<+1+k" ~spec:"s, i; k => s, 2*i<+1+k" (fun () ->
      let a = operand () in
      let w = kernel () in
      let%cd asn = { sm } =:+ a * w ~logic:"s, i; k => s, 2*i<+1+k" in
      fun () ->
        let ctx = Context.auto () in
        let ctx, routine = Context.compile ~name:"cd_offset" ctx asn Ir.Indexing.Empty in
        ignore (Context.run ctx routine : Context.t);
        ignore (sm : Tensor.t));
  refused "operand offset equal to the stride" ~axis:"2*o+2" ~spec:"s, 2*o+2 => s, o" (fun () ->
      forward (Operation.einsum1 "s, 2*o+2 => s, o" ~grad_spec:pg (a8 ()) ()));
  refused "operand unpadded window with an offset" ~axis:"2*o<+1+k" ~spec:"s, 2*o<+1+k; k => s, o"
    (fun () ->
      let x = a9 () in
      let w = kernel () in
      let%op r = x +* "s, 2*o<+1+k; k => s, o" w in
      forward r)

(* {2 The renderer}

   A padded operand window keeps its offset (the padding absorbs it). Under [%op] its spec used to
   be re-rendered as [2*o=+k+1], a parse error; now the [%op] route and a plain string reach the
   same routine. *)

let run_renderer () =
  case "padded operand window with an offset, %op vs a plain string" (fun () ->
      let run build =
        Tensor.unsafe_reinitialize ();
        let src = a8 () in
        (* [x] is computed, so the padded window can grant it margins (padding_lifecycle). *)
        let x = Operation.einsum1 "s, c => s, c" ~grad_spec:pg src () in
        let r = build x (kernel ()) in
        read (Train.forward_once (Context.auto ()) r) r
      in
      match
        ( run (fun x w -> [%op x +* "s, 2*o=+1+k; k => s, o" w]),
          run (fun x w -> Operation.einsum "s, 2*o=+1+k; k => s, o" ~grad_spec:pg x w ()) )
      with
      | exception e ->
          Stdio.eprintf "renderer: raised %s\n%!" (exn_message e);
          p "the %op route of 2*o=+1+k executes" false
      | (ppx_dims, ppx_vals), (raw_dims, raw_vals) ->
          p "the %op route of 2*o=+1+k has the plain string's extent"
            (Array.equal Int.equal ppx_dims raw_dims);
          p_all2 "the %op route of 2*o=+1+k computes the plain string's values" ppx_vals raw_vals
            ~f:Float.equal)

let () =
  run_forms ();
  run_gradients ();
  run_refusals ();
  run_renderer ()
