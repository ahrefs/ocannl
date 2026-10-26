(* gh-ocannl-1618: the execute-audit of result-side affine index forms. An einsum spec whose RESULT
   slot carries an affine axis (after [=>]) writes through that index: a stride-only axis [2*i]
   scatters with gaps, a stride and offset [2*i+1] interleaves, and an unpadded window [i<+k] is a
   transposed convolution. Each accepted form below runs and is compared against an oracle computed
   here in plain OCaml, extent included; the values are small integers, so every sum is exact in
   single precision on any backend and in any order.

   The forms: stride-only and stride-and-offset axes (unary, binary, and the [interleave] operation,
   which writes them from [%cd ~logic]); unpadded windows at stride 1, strided, dilated, and both; a
   multi-term index whose window label is bound by the same operand ([i<+3*j]); a window whose
   kernel label is captured and set, and one left unbound (it closes to extent 1: the identity); the
   ternary route; two windows in one result (a 2-D transposed convolution); a window on a batch and
   on an input axis; the legacy [+] under [use_padding = false]; a [%cd ~logic] assignment; and the
   gradients of a strided, dilated transposed convolution and of a stride-and-offset scatter. The
   padded result-side window ([=+]) is refused by [Shape.refuse_result_side_padded_window]
   (gh-ocannl-1550, test [result_side_padded_window]); the recipe its message gives for that layer,
   the full transposed convolution followed by a one-hot crop, runs here against PyTorch's
   [ConvTranspose1d] semantics with [padding] and [output_padding] (gh-ocannl-1619).

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
  executes "binary window i<+k"
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, i<+k" w]);
  executes "binary strided window 2*i<+k"
    ~dims:[| 1; (2 * (n - 1)) + kk |]
    ~want:(transposed ~stride:2 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+k" w]);
  executes "binary dilated window i<+2*k"
    ~dims:[| 1; n + (2 * (kk - 1)) |]
    ~want:(transposed ~dilation:2 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, i<+2*k" w]);
  executes "binary strided dilated window 2*i<+3*k"
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
  executes "padded transposed conv recipe: 2*i<+2*k, then a one-hot crop, padding 2"
    ~dims:[| 1; ((n - 1) * 2) - 4 + 4 + 1 |]
    ~want:(conv_transpose ~stride:2 ~dilation:2 ~padding:2 ~output_padding:0 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+2*k" w +* "s, o<+c; c => s, o" [ 0.; 0.; 1.; 0.; 0. ]]);
  executes "padded transposed conv recipe: 2*i<+k, then a one-hot crop, padding 1, output_padding 1"
    ~dims:[| 1; 2 * n |]
    ~want:(conv_transpose ~stride:2 ~dilation:1 ~padding:1 ~output_padding:1 a_vals kernel_vals)
    (fun () ->
      let a = operand () in
      let w = kernel () in
      [%op a +* "s, i; k => s, 2*i<+k" w +* "s, o<+c; c => s, o" [ 0.; 1. ]]);
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
  executes "unary window, kernel captured and set"
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
  executes "ternary window i<+k"
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals (Array.map2_exn kernel_vals kernel2_vals ~f:( *. )))
    (fun () ->
      Operation.einsum3 "s, i; k; k => s, i<+k" ~grad_spec:pg (operand ()) (kernel ())
        (ndarray [ kk ] kernel2_vals) ());
  executes "binary two windows i<+k, j<+l"
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
  executes "batch-axis window b<+k"
    ~dims:[| n + kk - 1; 1 |]
    ~want:(transposed a_vals kernel_vals)
    (fun () ->
      let a = ndarray ~batch_dims:[ n ] [ 1 ] a_vals in
      let w = kernel () in
      [%op a +* "b|s; k => b<+k|s" w]);
  executes "input-axis window i<+k"
    ~dims:[| 1; n + kk - 1 |]
    ~want:(transposed a_vals kernel_vals)
    (fun () ->
      let a = ndarray ~input_dims:[ n ] [ 1 ] a_vals in
      let w = kernel () in
      [%op a +* "i->s; k => i<+k->s" w]);
  executes "legacy + with use_padding = false"
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
      match
        let a = operand () in
        let w = kernel () in
        let%cd asn = { sm } =:+ a * w ~logic:"s, i; k => s, 2*i<+k" in
        Train.set_materialized sm.Tensor.value;
        let ctx = Context.auto () in
        let ctx, routine = Context.compile ~name:"cd_window" ctx asn Ir.Indexing.Empty in
        read (Context.run ctx routine) sm
      with
      | exception e ->
          Stdio.eprintf "%s: raised %s\n%!" label (exn_message e);
          pf "%s: executes" label false
      | got ->
          matches label
            ~dims:[| 1; (2 * (n - 1)) + kk |]
            ~want:(transposed ~stride:2 a_vals kernel_vals)
            got)

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
      let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
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
