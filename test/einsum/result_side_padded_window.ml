(* gh-ocannl-1550: a padded ([=+]) window on the RESULT side of an einsum spec, e.g. ["s, i; k => s,
   i=+k"], is refused at shape inference with a message that names the window and says what to write
   instead. It would be a padded transposed convolution, which OCANNL does not implement; before the
   refusal such a spec built fine and then died at compile time in projection inference with the
   internal [Shape_error "projection_of_solved_dims: unknown projection"].

   The refusal is structural over the parsed result slot, so the cases below cover every route into
   it -- binary, unary ([Permute]), ternary and block einsum, a [%cd ~logic] assignment, the legacy
   [+] that the ppx resolves from [use_padding = true] -- and every place in the slot: an output, a
   batch and an input axis, and an axis written before a row variable. Each case records WHEN it was
   refused; the claim is "refused while the tensor was being built, naming the window and the spec",
   so the base commit's late internal error fails it.

   The positive controls pin what the message tells the user to write instead, by execution against
   oracles computed here (small integers: every sum is exact in single precision, in any order): the
   operand-side padded window with the kernel reversed is the padded transposed convolution (stride
   1, dilation 1, odd kernel size), and the unpadded result-side window ([<+]) is the full
   transposed convolution, extent included. A stride-only result axis still builds and runs. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let n = 4
let a_vals = Array.init n ~f:(fun i -> Float.of_int (i + 1))
let kernel_vals = [| 1.; 2.; 4. |]
let kk = Array.length kernel_vals

let ndarray ?(batch_dims = []) ?(input_dims = []) output_dims values =
  Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad values ~batch_dims ~input_dims ~output_dims ()

let operand () = ndarray [ 1; n ] a_vals
let kernel () = ndarray [ kk ] kernel_vals

(* The full transposed convolution [res[i + k] += a[i] * w[k]], extent [n + K - 1]. *)
let full_transposed =
  let res = Array.create ~len:(n + kk - 1) 0. in
  Array.iteri a_vals ~f:(fun i av ->
      Array.iteri kernel_vals ~f:(fun k kv -> res.(i + k) <- res.(i + k) +. (av *. kv)));
  res

(* The padded transposed convolution a result-side [i=+k] would denote at stride 1: padded mode's
   left margin for an odd kernel is [(K - 1) / 2], so [res[i + k - left] += a[i] * w[k]] within the
   operand's extent. *)
let padded_transposed =
  let left = (kk - 1) / 2 in
  let res = Array.create ~len:n 0. in
  Array.iteri a_vals ~f:(fun i av ->
      Array.iteri kernel_vals ~f:(fun k kv ->
          let o = i + k - left in
          if o >= 0 && o < n then res.(o) <- res.(o) +. (av *. kv)));
  res

let run_positive_controls () =
  Tensor.unsafe_reinitialize ();
  case "operand-side padded window, reversed kernel" (fun () ->
      let src = operand () in
      let reversed = ndarray [ kk ] (Array.rev kernel_vals) in
      (* [x] is computed, so its layout is not committed at creation and the padded window can grant
         it margins (padding_lifecycle). *)
      let%op x = src ++ "s, c => s, c" in
      let%op y = x +* "s, o=+k; k => s, o" reversed in
      let ctx = Train.forward_once (Context.auto ()) y in
      pf_all2
        "the operand-side padded window with the reversed kernel is the padded transposed \
         convolution"
        (Context.get_values ctx y.Tensor.value)
        padded_transposed ~f:Float.equal);
  case "result-side unpadded window" (fun () ->
      let a = operand () in
      let w = kernel () in
      let%op r = a +* "s, i; k => s, i<+k" w in
      let ctx = Train.forward_once (Context.auto ()) r in
      p "the result-side unpadded window's extent is n + K - 1"
        (Array.equal Int.equal (Ir.Tnode.dims_without_padding r.Tensor.value) [| 1; n + kk - 1 |]);
      pf_all2 "the result-side unpadded window is the full transposed convolution"
        (Context.get_values ctx r.Tensor.value)
        full_transposed ~f:Float.equal);
  case "result-side stride-only axis" (fun () ->
      let a = operand () in
      let%op r = a ++ "s, i => s, 2*i" in
      let _ctx = Train.forward_once (Context.auto ()) r in
      p "a result-side stride-only axis builds and runs" true)

type outcome = Refused_at_construction of string | Refused_later of string | Ran

let exn_message = function Row.Shape_error (msg, _) -> msg | e -> Stdlib.Printexc.to_string e

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

let refused ?(print_message = false) label ~window ~spec build =
  case label (fun () ->
      let claim_label =
        Printf.sprintf "%s: refused at construction, naming the result-side padded window %s" label
          window
      in
      match attempt build with
      | Refused_at_construction msg ->
          if print_message then Stdio.printf "%s: %s\n" label msg;
          let names_it =
            String.is_substring msg ~substring:("result-side padded window `" ^ window ^ "`")
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

let run_refusals () =
  Tensor.unsafe_reinitialize ();
  refused ~print_message:true "binary (the reported spec)" ~window:"i=+k" ~spec:"s, i; k => s, i=+k"
    (fun () ->
      let a = operand () in
      let w = kernel () in
      let%op r = a +* "s, i; k => s, i=+k" w in
      forward r);
  refused "binary, strided and dilated" ~window:"2*i=+2*k" ~spec:"s, i; k => s, 2*i=+2*k" (fun () ->
      let a = operand () in
      let w = kernel () in
      let%op r = a +* "s, i; k => s, 2*i=+2*k" w in
      forward r);
  refused "unary" ~window:"i=+k" ~spec:"s, i => s, i=+k" (fun () ->
      let a = operand () in
      let%op r = a ++ "s, i => s, i=+k" in
      forward r);
  refused "ternary" ~window:"i=+k" ~spec:"s, i; k; k => s, i=+k" (fun () ->
      let r =
        Operation.einsum3 "s, i; k; k => s, i=+k" ~grad_spec:Tensor.Prohibit_grad (operand ())
          (kernel ()) (kernel ()) ()
      in
      forward r);
  (* The [++^] ppx re-renders the concatenation as [c ^ d] in the spec it passes on. *)
  refused "block" ~window:"s=+k" ~spec:"s, c; s, d => s=+k, c ^ d" (fun () ->
      let a = operand () in
      let b = operand () in
      let%op r = (a, b) ++^ "s, c; s, d => s=+k, c^d" in
      forward r);
  refused "batch axis" ~window:"b=+k" ~spec:"b|s; k => b=+k|s" (fun () ->
      let a = ndarray ~batch_dims:[ n ] [ 1 ] a_vals in
      let w = kernel () in
      let%op r = a +* "b|s; k => b=+k|s" w in
      forward r);
  refused "input axis" ~window:"i=+k" ~spec:"i->s; k => i=+k->s" (fun () ->
      let a = ndarray ~input_dims:[ n ] [ 1 ] a_vals in
      let w = kernel () in
      let%op r = a +* "i->s; k => i=+k->s" w in
      forward r);
  refused "axis before a row variable" ~window:"i=+k" ~spec:"i, ..r..; k => i=+k, ..r.." (fun () ->
      let a = ndarray [ n; 1 ] a_vals in
      let w = kernel () in
      let%op r = a +* "i, ..r..; k => i=+k, ..r.." w in
      forward r);
  refused "legacy + with use_padding = true" ~window:"i=+k" ~spec:"s, i; k => s, i=+k" (fun () ->
      let use_padding = true in
      let a = operand () in
      let w = kernel () in
      let%op r = a +* "s, i; k => s, i+k" w in
      forward r);
  refused "%cd ~logic" ~window:"i=+k" ~spec:"s, i; k => s, i=+k" (fun () ->
      let a = operand () in
      let w = kernel () in
      let%cd asn = { sm } =:+ a * w ~logic:"s, i; k => s, i=+k" in
      fun () ->
        let ctx = Context.auto () in
        let ctx, routine = Context.compile ~name:"cd_window" ctx asn Ir.Indexing.Empty in
        ignore (Context.run ctx routine : Context.t);
        ignore (sm : Tensor.t))

let () =
  run_positive_controls ();
  run_refusals ()
