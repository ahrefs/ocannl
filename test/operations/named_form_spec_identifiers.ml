(* gh-ocannl-2060: under [%op] and [%cd], a named einsum form ([einsum], [einsum1], [einsum3], ...)
   reads the identifiers of its literal spec as an operator does: a stride or dilation identifier
   takes its run-time value, a plain [+] window reads [use_padding]. Before, the spec reached the
   shape solver as written and died with "Invalid stride value (expected integer): stride".

   Each named form is checked against the operator form over the same inputs and against the values
   computed by hand. The inputs are distinct integers and the kernel taps 1 and 10, so a wrong
   stride, dilation or window lands on other cells. The sums are integer-valued, exact in any
   order. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

let n = 12
let xs = Array.init n ~f:(fun i -> Float.of_int (i + 1))
let taps = [| 1.; 10. |]
let x = TDSL.ndarray xs ~label:[ "x" ] ~output_dims:[ n ] ()
let w = TDSL.ndarray taps ~label:[ "w" ] ~output_dims:[ 2 ] ()

let values t =
  let ctx = Ocannl.Train.forward_once (Context.cpu ()) t in
  Context.get_values ctx t.Tensor.value

let exact = Float.equal

(* A stride: [out.(o) = x.(stride * o)]. *)
let%op named_strided ~stride x = einsum1 "stride*o => o" x
let%op op_strided ~stride x = x ++ "stride*o => o"

(* A stride and a dilation over a valid window: [out.(o) = x.(s*o) + 10 x.(s*o + d)]. *)
let%op named_dilated ~stride ~dil x w = einsum "stride*o<+dil*k; k => o" x w
let%op op_dilated ~stride ~dil x w = x +* "stride*o<+dil*k; k => o" w

(* A plain [+] window reads [use_padding]. *)
let%op named_window ~use_padding x w = einsum "o+k; k => o" x w
let%op op_window ~use_padding x w = x +* "o+k; k => o" w

(* A capture alongside a stride identifier. *)
let%op named_captured ~stride ~d x = einsum1 ~capture_dims:[%oc [ d ]] "stride*o => o" x

(* The same in [%cd]. *)
let%cd cd_strided ~stride x = einsum1 "stride*o => o" x

(* The ternary [einsum3], which has no operator, over the same window with the kernel read twice:
   [out.(o) = x.(s*o) + 100 x.(s*o + d)]. *)
let%op named_tern ~stride ~dil x w = einsum3 "stride*o<+dil*k; k; k => o" x w w
let%op op_tern ~stride ~dil x w = x +* "stride*o<+dil*k; k => o" (w *. w)
let%cd cd_tern ~stride ~dil x w = einsum3 "stride*o<+dil*k; k; k => o" x w w

let () =
  let stride = 3 in
  let want = Array.init (n / stride) ~f:(fun o -> xs.(stride * o)) in
  let named = values (named_strided ~stride x) in
  p_all2 "einsum1 with a stride identifier computes x.(stride * o)" named want ~f:exact;
  p_all2 "einsum1 agrees with ++ on a stride identifier" named
    (values (op_strided ~stride x))
    ~f:exact;
  p_all2 "the %cd einsum1 with a stride identifier computes x.(stride * o)"
    (values (cd_strided ~stride x))
    want ~f:exact

let () =
  let stride = 2 and dil = 3 in
  (* Valid window: n = stride * (out - 1) + dil + 1. *)
  let out = ((n - dil - 1) / stride) + 1 in
  let want =
    Array.init out ~f:(fun o ->
        (taps.(0) *. xs.(stride * o)) +. (taps.(1) *. xs.((stride * o) + dil)))
  in
  let named = values (named_dilated ~stride ~dil x w) in
  p_all2 "einsum with stride and dilation identifiers computes the dilated window" named want
    ~f:exact;
  p_all2 "einsum agrees with +* on stride and dilation identifiers" named
    (values (op_dilated ~stride ~dil x w))
    ~f:exact

let () =
  let stride = 2 and dil = 3 in
  let out = ((n - dil - 1) / stride) + 1 in
  let want =
    Array.init out ~f:(fun o ->
        (taps.(0) *. taps.(0) *. xs.(stride * o))
        +. (taps.(1) *. taps.(1) *. xs.((stride * o) + dil)))
  in
  let named = values (named_tern ~stride ~dil x w) in
  p_all2 "einsum3 with stride and dilation identifiers computes the dilated window" named want
    ~f:exact;
  p_all2 "einsum3 agrees with +* over the squared kernel" named
    (values (op_tern ~stride ~dil x w))
    ~f:exact;
  p_all2 "the %cd einsum3 with stride and dilation identifiers computes the dilated window"
    (values (cd_tern ~stride ~dil x w))
    want ~f:exact

let () =
  let use_padding = false in
  let want = Array.init (n - 1) ~f:(fun o -> (taps.(0) *. xs.(o)) +. (taps.(1) *. xs.(o + 1))) in
  let named = values (named_window ~use_padding x w) in
  p_all2 "einsum with a plain + window reads use_padding = false" named want ~f:exact;
  p_all2 "einsum agrees with +* on a plain + window" named
    (values (op_window ~use_padding x w))
    ~f:exact

let () =
  let stride = 4 in
  let d = Shape.get_variable_ref "o" in
  let named = values (named_captured ~stride ~d x) in
  let want = Array.init (n / stride) ~f:(fun o -> xs.(stride * o)) in
  p_all2 "einsum1 with a capture and a stride identifier computes x.(stride * o)" named want
    ~f:exact;
  p "the capture names the output dimension, n / stride"
    (Option.equal Int.equal d.var_ref.solved_dim (Some (n / stride)))
