(* gh-ocannl-2087: two slots of one einsum may list the segments of one concatenation differently.
   Each slot's concatenated axis resolves by its own components, so a component one slot does not
   list is a segment that slot takes no part in:

   - dropping a segment, [a^b^c => a^c]: the [b] cells of the operand are read by nothing; - writing
   part of the axis, [a^c => a^b^c]: the [b] cells of the result have no operand, so a fresh [++]
   result holds its zero initialization there; - reordering, with or without a drop, [a^b^c => c^a]:
   each segment lands at its own offset.

   These died in lowering with "Concat index could not be resolved to an active component during
   Block lowering". Full reorders ([a^b^c => c^b^a]) computed already, and still do; the
   equal-length ones pin that the components are matched by label, never by position.

   Every value is a small integer copied, multiplied or left at zero, so equality is exact. These
   are claims about projection derivation and lowering, so every context is [Context.cpu ()]. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let leaf ?(grad_spec = Tensor.Prohibit_grad) ~o ~base () =
  let n = List.fold o ~init:1 ~f:( * ) in
  Tensor.ndarray ~grad_spec
    (Array.init n ~f:(fun k -> Float.of_int (base + k)))
    ~batch_dims:[] ~input_dims:[] ~output_dims:o ()

let vec ?(base = 1) n = leaf ~o:[ n ] ~base ()

let values t =
  Train.set_materialized t.Tensor.value;
  Context.get_values (Train.forward_once (Context.cpu ()) t) t.Tensor.value

let capture name n =
  let r = Shape.get_variable_ref name in
  Shape.set_dim r n;
  r

let computes label make ~want =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      pf_close "%s" label (make ()) want ~tol:(fun _ -> 0.))

(* The operand is [1 .. 6]: a = [1 2], b = [3], c = [4 5 6]. *)
let abc spec = NTDSL.einsum1 ~capture_dims:[ capture "a" 2; capture "b" 1 ] spec (vec 6) ()

let forward () =
  computes "a dropped middle segment, a^b^c => a^c"
    (fun () -> values (abc "a^b^c => a^c"))
    ~want:[| 1.; 2.; 4.; 5.; 6. |];
  computes "a dropped first segment, a^b^c => b^c"
    (fun () -> values (abc "a^b^c => b^c"))
    ~want:[| 3.; 4.; 5.; 6. |];
  computes "a dropped segment and a reorder, a^b^c => c^a"
    (fun () -> values (abc "a^b^c => c^a"))
    ~want:[| 4.; 5.; 6.; 1.; 2. |];
  computes "a written middle segment left at zero, a^c => a^b^c"
    (fun () ->
      values
        (NTDSL.einsum1 ~capture_dims:[ capture "a" 2; capture "b" 1 ] "a^c => a^b^c" (vec 5) ()))
    ~want:[| 1.; 2.; 0.; 3.; 4.; 5. |];
  computes "a written segment between reordered ones, a^c => c^b^a"
    (fun () ->
      values
        (NTDSL.einsum1 ~capture_dims:[ capture "a" 2; capture "b" 1 ] "a^c => c^b^a" (vec 5) ()))
    ~want:[| 3.; 4.; 5.; 0.; 1.; 2. |];
  computes "a segment replaced by a zero segment of another size, a^b^c => a^d^c"
    (fun () ->
      values
        (NTDSL.einsum1
           ~capture_dims:[ capture "a" 2; capture "b" 1; capture "d" 2 ]
           "a^b^c => a^d^c" (vec 6) ()))
    ~want:[| 1.; 2.; 0.; 0.; 4.; 5.; 6. |];
  computes "an integer segment dropped, a^2^c => a^c"
    (fun () -> values (NTDSL.einsum1 ~capture_dims:[ capture "a" 2 ] "a^2^c => a^c" (vec 6) ()))
    ~want:[| 1.; 2.; 5.; 6. |];
  computes "an integer segment written, a^c => a^2^c"
    (fun () -> values (NTDSL.einsum1 ~capture_dims:[ capture "a" 2 ] "a^c => a^2^c" (vec 4) ()))
    ~want:[| 1.; 2.; 0.; 0.; 3.; 4. |];
  computes "a full reorder, a^b^c => b^a^c"
    (fun () -> values (abc "a^b^c => b^a^c"))
    ~want:[| 3.; 1.; 2.; 4.; 5.; 6. |];
  computes "a full reversal, a^b^c => c^b^a"
    (fun () -> values (abc "a^b^c => c^b^a"))
    ~want:[| 4.; 5.; 6.; 3.; 1.; 2. |];
  (* Equal lengths: a positional pairing of the two concatenations would be the identity. *)
  computes "an equal-length swap, a^b => b^a"
    (fun () -> values (NTDSL.einsum1 ~capture_dims:[ capture "a" 3 ] "a^b => b^a" (vec 6) ()))
    ~want:[| 4.; 5.; 6.; 1.; 2.; 3. |];
  computes "an equal-length reversal, a^b^c => c^b^a"
    (fun () ->
      values
        (NTDSL.einsum1 ~capture_dims:[ capture "a" 2; capture "b" 2 ] "a^b^c => c^b^a" (vec 6) ()))
    ~want:[| 5.; 6.; 3.; 4.; 1.; 2. |];
  computes "a binary einsum over a dropped segment, a^b^c; a^c => a^c"
    (fun () ->
      values
        (NTDSL.einsum
           ~capture_dims:[ capture "a" 2; capture "b" 1 ]
           "a^b^c; a^c => a^c" (vec 6) (vec ~base:10 5) ()))
    ~want:[| 10.; 22.; 48.; 65.; 84. |];
  computes "a concatenation that drops a segment of its first operand, a^b; c => a^c"
    (fun () ->
      values
        (NTDSL.concat
           ~capture_dims:[ capture "b" 1 ]
           "a^b; c => a^c"
           [| vec 3; vec ~base:10 3 |]
           ()))
    ~want:[| 1.; 2.; 10.; 11.; 12. |];
  computes "a concatenation with a written segment between its operands, a; c => a^b^c"
    (fun () ->
      values
        (NTDSL.concat
           ~capture_dims:[ capture "b" 1 ]
           "a; c => a^b^c"
           [| vec 2; vec ~base:10 3 |]
           ()))
    ~want:[| 1.; 2.; 0.; 10.; 11.; 12. |]

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(* The gradient of [sum (t *. w)] with distinct weights [w]: an operand cell gets the weight of the
   result cell it lands in, and a dropped cell gets zero. *)
let gradient label ~src ~make ~want_forward ~want_grad =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let x = leaf ~grad_spec:Tensor.Require_grad ~o:[ src ] ~base:1 () in
      let t = make x in
      let dst = Array.length want_forward in
      let w =
        Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad
          (Array.init dst ~f:(fun k -> Float.of_int (100 * (k + 1))))
          ~batch_dims:[] ~input_dims:[] ~output_dims:[ dst ] ()
      in
      let%op loss = (t *. w) ++ "...|... => |->0" in
      Train.set_materialized t.Tensor.value;
      Train.set_materialized (grad_of x);
      let ctx = Train.update_once ~output_cd_file:false (Context.cpu ()) loss in
      pf_close "%s: the forward" label (Context.get_values ctx t.value) want_forward ~tol:(fun _ ->
          0.);
      pf_close "%s: the operand's gradient" label
        (Context.get_values ctx (grad_of x))
        want_grad
        ~tol:(fun _ -> 0.))

let gradients () =
  let einsum1 spec x = TDSL.einsum1 ~capture_dims:[ capture "a" 2; capture "b" 1 ] spec x () in
  gradient "the gradient of a dropped segment, a^b^c => a^c" ~src:6 ~make:(einsum1 "a^b^c => a^c")
    ~want_forward:[| 1.; 2.; 4.; 5.; 6. |]
    ~want_grad:[| 100.; 200.; 0.; 300.; 400.; 500. |];
  gradient "the gradient of a partial write, a^c => a^b^c" ~src:5 ~make:(einsum1 "a^c => a^b^c")
    ~want_forward:[| 1.; 2.; 0.; 3.; 4.; 5. |]
    ~want_grad:[| 100.; 200.; 400.; 500.; 600. |];
  gradient "the gradient of a drop and a reorder, a^b^c => c^a" ~src:6
    ~make:(einsum1 "a^b^c => c^a") ~want_forward:[| 4.; 5.; 6.; 1.; 2. |]
    ~want_grad:[| 400.; 500.; 0.; 100.; 200.; 300. |]

let () =
  forward ();
  gradients ()
