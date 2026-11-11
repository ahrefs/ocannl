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

   Listing one component twice in a concatenation, [a => a^a], has no such reading: a segment is one
   range of positions with one loop index, so lowering read or wrote one copy only, at the last
   copy's offset -- silently: [a => a^a] gave [0 0 1 2]. It is refused from the spec, by every entry
   point, while a tensor passed twice under distinct components still concatenates.

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
let open_leaf () = NTDSL.term ~fetch_op:(Constant 2.) ()

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

let cd_partial label make =
  computes label
    (fun () ->
      let tgt =
        NTDSL.init ~l:"cd_sentinel" ~prec:Ir.Ops.single ~o:[ 6 ]
          ~f:(fun i -> Float.of_int (-100 + i.(0)))
          ()
      in
      Train.set_materialized tgt.Tensor.value;
      let comp = make tgt (vec 5) in
      let ctx, routine = Context.compile (Context.cpu ()) comp Ir.Indexing.Empty in
      Context.get_values (Context.run ctx routine) tgt.Tensor.value)
    ~want:[| 1.; 2.; -98.; 3.; 4.; 5. |]

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
  cd_partial "%cd captures concat sizes and preserves the untouched sentinel" (fun tgt x ->
      let%cd comp = tgt =: id x ~logic:"a^c => a^b^c" [ "a"; "b" ] in
      Shape.set_dim a 2;
      Shape.set_dim b 1;
      comp);
  cd_partial "%cd parenthesized primitive captures preserve the untouched sentinel" (fun tgt x ->
      let%cd comp = tgt =: (id x) ~logic:"a^c => a^b^c" [ "a"; "b" ] in
      Shape.set_dim a 2;
      Shape.set_dim b 1;
      comp);
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

(* A refusal at construction, a [Row.Shape_error] with [needle] in its message: a value, or any
   other exception (the lowering errors above among them), fails the claim. *)
let refused label make =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      match make () with
      | (_ : float array) -> pf "%s: refused" label false
      | exception Row.Shape_error (msg, _) ->
          Stdio.printf "%s: %s\n" label msg;
          pf "%s: the refusal says why" label
            (String.is_substring msg ~substring:"appears twice in the concatenated axis")
      | exception e ->
          Stdio.printf "%s: raised %s\n" label (Exn.to_string e);
          pf "%s: refused by the spec check" label false)

let doubling () =
  refused "a component doubled in the result, a => a^a" (fun () ->
      values (NTDSL.einsum1 "a => a^a" (vec 2) ()));
  refused "a component doubled around another, a^b => a^b^a" (fun () ->
      values (NTDSL.einsum1 ~capture_dims:[ capture "a" 2 ] "a^b => a^b^a" (vec 5) ()));
  refused "a component doubled in the operand, a^b^a => a^b" (fun () ->
      values (NTDSL.einsum1 ~capture_dims:[ capture "a" 2 ] "a^b^a => a^b" (vec 6) ()));
  refused "a doubled component over an open operand" (fun () ->
      values (NTDSL.einsum1 "a => a^a" (open_leaf ()) ()));
  refused "a concatenation filling one component twice, a; a => a^a" (fun () ->
      values (NTDSL.concat "a; a => a^a" [| vec 2; vec ~base:10 2 |] ()));
  refused "a concatenation repeating its first segment, a; b => a^b^a" (fun () ->
      values (NTDSL.concat "a; b => a^b^a" [| vec 2; vec ~base:10 3 |] ()));
  refused "through %op" (fun () ->
      let x = vec 2 in
      let%op t = x ++ "a => a^a" in
      values t);
  refused "through %cd" (fun () ->
      let x = vec 2 and tgt = vec ~base:50 4 in
      let comp = [%cd tgt =: id x ~logic:"a => a^a"] in
      ignore (comp : Ir.Assignments.comp);
      [||])

let doubling_neighbours () =
  computes "a tensor passed twice under distinct components, (x, x) ++^ \"a; b => a^b\""
    (fun () ->
      let x = vec 2 in
      values (NTDSL.concat "a; b => a^b" [| x; x |] ()))
    ~want:[| 1.; 2.; 1.; 2. |];
  computes "a repeated integer component, 2^a^2 => a"
    (fun () -> values (NTDSL.einsum1 "2^a^2 => a" (vec 5) ()))
    ~want:[| 3. |];
  (* No reinitialization between the refusal and the computation, which uses an operand that
     predates it. *)
  case "a refused doubling leaves inference usable" (fun () ->
      Tensor.unsafe_reinitialize ();
      let x = vec 2 in
      p "the doubling is refused"
        (match NTDSL.einsum1 "a => a^a" x () with
        | (_ : Tensor.t) -> false
        | exception Row.Shape_error _ -> true);
      let label = "the same operand, a valid operation" in
      pf_close "%s" label
        (values (NTDSL.einsum1 ~capture_dims:[ capture "b" 1 ] "a => a^b" x ()))
        [| 1.; 2.; 0. |]
        ~tol:(fun _ -> 0.))

(* gh-ocannl-2088: a lowering refusal, a [Utils.User_error] carrying [needle] and not [unlike]: each
   of the two refusals below is checked to be told apart from the other. *)
let refused_lowering label ~needle ~unlike make =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      match make () with
      | (_ : float array) -> pf "%s: refused" label false
      | exception Utils.User_error msg ->
          Stdio.printf "%s: %s\n" label msg;
          pf "%s: the refusal says why" label
            (String.is_substring msg ~substring:needle
            && not (String.is_substring msg ~substring:unlike))
      | exception e ->
          Stdio.printf "%s: raised %s\n" label (Exn.to_string e);
          pf "%s: refused in lowering" label false)

let missing_operand = "take no part in the block"
let ambiguous = "Ambiguous indices in concatenation"

(* A block of a concatenated axis is one segment's positions. An operand that does not use the
   segment contributes nothing there, which is what concatenation wants, but a multiplicative einsum
   needs every operand at every position: with one of them absent there is nothing to multiply, so
   the block is refused, never left at the result's initialization. These read "Ambiguous indices in
   concatenation: multiple blocks viable for same position", which is the opposite mistake: more
   operands than the operation takes, as when two operands of a concatenation can fill one segment.
   ([p^q; p, q => p^q], the issue's other example, no longer reaches lowering: the spec check
   refuses a tensor whose axes need both components.) *)
let operand_shortfall () =
  refused_lowering "each operand fills one segment of a product, a; b => a^b"
    ~needle:missing_operand ~unlike:ambiguous (fun () ->
      values (NTDSL.einsum "a; b => a^b" (vec 2) (vec ~base:10 3) ()));
  refused_lowering "through %op, x +* \"a; b => a^b\" y" ~needle:missing_operand ~unlike:ambiguous
    (fun () ->
      let x = vec 2 and y = vec ~base:10 3 in
      let%op t = x +* "a; b => a^b" y in
      values t);
  refused_lowering "an operand lacking a segment the result lists, a^c; a^b^c => a^b^c"
    ~needle:missing_operand ~unlike:ambiguous (fun () ->
      values
        (NTDSL.einsum
           ~capture_dims:[ capture "a" 2; capture "b" 1 ]
           "a^c; a^b^c => a^b^c" (vec 5) (vec ~base:10 6) ()));
  refused_lowering "two operands viable for one segment of a concatenation, a^b; c => a^b"
    ~needle:ambiguous ~unlike:missing_operand (fun () ->
      values
        (NTDSL.concat
           ~capture_dims:[ capture "a" 2 ]
           "a^b; c => a^b"
           [| vec 5; vec ~base:10 2 |]
           ()));
  computes "every operand listing every segment, a^b; a^b => a^b"
    (fun () ->
      values
        (NTDSL.einsum ~capture_dims:[ capture "a" 2 ] "a^b; a^b => a^b" (vec 3) (vec ~base:10 3) ()))
    ~want:[| 10.; 22.; 36. |];
  (* A segment no other slot names is a slice, not a block: [3 * 10 + 4 * 11 + 5 * 12]. *)
  computes "a segment only one operand names, sliced off, a^b; b => "
    (fun () ->
      values (NTDSL.einsum ~capture_dims:[ capture "a" 2 ] "a^b; b => " (vec 5) (vec ~base:10 3) ()))
    ~want:[| 134. |]

let () =
  forward ();
  gradients ();
  doubling ();
  doubling_neighbours ();
  operand_shortfall ()
