(* gh-ocannl-2019: a concatenated axis of integer components alone, on an operand. Nothing in [2^3
   =>] is a variable, so substitution used to leave its [Concat] in the operand's row, where the
   closure kept it, and the step died on an assertion in [Row.get_dim_padding]. Any other solved
   dimension summed the axis on the way, which is why a closed operand worked. Now substitution sums
   it as it sums one with a variable component, and the axis is a plain one of 5: for the reduction
   that writes it, and for everything else reading the operand in the same inference, as a capture,
   a fixed index, a window or a pointwise operation.

   The operand here is open, so that the spec is what sizes it, and holds its own offsets, so that
   each cell read can be told from the rest. Claims are about shape closure and projection
   derivation: every execution context is [Context.cpu ()]. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let leaf ?(b = []) ~o ~base () =
  let n = List.fold (b @ o) ~init:1 ~f:( * ) in
  Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad
    (Array.init n ~f:(fun k -> Float.of_int (base + k)))
    ~batch_dims:b ~input_dims:[] ~output_dims:o ()

let open_operand () = NTDSL.term ~label:[ "x" ] ~fetch_op:Range_over_offsets ()
let dims t = Array.to_list (Lazy.force t.Tensor.value.dims)

let values t =
  Train.set_materialized t.Tensor.value;
  Context.get_values (Train.forward_once (Context.cpu ()) t) t.Tensor.value

(* Every value is a small integer, so equality is exact. *)
let computes label make ~want =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      pf_close "%s" label (make ()) want ~tol:(fun _ -> 0.))

let refused label ~needle make =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let outcome =
        match make () with
        | (_ : float array) -> `Accepted
        | exception Row.Shape_error (msg, _) -> `Shape_error msg
        | exception e -> `Other (Exn.to_string e)
      in
      (match outcome with
      | `Shape_error msg -> Stdio.printf "%s: refused: %s\n" label msg
      | `Accepted -> Stdio.eprintf "%s: accepted\n" label
      | `Other e -> Stdio.eprintf "%s: raised %s\n" label e);
      pf "%s: refused with a shape error" label
        (match outcome with
        | `Shape_error msg -> String.is_substring msg ~substring:needle
        | `Accepted | `Other _ -> false))

let%op scheme_reduction () x = x ++ "2^3 => "

let routes =
  [
    ( "as written",
      fun x ->
        let%op r = x ++ "2^3 => " in
        r );
    ("made by a shape scheme", fun x -> scheme_reduction () x);
  ]

let reductions () =
  computes "a closed operand, for reference"
    (fun () -> values (TDSL.einsum1 "2^3 => " (leaf ~o:[ 5 ] ~base:1 ()) ()))
    ~want:[| 15. |];
  List.iter routes ~f:(fun (route, reduce) ->
      case (route ^ ": the spec sizes an open operand") (fun () ->
          Tensor.unsafe_reinitialize ();
          let x = open_operand () in
          let r = reduce x in
          pf_close "%s: the sum of the axis" route (values r) [| 10. |] ~tol:(fun _ -> 0.);
          pf "%s: the operand takes the one axis" route (List.equal Int.equal (dims x) [ 5 ])));
  computes "a named result of the same reduction"
    (fun () ->
      let x = open_operand () in
      let%op r = x ++ "2^3 => a" [ "a" ] in
      Shape.set_dim a 2;
      values r)
    ~want:[| 10.; 10. |];
  computes "beside a batch axis"
    (fun () ->
      let x = NTDSL.term ~label:[ "x" ] ~fetch_op:Range_over_offsets ~batch_dims:[ 2 ] () in
      let r = TDSL.einsum1 "b|2^3 => b|" x () in
      values r)
    ~want:[| 10.; 35. |];
  case "the gradient reaches every segment" (fun () ->
      Tensor.unsafe_reinitialize ();
      let x =
        Tensor.term ~label:[ "x" ] ~grad_spec:Tensor.Require_grad ~fetch_op:(Constant 2.) ()
      in
      let%op t = x ++ "2^3 => " in
      let%op loss = t *. t in
      Train.set_materialized (grad_of x);
      let ctx = Train.update_once ~output_cd_file:false (Context.cpu ()) loss in
      pf_close "the gradient of the squared sum"
        (Context.get_values ctx (grad_of x))
        (Array.create ~len:5 20.)
        ~tol:(fun _ -> 0.))

(* The reduction's scalar is added in, so that its step derives alongside the consumer's. *)
let consumers () =
  let with_reduction route reduce label ~want consumer =
    computes
      (Printf.sprintf "%s: %s" route label)
      (fun () ->
        let x = open_operand () in
        let r = reduce x in
        let c = consumer x in
        let%op s = c + r in
        values s)
      ~want:(Array.map want ~f:(fun v -> v +. 10.))
  in
  List.iter routes ~f:(fun (route, reduce) ->
      let read = with_reduction route reduce in
      read "a plain read" ~want:[| 0.; 1.; 2.; 3.; 4. |] (fun x -> TDSL.einsum1 "o => o" x ());
      read "a fixed index" ~want:[| 3. |] (fun x -> TDSL.einsum1 "3 => 0" x ());
      read "a pointwise operation" ~want:[| 0.; 2.; 6.; 12.; 20. |] (fun x ->
          let w = leaf ~o:[ 5 ] ~base:1 () in
          let%op c = x *. w in
          c);
      read "another concatenation of the size" ~want:[| 10. |] (fun x ->
          TDSL.einsum1 "1^4 => " x ());
      read "a capture of its dimension" ~want:[| 0.; 1.; 2.; 3.; 4. |] (fun x ->
          let d = Shape.get_variable_ref "d" in
          let c = TDSL.einsum1 ~capture_dims:[ d ] "d => d" x () in
          Shape.set_dim d 5;
          c);
      read "a capture of its row" ~want:[| 0.; 1.; 2.; 3.; 4. |] (fun x ->
          let r = Shape.get_variable_ref "r" in
          let c = TDSL.einsum1 ~capture_dims:[ r ] "..r.. => ..r.." x () in
          Shape.set_dim r 5;
          c));
  computes "a window, a size it divides"
    (fun () ->
      let x = open_operand () in
      let r = TDSL.einsum1 "3^3 => " x () in
      let c = TDSL.einsum1 "2*o => o" x () in
      let%op s = c + r in
      values s)
    ~want:[| 15.; 17.; 19. |];
  computes "a row's total over two of them"
    (fun () ->
      let x = open_operand () in
      let r = TDSL.einsum1 "2^3, 1^2 => " x () in
      let v = Shape.get_variable_ref "r" in
      let c = TDSL.einsum1 ~capture_dims:[ v ] "..r.. => ..r.." x () in
      Shape.set_dim v 15;
      let%op s = c ++ "a, b => " in
      let%op total = s + r in
      values total)
    ~want:[| 210. |]

let refusals () =
  let reduced x = TDSL.einsum1 "2^3 => " x () in
  refused "a closed operand of another size" ~needle:"mismatch" (fun () ->
      values (reduced (leaf ~o:[ 6 ] ~base:1 ())));
  refused "a fixed index past the axis" ~needle:"At_least_dim" (fun () ->
      let x = open_operand () in
      let r = reduced x in
      let c = TDSL.einsum1 "7 => 0" x () in
      let%op s = c + r in
      values s);
  refused "a window of a stride that does not divide it" ~needle:"incompatible stride" (fun () ->
      let x = open_operand () in
      let r = reduced x in
      let c = TDSL.einsum1 "2*o => o" x () in
      let%op s = c + r in
      values s);
  refused "a capture of another size" ~needle:"Cannot set dimension" (fun () ->
      let x = open_operand () in
      let r = reduced x in
      let d = Shape.get_variable_ref "d" in
      let c = TDSL.einsum1 ~capture_dims:[ d ] "d => d" x () in
      Shape.set_dim d 6;
      let%op s = c + r in
      values s);
  refused "a row's total of another size" ~needle:"Cannot set dimension" (fun () ->
      let x = open_operand () in
      let r = TDSL.einsum1 "2^3, 1^2 => " x () in
      let v = Shape.get_variable_ref "r" in
      let c = TDSL.einsum1 ~capture_dims:[ v ] "..r.. => ..r.." x () in
      Shape.set_dim v 14;
      let%op s = c ++ "a, b => " in
      let%op total = s + r in
      values total);
  (* All-integer components claim the [default] basis, as a plain 5 does: a short-term rule
     (gh-ocannl-2093). Their sum is a solved size, and a solved size meeting another records no
     basis, so a claim-free sum that took [rgb 5] could take [hsv 5] as well, merging the two
     silently. A named segment is a variable, which records the claim it takes, once: the two cases
     after this one. A sound claim-free integer component is gh-ocannl-2095's algebra of
     concatenation bases. *)
  refused "a named basis on the operand" ~needle:"different bases" (fun () ->
      values (reduced (Operation.range ~axis_basis:"rgb" 5)));
  (* Added to an [rgb 3], which a [default] segment would refuse. *)
  computes "a named segment takes the operand's basis"
    (fun () ->
      let c = TDSL.einsum1 "2^a => a" (Operation.range ~axis_basis:"rgb" 5) () in
      let n = Operation.range ~axis_basis:"rgb" 3 in
      let%op s = c + n in
      values s)
    ~want:[| 2.; 4.; 6. |];
  refused "the named segment's claim, against another basis" ~needle:"different bases" (fun () ->
      let c = TDSL.einsum1 "2^a => a" (Operation.range ~axis_basis:"rgb" 5) () in
      let h = Operation.range ~axis_basis:"hsv" 3 in
      let%op s = c + h in
      values s)

let () =
  reductions ();
  consumers ();
  refusals ()
