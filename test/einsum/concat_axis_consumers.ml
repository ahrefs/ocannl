(* gh-ocannl-1971: gh-ocannl-1889 made [Row.solve_proj_equations] verify the equations between a
   window or a concatenation and a solved index, or another window, and a [Concat] index never
   structurally equals a fixed or affine one. So the question was which programs bring a [Concat]
   side to that check: slicing a concatenated batch axis, a fixed index on a concatenated axis, a
   window over one.

   Through [Shape], a valid program does not. A concatenation is a [Concat] dim of the step that
   writes it, against a tensor axis, which makes the tensor axis the concat target; once the shape
   closes, the tensor's axis is a plain [Dim] of the summed extent, so every later access is an
   access to a plain axis. The first half executes each such consumer against the concatenation
   built by hand, for the concatenation as written, read through a pointwise copy (whose row the
   solver filled in) and made by a shape scheme, whose row is instantiated. Then the forms holding a
   window or a fixed index in the step that concatenates.

   Substitution is what sums the components, so two closed rows keep their [Concat]. One is a row of
   integer components alone (["=> 2^3"]), with no variable to substitute: its own step does not
   derive, which is not pinned here. The other has a symbolic component, with no [Dim] to sum to. No
   such program ever ran: the concatenation's own step died on an assertion in
   [Row.get_dim_padding], and a slice or a fixed index reading it, derived first, got the verifier's
   refusal about a window missing a solved index, which says nothing of what is wrong. Where the
   segments after a symbolic component start depends on the launch-time extent, so it stays refused,
   by [Row.get_proj_equations], under a message naming that.

   These are claims about projection derivation, so every execution context is [Context.cpu ()]. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims
module IDX = Train.IDX

let leaf ?(grad_spec = Tensor.Prohibit_grad) ?(b = []) ~o ~base () =
  let n = List.fold (b @ o) ~init:1 ~f:( * ) in
  Tensor.ndarray ~grad_spec
    (Array.init n ~f:(fun k -> Float.of_int (base + k)))
    ~batch_dims:b ~input_dims:[] ~output_dims:o ()

let values ?bindings ?launch_values t =
  Train.set_materialized t.Tensor.value;
  Context.get_values (Train.forward_once ?bindings ?launch_values (Context.cpu ()) t) t.Tensor.value

(* Every value is a small integer copied or left at the zero initialization, so equality is
   exact. *)
let computes label make ~want =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      pf_close "%s" label (make ()) want ~tol:(fun _ -> 0.))

let%op scheme_concat () a c = (a, c) ++^ "x; y => x^y"

let routes =
  [
    ( "as written",
      fun a c ->
        let%op r = (a, c) ++^ "x; y => x^y" in
        r );
    ( "through a pointwise copy",
      fun a c ->
        let%op r = (a, c) ++^ "x; y => x^y" in
        let%op copy = r *. 1.0 in
        copy );
    ("made by a shape scheme", fun a c -> scheme_concat () a c);
  ]

let batch_routes =
  [
    ( "as written",
      fun a c ->
        let%op r = (a, c) ++^ "x|i; y|i => x^y|i" in
        r );
    ( "through a pointwise copy",
      fun a c ->
        let%op r = (a, c) ++^ "x|i; y|i => x^y|i" in
        let%op copy = r *. 1.0 in
        copy );
  ]

let consumers () =
  let concatenated = [| 10.; 11.; 30.; 31.; 32.; 33. |] in
  let operands () = (leaf ~o:[ 2 ] ~base:10 (), leaf ~o:[ 4 ] ~base:30 ()) in
  List.iter routes ~f:(fun (route, concat) ->
      computes
        (Printf.sprintf "%s: every axis read as a plain axis" route)
        (fun () ->
          let a, c = operands () in
          let r = concat a c in
          let%op s = r ++ "o => o" in
          values s)
        ~want:concatenated;
      Array.iteri concatenated ~f:(fun k cell ->
          computes
            (Printf.sprintf "%s: a fixed index, %d" route k)
            (fun () ->
              let a, c = operands () in
              values (TDSL.einsum1 (Printf.sprintf "%d => 0" k) (concat a c) ()))
            ~want:[| cell |]);
      computes
        (Printf.sprintf "%s: a window over it" route)
        (fun () ->
          let a, c = operands () in
          let r = concat a c in
          let%op s = r ++ "2*o => o" in
          values s)
        ~want:[| 10.; 30.; 32. |]);
  (* Rows of three: two rows from [a], three from [c]. *)
  let rows = Array.init 15 ~f:(fun k -> Float.of_int (if k < 6 then 10 + k else 30 + k - 6)) in
  List.iter batch_routes ~f:(fun (route, concat) ->
      for row = 0 to 4 do
        computes
          (Printf.sprintf "%s: slicing the concatenated batch axis at %d" route row)
          (fun () ->
            let a = leaf ~b:[ 2 ] ~o:[ 3 ] ~base:10 () in
            let c = leaf ~b:[ 3 ] ~o:[ 3 ] ~base:30 () in
            let n, bindings = IDX.get_static_symbol ~static_range:5 IDX.empty in
            (* A slice is a view of its parent: read a copy. *)
            let r = concat a c in
            let%op s = (r @| n) *. 1.0 in
            values ~bindings ~launch_values:[ (n, row) ] s)
          ~want:(Array.sub rows ~pos:(3 * row) ~len:3)
      done)

let same_step () =
  computes "a window as one concatenated component"
    (fun () ->
      let a = leaf ~o:[ 4 ] ~base:10 () and c = leaf ~o:[ 3 ] ~base:30 () in
      let%op s = (a, c) ++^ "2*p; q => p^q" in
      values s)
    ~want:[| 10.; 12.; 30.; 31.; 32. |];
  computes "a fixed index beside the concatenated axis"
    (fun () ->
      let a = leaf ~o:[ 2; 2 ] ~base:10 () and c = leaf ~o:[ 3; 2 ] ~base:30 () in
      let%op s = (a, c) ++^ "x, 1; y, 1 => x^y" in
      values s)
    ~want:[| 11.; 13.; 31.; 33.; 35. |];
  computes "a result-side window over an extracted component"
    (fun () ->
      let a = leaf ~o:[ 5 ] ~base:10 () in
      let%op s = a ++ "p^o => 2*o" [ "o" ] in
      Shape.set_dim o 2;
      values s)
    ~want:[| 13.; 0.; 14.; 0. |]

let symbolic_output seq =
  let%op x = { x = 0.5 } in
  let%op y = (2. *. x) ++ "s => s" [ "s" ] in
  Shape.set_sym_dim s seq;
  y

let symbolic_batch seq =
  let%op x = { x = 0.5 } in
  let%op y = (2. *. x) ++ "s|i => s|i" [ "s" ] in
  Shape.set_sym_dim s seq;
  y

let refuses label make =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      match make () with
      | (_ : float array) -> pf "%s: refused" label false
      | exception Row.Shape_error (msg, _) ->
          Stdio.printf "%s: Shape_error: %s\n" label msg;
          pf "%s: the refusal names the symbolic component" label
            (String.is_substring msg ~substring:"symbolic extent"))

let symbolic_component () =
  let concatenated seq =
    let y = symbolic_output seq in
    let c = leaf ~grad_spec:Tensor.Require_grad ~o:[ 2 ] ~base:30 () in
    let%op r = (y, c) ++^ "a; b => a^b" in
    r
  in
  refuses "a symbolic component, computed" (fun () ->
      let seq, bindings = IDX.get_static_symbol ~static_range:4 IDX.empty in
      values ~bindings ~launch_values:[ (seq, 2) ] (concatenated seq));
  refuses "a symbolic component, at a fixed index" (fun () ->
      let seq, bindings = IDX.get_static_symbol ~static_range:4 IDX.empty in
      let r = concatenated seq in
      let%op s = r ++ "2 => 0" in
      values ~bindings ~launch_values:[ (seq, 2) ] s);
  refuses "a symbolic component, sliced" (fun () ->
      let seq, bindings = IDX.get_static_symbol ~static_range:4 IDX.empty in
      let n, bindings = IDX.get_static_symbol bindings in
      let y = symbolic_batch seq in
      let c = leaf ~grad_spec:Tensor.Require_grad ~b:[ 3 ] ~o:[ 2 ] ~base:30 () in
      let%op r = (y, c) ++^ "a|i; b|i => a^b|i" in
      let%op s = (r @| n) *. 1.0 in
      values ~bindings ~launch_values:[ (seq, 2); (n, 1) ] s)

let () =
  consumers ();
  same_step ();
  symbolic_component ()
