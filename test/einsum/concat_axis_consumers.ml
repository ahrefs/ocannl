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

   Substitution is what sums the components, so two closed rows keep their [Concat]. One is an
   operand's row of integer components alone (["2^3 => "]), with no variable to substitute: its own
   step does not derive, which is not pinned here. The other has a symbolic component, with no [Dim]
   to sum to. No such program ever ran: the concatenation's own step died on an assertion in
   [Row.get_dim_padding], and a slice or a fixed index reading it, derived first, got the verifier's
   refusal about a window missing a solved index, which says nothing of what is wrong. Where the
   segments after a symbolic component start depends on the launch-time extent, so it stays refused,
   by [Row.get_proj_equations], under a message naming that.

   gh-ocannl-2017: the forms with no meaning used to compute one. A concatenated axis is one axis of
   alternatives, so a tensor needing two of its components got no value at all (all zeros), and a
   result axis no operand names took the reduced operand in every cell, or crashed on an open
   operand. They are refused, whatever the extents and however the operand is closed, and what they
   sit next to is executed against values worked out by hand: reductions over a concatenated operand
   axis, slices, joins and reorderings.

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

let capture name n =
  let r = Shape.get_variable_ref name in
  Shape.set_dim r n;
  r

(* A refusal with [needle] in its message: anything else, a value above all, fails the claim. *)
let refused ?(exn_message = function Row.Shape_error (msg, _) -> Some msg | _ -> None) label ~needle
    make =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      match make () with
      | (_ : float array) -> pf "%s: refused" label false
      | exception e -> (
          match exn_message e with
          | Some msg ->
              Stdio.printf "%s: %s\n" label msg;
              pf "%s: the refusal says why" label (String.is_substring msg ~substring:needle)
          | None ->
              Stdio.printf "%s: raised %s\n" label (Exn.to_string e);
              pf "%s: refused by the intended check" label false))

let refused_lowering =
  refused ~exn_message:(function Utils.User_error msg -> Some msg | _ -> None)

let open_leaf () = NTDSL.term ~fetch_op:(Constant 2.) ()
let vec ?(base = 1) n = leaf ~o:[ n ] ~base ()
let two_components = "one tensor's axes need both"
let unnamed = "no operand names a component"
let out_of_order = "not in the order of the segments"
let unreached = "no segment of a concatenated axis reaches"

let two_components_at_once () =
  refused "the result needs both components, a binary einsum" ~needle:two_components (fun () ->
      values (NTDSL.einsum "p^q; 2*p => p, q" (vec 5) (vec ~base:10 4) ()));
  refused "the result needs both components, through %op" ~needle:two_components (fun () ->
      let a = vec 5 and c = vec ~base:10 4 in
      let%op t = a +* "p^q; 2*p => p, q" c in
      values t);
  refused "the result needs both components, sized by captures" ~needle:two_components (fun () ->
      values (NTDSL.einsum1 ~capture_dims:[ capture "p" 2; capture "q" 3 ] "p^q => p, q" (vec 5) ()));
  refused "a window needs one of them" ~needle:two_components (fun () ->
      values (NTDSL.einsum1 "2*a, b => a^b" (leaf ~o:[ 4; 3 ] ~base:1 ()) ()));
  refused "an operand needs both components" ~needle:two_components (fun () ->
      values (NTDSL.einsum1 "a, b => a^b" (leaf ~o:[ 2; 3 ] ~base:1 ()) ()));
  refused "an operand needs both, and one component is a single cell" ~needle:two_components
    (fun () -> values (NTDSL.einsum1 "a, b => a^b" (leaf ~o:[ 2; 1 ] ~base:1 ()) ()));
  refused "an operand needs two of three" ~needle:two_components (fun () ->
      values (NTDSL.concat "a, b; c => a^b^c" [| leaf ~o:[ 2; 3 ] ~base:1 (); vec ~base:10 2 |] ()));
  refused "an operand needs both, another fills one" ~needle:two_components (fun () ->
      values (NTDSL.concat "a, b; b => a^b" [| leaf ~o:[ 2; 3 ] ~base:1 (); vec ~base:10 3 |] ()))

(* The operand summed is 10, so a plain axis would hold 10 in every cell. *)
let unnamed_result () =
  refused "integers alone, a closed operand" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 "a => 2^3" (vec 4) ()));
  refused "integers alone, the operand as long as the axis" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 "a => 2^3" (vec 5) ()));
  refused "integers alone, an open operand" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 "a => 2^3" (open_leaf ()) ()));
  refused "integers alone, through %op" ~needle:unnamed (fun () ->
      let x = vec 4 in
      let%op t = x ++ "a => 2^3" in
      values t);
  refused "integers alone, no operand axis" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 " => 2^3" (open_leaf ()) ()));
  refused "beside a plain axis" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 "a, c => 2^3, c" (leaf ~o:[ 2; 2 ] ~base:1 ()) ()));
  refused "integers both ways" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 "2^3 => 2^3" (vec 5) ()));
  refused "two operands" ~needle:unnamed (fun () ->
      values (NTDSL.concat "a; b => 2^3" [| vec 2; vec ~base:10 3 |] ()));
  refused "a captured component" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 ~capture_dims:[ capture "b" 3 ] "a => 2^b" (vec 4) ()));
  refused "captured components" ~needle:unnamed (fun () ->
      values (NTDSL.einsum1 ~capture_dims:[ capture "a" 2; capture "b" 3 ] "c => a^b" (vec 4) ()));
  refused "other components on both sides" ~needle:unnamed (fun () ->
      values
        (NTDSL.einsum1
           ~capture_dims:[ capture "p" 2; capture "q" 3; capture "r" 2; capture "s" 3 ]
           "p^q => r^s" (vec 5) ()));
  refused "other components, other counts" ~needle:unnamed (fun () ->
      values
        (NTDSL.einsum1
           ~capture_dims:
             [ capture "a" 1; capture "b" 2; capture "c" 2; capture "d" 2; capture "e" 3 ]
           "a^b^c => d^e" (vec 5) ()));
  refused "through %cd" ~needle:unnamed (fun () ->
      let x = vec 4 and tgt = vec ~base:50 5 in
      let comp = [%cd tgt =: id x ~logic:"a => 2^3"] in
      ignore (comp : Ir.Assignments.comp);
      [||])

let operand_order () =
  refused "swapped operands" ~needle:out_of_order (fun () ->
      values (NTDSL.concat "b; a => a^b" [| vec 2; vec ~base:10 3 |] ()));
  refused "swapped single cells, which nothing else tells apart" ~needle:out_of_order (fun () ->
      values (NTDSL.concat "b; a => a^b" [| vec 1; vec ~base:10 1 |] ()));
  refused "swapped, one of them sliced" ~needle:out_of_order (fun () ->
      values (NTDSL.concat "b; 1^a => a^b" [| vec 2; vec ~base:10 4 |] ()));
  computes "in the order of the segments, the same operands"
    (fun () -> values (NTDSL.concat "a; b => a^b" [| vec ~base:10 3; vec 2 |] ()))
    ~want:[| 10.; 11.; 12.; 1.; 2. |]

let unreached_operand () =
  refused_lowering "the result needs the segment only the other operand has" ~needle:unreached
    (fun () -> values (NTDSL.concat "a; b => a^b, b" [| vec 2; vec ~base:10 3 |] ()));
  refused_lowering "the result and the operand need different segments" ~needle:unreached (fun () ->
      values
        (NTDSL.einsum1
           ~capture_dims:[ capture "a" 2 ]
           "a^b, b => a"
           (leaf ~o:[ 5; 3 ] ~base:1 ())
           ()));
  refused_lowering "a window over the segment the result does not need" ~needle:unreached (fun () ->
      values (NTDSL.concat "p^q; 2*p => q" [| vec 5; vec ~base:10 4 |] ()));
  (* Both operands can fill the second segment, and the positions used to pick one of them: the
     second operand's sum went in there, the first operand's first segment summed once per cell of
     the second operand. *)
  refused_lowering "two operands viable for one segment" ~needle:"Ambiguous" (fun () ->
      values
        (NTDSL.concat
           ~capture_dims:[ capture "a" 2 ]
           "a^b; c => a^b"
           [| vec 5; vec ~base:10 2 |]
           ()))

(* [1 2 3 4 5] reduced over all five cells is 15; cell [p] of the slice, times cell [2 * p] of [10
   11 12 13]. *)
let neighbours () =
  computes "an operand's integer components, reduced"
    (fun () -> values (NTDSL.einsum1 "2^3 => " (vec 5) ()))
    ~want:[| 15. |];
  computes "an operand's integer components, reduced into a plain axis"
    (fun () -> values (NTDSL.einsum1 ~capture_dims:[ capture "a" 2 ] "2^3 => a" (vec 5) ()))
    ~want:[| 15.; 15. |];
  computes "an operand's captured components, reduced"
    (fun () ->
      values (NTDSL.einsum1 ~capture_dims:[ capture "p" 2; capture "q" 3 ] "p^q => " (vec 5) ()))
    ~want:[| 15. |];
  computes "a plain axis repeats a reduced operand, which is what a => 2^3 is refused in favour of"
    (fun () -> values (NTDSL.einsum1 ~capture_dims:[ capture "b" 5 ] "a => b" (vec 4) ()))
    ~want:[| 10.; 10.; 10.; 10.; 10. |];
  computes "one component named by the result and the other operand"
    (fun () ->
      values
        (NTDSL.einsum ~capture_dims:[ capture "q" 3 ] "p^q; 2*p => p" (vec 5) (vec ~base:10 4) ()))
    ~want:[| 10.; 24. |];
  computes "a diagonal of a slice"
    (fun () ->
      values
        (NTDSL.einsum1
           ~capture_dims:[ capture "a" 2 ]
           "a^b, b => b"
           (leaf ~o:[ 5; 3 ] ~base:1 ())
           ()))
    ~want:[| 7.; 11.; 15. |];
  computes "the same concatenation on two axes"
    (fun () ->
      values
        (NTDSL.einsum1
           ~capture_dims:[ capture "a" 2 ]
           "a^b, a^b => a^b"
           (leaf ~o:[ 5; 5 ] ~base:1 ())
           ()))
    ~want:[| 1.; 7.; 13.; 19.; 25. |];
  computes "the integers of a slice are skipped on both sides"
    (fun () -> values (NTDSL.einsum1 "2^b => 2^b" (vec 5) ()))
    ~want:[| 0.; 0.; 3.; 4.; 5. |];
  computes "and move with the segment"
    (fun () -> values (NTDSL.einsum1 "2^b => b^2" (vec 5) ()))
    ~want:[| 3.; 4.; 5.; 0.; 0. |];
  computes "two segments swapped"
    (fun () -> values (NTDSL.einsum1 ~capture_dims:[ capture "a" 2 ] "a^b => b^a" (vec 5) ()))
    ~want:[| 3.; 4.; 5.; 1.; 2. |];
  computes "a segment between skipped ones, an unnamed component among them"
    (fun () -> values (NTDSL.einsum1 ~capture_dims:[ capture "b" 1 ] "a => 2^a^3^b" (vec 2) ()))
    ~want:[| 0.; 0.; 1.; 2.; 0.; 0.; 0.; 0. |];
  computes "a pointwise product of two concatenated operands"
    (fun () ->
      values
        (NTDSL.einsum ~capture_dims:[ capture "a" 2 ] "a^b; a^b => a^b" (vec 5) (vec ~base:10 5) ()))
    ~want:[| 10.; 22.; 36.; 52.; 70. |];
  computes "two operands multiplied into one segment"
    (fun () -> values (NTDSL.einsum "a; a => a^3" (vec 2) (vec ~base:10 2) ()))
    ~want:[| 10.; 22.; 0.; 0.; 0. |];
  computes "joining a single cell and a vector"
    (fun () -> values (NTDSL.concat "a; b => a^b" [| vec 1; vec ~base:10 3 |] ()))
    ~want:[| 1.; 10.; 11.; 12. |];
  computes "joining a vector and a single cell"
    (fun () -> values (NTDSL.concat "a; b => a^b" [| vec ~base:10 3; vec 1 |] ()))
    ~want:[| 10.; 11.; 12.; 1. |];
  computes "joining two single cells"
    (fun () -> values (NTDSL.concat "a; b => a^b" [| vec 1; vec ~base:10 1 |] ()))
    ~want:[| 1.; 10. |]

(* No reinitialization between the refusals and the computation, which uses operands that predate
   them. *)
let refusal_leaves_inference_usable () =
  Verdict.case "a refused concatenation leaves inference usable" (fun () ->
      Tensor.unsafe_reinitialize ();
      let x = vec 4 and y = leaf ~o:[ 2; 3 ] ~base:1 () in
      let shape_refused make =
        match make () with (_ : Tensor.t) -> false | exception Row.Shape_error _ -> true
      in
      p "the result concatenation is refused"
        (shape_refused (fun () -> NTDSL.einsum1 "a => 2^3" x ()));
      p "the two components are refused"
        (shape_refused (fun () -> NTDSL.einsum1 "a, b => a^b" y ()));
      let label = "the same operands, a valid operation" in
      pf_close "%s" label
        (values (NTDSL.einsum1 "a => 2^a^1" x ()))
        [| 0.; 0.; 1.; 2.; 3.; 4.; 0. |]
        ~tol:(fun _ -> 0.))

let () =
  consumers ();
  same_step ();
  symbolic_component ();
  two_components_at_once ();
  unnamed_result ();
  operand_order ();
  unreached_operand ();
  neighbours ();
  refusal_leaves_inference_usable ()
