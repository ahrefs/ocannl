(* gh-ocannl-1803: a window kernel nothing can size is refused at the definition. The refusal reads
   the spec and the step's captures alone, so it must agree with what projection derivation does
   under op_shape_schemes=off: each refused program below fails at the closure there, and each valid
   one, a repaired program included, passes in both modes. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

type mode = [ `Off | `Check ]

let modes : mode list = [ `Off; `Check ]
let mode_name = function `Off -> "off" | `Check -> "check"

let leaf ?(batch = []) ~output name =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

(* {2 Refused: the kernel occurs only as a padded window's} *)

let%op strided () x = x ++ "s, 2*o=+1+k => s, o"
let%op stride_one () x = x ++ "s, o=+k => s, o"
let%op binary () x = x +* "s, 2*o=+k; a => s, o, a" { c = 0.; o = [ 2 ] }

(* The same kernel in two windows: neither sizes the other. *)
let%op two_windows () x =
  let y = { c = 0.; o = [ 1; 8 ] } in
  x +* "s, 2*o=+k; t, 2*p=+k => s, o, t, p" y

let%op ternary () x =
  [%oc
    Ocannl.Operation.einsum3 "s, 2*o=+k; a; b => s, o, a, b" ~grad_spec:Prohibit_grad x
      (leaf ~output:[ 2 ] "a") (leaf ~output:[ 3 ] "b") ()]

(* A block spec: the kernel of a concatenation's component. *)
let%op block () x = [%oc NTDSL.concat "2*o=+k; a => o^a" [| x; leaf ~output:[ 2 ] "y" |] ()]

(* {2 Valid: what the refusal must leave alone} *)

(* The repair the refusal's message names. *)
let%op captured_and_set () x =
  let y = x ++ "s, 2*o=+1+k => s, o" [ "k" ] in
  [%oc Shape.set_dim k 3];
  y

(* The other repair it names. *)
let%op kernel_operand () x = x +* "s, 2*o=+k; k => s, o" { w = 0.; o = [ 3 ] }

(* A reference made outside: its value belongs to the application. *)
let%op external_capture k () x = [%oc NTDSL.einsum1 ~capture_dims:[ k ] "s, 2*o=+1+k => s, o" x ()]

(* A second, unpadded window sizes it from its extents. *)
let%op unpadded_window () x =
  let y = { c = 0.; o = [ 1; 8 ] } in
  x +* "s, 2*o=+k; t, 3*p<+k => s, o, t, p" y

(* An axis named like a kernel, in a spec with no window. *)
let%op no_window () x = x ++ "s, k => s, k"

(* {2 Not the refusal's to decide: left to the closure, whatever it says} *)

(* A capture nothing has set: an application could still. *)
let%op captured_unset () x = x ++ "s, 2*o=+1+k => s, o" [ "k" ]

(* An axis of the result carries it. *)
let%op kernel_in_result () x = x ++ "s, 2*o=+k => s, o, k"

type case = {
  name : string;
  apply : unit -> Tensor.t;  (** Defines the function, then applies it. *)
  define : (unit -> unit) option;  (** The definition alone. *)
}

let x18 () = leaf ~output:[ 1; 8 ] "x"

let unit_case name def =
  {
    name;
    define = Some (fun () -> ignore (def () : Tensor.t -> Tensor.t));
    apply =
      (fun () ->
        let f = def () in
        f (x18 ()));
  }

let refused =
  [
    unit_case "strided padded window" strided;
    unit_case "stride-one padded window" stride_one;
    unit_case "binary einsum" binary;
    unit_case "one kernel in two padded windows" two_windows;
    unit_case "ternary einsum" ternary;
    {
      (unit_case "block spec" block) with
      apply =
        (fun () ->
          let f = block () in
          f (leaf ~output:[ 8 ] "x"));
    };
  ]

let external_case =
  let def () = external_capture (Shape.get_variable_ref "k") () in
  {
    name = "external capture, set after the application";
    define = Some (fun () -> ignore (def () : Tensor.t -> Tensor.t));
    apply =
      (fun () ->
        let k = Shape.get_variable_ref "k" in
        let y = external_capture k () (x18 ()) in
        Shape.set_dim k 3;
        y);
  }

let valid =
  [
    unit_case "captured and set" captured_and_set;
    unit_case "kernel carried by an operand" kernel_operand;
    external_case;
    unit_case "second window unpadded" unpadded_window;
    unit_case "no window" no_window;
  ]

let left_to_closure =
  [
    unit_case "capture left unset" captured_unset;
    unit_case "kernel carried by the result" kernel_in_result;
  ]

type phase = Definition | Application | Closure | Accepted

let phase_name = function
  | Definition -> "definition"
  | Application -> "application"
  | Closure -> "closure"
  | Accepted -> "accepted"

let equal_phase a b = String.equal (phase_name a) (phase_name b)

let error_of = function
  | Row.Shape_error (msg, _) | Tensor.Session_error (msg, _) -> Some msg
  | _ -> None

let outcome (mode : mode) case =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  let at phase f =
    match f () with
    | v -> Ok v
    | exception exn -> (
        match error_of exn with Some msg -> Error (phase, msg) | None -> raise exn)
  in
  let result =
    Result.bind
      (match case.define with Some d -> at Definition d | None -> Ok ())
      ~f:(fun () ->
        Result.bind (at Application case.apply) ~f:(fun y ->
            at Closure (fun () -> ignore (Shape.to_dims y.Tensor.shape : int array))))
  in
  match result with Ok () -> (Accepted, None) | Error (phase, msg) -> (phase, Some msg)

let table cases =
  List.map cases ~f:(fun case ->
      let row = List.map modes ~f:(fun mode -> (mode, outcome mode case)) in
      Stdio.printf "%s | %s\n" case.name
        (String.concat ~sep:" | "
           (List.map row ~f:(fun (mode, (phase, _)) -> mode_name mode ^ ":" ^ phase_name phase)));
      List.iter row ~f:(fun (mode, (_, msg)) ->
          Option.iter msg ~f:(fun msg ->
              Stdio.eprintf "(not part of the golden) %s [%s]: %s\n" case.name (mode_name mode)
                (String.prefix msg 200)));
      (case, row))

let phase_of mode row = fst (List.Assoc.find_exn row mode ~equal:Poly.equal)
let message_of mode row = snd (List.Assoc.find_exn row mode ~equal:Poly.equal)
let contains msg substring = String.is_substring msg ~substring

let () =
  let refused_rows = table refused in
  let valid_rows = table valid in
  let left_rows = table left_to_closure in
  Verdict.case "a kernel nothing can size is refused at the definition" (fun () ->
      p_all "without schemes, projection derivation refuses each" refused_rows ~f:(fun (_, row) ->
          equal_phase (phase_of `Off row) Closure
          && Option.exists (message_of `Off row) ~f:(fun m -> contains m "has no size"));
      p_all "under a scheme, the definition does" refused_rows ~f:(fun (_, row) ->
          equal_phase (phase_of `Check row) Definition);
      p_all "the refusal names the function and keeps the closure's explanation" refused_rows
        ~f:(fun (_, row) ->
          Option.exists (message_of `Check row) ~f:(fun m ->
              contains m "tracing the shape scheme" && contains m "has no size")));
  Verdict.case "a kernel something can size is left alone" (fun () ->
      p_all "accepted in every mode" valid_rows ~f:(fun (_, row) ->
          List.for_all modes ~f:(fun m -> equal_phase (phase_of m row) Accepted));
      p_all "what the spec does not decide keeps its verdict" left_rows ~f:(fun (_, row) ->
          equal_phase (phase_of `Off row) (phase_of `Check row)))

(* A function without a [()] stage is traced at its first application. *)
let%op deferred x = x ++ "s, 2*o=+k => s, o"

let () =
  Tensor.For_testing.set_op_shape_schemes (Some `Check);
  Tensor.unsafe_reinitialize ();
  let refused_at_trace =
    match deferred (x18 ()) with
    | (_ : Tensor.t) -> false
    | exception Row.Shape_error (msg, _) -> contains msg "tracing the shape scheme deferred"
  in
  Verdict.case "no () stage" (fun () ->
      p "the first application traces and refuses, before the body registers" refused_at_trace)

(* The explicit layer has the same contract. *)
let () =
  Tensor.For_testing.set_op_shape_schemes (Some `Check);
  Tensor.unsafe_reinitialize ();
  let body = function [ x ] -> [ TDSL.einsum1 "s, 2*o=+k => s, o" x () ] | _ -> assert false in
  let refused =
    match Tensor.Scheme.trace ~label:"explicit" 1 body with
    | (_ : Tensor.Scheme.t) -> false
    | exception (Row.Shape_error _ as exn) ->
        let text = Stdlib.Printexc.to_string exn in
        contains text "tracing the shape scheme explicit" && contains text "window kernel k"
  in
  Verdict.case "the explicit layer" (fun () ->
      p "Tensor.Scheme.trace refuses it, and the printed error names the function and the kernel"
        refused)

(* A refused definition leaves open work as it found it: inference continues, to the same dims. *)
let () =
  let dims_after mode =
    Tensor.For_testing.set_op_shape_schemes (Some mode);
    Tensor.unsafe_reinitialize ();
    let a = leaf ~output:[ 3 ] "a" and b = leaf ~output:[ 3 ] "b" in
    let%op open_work = a + { c = 0. } in
    (match strided () with (_ : Tensor.t -> Tensor.t) -> () | exception Row.Shape_error _ -> ());
    let%op further = open_work + b in
    Shape.to_dims further.Tensor.shape
  in
  let off = dims_after `Off and check = dims_after `Check in
  Tensor.For_testing.set_op_shape_schemes None;
  Stdio.printf "open work after a refused definition: %s\n"
    (String.concat ~sep:"," (List.map (Array.to_list check) ~f:Int.to_string));
  Verdict.case "open work survives a refused definition" (fun () ->
      p "the dims are those without schemes" (Array.equal Int.equal off check))
