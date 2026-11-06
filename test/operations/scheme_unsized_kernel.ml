(* gh-ocannl-1803, gh-ocannl-1983: a window kernel nothing can size is refused where the operation
   is built, naming the kernel and the spec. The refusal reads the spec and the step's captures
   alone, so it holds in every mode: without a scheme at the application (the body's registration),
   under one at the definition, since a trace runs the body. It writes nothing first, so it taints
   no session. Each valid program, a repaired one included, passes in both modes. What the spec does
   not decide stays with projection derivation, at the closure, which refuses with the same message:
   a capture nothing set, and a kernel only the result carries, unless that axis is sized. *)

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

(* An axis of the result carries it, and something sizes that axis. *)
let%op result_axis_captured () x =
  let y = x ++ "s, 2*o=+k => s, o, k" [ "k" ] in
  [%oc Shape.set_dim k 3];
  y

let%op result_axis_consumer () x =
  let y = x ++ "s, 2*o=+k => s, o, k" in
  y +* "s, o, k; s, o, k => s, o, k" { c = 0.; o = [ 1; 4; 3 ] }

(* An explicit 1 is a size, not closure's guess. *)
let%op result_axis_set_to_one () x =
  let y = x ++ "s, 2*o=+k => s, o, k" [ "k" ] in
  [%oc Shape.set_dim k 1];
  y

(* {2 Not the refusal's to decide: left to the closure, whatever it says} *)

(* A capture nothing sets. The reference is made inside the body, so nothing else ever will, but the
   refusal reads whether a capture exists, not what becomes of it. *)
let%op captured_unset () x = x ++ "s, 2*o=+1+k => s, o" [ "k" ]

(* An axis of the result carries it, and nothing sizes that axis: closure would guess the kernel. *)
let%op kernel_in_result () x = x ++ "s, 2*o=+k => s, o, k"

(* The same at stride 1, where the window's extent equation is the identity. *)
let%op kernel_in_result_stride_one () x = x ++ "s, o=+k => s, o, k"

(* A broadcast sizes nothing: the result's axis closes to 1 and broadcasts into the consumer's 3. *)
let%op kernel_in_result_broadcast () x =
  let y = x ++ "s, 2*o=+k => s, o, k" in
  y + { c = 0.; o = [ 1; 4; 3 ] }

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
    unit_case "result axis captured and set" result_axis_captured;
    unit_case "result axis sized by an einsum consumer" result_axis_consumer;
    unit_case "result axis set to 1" result_axis_set_to_one;
  ]

let left_to_closure = [ unit_case "capture left unset" captured_unset ]

let result_carried =
  [
    unit_case "kernel carried by the result" kernel_in_result;
    unit_case "kernel carried by the result, stride 1" kernel_in_result_stride_one;
    unit_case "kernel carried by the result, broadcast consumer" kernel_in_result_broadcast;
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
  let carried_rows = table result_carried in
  Verdict.case "a kernel nothing can size is refused where the operation is built" (fun () ->
      p_all "without schemes, the application refuses each" refused_rows ~f:(fun (_, row) ->
          equal_phase (phase_of `Off row) Application);
      p_all "under a scheme, the definition does" refused_rows ~f:(fun (_, row) ->
          equal_phase (phase_of `Check row) Definition);
      p_all "either way the message names the kernel and the spec" refused_rows ~f:(fun (_, row) ->
          List.for_all modes ~f:(fun m ->
              Option.exists (message_of m row) ~f:(fun msg ->
                  contains msg "the window kernel k has no size in einsum spec \""
                  && contains msg "Shape.set_dim" && contains msg "kernel operand")));
      p_all "the refusal under a scheme names the function" refused_rows ~f:(fun (_, row) ->
          Option.exists (message_of `Check row) ~f:(fun m -> contains m "tracing the shape scheme"));
      p_all "a refused application taints no session" refused ~f:(fun case ->
          ignore (outcome `Off case : phase * string option);
          Option.is_none (Shape.taint_provenance ())));
  Verdict.case "a kernel only the result carries, unsized" (fun () ->
      p_all "is refused at the closure in every mode, not at construction" carried_rows
        ~f:(fun (_, row) -> List.for_all modes ~f:(fun m -> equal_phase (phase_of m row) Closure));
      p_all "naming the kernel and the spec, whatever the stride" carried_rows ~f:(fun (_, row) ->
          List.for_all modes ~f:(fun m ->
              Option.exists (message_of m row) ~f:(fun msg ->
                  contains msg "the window kernel k has no size in einsum spec \"s, ")));
      p_all "never as a clash of bases" carried_rows ~f:(fun (_, row) ->
          not
            (List.exists modes ~f:(fun m ->
                 Option.exists (message_of m row) ~f:(fun msg -> contains msg "different bases")))));
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

(* A refused construction leaves open work as it found it: inference continues, to the same dims,
   whether the application refused it or the definition did. *)
let () =
  let dims_after mode =
    Tensor.For_testing.set_op_shape_schemes (Some mode);
    Tensor.unsafe_reinitialize ();
    let a = leaf ~output:[ 3 ] "a" and b = leaf ~output:[ 3 ] "b" in
    let%op open_work = a + { c = 0. } in
    (match strided () (x18 ()) with (_ : Tensor.t) -> () | exception Row.Shape_error _ -> ());
    let%op further = open_work + b in
    Shape.to_dims further.Tensor.shape
  in
  let off = dims_after `Off and check = dims_after `Check in
  Tensor.For_testing.set_op_shape_schemes None;
  Stdio.printf "open work after a refused construction: %s\n"
    (String.concat ~sep:"," (List.map (Array.to_list check) ~f:Int.to_string));
  Verdict.case "open work survives a refused construction" (fun () ->
      p "the dims are those without schemes" (Array.equal Int.equal off check))
