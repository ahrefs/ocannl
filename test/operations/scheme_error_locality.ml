(* gh-ocannl-1752: which shape errors a shape scheme reports earlier. Each case restates a shape
   error from the test suite (test/einsum/test_inference_counterexamples.ml,
   test_param_shape_error.ml, concat_extraction.ml, op_schemes.ml, ...) as a schemed [%op] function
   applied to an argument, and records, under op_shape_schemes=off|check, where the error surfaces:
   at the definition (the [()] application, which traces the scheme), at the application (by the
   scheme's instantiation, or by the body's own registration), or only at the closure that forcing
   the result's dims runs. The table is the catalogue in docs/shape_inference.md. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

type mode = [ `Off | `Check ]

let modes : mode list = [ `Off; `Check ]
let mode_name = function `Off -> "off" | `Check -> "check"

let leaf ?(batch = []) ?output name =
  match output with
  | Some output ->
      Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()
  | None -> Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ()

(* {2 The functions} *)

(* A bias of the wrong width for the argument. *)
let%op bias5 () x = x + { bias = 0.; o = [ 5 ] }

(* Two parameters whose fixed widths contradict each other, whatever the argument. *)
let%op internal_mismatch () x = x + ({ c3 = 0.; o = [ 3 ] } + { c4 = 0.; o = [ 4 ] })

(* A residual connection around a hidden layer of fixed width. *)
let%op residual4 () x = x + relu (({ w; o = [ 4 ] } * x) + { b = 0.; o = [ 4 ] })

(* A spec that needs exactly two output axes. *)
let%op two_axes () x = x ++ "...|ab => ...|a"

(* A spec naming [v] both a dimension and a row variable. *)
let%op name_clash () x = x ++ "..v.., v => v, ..v.."

(* An empty concatenation component. *)
let%op concat_empty () x = x ++ "0^o => o"

(* A captured concatenation component larger than the argument's axis. *)
let%op concat_exceeds () x =
  let y = x ++ "p^o => o" [ "p" ] in
  [%oc Shape.set_dim p 6];
  y

(* A concatenation whose components only their sum determines. *)
let%op concat_undetermined () x = x ++ "p^o => o"

(* A weight whose output axes nothing determines. *)
let%op forgotten_hidden () x = { w } * x

(* A parameter axis sized only by a fixed-index result axis. *)
let%op fixed_index_param () x =
  let pooled = x ++ "... | h, w, ..c.. => ... | 0, 0, ..c.." in
  ({ w } * pooled) + { b = 0.; o = [ 5 ] }

(* A captured dimension set to a size its other use contradicts. *)
let%op set_dim_conflict () x =
  let y = x ++ "...|a => ...|a" [ "a" ] in
  [%oc Shape.set_dim a 3];
  y + { c4 = 0.; o = [ 4 ] }

(* The same, without a [()] stage: the scheme is traced at the first application. *)
let%op set_dim_twice x =
  let y = x ++ "...|a => ...|a" [ "a" ] in
  [%oc Shape.set_dim a 3];
  [%oc Shape.set_dim a 4];
  y

(* test_inference_counterexamples.ml's cyclic row permutation: satisfiable for an abstract argument
   too ([..r..] = [3]), but the closing policy closes [..r..] empty. *)
let%op rotation () x =
  let y = x + (x ++ "a,b,..r.. => ..r..,b,a" [ "a"; "b"; "r" ]) in
  [%oc Shape.set_dim a 3];
  [%oc Shape.set_dim b 5];
  y

(* A window kernel no operand axis sizes. *)
let%op kernel_unsized () x = x ++ "s, 2*o=+1+k => s, o"

type case = {
  name : string;
  depends : string;
      (** ["arg"]: the error depends on the argument's shape; ["int"]: it does not. *)
  apply : unit -> Tensor.t;  (** Defines the function, then applies it. *)
  define : (unit -> unit) option;  (** The definition alone, for a function with a [()] stage. *)
}

let unit_case name depends def arg =
  {
    name;
    depends;
    define = Some (fun () -> ignore (def () : Tensor.t -> Tensor.t));
    apply =
      (fun () ->
        let f = def () in
        f (arg ()));
  }

let x23 () = leaf ~batch:[ 2 ] ~output:[ 3 ] "x"

let cases =
  [
    unit_case "bias of the wrong width" "arg" bias5 x23;
    unit_case "contradicting parameters" "int" internal_mismatch x23;
    unit_case "residual around a wider layer" "arg" residual4 x23;
    unit_case "too few axes for the spec" "arg" two_axes x23;
    unit_case "dimension and row variable share a name" "int" name_clash x23;
    unit_case "empty concatenation component" "int" concat_empty x23;
    unit_case "concatenation component exceeds the axis" "arg" concat_exceeds (fun () ->
        leaf ~batch:[ 2 ] ~output:[ 5 ] "x");
    unit_case "undetermined concatenation components" "int" concat_undetermined (fun () ->
        leaf ~batch:[ 2 ] ~output:[ 5 ] "x");
    unit_case "forgotten hidden dimension" "int" forgotten_hidden x23;
    unit_case "parameter sized by a fixed index" "int" fixed_index_param (fun () ->
        leaf ~batch:[ 2 ] ~output:[ 3; 3; 4 ] "x");
    unit_case "set_dim against a fixed width" "int" set_dim_conflict x23;
    {
      name = "set_dim twice, no () stage";
      depends = "int";
      define = None;
      apply = (fun () -> set_dim_twice (x23 ()));
    };
    unit_case "cyclic row permutation, abstract argument" "arg" rotation (fun () -> leaf "x");
    unit_case "cyclic row permutation, concrete argument" "arg" rotation (fun () ->
        leaf ~output:[ 3; 5; 3 ] "x");
    unit_case "window kernel nothing sizes" "int" kernel_unsized (fun () ->
        leaf ~output:[ 1; 8 ] "x");
  ]

(* Where an error surfaced, in the order a program meets the places. *)
type phase = Definition | Application | Closure | Accepted

let rank = function Definition -> 0 | Application -> 1 | Closure -> 2 | Accepted -> 3
let equal_phase a b = rank a = rank b
let compare_phase a b = Int.compare (rank a) (rank b)

let phase_name = function
  | Definition -> "definition"
  | Application -> "application"
  | Closure -> "closure"
  | Accepted -> "accepted"

(* Who reported it, from the message's prefix. *)
let reporter msg =
  if String.is_substring msg ~substring:"tracing the shape scheme" then "trace"
  else if String.is_substring msg ~substring:"applying the shape scheme" then "scheme"
  else "body"

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
        match error_of exn with Some msg -> Error (phase, reporter msg, msg) | None -> raise exn)
  in
  let result =
    Result.bind
      (match case.define with Some d -> at Definition d | None -> Ok ())
      ~f:(fun () ->
        (* [apply] defines the function afresh: a definition that passed is traced again there. *)
        Result.bind (at Application case.apply) ~f:(fun y ->
            at Closure (fun () ->
                ignore (Shape.to_dims y.Tensor.shape : int array);
                Set.iter y.Tensor.params ~f:(fun p ->
                    ignore (Shape.to_dims p.Tensor.shape : int array)))))
  in
  match result with
  | Ok () -> (Accepted, "-", None)
  | Error (Closure, _, msg) -> (Closure, "closure", Some msg)
  | Error (phase, who, msg) -> (phase, who, Some msg)

let () =
  let table =
    List.map cases ~f:(fun case ->
        (case, List.map modes ~f:(fun mode -> (mode, outcome mode case))))
  in
  Tensor.For_testing.set_op_shape_schemes None;
  Stdio.printf "case | depends | %s\n" (String.concat ~sep:" | " (List.map modes ~f:mode_name));
  List.iter table ~f:(fun (case, row) ->
      Stdio.printf "%s | %s | %s\n" case.name case.depends
        (String.concat ~sep:" | "
           (List.map row ~f:(fun (_, (phase, who, _)) ->
                match phase with
                | Accepted | Closure -> phase_name phase
                | Definition | Application -> phase_name phase ^ ":" ^ who)));
      List.iter row ~f:(fun (mode, (_, _, msg)) ->
          Option.iter msg ~f:(fun msg ->
              Stdio.eprintf "(not part of the golden) %s [%s]: %s\n" case.name (mode_name mode)
                (String.prefix msg 300))));
  let phase_of mode row =
    let phase, _, _ = List.Assoc.find_exn row mode ~equal:Poly.equal in
    phase
  in
  Verdict.case "where schemes move shape errors" (fun () ->
      p_all "every mode accepts and refuses the same programs" table ~f:(fun (_, row) ->
          List.for_all modes ~f:(fun m ->
              Bool.equal
                (equal_phase (phase_of m row) Accepted)
                (equal_phase (phase_of `Off row) Accepted)));
      p_all "no error surfaces later under a scheme than without one" table ~f:(fun (_, row) ->
          compare_phase (phase_of `Check row) (phase_of `Off row) <= 0);
      p_none "without schemes, no error surfaces at a definition" table ~f:(fun (_, row) ->
          equal_phase (phase_of `Off row) Definition);
      p_exists "some error moves to the definition under a scheme" table ~f:(fun (_, row) ->
          equal_phase (phase_of `Check row) Definition);
      p_exists "some error moves from the closure to the application" table ~f:(fun (_, row) ->
          equal_phase (phase_of `Off row) Closure && equal_phase (phase_of `Check row) Application))
