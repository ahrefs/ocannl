(* gh-ocannl-1752: which shape errors a shape scheme reports earlier. Each case restates a shape
   error from the test suite (test/einsum/test_inference_counterexamples.ml,
   test_param_shape_error.ml, concat_extraction.ml, op_schemes.ml, ...) as a schemed [%op] function
   applied to an argument, and records, under op_shape_schemes=off|check, where the error surfaces:
   at the definition (the [()] application, which traces the scheme), at the application (by the
   scheme's instantiation, or by the body's own registration), or only at the closure that forcing
   the result's dims runs. The table is the catalogue in docs/shape_inference.md.

   gh-ocannl-1814: the catalogue again with each function nested in a schemed one, whose check
   stands for the nested application's. The second table puts it beside the control that checks the
   nested application as well, and beside an opted-out outer function, which vouches for nothing. *)

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

(* A trailing equality is checked by the scheme before the inline body's next closure. *)
let%op shape_equal5 () x =
  let y = x ++ "...|... => ...|..." in
  let reference =
    NTDSL.init ~l:"shape_five" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 5 ] ~f:(fun _ -> 0.) ()
  in
  [%oc Shape.infer_equal y.Tensor.shape reference.Tensor.shape];
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

(* [def]'s closure, applied in the body of a schemed function, and of an opted-out one. *)
let%op nested def () =
  let f = def () in
  fun x -> f x

let%op nested_plain def () =
  let f = def () in
  fun x -> f x
[@@ocannl.no_scheme]

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

(* The catalogue, each function passed through [wrap]. *)
let catalogue wrap =
  let unit_case name depends def arg = unit_case name depends (wrap def) arg in
  [
    unit_case "bias of the wrong width" "arg" bias5 x23;
    unit_case "contradicting parameters" "int" internal_mismatch x23;
    unit_case "residual around a wider layer" "arg" residual4 x23;
    unit_case "too few axes for the spec" "arg" two_axes x23;
    unit_case "dimension and row variable share a name" "int" name_clash x23;
    unit_case "empty concatenation component" "int" concat_empty x23;
    unit_case "concatenation component exceeds the axis" "arg" concat_exceeds (fun () ->
        leaf ~batch:[ 2 ] ~output:[ 5 ] "x");
    unit_case "trailing equality with a fixed-width tensor" "arg" shape_equal5 x23;
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
      apply = (fun () -> wrap (fun () -> set_dim_twice) () (x23 ()));
    };
    unit_case "cyclic row permutation, abstract argument" "arg" rotation (fun () -> leaf "x");
    unit_case "cyclic row permutation, concrete argument" "arg" rotation (fun () ->
        leaf ~output:[ 3; 5; 3 ] "x");
    unit_case "window kernel nothing sizes" "int" kernel_unsized (fun () ->
        leaf ~output:[ 1; 8 ] "x");
  ]

let cases = catalogue Fn.id

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
  (* gh-ocannl-1814. Rows in the catalogue's order; "re-checked" is the control. *)
  let nested_outcome ?(recheck = false) case =
    let run () = outcome `Check case in
    if recheck then Tensor.For_testing.with_nested_rechecks run else run ()
  in
  let nested_table =
    List.map (catalogue nested) ~f:(fun case ->
        (case, nested_outcome case, nested_outcome ~recheck:true case))
  in
  let plain_table = List.map (catalogue nested_plain) ~f:(fun case -> nested_outcome case) in
  Tensor.For_testing.set_op_shape_schemes None;
  let cell (phase, who, _) =
    match phase with
    | Accepted | Closure -> phase_name phase
    | Definition | Application -> phase_name phase ^ ":" ^ who
  in
  Stdio.printf "nested case | depends | check | re-checked | under an opted-out function\n";
  List.iter2_exn nested_table plain_table ~f:(fun (case, skipped, control) plain ->
      Stdio.printf "%s | %s | %s | %s | %s\n" case.name case.depends (cell skipped) (cell control)
        (cell plain));
  let phase_in (phase, _, _) = phase in
  (* A message spells the session's variable ids ([$214]), which go on growing between the runs. *)
  let message_in (_, _, msg) =
    let without_ids m =
      let after_dollar = ref false in
      String.filter m ~f:(fun c ->
          after_dollar := (!after_dollar && Char.is_digit c) || Char.equal c '$';
          not !after_dollar)
    in
    Option.map msg ~f:without_ids
  in
  let named_by who (_, reporter, _) = String.equal who reporter in
  let names name o = Option.exists (message_in o) ~f:(String.is_substring ~substring:name) in
  let refused_by_scheme = List.filter nested_table ~f:(fun (_, o, _) -> named_by "scheme" o) in
  Verdict.case "a nested application's errors under the outer check" (fun () ->
      p_all "skipping the nested checks moves no error: phase, reporter and message" nested_table
        ~f:(fun (_, skipped, control) ->
          equal_phase (phase_in skipped) (phase_in control)
          && String.equal (cell skipped) (cell control)
          && Option.equal String.equal (message_in skipped) (message_in control));
      p_all "nesting accepts and refuses what the function alone does"
        (List.zip_exn table nested_table) ~f:(fun ((_, row), (_, nested, _)) ->
          Bool.equal
            (equal_phase (phase_of `Check row) Accepted)
            (equal_phase (phase_in nested) Accepted));
      p_all "no nested error surfaces later than the function's own"
        (List.zip_exn table nested_table) ~f:(fun ((_, row), (_, nested, _)) ->
          compare_phase (phase_in nested) (phase_of `Check row) <= 0);
      p "some nested error is refused by the outer application's scheme"
        (not (List.is_empty refused_by_scheme));
      p_all "each such error names the outer function" refused_by_scheme ~f:(fun (_, o, _) ->
          names "applying the shape scheme nested to" o);
      p_exists "an opted-out outer function leaves the nested function's own name" plain_table
        ~f:(fun o ->
          named_by "scheme" o
          && (not (names "scheme nested" o))
          && names "applying the shape scheme" o));
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
