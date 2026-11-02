(* gh-ocannl-1799: a shape error raised inside a shape-scheme trace or check prints readably -- what
   [Printexc.to_string], and so the toplevel, shows for it. Each case is a tutorial-style mistake
   (test/operations/reusable_tensor_programs.ml, scheme_error_locality.ml): the rendering names the
   function, the conflicting sizes and the tensors involved with their axis kinds, and spells none
   of the reserved-range ids a trace or check draws its variables and tensors from. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

let input ?(batch = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

let%op mlp_layer ~hid () x = relu (({ w; o = [ hid ] } * x) + { b = 0.; o = [ hid ] })

let%op residual_block ~hid () =
  let layer = mlp_layer ~hid () in
  fun x -> x + layer x

(* The second bias has the hidden size instead of the output size: no argument can fix it. *)
let%op two_layer ~hid ~out () x =
  let h = relu (({ w1; o = [ hid ] } * x) + { b1 = 0.; o = [ hid ] }) in
  ({ w2; o = [ out ] } * h) + { b2 = 0.; o = [ hid ] }

let%op two_layer_unchecked ~hid ~out () x =
  let h = relu (({ w1; o = [ hid ] } * x) + { b1 = 0.; o = [ hid ] }) in
  ({ w2; o = [ out ] } * h) + { b2 = 0.; o = [ hid ] }
[@@ocannl.no_scheme]

let%op two_axes () x = x ++ "...|ab => ...|a"
let%op name_clash () x = x ++ "..v.., v => v, ..v.."
let%op concat_empty () x = x ++ "0^o => o"

(* Only the components' sum determines them: found when the closure forces the result's dims. *)
let%op concat_undetermined () x = x ++ "p^o => o"

type case = { name : string; in_scheme : bool; run : unit -> unit }

let cases =
  [
    {
      name = "a mistake no argument can fix, at the trace";
      in_scheme = true;
      run = (fun () -> ignore (two_layer ~hid:8 ~out:2 () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "the same mistake, found by a definition check";
      in_scheme = true;
      run =
        (fun () ->
          ignore
            (Tensor.Scheme.check_definition ~name:"two_layer" (fun () ->
                 ignore (two_layer ~hid:8 ~out:2 () : Tensor.t -> Tensor.t))
              : string list));
    };
    {
      name = "an argument the scheme refuses, at the application";
      in_scheme = true;
      run = (fun () -> ignore (residual_block ~hid:4 () (input ~batch:[ 2 ] "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "an argument with too few axes for the spec";
      in_scheme = true;
      run = (fun () -> ignore (two_axes () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "a dimension and a row variable sharing a name";
      in_scheme = true;
      run = (fun () -> ignore (name_clash () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "an empty concatenation component";
      in_scheme = true;
      run = (fun () -> ignore (concat_empty () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "undetermined concatenation components, at the closure (a variable in the message)";
      in_scheme = false;
      run =
        (fun () ->
          let y = concat_undetermined () (input "x" [ 3 ]) in
          ignore (Shape.to_dims y.Tensor.shape : int array));
    };
    {
      name = "the first mistake without a scheme (session ids, for contrast)";
      in_scheme = false;
      run = (fun () -> ignore (two_layer_unchecked ~hid:8 ~out:2 () (input "x" [ 3 ]) : Tensor.t));
    };
  ]

(* Whether a text spells a number from the reserved ranges (every one starts at 2^40). *)
let spells_reserved_id text =
  String.map text ~f:(fun c -> if Char.is_digit c then c else ' ')
  |> String.split ~on:' '
  |> List.exists ~f:(fun run ->
      (not (String.is_empty run))
      && (String.length run > 18 || Int.of_string run >= Shape.reserved_id_base))

let () =
  let results =
    List.map cases ~f:(fun case ->
        Tensor.unsafe_reinitialize ();
        match case.run () with
        | () -> (case, None)
        | exception (Row.Shape_error (msg, trace) as exn) -> (case, Some (exn, msg, trace)))
  in
  List.iter results ~f:(fun (case, outcome) ->
      Stdio.printf "--- %s\n%s\n" case.name
        (match outcome with
        | None -> "(no error)"
        | Some (exn, _, _) -> Stdlib.Printexc.to_string exn));
  let errors = List.filter_map results ~f:(fun (case, o) -> Option.map o ~f:(fun e -> (case, e))) in
  let in_scheme = List.filter errors ~f:(fun (case, _) -> case.in_scheme) in
  (* The control: the raw constraint sexp of the same errors spells reserved-range ids. *)
  let raw (msg, trace) =
    Sexp.to_string (Sexp.List (Sexp.Atom msg :: List.map trace ~f:Row.sexp_of_error_trace))
  in
  Verdict.case "readable shape errors" (fun () ->
      p "every case raised a shape error" (List.length errors = List.length cases);
      p_exists "the raw constraint sexp of an error in a scheme spells a reserved-range id"
        in_scheme ~f:(fun (_, (_, msg, trace)) -> spells_reserved_id (raw (msg, trace)));
      p_none "no rendering spells a reserved-range id" errors ~f:(fun (_, (exn, _, _)) ->
          spells_reserved_id (Stdlib.Printexc.to_string exn));
      p_all "every rendering fits one screen (at most 24 lines)" errors ~f:(fun (_, (exn, _, _)) ->
          List.length (String.split_lines (Stdlib.Printexc.to_string exn)) <= 24);
      (* [Exn.to_string] (the sexp converter) carries the same lines, one atom each. *)
      p_all "the exception's sexp carries the rendering's lines" errors
        ~f:(fun (_, (exn, msg, trace)) ->
          Sexp.equal (Exn.sexp_of_t exn)
            (Sexp.List
               (Sexp.Atom "Shape_error"
               :: List.map
                    (String.split_lines (Shape.error_to_string_hum msg trace))
                    ~f:(fun l -> Sexp.Atom (String.strip l))))))
