open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* Spec-only refusals must precede every registration write, including binding a valid capture
   before discovering a later missing label. These use the public operation API. *)
let leaf () = NTDSL.term ~fetch_op:(Constant 2.) ()
let contains s substring = String.is_substring s ~substring

let () =
  let forms =
    [
      ( "unary",
        "a => a",
        "a; a => a",
        fun spec refs xs -> NTDSL.einsum1 ~capture_dims:refs spec (List.hd_exn xs) () );
      ( "binary",
        "a; a => a",
        "a => a",
        fun spec refs xs ->
          NTDSL.einsum ~capture_dims:refs spec (List.nth_exn xs 0) (List.nth_exn xs 1) () );
      ( "ternary",
        "a; a; a => a",
        "a; a => a",
        fun spec refs xs ->
          NTDSL.einsum3 ~capture_dims:refs spec (List.nth_exn xs 0) (List.nth_exn xs 1)
            (List.nth_exn xs 2) () );
      ( "block",
        "a; a; a => a",
        "a; a => a",
        fun spec refs xs -> NTDSL.concat_sum ~capture_dims:refs spec (Array.of_list xs) () );
    ]
  in
  List.iter forms ~f:(fun (name, valid, wrong_count, build) ->
      List.iter [ "arity"; "clash"; "capture" ] ~f:(fun kind ->
          let xs = List.init 3 ~f:(fun _ -> leaf ()) in
          let spec, refs, message =
            match kind with
            | "arity" -> (wrong_count, [], "expected")
            | "clash" ->
                ( String.substr_replace_all valid ~pattern:"a" ~with_:"..a..a",
                  [],
                  "both a dimension variable and a row variable" )
            | _ ->
                ( valid,
                  List.map [ "a"; "missing" ] ~f:Shape.get_variable_ref,
                  "Variable missing not found" )
          in
          let error =
            match build spec refs xs with
            | _ -> None
            | exception Row.Shape_error (msg, _) -> Some msg
          in
          p
            (name ^ " " ^ kind ^ " reports its spec error")
            (Option.exists error ~f:(fun msg -> contains msg message));
          p
            (name ^ " " ^ kind ^ " leaves the session untainted")
            (Option.is_none (Shape.taint_provenance ()));
          let unrelated =
            match NTDSL.einsum1 "a => a" (leaf ()) () with
            | _ -> true
            | exception Row.Shape_error _ -> false
          in
          p (name ^ " " ^ kind ^ " permits an unrelated operation") unrelated;
          if Option.is_some (Shape.taint_provenance ()) then Tensor.rewind_to_last_closure ()))
