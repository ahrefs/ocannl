(** Spec-side regression for the beg_dims-on-Row.t refactor.

    Guards against a regression in [tensor/shape.ml::axes_spec_to_dims_bio] (lines around 196–207),
    where parser-derived leading axes are now stored in [Row.beg_dims] instead of being prepended
    to [Row.dims]. The pre-fix behavior was: build a row with [bcast = Row_var { v; beg_dims }] and
    then flatten [beg_dims @ dims] into the top-level [dims]. Under the new layout, [beg_dims] is
    a top-level field and the spec parser must place leading axes there directly.

    This test exercises the spec-parsing path end to end (via [Shape.of_spec]), not just direct
    [Row] constructors. *)

open! Base
open Ocannl

let dummy_origin : Row.constraint_origin list =
  [
    {
      lhs_name = "test";
      lhs_kind = `Output;
      rhs_name = "test";
      rhs_kind = `Output;
      operation = None;
    };
  ]

(* Match a Row.t against the expected structural shape: non-empty beg_dims of given size, then a
   middle Row_var, then trailing dims of given size. Returns Ok () or Error explanation. *)
let dim_sizes_of dims =
  List.map dims ~f:(function Row.Dim { d; _ } -> Some d | _ -> None)

let int_opts_equal xs ys =
  List.length xs = List.length ys
  && List.for_all2_exn xs ys ~f:(fun a b ->
      match (a, b) with
      | Some i, Some j -> i = j
      | None, None -> true
      | _ -> false)

let row_has_structure (row : Row.t) ~expected_beg ~expected_trailing =
  let beg_sizes = dim_sizes_of row.beg_dims in
  let dim_sizes = dim_sizes_of row.dims in
  let expected_beg_opts = List.map expected_beg ~f:Option.some in
  let expected_trail_opts = List.map expected_trailing ~f:Option.some in
  let opts_to_str os =
    String.concat ~sep:";"
      (List.map os ~f:(function Some i -> Int.to_string i | None -> "?"))
  in
  if int_opts_equal beg_sizes expected_beg_opts
     && int_opts_equal dim_sizes expected_trail_opts
  then Ok ()
  else
    Error
      (Printf.sprintf
         "expected beg_dims=[%s] dims=[%s]; got beg_dims=[%s] dims=[%s]"
         (String.concat ~sep:";" (List.map expected_beg ~f:Int.to_string))
         (String.concat ~sep:";" (List.map expected_trailing ~f:Int.to_string))
         (opts_to_str beg_sizes) (opts_to_str dim_sizes))

(* Test 1: parser places leading axes in t.beg_dims, not in t.dims.
   Spec "3, ..rho.., 7" on the output kind: leading [Fixed_index 3], row var rho, trailing
   [Fixed_index 7]. Under the new layout this must build
   { beg_dims = [Dim 3]; dims = [Dim 7]; bcast = Row_var rho }, NOT the pre-fix
   { beg_dims = []; dims = [Dim 3; Dim 7]; bcast = Row_var rho }. *)
let test_parser_places_leading_in_beg_dims () =
  Stdio.printf
    "Test: einsum/shape spec parser places leading axes into Row.beg_dims\n";
  Tensor.unsafe_reinitialize ();
  let shape = Shape.of_spec ~debug_name:"spec_test" ~id:1 "3, ..rho.., 7" in
  let row = shape.output in
  match row.bcast with
  | Row.Row_var _ -> (
      match row_has_structure row ~expected_beg:[ 3 ] ~expected_trailing:[ 7 ] with
      | Ok () -> Stdio.printf "  PASS\n"
      | Error msg -> Stdio.printf "  FAIL: %s\n" msg)
  | Row.Broadcastable ->
      Stdio.printf
        "  FAIL: bcast should be Row_var (named row var rho), got Broadcastable\n"

(* Test 2: spec-driven outer-left mismatch raises Shape_error.
   Build two rows from specs that share the same row variable but disagree on the outer-left
   leading axis. After unification, the disagreement must surface as a Shape_error from the
   solver — i.e., the spec/parser path produces rows where outer-left alignment is enforced by
   the row inference machinery.

   Concretely: r1 has leading flank [Dim 5; Dim 2] and r2 (open) has leading flank [Dim 2].
   Under outer-left alignment, r2's first leading axis (Dim 2) aligns against r1's first leading
   axis (Dim 5) — incompatible. *)
let test_spec_outer_left_mismatch () =
  Stdio.printf "Test: outer-left alignment rejects spec-derived mismatch\n";
  Tensor.unsafe_reinitialize ();
  let prov = Row.empty_provenance in
  let res =
    {
      Row.beg_dims =
        [ Row.get_default_dim ~d:5 (); Row.get_default_dim ~d:2 () ];
      dims = [ Row.get_default_dim ~d:4 () ];
      bcast = Broadcastable;
      prov;
    }
  in
  let rho = Row.get_row_var () in
  let opnd =
    {
      Row.beg_dims = [ Row.get_default_dim ~d:2 () ];
      dims = [ Row.get_default_dim ~d:4 () ];
      bcast = Row_var rho;
      prov;
    }
  in
  let ineq = Row.Row_ineq { res; opnd; origin = dummy_origin } in
  try
    let _remaining, _env =
      Row.solve_inequalities ~stage:Stage1 [ ineq ] Row.empty_env
    in
    Stdio.printf "  FAIL: inequality should have raised Shape_error (5 vs 2 outer-left)\n"
  with Row.Shape_error (msg, _) ->
    Stdio.printf "  PASS: got Shape_error: %s\n" msg

(* Test 3: when the spec parser populates beg_dims, downstream subst preserves them.
   Spec "3, ..rho.., 7" builds row with beg_dims=[Dim 3] and dims=[Dim 7]. Substituting rho :=
   {beg_dims=[]; dims=[Dim 4]; Broadcastable} via Row_eq must yield a closed row with
   beg_dims=[Dim 3] and dims=[Dim 4; Dim 7]. This exercises the spec → row → substitution chain
   and would fail if the spec parser had reverted to flattening leading axes into dims. *)
let test_spec_substitution_preserves_leading () =
  Stdio.printf "Test: spec-derived leading flank survives downstream substitution\n";
  Tensor.unsafe_reinitialize ();
  let shape = Shape.of_spec ~debug_name:"spec_subst" ~id:2 "3, ..rho.., 7" in
  let row = shape.output in
  let rho =
    match row.bcast with
    | Row.Row_var v -> v
    | Row.Broadcastable ->
        failwith "Test: spec did not produce a row variable; cannot continue"
  in
  let prov = Row.empty_provenance in
  let value : Row.t =
    {
      beg_dims = [];
      dims = [ Row.get_default_dim ~d:4 () ];
      bcast = Broadcastable;
      prov;
    }
  in
  let eq =
    Row.Row_eq
      { r1 = { beg_dims = []; dims = []; bcast = Row_var rho; prov }; r2 = value; origin = dummy_origin }
  in
  let _remaining, env =
    Row.solve_inequalities ~stage:Stage1 [ eq ] Row.empty_env
  in
  let result = Row.subst_row env row in
  match
    row_has_structure result ~expected_beg:[ 3 ] ~expected_trailing:[ 4; 7 ]
  with
  | Ok () -> (
      match result.bcast with
      | Row.Broadcastable -> Stdio.printf "  PASS\n"
      | Row.Row_var _ ->
          Stdio.printf
            "  FAIL: expected Broadcastable after substitution, got Row_var\n")
  | Error msg -> Stdio.printf "  FAIL: %s\n" msg

let () =
  test_parser_places_leading_in_beg_dims ();
  test_spec_outer_left_mismatch ();
  test_spec_substitution_preserves_leading ()
