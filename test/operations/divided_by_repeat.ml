open Base
open Ocannl
open Verdict.Claims

(* gh-ocannl-2034: a [Total_elems] constraint reduced over a row that holds a variable its
   [divided_by] already lists. For a total of [n] over [w, ..r1..], [r1] carries [n / w]; once [r1]
   opens as [w, ..r2..], [r2] carries [n / (w * w)]: [divided_by] is a multiset, so the repeat is an
   equation, not a contradiction. The reduction used to drop the whole constraint there (a guard
   from when [divided_by] was a set), so the solver stopped enforcing the total on [r2]: a wrong
   size went through and a size the total fixes stayed unsolved. The same equation reaches a closed
   row with two or more [divided_by] variables, which the closed-row check used to leave waiting
   rather than cancel. *)

let origin : Row.constraint_origin list =
  [ { lhs_name = "t"; lhs_kind = `Output; rhs_name = "t"; rhs_kind = `Output; operation = None } ]

let dim d = Row.get_dim ~d ~basis:Row.default_basis ()
let prov = Row.empty_provenance
let closed_row dims : Row.t = { beg_dims = []; dims; bcast = Broadcastable; prov }
let open_row dims r : Row.t = { beg_dims = []; dims; bcast = Row_var r; prov }

let total ?(divided_by = []) n =
  Row.Total_elems { numerator = Num_elems n; divided_by; keep_axis = false }

let solve ?(stage = Row.Stage1) cs = Row.solve_inequalities ~stage cs Row.empty_env
let value env v = Row.get_dim_val env v
let is_value env v d = Option.equal Int.equal (value env v) (Some d)
let refusal f = match f () with _ -> None | exception Row.Shape_error (msg, _) -> Some msg

let refused name f =
  let message = refusal f in
  Stdio.printf "refusal: %s\n" (Option.value message ~default:"none");
  p name (Option.exists message ~f:(String.is_substring ~substring:"Total_elems"))

(* [n] elements over [w, ..r1..], then [..r1..] opened as [w, ..r2..], in one solver call: [r2]
   holds [n / (w * w)]. The rest of the row closes, and [w] takes its size, in a later call, as the
   incremental solver meets them when a later operation brings them: by then only [r2]'s entry
   carries the total. Returns [w], [r2]'s entry after the first call, and the second call. *)
let repeated ~n ~w_size ~rest =
  let w = Row.get_var () and r1 = Row.get_row_var () and r2 = Row.get_row_var () in
  let _, env =
    solve
      [
        Row.Rows_constr { r = [ open_row [ Var w ] r1 ]; constr = total n; origin };
        Row_eq { r1 = open_row [] r1; r2 = open_row [ Var w ] r2; origin };
      ]
  in
  let entry = Row.find_row_entry env r2 in
  let later () =
    Row.solve_inequalities ~stage:Stage1
      [
        Row_eq { r1 = open_row [] r2; r2 = closed_row rest; origin };
        Dim_eq { d1 = Var w; d2 = dim w_size; origin };
      ]
      env
  in
  (w, entry, later)

let test_reduction () =
  Stdio.printf "A repeated variable reduced into the rest of the row:\n";
  let w, entry, _ = repeated ~n:12 ~w_size:2 ~rest:[] in
  p "[..r2..] in [w, w, ..r2..] keeps the total 12 / (w * w)"
    (match entry with
    | Some (Bounds_row { constr = Total_elems { numerator = Num_elems 12; divided_by; _ }; _ }) ->
        List.equal Row.equal_dim_var divided_by [ w; w ]
    | _ -> false);
  let _, _, later = repeated ~n:12 ~w_size:2 ~rest:[ dim 6 ] in
  refused "12 elements over [w, w, 6] with w = 2 are refused (24 elements)" later;
  let u = Row.get_var () in
  let w, _, later = repeated ~n:12 ~w_size:2 ~rest:[ Var u ] in
  let _, env = later () in
  Stdio.printf "u = %s\n" (Option.value_map (value env u) ~default:"unsolved" ~f:Int.to_string);
  p "12 elements over [w, w, u] with w = 2 solve u = 3" (is_value env w 2 && is_value env u 3);
  let _, _, later = repeated ~n:12 ~w_size:2 ~rest:[ dim 3 ] in
  p "12 elements over [w, w, 3] with w = 2 are accepted" (Option.is_none (refusal later))

(* A closed row against a total divided by two or more variables. *)
let test_closed_row () =
  Stdio.printf "A closed row against two divided_by variables:\n";
  let w = Row.get_var () and v = Row.get_var () in
  let cs =
    [ Row.Rows_constr { r = [ closed_row [] ]; constr = total ~divided_by:[ w; v ] 1; origin } ]
  in
  let remaining, env = solve cs in
  p "1 / (w * v) over the empty row solves w = v = 1" (is_value env w 1 && is_value env v 1);
  p_empty "1 / (w * v) over the empty row, nothing left to wait on" ~over:cs remaining;
  refused "6 / (w * w) over a row of 4 is refused" (fun () ->
      let w = Row.get_var () in
      solve
        [
          Row.Rows_constr
            { r = [ closed_row [ dim 4 ] ]; constr = total ~divided_by:[ w; w ] 6; origin };
        ])

let () =
  test_reduction ();
  test_closed_row ()
