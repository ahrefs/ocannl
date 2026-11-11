open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* The first call must stand on its own: replaying the original total along with the later equations
   hides the lost constraint (gh-ocannl-2127). These are solver tests, with no backend. *)
let origin : Row.constraint_origin list = []
let dim d = Row.get_default_dim ~d ()

let row ?(beg_dims = []) ?(dims = []) bcast : Row.t =
  { beg_dims; dims; bcast; prov = Row.empty_provenance }

let eq d1 d2 = Row.Dim_eq { d1; d2; origin }
let row_eq r1 r2 = Row.Row_eq { r1; r2; origin }
let solve cs env = Row.solve_inequalities ~stage:Stage1 cs env
let refused f = match f () with _ -> false | exception Row.Shape_error _ -> true

let test label axis ~axis_size ~leading =
  let x = Row.get_var () and r = Row.get_row_var () in
  let axis = axis x in
  let open_row =
    if leading then row ~beg_dims:[ axis ] (Row_var r) else row ~dims:[ axis ] (Row_var r)
  in
  let constr =
    Row.Total_elems { numerator = Num_elems (axis_size * 3); divided_by = []; keep_axis = false }
  in
  let pending, env = solve [ Row.Rows_constr { r = [ open_row ]; constr; origin } ] Row.empty_env in
  p
    (label ^ ": unresolved axis keeps the total pending")
    (match pending with [ Row.Rows_constr _ ] -> true | _ -> false);
  let pending, env =
    List.fold [ Row.Stage2; Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ] ~init:(pending, env)
      ~f:(fun (pending, env) stage -> Row.solve_inequalities ~stage pending env)
  in
  p
    (label ^ ": unresolved total survives closing stages")
    (match pending with [ Row.Rows_constr _ ] -> true | _ -> false);
  let later rest =
    solve
      (pending @ [ eq (Var x) (dim 2); row_eq (row (Row_var r)) (row ~dims:rest Broadcastable) ])
      env
  in
  p (label ^ ": a later wrong total is refused") (refused (fun () -> later [ dim 4 ]));
  let u = Row.get_var () in
  let remaining, solved = later [ Var u ] in
  p
    (label ^ ": a later extent solves the remaining factor")
    (Option.equal Int.equal (Row.get_dim_val solved u) (Some 3));
  p_empty (label ^ ": the satisfied total leaves no residual") ~over:pending remaining

let () =
  List.iter [ false; true ] ~f:(fun leading ->
      let flank = if leading then "leading" else "trailing" in
      Verdict.case ("concat " ^ flank) (fun () ->
          test ("concat " ^ flank) (fun x -> Row.Concat [ Var x; dim 2 ]) ~axis_size:4 ~leading);
      Verdict.case ("valid window " ^ flank) (fun () ->
          test ("valid window " ^ flank)
            (fun x ->
              Row.Affine
                {
                  stride = 2;
                  over = Var x;
                  stride_offset = 0;
                  conv = Some { dilation = 1; kernel = dim 3; use_padding = false };
                })
            ~axis_size:5 ~leading))
