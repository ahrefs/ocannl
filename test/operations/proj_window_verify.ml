(* gh-ocannl-1889: [Row.solve_proj_equations] cannot bind an equation between a window ([Affine]
   dim, a [Conv_input] projection) and a solved index, or between two windows of different strides:
   it set them aside to verify once the indices were known. The verification ran under [with _ ->
   ()], so a window equated to an index it does not reach passed silently, and the program lowered
   with one of the two indices dropped. It also ran before the indices of the windows' own targets
   existed, so a window over another window's target could not be indexed there at all: the
   catch-all hid that too.

   The solver now verifies these equations against the final environment and lets every error
   through. The consistent cases are the controls: they solve, and index as the equations say. The
   inconsistent ones are refused with a [Shape_error]; on the base they solve.

   [Solved] sides come from a projection-axis environment (a batch slice's static index, in
   [Shape]); the equations are built by hand, as in [proj_var_cycle], each under a deadline
   ([Bounded_run]). *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Idx = Ir.Indexing

let x_dim = Row.get_default_dim ~d:5 ~proj_id:1_889_001 ()
let one_dim = Row.get_default_dim ~d:1 ~proj_id:1_889_002 ()
let target_dim = Row.get_default_dim ~d:5 ~proj_id:1_889_003 ()
let var name = Row.get_var ~name ()
let eq d1 d2 : Row.constraint_ = Dim_eq { d1; d2; origin = [] }

let window ?(offset = 0) stride over : Row.dim =
  Affine { stride; over; conv = None; stride_offset = offset }

let solve ?(solved = []) constraints =
  let constraints = (Terminal_dim (false, x_dim, []) : Row.constraint_) :: constraints in
  let proj_axis_env =
    List.fold solved ~init:Row.dim_map_empty ~f:(fun env (v, idx) ->
        Map.add_exn env ~key:v ~data:idx)
  in
  let eqs = Row.get_proj_equations constraints proj_axis_env Row.empty_env in
  Row.solve_proj_equations eqs ~resolved_padding:[] ~inferred_padding:[]

let index_to_string i = Sexp.to_string_hum (Idx.sexp_of_axis_index i)

let solves label ?solved constraints ~check =
  case label (fun () ->
      match Bounded_run.run label (fun () -> solve ?solved constraints) with
      | env -> check env
      | exception Row.Shape_error (msg, _) ->
          Stdio.eprintf "%s: Shape_error: %s\n" label msg;
          pf "%s: solves" label false)

let refuses label ?solved constraints =
  case label (fun () ->
      match Bounded_run.run label (fun () -> solve ?solved constraints) with
      | _ -> pf "%s: refused" label false
      | exception Row.Shape_error (msg, _) ->
          Stdio.printf "%s: Shape_error: %s\n" label msg;
          pf "%s: the refusal is the projection verifier's" label
            (String.is_prefix msg ~prefix:"solve_proj_equations: "))

let () =
  (let s = var "s" in
   (* [1*y + 3] over a dimension-1 axis is the fixed index 3: the slice's own. *)
   solves "a window reaching the slice index, s = 1*y + 3 at s = 3"
     ~solved:[ (s, Idx.Fixed_idx 3) ]
     [ eq (Var s) (window ~offset:3 1 one_dim) ]
     ~check:(fun _ -> pf "a window reaching the slice index: solves" true));
  (let s = var "s" in
   refuses "a window missing the slice index, s = 1*x at s = 3"
     ~solved:[ (s, Idx.Fixed_idx 3) ]
     [ eq (Var s) (window 1 x_dim) ]);
  (let y = var "y" in
   (* [t] is a window's target, indexed by the window ([t = 1*x]), so [2*t] and [1*y] with [y = 2*x]
      both index as [2*x]. [t] had no index when the verification ran early. *)
   solves "a window over a window's target, 2*t = 1*y with t = 1*x and y = 2*x"
     [
       eq target_dim (window 1 x_dim);
       eq (window 2 target_dim) (window 1 (Var y));
       eq (Var y) (window 2 x_dim);
     ]
     ~check:(fun env ->
       let x = Row.get_dim_index env x_dim and t = Row.get_dim_index env target_dim in
       Stdio.eprintf "a window over a window's target: x %s, t %s (not part of the golden)\n"
         (index_to_string x) (index_to_string t);
       pf "a window over a window's target: x is iterated"
         (match x with Idx.Iterator _ -> true | _ -> false);
       pf "a window over a window's target: t indexes as x" (Idx.equal_axis_index t x)));
  refuses "two strides over one axis, 2*x = 3*x" [ eq (window 2 x_dim) (window 3 x_dim) ]
