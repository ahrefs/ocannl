(* gh-ocannl-1790: [Row.solve_proj_equations] bound a projection variable to whatever projection an
   equation paired it with, without looking at what that projection already reached through the
   bindings. Two equations stating one equality in both directions ([a = b], [b = a]) bound [a] to
   [b] and [b] to [a]; an equation whose other side held the variable inside a window ([v = 2*v]),
   directly or through another variable, bound [v] under itself. Every reader follows bindings
   without a visited set, so the solver's own deferred passes, [Row.get_proj_index] and
   [Row.get_dim_index] then looped forever (or recursed until the stack ran out).

   The solver now binds a variable to the projection its partner resolves to: an equation that
   already holds through the bindings is dropped, and one under which the variable would occur in
   its own projection is refused with a [Shape_error] naming the variables of the cycle. So the
   bindings stay acyclic. [Row.get_dim_index]'s [assert false] for an iterated axis without a
   projection id is a [Shape_error] too: every route through [Shape] freshens the projection ids of
   the step's shapes first, so only a hand-built call reaches it.

   No route through [Shape] is known to produce these equations: the dimension solver's own
   bookkeeping keeps such variables apart. So the constraints are built here by hand, as in
   [affine_dim_index_kernel]. Each case runs on its own domain under a deadline ([Bounded_run], in
   [test/support]): a solver that never returns is a failed claim, not a stalled test, and the run
   stops there (the looping domain cannot be cancelled). *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Idx = Ir.Indexing

let x_dim = Row.get_default_dim ~d:5 ~proj_id:1_790_001 ()
let var name : Row.dim = Var (Row.get_var ~name ())
let eq d1 d2 : Row.constraint_ = Dim_eq { d1; d2; origin = [] }
let strided over : Row.dim = Affine { stride = 2; over; conv = None; stride_offset = 0 }

let solve constraints =
  let constraints = (Terminal_dim (false, x_dim, []) : Row.constraint_) :: constraints in
  let eqs = Row.get_proj_equations constraints Row.dim_map_empty Row.empty_env in
  Row.solve_proj_equations eqs ~resolved_padding:[] ~inferred_padding:[]

let index_to_string i = Sexp.to_string_hum (Idx.sexp_of_axis_index i)

(* [a] and [b] index as [x], which the terminal constraint makes iterated: an iterator of its own,
   that neither variable could reach unless the bindings carry it there. *)
let resolves label constraints ~a ~b =
  case label (fun () ->
      let indices =
        Bounded_run.run label (fun () ->
            let env = solve constraints in
            List.map [ x_dim; a; b ] ~f:(Row.get_dim_index env))
      in
      match indices with
      | [ x; ia; ib ] ->
          Stdio.eprintf "%s: x %s, a %s, b %s (not part of the golden)\n" label (index_to_string x)
            (index_to_string ia) (index_to_string ib);
          pf "%s: x is iterated" label (match x with Idx.Iterator _ -> true | _ -> false);
          pf "%s: a and b index as x" label (Idx.equal_axis_index ia x && Idx.equal_axis_index ib x)
      | _ -> pf "%s: three indices" label false)

let refuses label constraints ~indexed ~names =
  case label (fun () ->
      match
        Bounded_run.run label (fun () ->
            let env = solve constraints in
            Row.get_dim_index env indexed)
      with
      | i ->
          Stdio.eprintf "%s: indexed as %s\n" label (index_to_string i);
          pf "%s: refused" label false
      | exception Row.Shape_error (msg, _) ->
          Stdio.printf "%s: Shape_error: %s\n" label msg;
          p_all
            (Printf.sprintf "%s: the refusal names %s" label (String.concat ~sep:" and " names))
            names
            ~f:(fun name -> String.is_substring msg ~substring:name))

let () =
  (let a = var "a" and b = var "b" in
   (* The negative control: the same variables and the same final equality, stated once. *)
   resolves "a chain a = b = x" [ eq a b; eq b x_dim ] ~a ~b);
  (let a = var "a" and b = var "b" in
   resolves "one equality stated both ways, a = b and b = a, then b = x"
     [ eq a b; eq b a; eq b x_dim ]
     ~a ~b);
  (let v = var "v" in
   refuses "a variable inside its own window, v = 2*v"
     [ eq v (strided v) ]
     ~indexed:v ~names:[ "the projection variable v" ]);
  (let v = var "v" and w = var "w" in
   refuses "a variable inside its own window through another, w = 2*v and v = w"
     [ eq w (strided v); eq v w ]
     ~indexed:v
     ~names:[ "the projection variable v"; "(through w)" ]);
  (* gh-ocannl-1790's second half: an iterated axis the projection environment never saw. *)
  refuses "an iterated axis without a projection id" [] ~indexed:(Row.get_default_dim ~d:5 ())
    ~names:[ "projection id"; "d=5" ]
