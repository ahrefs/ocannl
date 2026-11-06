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
   ([Bounded_run]).

   gh-ocannl-1971: the pass giving each window's target its index read an environment taken before
   the pass, so a window over another window's target ([u = 2*t] with [t = 1*x]) was an unknown
   projection, whichever equation came first, and a target of extent 1 read as [Fixed_idx 0], a
   silently wrong index. The pass now goes in dependency order. The chains' cases execute the index
   each axis ends up with against the composition of its windows, for every value of the iterator.

   No route through [Shape] states such a chain: within a step each tensor axis has one role, plain
   label or window, operands that alias are de-aliased ([Shape.dealias_operands]), and a window
   ranges over a label, so a target is never what a window ranges over. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Idx = Ir.Indexing

let x_dim = Row.get_default_dim ~d:5 ~proj_id:1_889_001 ()
let one_dim = Row.get_default_dim ~d:1 ~proj_id:1_889_002 ()
let target_dim = Row.get_default_dim ~d:5 ~proj_id:1_889_003 ()
let outer_dim = Row.get_default_dim ~d:5 ~proj_id:1_889_004 ()
let outermost_dim = Row.get_default_dim ~d:5 ~proj_id:1_889_005 ()
let var name = Row.get_var ~name ()
let eq d1 d2 : Row.constraint_ = Dim_eq { d1; d2; origin = [] }

let window ?(offset = 0) stride over : Row.dim =
  Affine { stride; over; conv = None; stride_offset = offset }

let kernel_window stride over dilation kernel : Row.dim =
  Affine { stride; over; conv = Some { dilation; kernel; use_padding = false }; stride_offset = 0 }

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

let refuses label ?solved ?(by = ("projection verifier", "solve_proj_equations: ")) constraints =
  case label (fun () ->
      match Bounded_run.run label (fun () -> solve ?solved constraints) with
      | _ -> pf "%s: refused" label false
      | exception Row.Shape_error (msg, _) ->
          Stdio.printf "%s: Shape_error: %s\n" label msg;
          let who, prefix = by in
          pf "%s: the refusal is the %s's" label who (String.is_prefix msg ~prefix))

(* The position an index reaches at a value of the iterator [x] of [x_dim]. *)
let position (i : Idx.axis_index) ~x ~at =
  let weight s = if Idx.equal_symbol s x then at else 0 in
  match i with
  | Fixed_idx k -> Some k
  | Iterator s -> Some (weight s)
  | Affine { symbols; offset } ->
      Some (offset + List.sum (module Int) symbols ~f:(fun (c, s) -> c * weight s))
  | Sub_axis | Concat _ -> None

(* Each of [axes] is an axis and the function its windows compose to. Both orders of the equations
   are solved: the order they reach the pass in is the only thing that could have let a chain
   through. *)
let chain label ?solved constraints ~axes =
  List.iter
    [ ("", constraints); (", equations reversed", List.rev constraints) ]
    ~f:(fun (order, constraints) ->
      let label = label ^ order in
      solves label ?solved constraints ~check:(fun env ->
          match Row.get_dim_index env x_dim with
          | Iterator x ->
              p_all (label ^ ": every axis indexes as its windows compose") axes
                ~f:(fun (dim, composed) ->
                  let i = Row.get_dim_index env dim in
                  Stdio.eprintf "%s: %s (not part of the golden)\n" label (index_to_string i);
                  List.for_all (List.range 0 5) ~f:(fun at ->
                      Option.equal Int.equal (position i ~x ~at) (Some (composed at))))
          | i -> pf "%s: x is iterated, not %s" label (index_to_string i) false))

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
  refuses "two strides over one axis, 2*x = 3*x" [ eq (window 2 x_dim) (window 3 x_dim) ];
  chain "a chain of two windows, u = 2*t with t = 1*x"
    [ eq target_dim (window 1 x_dim); eq outer_dim (window 2 target_dim) ]
    ~axes:[ (target_dim, Fn.id); (outer_dim, fun x -> 2 * x) ];
  chain "a chain of three windows with offsets, v = 3*u + 1, u = 2*t + 2, t = 1*x + 1"
    [
      eq target_dim (window ~offset:1 1 x_dim);
      eq outer_dim (window ~offset:2 2 target_dim);
      eq outermost_dim (window ~offset:1 3 outer_dim);
    ]
    ~axes:
      [
        (target_dim, fun x -> x + 1);
        (outer_dim, fun x -> (2 * (x + 1)) + 2);
        (outermost_dim, fun x -> (3 * ((2 * (x + 1)) + 2)) + 1);
      ];
  (let w = var "w" in
   chain "a chain through a variable, u = 2*w with w = t and t = 1*x"
     [ eq target_dim (window 1 x_dim); eq (Var w) target_dim; eq outer_dim (window 2 (Var w)) ]
     ~axes:[ (outer_dim, fun x -> 2 * x) ]);
  (* The kernel of a window is what the chain goes through: [u = 2*y + 3*k] with [k = 1*x], where
     [y] is fixed at 1, so only the window's kernel term follows the iterator. *)
  (let y = var "y" in
   chain "a chain through a kernel, u = 2*y + 3*k with k = 1*x and y at 1"
     ~solved:[ (y, Idx.Fixed_idx 1) ]
     [ eq target_dim (window 1 x_dim); eq outer_dim (kernel_window 2 (Var y) 3 target_dim) ]
     ~axes:[ (outer_dim, fun x -> 2 + (3 * x)) ]);
  (* Extent 1 is what a stale read did not refuse: [u] read [t] as index 0, not as the index its
     window gives it. *)
  (let y_one = Row.get_default_dim ~d:1 ~proj_id:1_889_006 () in
   chain "a chain over a target of extent 1, u = 2*t with t = 1*y + 3"
     [ eq one_dim (window ~offset:3 1 y_one); eq outer_dim (window 2 one_dim) ]
     ~axes:[ (outer_dim, Fn.const 6) ]);
  refuses "a cycle of windows, t = 2*u and u = 2*t"
    ~by:("projection reader", "projection_of_solved_dims: unknown projection")
    [ eq target_dim (window 2 outer_dim); eq outer_dim (window 2 target_dim) ]
