(* gh-ocannl-1734: [Row.get_dim_index] on a window axis whose kernel the row holds as a variable.
   [Row.dim_to_proj] read the kernel's extent (which sizes a padded window's margins) off the dim as
   given, and asserted it was a solved [Dim]: a variable kernel died with [Assert_failure]. It now
   reads a variable through the projection environment, the way [Row.get_proj_index] reads one, and
   refuses a kernel that environment does not size with the same [Shape_error] that
   [Row.get_proj_equations] raises for one.

   No route through [Shape] is known to reach this with a variable kernel: every shape dim goes
   through [get_proj_equations] first, which sizes or refuses the kernel, and a closed shape's
   window axes have been evaluated to plain extents. So the projection environment is built here by
   hand: a window over an axis of 5 whose kernel variable is equated to an axis of 4. The kernel is
   even, so a misread extent moves the padded window's offset: the left margin is [(4 - 1) / 2 = 1],
   where an extent read as 1 would give 0. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Idx = Ir.Indexing

let over_dim = Row.get_default_dim ~d:5 ~proj_id:1_734_001 ()
let kernel_dim = Row.get_default_dim ~d:4 ~proj_id:1_734_002 ()
let k = Row.get_var ~name:"k" ()

let window ~use_padding kernel : Row.dim =
  Affine
    {
      stride = 1;
      over = over_dim;
      conv = Some { dilation = 1; kernel; use_padding };
      stride_offset = 0;
    }

let proj_env () =
  let constraints : Row.constraint_ list =
    [
      Terminal_dim (false, over_dim, []);
      Terminal_dim (false, kernel_dim, []);
      Dim_eq { d1 = Var k; d2 = kernel_dim; origin = [] };
    ]
  in
  let eqs = Row.get_proj_equations constraints Row.dim_map_empty Row.empty_env in
  Row.solve_proj_equations eqs ~resolved_padding:[] ~inferred_padding:[]

let index_to_string i = Sexp.to_string_hum (Idx.sexp_of_axis_index i)

let index env dim =
  match Row.get_dim_index env dim with
  | i -> Ok i
  | exception Row.Shape_error (msg, _) -> Error ("Shape_error: " ^ msg)
  | exception e -> Error (Stdlib.Printexc.to_string e)

let resolves label ~use_padding ~offset =
  case label (fun () ->
      let env = proj_env () in
      match
        (index env (window ~use_padding (Var k)), index env (window ~use_padding kernel_dim))
      with
      | Ok via_var, Ok via_dim ->
          if not (Idx.equal_axis_index via_var via_dim) then
            Stdio.eprintf "%s: variable kernel %s, sized kernel %s\n" label
              (index_to_string via_var) (index_to_string via_dim);
          pf "%s: the variable kernel indexes as the axis it is equated to" label
            (Idx.equal_axis_index via_var via_dim);
          let got_offset =
            match via_var with Idx.Affine { offset; _ } -> Some offset | _ -> None
          in
          pf "%s: the window's offset is %d" label offset
            (Option.equal Int.equal got_offset (Some offset))
      | Error e, _ | _, Error e ->
          Stdio.eprintf "%s: raised %s\n" label e;
          pf "%s: indexes" label false)

let refuses label kernel ~substring =
  case label (fun () ->
      match index (proj_env ()) (window ~use_padding:true kernel) with
      | Ok i ->
          Stdio.eprintf "%s: indexed as %s\n" label (index_to_string i);
          pf "%s: refused" label false
      | Error e ->
          Stdio.printf "%s: %s\n" label e;
          pf "%s: refused with a Shape_error naming it" label
            (String.is_prefix e ~prefix:"Shape_error: " && String.is_substring e ~substring))

let () =
  resolves "unpadded window, kernel variable" ~use_padding:false ~offset:0;
  resolves "padded window, kernel variable" ~use_padding:true ~offset:(-1);
  refuses "padded window, kernel variable nothing sizes"
    (Var (Row.get_var ~name:"j" ()))
    ~substring:"the window kernel j has no size";
  let sym, _ = Idx.get_static_symbol ~static_range:4 Idx.Empty in
  refuses "padded window, symbolic kernel" (Row.get_sym_dim sym)
    ~substring:"symbolic dimension not supported as convolution kernel"
