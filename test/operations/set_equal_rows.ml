open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* [Shape.set_equal] on two row references ties the rows themselves, [Row_eq], whatever either
   reference's solved state (gh-ocannl-1617). A row reference's [solved_dim] holds only its row's
   product, so before the fix a pair with one side already solved -- by a fixed operand at its
   registration step, or by [Shape.set_dim] -- was tied by that product alone (or, both solved, by
   comparing the two products), while an unsolved pair got [Row_eq]: the outcome depended on what
   inference had solved by the time of the call. Each case below holds under both registration
   orders of the two operations and both argument orders of the call. Product equality stays the
   meaning of a dimension paired with a row only (the Dim-Row case, and [set_dim] on a row). *)

let fixed name dims = NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~o:dims ~f:(fun _ -> 1.) ()
let open_leaf name = NTDSL.term ~label:[ name ] ~fetch_op:(Constant 2.) ()

(* Captures [t]'s whole output row as the row variable [r]; returns the capturing operation. *)
let capture_row (r : Shape.delayed_var_ref) t =
  let l = r.var_ref.ref_label in
  NTDSL.einsum1 ~capture_dims:[ r ] (Printf.sprintf "..%s.. => ..%s.." l l) t ()

(* An independent capture fixes [t]'s axes to [sizes], one axis per size. *)
let fix t sizes =
  let labels = List.mapi sizes ~f:(fun i _ -> Printf.sprintf "f%d" i) in
  let refs = List.map labels ~f:Shape.get_variable_ref in
  List.iter2_exn refs sizes ~f:Shape.set_dim;
  let axes = String.concat ~sep:", " labels in
  ignore (NTDSL.einsum1 ~capture_dims:refs (axes ^ " => " ^ axes) t () : Tensor.t)

let dims t = Array.to_list (Lazy.force t.Tensor.value.dims)
let dims_str t = String.concat ~sep:"x" (List.map (dims t) ~f:Int.to_string)
let show name ts = List.iter ts ~f:(fun (l, t) -> Stdio.printf "%s: %s is %s\n" name l (dims_str t))

let shape_error f =
  match f () with
  | () -> None
  | exception Row.Shape_error (msg, _) ->
      Tensor.rewind_to_last_closure ~reason:"gh-1617 refusal case" ();
      Some msg

(* Registers the capture of [r1] over [t1] and of [r2] over [t2] in the given order; returns the
   capturing operations in argument order. *)
let register ~first_r1 (r1, t1) (r2, t2) =
  if first_r1 then
    let y1 = capture_row r1 t1 in
    (y1, capture_row r2 t2)
  else
    let y2 = capture_row r2 t2 in
    (capture_row r1 t1, y2)

let set_equal ~swap r1 r2 = if swap then Shape.set_equal r2 r1 else Shape.set_equal r1 r2

let orders =
  List.concat_map [ true; false ] ~f:(fun first_r1 ->
      List.map [ false; true ] ~f:(fun swap -> (first_r1, swap)))

let order_name (first_r1, swap) =
  Printf.sprintf "%s registered first, set_equal %s"
    (if first_r1 then "r" else "s")
    (if swap then "s r" else "r s")

let () =
  (* A fixed operand solves its reference at its own registration step; the open one is unsolved.
     Before the fix the open row got only the product, 6, and closed to one axis. *)
  List.iter orders ~f:(fun ((first_r1, swap) as order) ->
      let name = "solved by a fixed operand, " ^ order_name order in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
          let x = fixed "x" [ 2; 3 ] and w = open_leaf "w" in
          ignore (register ~first_r1 (r1, x) (r2, w) : Tensor.t * Tensor.t);
          p (name ^ ": r is solved at the call") (Option.is_some r1.var_ref.solved_dim);
          set_equal ~swap r1 r2;
          show name [ ("x", x); ("w", w) ];
          p
            (name ^ ": the open row takes the fixed row's structure")
            (List.equal Int.equal (dims w) [ 2; 3 ])));
  (* [Shape.set_dim] on a row reference solves it to a product: that stays a product constraint on
     its own row, but it no longer turns a later Row-Row [set_equal] into a product comparison. *)
  List.iter orders ~f:(fun ((first_r1, swap) as order) ->
      let name = "solved by set_dim, " ^ order_name order in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
          let w = open_leaf "w" and x = fixed "x" [ 2; 3 ] in
          ignore (register ~first_r1 (r1, w) (r2, x) : Tensor.t * Tensor.t);
          Shape.set_dim r1 6;
          set_equal ~swap r1 r2;
          show name [ ("w", w); ("x", x) ];
          p
            (name ^ ": the set_dim row takes the other row's structure")
            (List.equal Int.equal (dims w) [ 2; 3 ])));
  (* The same tie when neither reference is solved at the call and the structure arrives later: the
     outcome does not depend on whether set_equal came before or after the solution. *)
  List.iter orders ~f:(fun ((first_r1, swap) as order) ->
      let name = "both open, fixed afterwards, " ^ order_name order in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
          let x = open_leaf "x" and w = open_leaf "w" in
          ignore (register ~first_r1 (r1, x) (r2, w) : Tensor.t * Tensor.t);
          p
            (name ^ ": neither reference is solved at the call")
            (Option.is_none r1.var_ref.solved_dim && Option.is_none r2.var_ref.solved_dim);
          set_equal ~swap r1 r2;
          fix x [ 2; 3 ];
          show name [ ("x", x); ("w", w) ];
          p
            (name ^ ": the open row takes the later structure")
            (List.equal Int.equal (dims w) [ 2; 3 ])));
  (* Equal products are not equal rows. Before the fix two solved rows of product 6 were
     accepted. *)
  List.iter orders ~f:(fun ((first_r1, swap) as order) ->
      let name = "equal products, different structure, " ^ order_name order in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
          let x = fixed "x" [ 2; 3 ] and y = fixed "y" [ 6 ] in
          let y1, y2 = register ~first_r1 (r1, x) (r2, y) in
          p
            (name ^ ": the rows 2x3 and 6 do not unify")
            (Option.is_some
               (shape_error (fun () ->
                    set_equal ~swap r1 r2;
                    ignore (dims y1 : int list);
                    ignore (dims y2 : int list))))));
  (* Across a closure a row reference holds only its product, and its variable belongs to the reset
     environment: the row itself is gone, so the call is refused rather than tied by product. *)
  List.iter [ false; true ] ~f:(fun swap ->
      let name =
        Printf.sprintf "one reference closed, set_equal %s" (if swap then "s r" else "r s")
      in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
          let x = fixed "x" [ 2; 3 ] in
          (* Reading the capturing operation's dims closes the inference that bound [r]. *)
          ignore (dims (capture_row r1 x) : int list);
          p
            (name ^ ": r is solved to its product by the closure")
            (Option.equal Int.equal r1.var_ref.solved_dim (Some 6));
          let w = open_leaf "w" in
          ignore (capture_row r2 w : Tensor.t);
          let refused = shape_error (fun () -> set_equal ~swap r1 r2) in
          Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
          p (name ^ ": the call is refused") (Option.is_some refused)))
