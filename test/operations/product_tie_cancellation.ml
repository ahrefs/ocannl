open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* A product tie (gh-ocannl-1890) states that a row's elements multiply to a dimension [n], as a
   [Total_elems] over a [Strided_var] on [n]. When the row itself holds [n], the variable cancels:
   [(coeff * n / denom) / (n * rest) = (coeff / denom) / rest]. So [n] stays free, and what is left
   to check is that [rest] multiplies to [coeff / denom] (gh-ocannl-1960). The solver used to assign
   [n := coeff / denom] instead, at Stage1 for a tie's constant coefficient: a row closing around
   [n] forced [n = 1], refusing the sizes that arrived afterwards. The first part drives the tie
   through capture references, the only source of ties besides [Shape.set_equal]; the second pins
   each arithmetic outcome on hand-built constraints, for the coefficients and rows no capture
   reaches. *)

let open_leaf name = NTDSL.term ~label:[ name ] ~fetch_op:(Constant 2.) ()
let fixed name dims = NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~o:dims ~f:(fun _ -> 1.) ()
let dims t = Array.to_list (Lazy.force t.Tensor.value.dims)
let dims_str t = String.concat ~sep:"x" (List.map (dims t) ~f:Int.to_string)
let all_of_5 name ts = p_all name ts ~f:(fun t -> List.equal Int.equal (dims t) [ 5 ])

(* The message of the refusal [f] ends in, rewinding the session past it. *)
let refusal f =
  match f () with
  | () -> None
  | exception Row.Shape_error (msg, _) ->
      Tensor.rewind_to_last_closure ~reason:"gh-1960 refusal case" ();
      Some msg

let refused_with name f ~sub =
  let message = refusal f in
  Stdio.printf "refusal: %s\n" (Option.value message ~default:"none");
  p name (Option.exists message ~f:(fun msg -> String.is_substring msg ~substring:sub))

(* Binds [n] as a dimension of [..c.., n], then as a row, on the very row that holds it. *)
let tied () =
  let n = Shape.get_variable_ref "n" and c = Shape.get_variable_ref "c" in
  let y1 = NTDSL.einsum1 ~capture_dims:[ c; n ] "..c.., n => ..c.., n" (open_leaf "q") () in
  let y2 = NTDSL.einsum1 ~capture_dims:[ n ] "..n.. => ..n.." y1 () in
  (c, y1, y2)

let tied_closed_row spec_dims dim_refs =
  let n = Shape.get_variable_ref "n" in
  let spec = String.concat ~sep:", " (spec_dims @ [ "n" ]) in
  let y1 =
    NTDSL.einsum1 ~capture_dims:(dim_refs @ [ n ]) (spec ^ " => " ^ spec) (open_leaf "q") ()
  in
  (y1, NTDSL.einsum1 ~capture_dims:[ n ] "..n.. => ..n.." y1 ())

(* A case's own verdict: the rewind keeps a refusal from reaching the cases after it. *)
let case label f =
  Verdict.case label (fun () ->
      try f ()
      with Row.Shape_error _ as e ->
        Tensor.rewind_to_last_closure ~reason:"gh-1960 refused case" ();
        raise e)

(* An axis of 3 beside [n]: [3 * n = n] has no solution. *)
let open_row_with_3 () =
  let n = Shape.get_variable_ref "n" and c = Shape.get_variable_ref "c" in
  let w = Shape.get_variable_ref "w" in
  Shape.set_dim w 3;
  let spec = "..c.., w, n" in
  let y1 = NTDSL.einsum1 ~capture_dims:[ c; w; n ] (spec ^ " => " ^ spec) (open_leaf "q") () in
  (c, y1, NTDSL.einsum1 ~capture_dims:[ n ] "..n.. => ..n.." y1 ())

let test_through_captures () =
  Stdio.printf "Ties through capture references:\n";
  (* The row closes around [n] at the registration of [y3], at its Stage1, before anything sizes
     [n]. *)
  case "a size set after the row closed" (fun () ->
      let _, y1, y2 = tied () in
      let k = Shape.get_variable_ref "k" in
      let y3 = NTDSL.einsum1 ~capture_dims:[ k ] "k => k" y1 () in
      Shape.set_dim k 5;
      Stdio.printf "set_dim after the row closed: %s %s %s\n" (dims_str y1) (dims_str y2)
        (dims_str y3);
      all_of_5 "a size set after the row closed reaches the tied dimension" [ y1; y2; y3 ]);
  case "an operand met after the row closed" (fun () ->
      let _, y1, y2 = tied () in
      let y3 = NTDSL.einsum1 "x => x" y1 () in
      let sum = NTDSL.O.(y3 + fixed "g" [ 5 ]) in
      Stdio.printf "an operand met after the row closed: %s %s %s %s\n" (dims_str y1) (dims_str y2)
        (dims_str y3) (dims_str sum);
      all_of_5 "an operand met after the row closed sizes the tied dimension" [ y1; y2; y3; sum ]);
  (* Controls: the size is there before the row closes, so the cancellation does not decide them. *)
  case "a row left open" (fun () ->
      let _, y1, y2 = tied () in
      let sum = NTDSL.O.(y1 + fixed "g" [ 5 ]) in
      Stdio.printf "a row left open: %s %s %s\n" (dims_str y1) (dims_str y2) (dims_str sum);
      all_of_5 "a row left open closes empty around the tied dimension" [ y1; y2; sum ]);
  case "a row closed from the start" (fun () ->
      let y1, y2 = tied_closed_row [] [] in
      let sum = NTDSL.O.(y1 + fixed "g" [ 5 ]) in
      Stdio.printf "a row closed from the start: %s %s %s\n" (dims_str y1) (dims_str y2)
        (dims_str sum);
      all_of_5 "a row closed from the start holds the tied dimension" [ y1; y2; sum ]);
  let cancels = "the dimension cancels" in
  refused_with "a row closed with a second axis of 3 refuses the tie" ~sub:cancels (fun () ->
      let w = Shape.get_variable_ref "w" in
      Shape.set_dim w 3;
      let y1, y2 = tied_closed_row [ "w" ] [ w ] in
      ignore ((dims y1, dims y2) : int list * int list));
  refused_with "an open row with an axis of 3 refuses the tie" ~sub:cancels (fun () ->
      let _, y1, y2 = open_row_with_3 () in
      ignore ((dims y1, dims y2) : int list * int list));
  refused_with "an open row with an axis of 3 refuses the tie, its rest constrained" ~sub:cancels
    (fun () ->
      let c, y1, y2 = open_row_with_3 () in
      Shape.set_dim c 1;
      ignore ((dims y1, dims y2) : int list * int list))

let origin : Row.constraint_origin list =
  [ { lhs_name = "t"; lhs_kind = `Output; rhs_name = "t"; rhs_kind = `Output; operation = None } ]

let dim d = Row.get_dim ~d ~basis:Row.default_basis ()

let closed_row dims : Row.t =
  { beg_dims = []; dims; bcast = Broadcastable; prov = Row.empty_provenance }

let strided ?(round_up = false) ~coeff ~denom ~divided_by v =
  Row.Total_elems
    {
      numerator = Strided_var { coeff = Utils.safe_value "gh-1960" coeff; var = v; denom; round_up };
      divided_by;
      keep_axis = false;
    }

let solve ?(stage = Row.Stage1) cs = Row.solve_inequalities ~stage cs Row.empty_env
let value env v = Row.get_dim_val env v

(* The hand-built constraint leaves [v] free and nothing to wait on. *)
let discharged name constr row v =
  let cs = [ Row.Rows_constr { r = [ row ]; constr; origin } ] in
  let remaining, env = solve cs in
  p name (Option.is_none (value env v));
  p_empty (name ^ ", nothing left to wait on") ~over:cs remaining

let refused name constr row =
  refused_with name ~sub:"Total_elems" (fun () ->
      ignore (solve [ Row.Rows_constr { r = [ row ]; constr; origin } ]))

let test_on_hand_built_constraints () =
  Stdio.printf "Hand-built constraints:\n";
  let v = Row.get_var () in
  let self = strided ~divided_by:[ v ] v in
  discharged "1 v / v over the empty row is satisfied, v free" (self ~coeff:1 ~denom:1)
    (closed_row []) v;
  discharged "2 v / v over a row of 2 is satisfied, v free" (self ~coeff:2 ~denom:1)
    (closed_row [ dim 2 ])
    v;
  discharged "1 v over the row [v] is satisfied, v free"
    (strided ~coeff:1 ~denom:1 ~divided_by:[] v)
    (closed_row [ Var v ]) v;
  refused "2 v / v over the empty row is refused" (self ~coeff:2 ~denom:1) (closed_row []);
  refused "1 v / v over a row of 3 is refused" (self ~coeff:1 ~denom:1) (closed_row [ dim 3 ]);
  refused "3 v / 2 v is no whole number of elements" (self ~coeff:3 ~denom:2) (closed_row []);
  (* What cancels leaves the rest of the product to the check. *)
  let w = Row.get_var () in
  let _, env =
    solve
      [
        Rows_constr
          {
            r = [ closed_row [ Var w; Var v ] ];
            constr = strided ~coeff:1 ~denom:1 ~divided_by:[] v;
            origin;
          };
      ]
  in
  p "the other variable takes the quotient, v stays free"
    (Option.equal Int.equal (value env w) (Some 1) && Option.is_none (value env v));
  (* A round-up total fixes [v] only up to the slack window: nothing cancels. *)
  let remaining, env =
    solve
      [
        Rows_constr
          {
            r = [ closed_row [] ];
            constr = strided ~round_up:true ~coeff:2 ~denom:1 ~divided_by:[ v ] v;
            origin;
          };
      ]
  in
  p "a round-up total stays waiting" (Option.is_none (value env v) && not (List.is_empty remaining));
  (* Once cancelled, the constraint holds no guess back: the free variable closes. *)
  let cs =
    [
      Row.Rows_constr
        {
          r = [ closed_row [ Var v ] ];
          constr = strided ~coeff:1 ~denom:1 ~divided_by:[] v;
          origin;
        };
      Terminal_dim (false, Var v, origin);
    ]
  in
  let remaining, env = solve ~stage:Stage7 cs in
  p "the cancelled variable closes at the end" (Option.equal Int.equal (value env v) (Some 1));
  p_empty "the cancelled variable closes, nothing left to wait on" ~over:cs remaining;
  (* An open row closes at the end to the quotient of the cancellation. *)
  let u = Row.get_var () and rest = Row.get_row_var () in
  let open_row : Row.t =
    { beg_dims = []; dims = [ Var u ]; bcast = Row_var rest; prov = Row.empty_provenance }
  in
  let cs =
    [
      Row.Rows_constr
        { r = [ open_row ]; constr = strided ~coeff:1 ~denom:1 ~divided_by:[] u; origin };
      Terminal_row (false, open_row, origin);
    ]
  in
  let remaining, env = solve ~stage:Stage7 cs in
  let closed = Row.subst_row env open_row in
  p "an open row closes empty around the cancelled variable"
    (Row.equal_bcast closed.bcast Broadcastable && List.length closed.dims = 1);
  p_empty "the open row closes, nothing left to wait on" ~over:cs remaining

let () =
  test_through_captures ();
  test_on_hand_built_constraints ()
