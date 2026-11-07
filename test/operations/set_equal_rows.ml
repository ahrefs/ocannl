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

(* A refusal whose message satisfies [f]. *)
let refused_with name refused ~f = p name (Option.exists refused ~f)
let set_equal ~swap r1 r2 = if swap then Shape.set_equal r2 r1 else Shape.set_equal r1 r2

let orders =
  List.concat_map [ true; false ] ~f:(fun first_r1 ->
      List.map [ false; true ] ~f:(fun swap -> (first_r1, swap)))

let order_name (first_r1, swap) =
  Printf.sprintf "%s registered first, set_equal %s"
    (if first_r1 then "r" else "s")
    (if swap then "s r" else "r s")

(* Binds [r] as a dimension and as a row in one inference, in the given order, and closes it: the
   last binding decides what the closed reference is. A row 2x3 and a dimension of 6. *)
let close_both_kinds order r =
  let dim_capture () = NTDSL.einsum1 ~capture_dims:[ r ] "r => r" (open_leaf "v") () in
  let ys =
    match order with
    | `Dim_then_row ->
        let y = dim_capture () in
        [ y; capture_row r (fixed "x" [ 2; 3 ]) ]
    | `Row_then_dim ->
        let y = capture_row r (fixed "x" [ 2; 3 ]) in
        [ y; dim_capture () ]
  in
  List.iter ys ~f:(fun y -> ignore (dims y : int list))

let order_kind = function `Dim_then_row -> "a row" | `Row_then_dim -> "a dimension"

(* Closes [r] on the row of [operand x], which the scheme's body makes and does not return: an
   explicit instance stands for the body, so no update step holds that row and no projections run on
   it. This is the one way a row with an [Affine] axis, or a [Concat] with a symbolic component,
   closes: a step holding one fails its projections. Returns the scheme's rendering. *)
let close_scheme_internal_row r ~arg operand =
  let s =
    Tensor.Scheme.trace ~label:"internal" 1 (fun args ->
        let x = List.hd_exn args in
        ignore (capture_row r (operand x) : Tensor.t);
        [ x ])
  in
  let x = arg () and res = open_leaf "res" in
  Tensor.Scheme.instantiate s ~args:[ x.Tensor.shape ] ~results:[ res.Tensor.shape ];
  ignore (dims res : int list);
  Tensor.Scheme.to_string s

(* The row of an operand reduced over a concatenated axis, which substitution sums though no
   variable of the spec gets solved. *)
let literal_concat _x =
  let c = open_leaf "c" in
  ignore (NTDSL.einsum1 "2^3 => " c () : Tensor.t);
  c

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
  (* Across a closure a row reference's variable belongs to the reset environment, but the reference
     carries its row (gh-ocannl-1689): the call equates the other row with it, not with its product.
     Before, the call was refused. *)
  List.iter [ false; true ] ~f:(fun swap ->
      List.iter
        [ None; Some [ 2; 3 ]; Some [ 3; 2 ] ]
        ~f:(fun later ->
          let name =
            Printf.sprintf "one reference closed, set_equal %s%s"
              (if swap then "s r" else "r s")
              (match later with
              | None -> ""
              | Some sizes ->
                  ", the open row fixed to "
                  ^ String.concat ~sep:"x" (List.map sizes ~f:Int.to_string))
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
              let x = fixed "x" [ 2; 3 ] in
              (* Reading the capturing operation's dims closes the inference that bound [r]. *)
              ignore (dims (capture_row r1 x) : int list);
              p
                (name ^ ": the closure carried r's row and its product")
                (Option.equal Int.equal r1.var_ref.solved_dim (Some 6)
                && Option.exists r1.carried_row ~f:(fun row -> List.length row = 2));
              let w = open_leaf "w" in
              ignore (capture_row r2 w : Tensor.t);
              let outcome =
                shape_error (fun () ->
                    set_equal ~swap r1 r2;
                    Option.iter later ~f:(fix w);
                    ignore (dims w : int list))
              in
              match (later, outcome) with
              | (None | Some [ 2; 3 ]), None ->
                  show name [ ("w", w) ];
                  p
                    (name ^ ": the open row takes the closed row")
                    (List.equal Int.equal (dims w) [ 2; 3 ])
              | Some [ 3; 2 ], refused ->
                  Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
                  p (name ^ ": a row of the same product is refused") (Option.is_some refused)
              | _, refused ->
                  Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
                  p (name ^ ": the open row takes the closed row") false)));
  (* A closed reference is what its binding at the closure was: bound last as a row, it carries the
     row; bound last as a dimension, it carries a number. Projection derivation runs newest step
     first, and used to leave [var] holding the oldest step's kind (gh-ocannl-1969). *)
  List.iter
    [ (`Dim_then_row, [ 3; 2 ], false); (`Row_then_dim, [ 3; 2 ], true) ]
    ~f:(fun (order, sizes, accepted) ->
      List.iter [ false; true ] ~f:(fun swap ->
          let name =
            Printf.sprintf "closed after a %s, set_equal %s, the open row fixed to %s"
              (match order with
              | `Dim_then_row -> "dimension then a row capture"
              | `Row_then_dim -> "row then a dimension capture")
              (if swap then "s r" else "r s")
              (String.concat ~sep:"x" (List.map sizes ~f:Int.to_string))
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
              close_both_kinds order r1;
              let w = open_leaf "w" in
              ignore (capture_row r2 w : Tensor.t);
              let refused =
                shape_error (fun () ->
                    set_equal ~swap r1 r2;
                    fix w sizes;
                    ignore (dims w : int list))
              in
              Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
              if accepted then
                p
                  (name ^ ": tied by product, a row of the same product is accepted")
                  (Option.is_none refused)
              else p (name ^ ": a row of the same product is refused") (Option.is_some refused))));
  (* Both closed: each carries its row. The cases after the last below pin that the comparison is
     made at the call. *)
  List.iter
    [ ([ 2; 3 ], true); ([ 3; 2 ], false) ]
    ~f:(fun (sizes, equal) ->
      let name =
        Printf.sprintf "both references closed, rows 2x3 and %s"
          (String.concat ~sep:"x" (List.map sizes ~f:Int.to_string))
      in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
          ignore (dims (capture_row r1 (fixed "x" [ 2; 3 ])) : int list);
          ignore (dims (capture_row r2 (fixed "y" sizes)) : int list);
          let refused =
            shape_error (fun () ->
                Shape.set_equal r1 r2;
                ignore (dims (open_leaf "z") : int list))
          in
          Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
          if equal then p (name ^ ": equal rows are accepted") (Option.is_none refused)
          else p (name ^ ": different rows are refused") (Option.is_some refused)));
  (* An unbound reference [set_equal] relates to a row reference is a number to the call -- the
     row's product, given to it or taken from it -- so bound to a row afterwards it would leave the
     two rows tied by product only, where binding it first makes the same call a [Row_eq] (the cases
     above): that binding is refused (gh-ocannl-1655). Bound to a dimension, product equality is the
     meaning either way, so it stands. *)
  let numbered ~row_solved ~swap name =
    let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
    if row_solved then ignore (capture_row r2 (fixed "x" [ 2; 3 ]) : Tensor.t)
    else (
      Shape.set_dim r1 6;
      ignore (capture_row r2 (open_leaf "w") : Tensor.t));
    p (name ^ ": r is unbound at the call") (match r1.var with `Not_set_yet -> true | _ -> false);
    set_equal ~swap r1 r2;
    r1
  in
  List.iter [ false; true ] ~f:(fun row_solved ->
      List.iter [ false; true ] ~f:(fun swap ->
          let name =
            Printf.sprintf "unbound %s, set_equal %s, then bound to a row"
              (if row_solved then "r, s solved by a fixed operand" else "r set_dim'd")
              (if swap then "s r" else "r s")
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r1 = numbered ~row_solved ~swap name in
              let refused =
                shape_error (fun () -> ignore (capture_row r1 (open_leaf "v") : Tensor.t))
              in
              Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
              refused_with (name ^ ": the row binding is refused, naming the row reference") refused
                ~f:(fun msg ->
                  String.is_substring msg ~substring:"Variable reference r "
                  && String.is_substring msg ~substring:"the row reference s ");
              (* The refusal was rewound; fresh work in the same session still infers. *)
              let v = open_leaf "v" in
              fix v [ 2; 3 ];
              p
                (name ^ ": fresh work after the refusal infers")
                (List.equal Int.equal (dims v) [ 2; 3 ]));
          let name =
            Printf.sprintf "unbound %s, set_equal %s, then bound to a dimension"
              (if row_solved then "r, s solved by a fixed operand" else "r set_dim'd")
              (if swap then "s r" else "r s")
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r1 = numbered ~row_solved ~swap name in
              let v = open_leaf "v" in
              let y = NTDSL.einsum1 ~capture_dims:[ r1 ] "r => r" v () in
              show name [ ("v", v) ];
              p
                (name ^ ": the dimension takes the row's product")
                (List.equal Int.equal (dims y) [ 6 ]))));
  (* A row holding a symbolic extent has no numeric product, so the closure of the inference that
     bound its reference records nothing: equating that reference with a dimension afterwards would
     tie the dimension to a dead row variable (gh-ocannl-1655). Refused, naming the closure. A
     closed row that kept its product still gives it to the dimension. *)
  let closed_row ~symbolic =
    let r = Shape.get_variable_ref "r" in
    let x =
      if symbolic then (
        let seq = fst (Ir.Indexing.get_static_symbol ~static_range:6 Ir.Indexing.Empty) in
        let e = Shape.get_variable_ref "e" in
        Shape.set_sym_dim e seq;
        let x = open_leaf "x" in
        ignore (NTDSL.einsum1 ~capture_dims:[ e ] "e => e" x () : Tensor.t);
        x)
      else fixed "x" [ 2; 3 ]
    in
    (* Reading the capturing operation's dims closes the inference that bound [r]. *)
    ignore (dims (capture_row r x) : int list);
    r
  in
  List.iter [ false; true ] ~f:(fun swap ->
      let order = if swap then "d r" else "r d" in
      let name = "closed row holding a symbolic extent, set_equal " ^ order in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r = closed_row ~symbolic:true in
          p
            (name ^ ": the closure recorded nothing for r")
            (Option.is_none r.var_ref.solved_dim && Option.is_none r.var_ref.solved_sym);
          let d = Shape.get_variable_ref "d" in
          ignore (NTDSL.einsum1 ~capture_dims:[ d ] "d => d" (open_leaf "w") () : Tensor.t);
          let refused = shape_error (fun () -> set_equal ~swap r d) in
          Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
          refused_with (name ^ ": the call is refused, naming the closure") refused ~f:(fun msg ->
              String.is_substring msg ~substring:"reference r was bound by shape inference"
              && String.is_substring msg ~substring:"has since closed"));
      let name =
        "closed row holding a symbolic extent, set_equal with a row, "
        ^ if swap then "s r" else "r s"
      in
      Verdict.case name (fun () ->
          (* A row holding a symbolic extent has no product, but its closure carries the row
             (gh-ocannl-1689): a Row-Row call relates it. *)
          Tensor.unsafe_reinitialize ();
          let r = closed_row ~symbolic:true in
          let s = Shape.get_variable_ref "s" in
          let w = open_leaf "w" in
          ignore (capture_row s w : Tensor.t);
          let refused =
            shape_error (fun () ->
                set_equal ~swap r s;
                ignore (dims w : int list))
          in
          Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
          p
            (name ^ ": the open row takes the symbolic row")
            (Option.is_none refused
            && String.is_substring
                 (Shape.to_string_hum ~style:Axis_size w.Tensor.shape)
                 ~substring:"<=6"));
      let name = "closed row that kept its product, set_equal " ^ order in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r = closed_row ~symbolic:false in
          let d = Shape.get_variable_ref "d" in
          let w = open_leaf "w" in
          let y = NTDSL.einsum1 ~capture_dims:[ d ] "d => d" w () in
          set_equal ~swap r d;
          show name [ ("w", w) ];
          p
            (name ^ ": the dimension takes the closed product")
            (List.equal Int.equal (dims y) [ 6 ])));
  (* What a closed reference is, for the uses reading its kind (gh-ocannl-1969): [set_scale] refuses
     a row and takes a dimension, at either position and for a factor of 1, which goes through
     [set_equal]. Each of the two orders used to read as the other. *)
  List.iter
    [ (`Dim_then_row, None); (`Row_then_dim, Some 6) ]
    ~f:(fun (order, closed_dim) ->
      List.iter
        [ (1, `Large, 6); (2, `Large, 3); (1, `Small, 6); (2, `Small, 12) ]
        ~f:(fun (factor, position, want) ->
          let name =
            Printf.sprintf "set_scale ~factor:%d, %s closed reference is %s" factor
              (match position with `Large -> "the larger," | `Small -> "the smaller,")
              (order_kind order)
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r = Shape.get_variable_ref "r" and s = Shape.get_variable_ref "s" in
              close_both_kinds order r;
              let refused =
                shape_error (fun () ->
                    match position with
                    | `Large -> Shape.set_scale ~factor r s
                    | `Small -> Shape.set_scale ~factor s r)
              in
              Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
              match closed_dim with
              | None ->
                  refused_with (name ^ ": a row is refused") refused ~f:(fun msg ->
                      String.is_substring msg ~substring:"requires two dimension variables")
              | Some _ ->
                  p
                    (name ^ ": the dimension relates the other reference")
                    (Option.is_none refused
                    && Option.equal Int.equal s.var_ref.solved_dim (Some want)))));
  (* The setters dispatch on the solved value first, so a closed reference never reaches their [var]
     arms: an equal number is no change, and a different one or a symbol is refused, whichever kind
     the reference closed as. *)
  List.iter [ `Dim_then_row; `Row_then_dim ] ~f:(fun order ->
      let name = "setters on a closed reference that is " ^ order_kind order in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r = Shape.get_variable_ref "r" in
          close_both_kinds order r;
          Shape.set_dim r 6;
          p
            (name ^ ": the number it holds is no change")
            (Option.equal Int.equal r.var_ref.solved_dim (Some 6));
          p
            (name ^ ": a different number is refused")
            (Option.is_some (shape_error (fun () -> Shape.set_dim r 7)));
          let seq = fst (Ir.Indexing.get_static_symbol ~static_range:6 Ir.Indexing.Empty) in
          p (name ^ ": a symbol is refused")
            (Option.is_some (shape_error (fun () -> Shape.set_sym_dim r seq)))));
  (* An unbound reference [set_equal] pairs with a closed row is a number to the call, so bound to a
     row afterwards it would leave the two tied by product: refused, as for an open row. Paired with
     a closed dimension, a row bound afterwards is tied by product by design. *)
  List.iter
    [ (`Dim_then_row, false); (`Row_then_dim, true) ]
    ~f:(fun (order, accepted) ->
      List.iter [ false; true ] ~f:(fun swap ->
          let name =
            Printf.sprintf
              "unbound u, set_equal %s with a closed reference that is %s, then u a row"
              (if swap then "r u" else "u r")
              (order_kind order)
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r = Shape.get_variable_ref "r" and u = Shape.get_variable_ref "u" in
              close_both_kinds order r;
              set_equal ~swap u r;
              let refused =
                shape_error (fun () -> ignore (capture_row u (fixed "z" [ 3; 2 ]) : Tensor.t))
              in
              Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
              if accepted then
                p (name ^ ": tied by product, 3x2 is accepted") (Option.is_none refused)
              else
                refused_with (name ^ ": the row binding is refused") refused ~f:(fun msg ->
                    String.is_substring msg ~substring:"Variable reference u "
                    && String.is_substring msg ~substring:"the row reference r "))));
  (* A closed row with a [Concat] axis of concrete components is a row of their sum: the axis is 5,
     and the row is that row, not the product 5 (gh-ocannl-1969, gh-ocannl-2019). *)
  let closed_concat name r =
    let scheme = close_scheme_internal_row r ~arg:(fun () -> fixed "x" []) literal_concat in
    p
      (name ^ ": the row closed with its axis summed")
      (String.is_substring scheme ~substring:"capture r = 5");
    p
      (name ^ ": the closure carried the sum")
      (Option.equal (List.equal Int.equal)
         (Option.map r.Shape.carried_row ~f:(List.map ~f:Row.dim_to_int_exn))
         (Some [ 5 ]))
  in
  List.iter
    [ ([ 5 ], true); ([ 1; 5 ], false); ([ 6 ], false) ]
    ~f:(fun (sizes, accepted) ->
      let name =
        Printf.sprintf "closed row with a concatenated axis, captured again over %s"
          (String.concat ~sep:"x" (List.map sizes ~f:Int.to_string))
      in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r = Shape.get_variable_ref "r" in
          closed_concat name r;
          let refused =
            shape_error (fun () -> ignore (dims (capture_row r (fixed "w" sizes)) : int list))
          in
          Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
          p
            (name ^ if accepted then ": the same row is accepted" else ": another row is refused")
            (Bool.equal accepted (Option.is_none refused))));
  let name = "closed row with a concatenated axis, captured again over an open operand" in
  Verdict.case name (fun () ->
      Tensor.unsafe_reinitialize ();
      let r = Shape.get_variable_ref "r" in
      closed_concat name r;
      let w = open_leaf "w" in
      ignore (capture_row r w : Tensor.t);
      show name [ ("w", w) ];
      p (name ^ ": the open operand takes the one axis") (List.equal Int.equal (dims w) [ 5 ]));
  List.iter
    [ ([ 5 ], true); ([ 1; 5 ], false) ]
    ~f:(fun (sizes, accepted) ->
      List.iter [ false; true ] ~f:(fun swap ->
          let name =
            Printf.sprintf "closed row with a concatenated axis, set_equal %s with a closed %s"
              (if swap then "s r" else "r s")
              (String.concat ~sep:"x" (List.map sizes ~f:Int.to_string))
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r = Shape.get_variable_ref "r" and s = Shape.get_variable_ref "s" in
              closed_concat name r;
              ignore (dims (capture_row s (fixed "y" sizes)) : int list);
              let refused = shape_error (fun () -> set_equal ~swap r s) in
              Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
              p
                (name ^ if accepted then ": equal rows are accepted" else ": other rows are refused")
                (Bool.equal accepted (Option.is_none refused)))));
  (* Over a symbolic extent a compound axis has no sum and the row no product. Such a row is not
     carried: in the next inference it would reach projections, which no step holding it survives.
     Its reuse stays refused, naming the closure. *)
  List.iter
    [ ("a => a^3", "a concatenated axis"); ("a => 2*a", "an affine axis") ]
    ~f:(fun (spec, axis) ->
      let name = "closed row with " ^ axis ^ " over a symbolic extent" in
      Verdict.case name (fun () ->
          Tensor.unsafe_reinitialize ();
          let r = Shape.get_variable_ref "r" in
          let operand x =
            let a = Shape.get_variable_ref "a" in
            Shape.set_sym_dim a
              (fst (Ir.Indexing.get_static_symbol ~static_range:6 Ir.Indexing.Empty));
            NTDSL.einsum1 ~capture_dims:[ a ] spec x ()
          in
          ignore (close_scheme_internal_row r ~arg:(fun () -> open_leaf "x") operand : string);
          p
            (name ^ ": the closure carried nothing")
            (Option.is_none r.carried_row && Option.is_none r.var_ref.solved_dim);
          let refused = shape_error (fun () -> ignore (capture_row r (open_leaf "w") : Tensor.t)) in
          refused_with (name ^ ": a capture is refused, naming the closure") refused ~f:(fun msg ->
              String.is_substring msg ~substring:"has since closed")));
  (* Two closed rows are compared at the call (gh-ocannl-1969): the mismatch raises from [set_equal]
     itself, with no later operation to expose it, and leaves nothing behind -- no rewind, and the
     next closure passes. The verdict is the solver's, so it covers what no comparison of sizes
     would: a claim-free 1 is not a plain 1, and a symbol's maximum is not its size. *)
  let closed_operands =
    [
      ("2x3 and 3x2", (fun () -> fixed "x" [ 2; 3 ]), fun () -> fixed "y" [ 3; 2 ]);
      ("2x3 and 6", (fun () -> fixed "x" [ 2; 3 ]), fun () -> fixed "y" [ 6 ]);
      ("no axes and 1", (fun () -> fixed "x" []), fun () -> fixed "y" [ 1 ]);
      ("1 and a claim-free 1", (fun () -> fixed "x" [ 1 ]), fun () -> NTDSL.number 2.);
    ]
  in
  List.iter closed_operands ~f:(fun (rows, x, y) ->
      List.iter [ false; true ] ~f:(fun swap ->
          let name =
            Printf.sprintf "closed rows %s, set_equal %s" rows (if swap then "s r" else "r s")
          in
          Verdict.case name (fun () ->
              Tensor.unsafe_reinitialize ();
              let r1 = Shape.get_variable_ref "r" and r2 = Shape.get_variable_ref "s" in
              ignore (dims (capture_row r1 (x ())) : int list);
              ignore (dims (capture_row r2 (y ())) : int list);
              let refused =
                match set_equal ~swap r1 r2 with
                | () -> None
                | exception Row.Shape_error (msg, _) -> Some msg
              in
              Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
              refused_with (name ^ ": refused at the call, naming both") refused ~f:(fun msg ->
                  String.is_prefix msg ~prefix:"Shape.set_equal: the closed row references ");
              let z = open_leaf "z" in
              fix z [ 4; 5 ];
              p
                (name ^ ": the next closure passes, with no rewind")
                (List.equal Int.equal (dims z) [ 4; 5 ]))));
  Verdict.case "closed rows, a symbol and its maximum" (fun () ->
      Tensor.unsafe_reinitialize ();
      let name = "closed rows, a symbol and its maximum" in
      let r = closed_row ~symbolic:true and s = Shape.get_variable_ref "s" in
      ignore (dims (capture_row s (fixed "y" [ 6 ])) : int list);
      let refused =
        match Shape.set_equal r s with () -> None | exception Row.Shape_error (msg, _) -> Some msg
      in
      Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
      p (name ^ ": refused at the call") (Option.is_some refused));
  (* A row and a dimension share a number, and that tie does not pass through the dimension
     (gh-ocannl-1969, the maintainer's call; [capture_per_call]'s [rebind_scenarios] pins it for one
     reference bound row, dimension, row). Two rows tied to one dimension keep their own axes, over
     one product, within one inference and across closures. An explicit [set_equal] of the two rows
     is what makes them one row. *)
  let name = "row = dim = row, within one inference" in
  let tied ~explicit =
    Tensor.unsafe_reinitialize ();
    let r = Shape.get_variable_ref "r" and s = Shape.get_variable_ref "s" in
    let d = Shape.get_variable_ref "d" in
    let a = open_leaf "a" and b = open_leaf "b" in
    ignore (capture_row r a : Tensor.t);
    ignore (capture_row s b : Tensor.t);
    let v = NTDSL.einsum1 ~capture_dims:[ d ] "d => d" (open_leaf "v") () in
    Shape.set_equal r d;
    Shape.set_equal d s;
    if explicit then Shape.set_equal r s;
    fix a [ 2; 3 ];
    fix b [ 3; 2 ];
    (a, b, v)
  in
  Verdict.case name (fun () ->
      let a, b, v = tied ~explicit:false in
      show name [ ("a", a); ("b", b); ("the dimension", v) ];
      p
        (name ^ ": each row keeps its axes, over one product")
        (List.equal Int.equal (dims a) [ 2; 3 ]
        && List.equal Int.equal (dims b) [ 3; 2 ]
        && List.equal Int.equal (dims v) [ 6 ]));
  let name = name ^ ", then the rows set_equal" in
  Verdict.case name (fun () ->
      let refused =
        shape_error (fun () ->
            let a, b, _ = tied ~explicit:true in
            ignore (dims a : int list);
            ignore (dims b : int list))
      in
      Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
      p (name ^ ": 2x3 and 3x2 are refused") (Option.is_some refused));
  let name = "row = dim = row, across closures" in
  let closed_tied () =
    Tensor.unsafe_reinitialize ();
    let r = Shape.get_variable_ref "r" and s = Shape.get_variable_ref "s" in
    let d = Shape.get_variable_ref "d" in
    ignore (dims (capture_row r (fixed "a" [ 2; 3 ])) : int list);
    ignore (dims (capture_row s (fixed "b" [ 3; 2 ])) : int list);
    let v = NTDSL.einsum1 ~capture_dims:[ d ] "d => d" (open_leaf "v") () in
    Shape.set_equal r d;
    Shape.set_equal d s;
    (r, s, v)
  in
  Verdict.case name (fun () ->
      let _, _, v = closed_tied () in
      p (name ^ ": the dimension has the one product") (List.equal Int.equal (dims v) [ 6 ]));
  let name = name ^ ", then the rows set_equal" in
  Verdict.case name (fun () ->
      let r, s, _ = closed_tied () in
      let refused = shape_error (fun () -> Shape.set_equal r s) in
      Option.iter refused ~f:(Stdio.printf "%s: refused: %s\n" name);
      p (name ^ ": 2x3 and 3x2 are refused") (Option.is_some refused))
