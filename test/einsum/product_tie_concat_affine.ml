(* gh-ocannl-2086: a product tie over a dimension solved to a concatenation or a window. A capture
   used as a dimension in one spec ([n => n]) and as a row in another ([..n.. => ..n..]) ties the
   row's total to the dimension: a [Total_elems] over a [Strided_var n]. When a third spec solves
   [n] to a [Concat] ([a^2]) or an [Affine] ([2*a], [a<+k]) while a part is still unsolved, the
   substitution died on [Failure "NOT IMPLEMENTED YET"]. The tie now waits for the parts: it keeps
   [n], whose value the environment sums once the parts are known, and a closed row's total reaches
   [n] as an equation that solves the part.

   Every combination is pinned: the three specs, both registration orders (the dimension's solving
   spec before or after the row's), both forcing orders, and the part set before the specs, set
   after them, or left to a closed row of [2; 3] to determine. Open rows with known leading axes
   still determine the parts exactly; a downstream consumer still sees the unsized row's axes and
   values. A closed row no part fits is refused with a shape error. With nothing sized, affine parts
   close to 1 before the total sizes the row (gh-ocannl-2159); unsized concat components remain
   deliberately refused (gh-ocannl-1656). Every operand holds its own offsets, so each cell read can
   be told from the rest, and every value is a small integer, so equality is exact. Claims are about
   shape closure: every execution context is [Context.cpu ()]. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let open_operand label = NTDSL.term ~label:[ label ] ~fetch_op:Range_over_offsets ()

let closed_operand dims =
  let n = List.fold dims ~init:1 ~f:( * ) in
  Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad (Array.init n ~f:Float.of_int) ~batch_dims:[]
    ~input_dims:[] ~output_dims:dims ()

let dims t = Array.to_list (Lazy.force t.Tensor.value.dims)

let values t =
  Train.set_materialized t.Tensor.value;
  Context.get_values (Train.forward_once (Context.cpu ()) t) t.Tensor.value

let dims_string ds = "[" ^ String.concat ~sep:"; " (List.map ds ~f:Int.to_string) ^ "]"
let iota n = Array.init n ~f:Float.of_int

(* How [n] is made of [a], and what [a]'s result reads of the operand [0, 1, ..., n - 1]. *)
type form = {
  name : string;
  spec : string;
  kernel : bool;  (** The spec's window takes a kernel [k], captured and set to 3. *)
  part_of : int -> int;  (** [a] for a given [n]. *)
  total_of : int -> int;  (** [n] for a given [a]. *)
  reads : int -> float array;  (** The result for a given [a]. *)
}

let forms =
  [
    {
      name = "a concatenation";
      spec = "a^2 => a";
      kernel = false;
      part_of = (fun n -> n - 2);
      total_of = (fun a -> a + 2);
      reads = iota;
    };
    {
      name = "a stride";
      spec = "2*a => a";
      kernel = false;
      part_of = (fun n -> n / 2);
      total_of = (fun a -> 2 * a);
      reads = (fun a -> Array.init a ~f:(fun i -> Float.of_int (2 * i)));
    };
    {
      name = "a window";
      spec = "a<+k => a";
      kernel = true;
      part_of = (fun n -> n - 2);
      total_of = (fun a -> a + 2);
      reads = (fun a -> Array.init a ~f:(fun i -> Float.of_int ((3 * i) + 3)));
    };
  ]

type sizing = Early | Late | Closed_row of int list | Open_row_prefix of int | Unsized | Consumed

let sizing_name = function
  | Early -> "the part sized before the specs"
  | Late -> "the part sized after them"
  | Closed_row ds -> "a closed row of " ^ dims_string ds
  | Open_row_prefix d -> "an open row with leading axis " ^ Int.to_string d
  | Unsized -> "nothing sized"
  | Consumed -> "nothing sized, the row doubled by a consumer"

let label form ~part_first ~row_first_forced sizing =
  Printf.sprintf "%s, %s registered first, %s forced first, %s" form.name
    (if part_first then "the part's spec" else "the row's spec")
    (if row_first_forced then "the row" else "the part")
    (sizing_name sizing)

(* [y1] reads the operand along [n], [y2] reads [a] of it, [y3] reads the row of [n]; the values of
   [y2] and [y3], forced in the given order. *)
let build form ~part_first ~row_first_forced sizing =
  let n = Shape.get_variable_ref "n" and a = Shape.get_variable_ref "a" in
  let k = Shape.get_variable_ref "k" in
  if Poly.equal sizing Early then Shape.set_dim a 2;
  let y1 = NTDSL.einsum1 ~capture_dims:[ n ] "n => n" (open_operand "q") () in
  let part () =
    let captures = if form.kernel then [ a; k ] else [ a ] in
    let y2 = NTDSL.einsum1 ~capture_dims:captures form.spec y1 () in
    if form.kernel then Shape.set_dim k 3;
    y2
  in
  let row () =
    let z =
      match sizing with
      | Closed_row ds -> closed_operand ds
      | Open_row_prefix d ->
          let prefix = Shape.get_variable_ref "prefix" in
          let z =
            NTDSL.einsum1 ~capture_dims:[ prefix ] "prefix, ..tail.. => prefix, ..tail.."
              (open_operand "z") ()
          in
          Shape.set_dim prefix d;
          z
      | _ -> open_operand "z"
    in
    NTDSL.einsum1 ~capture_dims:[ n ] "..n.. => ..n.." z ()
  in
  let y2, y3 =
    if part_first then
      let y2 = part () in
      (y2, row ())
    else
      let y3 = row () in
      (part (), y3)
  in
  let y3 = match sizing with Consumed -> NTDSL.add y3 y3 () | _ -> y3 in
  if Poly.equal sizing Late then Shape.set_dim a 2;
  let v2, v3 =
    if row_first_forced then
      let v3 = values y3 in
      (values y2, v3)
    else
      let v2 = values y2 in
      (v2, values y3)
  in
  (y1, y2, y3, v2, v3)

let pin form ~part_first ~row_first_forced sizing =
  let label = label form ~part_first ~row_first_forced sizing in
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let y1, y2, y3, v2, v3 = build form ~part_first ~row_first_forced sizing in
      let want_n, want_row =
        match sizing with
        | Closed_row ds -> (List.fold ds ~init:1 ~f:( * ), ds)
        | Open_row_prefix d -> (d, [ d ])
        | Early | Late -> (4, [ 4 ])
        | Unsized | Consumed ->
            let n = form.total_of 1 in
            (n, if n = 1 then [] else [ n ])
      in
      let want_a = form.part_of want_n in
      Stdio.printf "%s: dims %s, %s, %s\n" label
        (dims_string (dims y1))
        (dims_string (dims y2))
        (dims_string (dims y3));
      pf "%s: the dimension is the row's total" label (List.equal Int.equal (dims y1) [ want_n ]);
      pf "%s: the part is solved" label (List.equal Int.equal (dims y2) [ want_a ]);
      pf "%s: the row keeps its axes" label (List.equal Int.equal (dims y3) want_row);
      pf_close "%s: the part reads its cells" label v2 (form.reads want_a) ~tol:(fun _ -> 0.);
      let row_values =
        if Poly.equal sizing Consumed then Array.map (iota want_n) ~f:(fun x -> 2. *. x)
        else iota want_n
      in
      pf_close "%s: the row reads its cells" label v3 row_values ~tol:(fun _ -> 0.))

(* A shape error, not a crash: a closed row whose total no value of the part makes, or a concat
   component whose size no operand or capture determines. *)
let refused form ~part_first ~row_first_forced sizing ~needle =
  let label = label form ~part_first ~row_first_forced sizing in
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let outcome =
        match build form ~part_first ~row_first_forced sizing with
        | _ -> `Accepted
        | exception Row.Shape_error (msg, _) -> `Shape_error msg
        | exception e -> `Other (Exn.to_string e)
      in
      (match outcome with
      | `Shape_error msg -> Stdio.printf "%s: refused: %s\n" label msg
      | `Accepted -> Stdio.eprintf "%s: accepted\n" label
      | `Other e -> Stdio.eprintf "%s: raised %s\n" label e);
      pf "%s: refused with a shape error" label
        (match outcome with
        | `Shape_error msg -> String.is_substring msg ~substring:needle
        | `Accepted | `Other _ -> false))

let concatenation, stride = (List.nth_exn forms 0, List.nth_exn forms 1)
let incompatible_stride = "incompatible stride"
let components_exceed = "components exceed the concatenated size"

let () =
  List.iter forms ~f:(fun form ->
      List.iter [ true; false ] ~f:(fun part_first ->
          List.iter [ true; false ] ~f:(fun row_first_forced ->
              List.iter
                [ Early; Late; Closed_row [ 2; 3 ]; Open_row_prefix 4; Open_row_prefix 6 ]
                ~f:(pin form ~part_first ~row_first_forced);
              if phys_equal form concatenation then
                refused form ~part_first ~row_first_forced Unsized
                  ~needle:"closure does not guess one"
              else (
                pin form ~part_first ~row_first_forced Unsized;
                pin form ~part_first ~row_first_forced Consumed))));
  let plain =
    {
      name = "a plain variable";
      spec = "a => a";
      kernel = false;
      part_of = Fn.id;
      total_of = Fn.id;
      reads = iota;
    }
  in
  List.iter [ true; false ] ~f:(fun part_first ->
      List.iter [ true; false ] ~f:(fun row_first_forced ->
          pin plain ~part_first ~row_first_forced Unsized));
  List.iter [ true; false ] ~f:(fun part_first ->
      refused stride ~part_first ~row_first_forced:true (Closed_row [ 5 ])
        ~needle:incompatible_stride;
      refused concatenation ~part_first ~row_first_forced:true (Closed_row [ 1 ])
        ~needle:components_exceed)
