open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* Fixtures pinning the order in which %op evaluates the components of the literals it gives a
   meaning of its own (gh-ocannl-1869): block tensors (list, array and tuple stacks), a concat's
   operand tuple, and an einsum's capture list. OCaml evaluates the elements of a list, array or
   tuple literal right to left, so a ppx that splices effectful components into one mints their
   tensors -- ids, and whatever else creation consumes -- in reverse source order. The expansion
   binds every effectful component but the last with [let], in source order.

   Both this file and its expansion golden are compiled and run: each case records the order its
   components were evaluated in and the ids they were given. *)

let evaluated = ref []

(* An effectful component: records its evaluation and mints a tensor. *)
let mk name =
  evaluated := name :: !evaluated;
  TDSL.range ~label:[ name ] 3

let id (t : Tensor.t) = t.value.id

let in_source_order what names f =
  evaluated := [];
  let (_ : Tensor.t) = f () in
  let order = List.rev !evaluated in
  Printf.printf "%s: evaluated %s\n" what (String.concat " " order);
  p (what ^ ": components are evaluated in source order") (order = names)

(* The ids follow: each component's tensor is minted when it is evaluated. *)
let ids_ascend what ts =
  p_alli (what ^ ": component ids ascend in source order") ts ~f:(fun i t ->
      i = 0 || id (List.nth ts (i - 1)) < id t)

let () =
  in_source_order "list block" [ "a"; "b"; "c" ] (fun () -> [%op [ mk "a"; mk "b"; mk "c" ]]);
  in_source_order "array block" [ "a"; "b"; "c" ] (fun () -> [%op [| mk "a"; mk "b"; mk "c" |]]);
  in_source_order "tuple block" [ "a"; "b"; "c" ] (fun () -> [%op mk "a", mk "b", mk "c"]);
  in_source_order "nested block" [ "a"; "b"; "c"; "d" ] (fun () ->
      [%op [ (mk "a", mk "b"); (mk "c", mk "d") ]]);
  in_source_order "concat operands" [ "a"; "b"; "c" ] (fun () ->
      [%op (mk "a", mk "b", mk "c") ++^ "i; j; k => i^j^k"]);
  in_source_order "concat operands with captures" [ "a"; "b" ] (fun () ->
      [%op (mk "a", mk "b") ++^ "i; j => i^j" [ "i"; "j" ]])

(* A component that is a variable or a constant has no effect to order, and stays where it was: only
   the effectful components before the last are bound. *)
let () =
  let x = mk "x" and y = mk "y" in
  in_source_order "variables among effectful components" [ "a"; "b" ] (fun () ->
      [%op [ x; mk "a"; y; mk "b" ]]);
  in_source_order "variables only" [] (fun () -> [%op [ x; y ]])

(* A binding scopes over the components after it, so it takes a name none of them mentions: here the
   user's own [__op_component1] is still the second component. *)
let () =
  let __op_component1 = mk "y" in
  let stacked = ref [] in
  let keep t =
    stacked := t :: !stacked;
    t
  in
  let (_ : Tensor.t) = [%op [ keep (mk "a"); keep __op_component1 ]] in
  p "a variable named like a binding is not captured"
    (match !stacked with
    | [ second; first ] -> second == __op_component1 && first != second
    | _ -> false)

(* The components' ids, read back through bindings the block tensor leaves in scope. *)
let () =
  let a = ref None and b = ref None and c = ref None in
  let keep r t =
    r := Some t;
    t
  in
  let (_ : Tensor.t) = [%op [ keep a (mk "a"); keep b (mk "b"); keep c (mk "c") ]] in
  ids_ascend "list block" (List.filter_map (fun r -> !r) [ a; b; c ])

(* A capture list mints its references, and passes them on, in the order it is written. *)
let () =
  let i, j, k =
    [%op
      let r = (mk "a", mk "b", mk "c") ++^ "i; j; k => i^j^k" [ "i"; "j"; "k" ] in
      ignore (r : Tensor.t);
      (i, j, k)]
  in
  p_alli "capture list: reference uids ascend in source order" [ i; j; k ] ~f:(fun n r ->
      n = 0 || (List.nth [ i; j; k ] (n - 1)).Shape.uid < r.Shape.uid)
