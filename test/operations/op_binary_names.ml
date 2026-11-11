open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* This is a DSL binding test, deliberately executed on cc. It introduces no backend operation. *)
let check name build =
  Verdict.case name (fun () ->
      Tensor.unsafe_reinitialize ();
      let x = Tensor.term_init [| 2.; 8.; 4.; -2. |] ~grad_spec:Prohibit_grad () in
      let y = Tensor.term_init [| 4.; 2.; 4.; 8. |] ~grad_spec:Prohibit_grad () in
      let curried, uncurried, infix, explicit = build x y in
      List.iter [ curried; uncurried; infix; explicit ] ~f:(fun t ->
          Train.set_materialized t.Tensor.value);
      let%op combined = curried + uncurried + infix + explicit in
      let ctx = Train.forward_once (Context.cpu ()) combined in
      let values t = Context.get_values ctx t.Tensor.value in
      let want = values infix in
      List.iter
        [ ("curried", curried); ("uncurried", uncurried); ("explicit", explicit) ]
        ~f:(fun (form, t) ->
          pf "%s %s equals its infix form" name form (Array.equal Float.equal (values t) want);
          pf "%s %s forwards its label" name form
            (List.mem t.Tensor.value.label form ~equal:String.equal)))

let () =
  check "add" (fun x y ->
      let%op curried = add x y in
      let%op uncurried = add (x, y) in
      let%op infix = x + y in
      let explicit = TDSL.O.add ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  check "sub" (fun x y ->
      let%op curried = sub x y in
      let%op uncurried = sub (x, y) in
      let%op infix = x - y in
      let explicit = TDSL.O.sub ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  check "mul" (fun x y ->
      let%op curried = mul x y in
      let%op uncurried = mul (x, y) in
      let%op infix = x *. y in
      let explicit = TDSL.O.mul ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  check "div" (fun x y ->
      let%op curried = div x y in
      let%op uncurried = div (x, y) in
      let%op infix = x /. y in
      let explicit = TDSL.O.div ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  check "lt" (fun x y ->
      let%op curried = lt x y in
      let%op uncurried = lt (x, y) in
      let%op infix = x < y in
      let explicit = TDSL.O.lt ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  check "le" (fun x y ->
      let%op curried = le x y in
      let%op uncurried = le (x, y) in
      let%op infix = x <= y in
      let explicit = TDSL.O.le ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  check "eq" (fun x y ->
      let%op curried = eq x y in
      let%op uncurried = eq (x, y) in
      let%op infix = x = y in
      let explicit = TDSL.O.eq ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  check "ne" (fun x y ->
      let%op curried = ne x y in
      let%op uncurried = ne (x, y) in
      let%op infix = x <> y in
      let explicit = TDSL.O.ne ?label:(Some [ "explicit" ]) x y in
      (curried, uncurried, infix, explicit));
  let left = 2 and right = 5 in
  (* Each call is in a translated position, not the untranslated body of [%op let ... in ...]. *)
  let max = Int.max in
  let%op chosen = max left right in
  p "local curried integer max retains its OCaml type" (Int.equal chosen 5);
  let%op piped = right |> max left in
  p "local piped integer max retains its OCaml type" (Int.equal piped 5);
  let max (left, right) = Int.max left right in
  let chosen = [%op max (left, right)] in
  p "local tuple integer max retains its OCaml type" (Int.equal chosen 5);
  let chosen = [%op fst (left, right)] in
  p "fst keeps its tuple argument" (Int.equal chosen left);
  let id x = x in
  let%op chosen = id left in
  p "local unary id retains its OCaml type" (Int.equal chosen left);
  let mul3 x y z = x * y * z in
  let%op chosen = mul3 left right right in
  p "local curried mul3 retains its OCaml type" (Int.equal chosen 50);
  let mul3 (x, y, z) = x * y * z in
  let chosen = [%op mul3 (left, right, right)] in
  p "local tuple mul3 retains its OCaml type" (Int.equal chosen 50)
