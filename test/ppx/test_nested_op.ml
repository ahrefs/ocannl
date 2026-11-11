open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* Each wrapper must translate its child and lift its parameter to the unit stage. *)
let%op under_try () fail =
  try if fail then [%oc raise Exit] else { normal } with Exit -> { recovered }
[@@ocannl.no_scheme]

let%op under_lazy () = lazy { delayed } [@@ocannl.no_scheme]
let%op under_constructor () = Some { wrapped } [@@ocannl.no_scheme]
let%op under_variant () = `Tensor { tagged } [@@ocannl.no_scheme]
let ( let* ) x f = f x
let ( and* ) x y = (x, y)

let%op under_letop () =
  let* x = { first } and* y = { second } in
  x + y + { body }
[@@ocannl.no_scheme]

(* A sibling binding's pattern is not in scope in these right-hand sides. *)
let%op letop_outer_scope ~n () =
  let* n = { sized; o = [ n ] } and* other = { also_sized; o = [ n ] } in
  n + other
[@@ocannl.no_scheme]

let () =
  let choose = under_try () in
  p "try body parameter is shared" (choose false == choose false);
  p "try handler parameter is shared" (choose true == choose true);
  p "try body and handler parameters differ" (choose false != choose true);
  let t = under_lazy () in
  p "lazy parameter is shared" (Lazy.force t == Lazy.force t);
  let wrapped = under_constructor () in
  p "constructor contains a tensor"
    (match wrapped with Some t -> List.mem "wrapped" t.Tensor.value.label | None -> false);
  p "variant contains a tensor"
    (match under_variant () with `Tensor t -> List.mem "tagged" t.Tensor.value.label);
  p "let operators construct a tensor" (List.mem "under_letop" (under_letop ()).Tensor.value.label)

let () =
  p "let operator right-hand sides retain outer scope"
    (List.mem "letop_outer_scope" (letop_outer_scope ~n:3 ()).Tensor.value.label)
