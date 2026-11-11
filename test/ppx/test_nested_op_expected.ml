open Ocannl.Operation.DSL_modules
open Verdict.Claims
let under_try =
  let open! TDSL.O in
    fun () ->
      let normal =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "normal") () in
      let recovered =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "recovered") () in
      fun fail ->
        try if fail then raise Exit else normal with | Exit -> recovered
let under_lazy =
  let open! TDSL.O in
    fun () ->
      let delayed =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "delayed") () in
      lazy delayed
let under_constructor =
  let open! TDSL.O in
    fun () ->
      let wrapped =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "wrapped") () in
      Some wrapped
let under_variant =
  let open! TDSL.O in
    fun () ->
      let tagged =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "tagged") () in
      `Tensor tagged
let ( let* ) x f = f x
let ( and* ) x y = (x, y)
let under_letop =
  let open! TDSL.O in
    fun () ->
      let first =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "first") () in
      let second =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "second") () in
      let body =
        (TDSL.param ?more_label:None ?value:None ?values:None
           ?param_init:None "body") () in
      let* x = first
      and* y = second in
      (+) ?label:(Some ["under_letop"]) ((+) ?label:None x y) body
let letop_outer_scope =
  let open! TDSL.O in
    fun ~n () ->
      let sized =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "sized") ~output_dims:[n]) () in
      let also_sized =
        ((TDSL.param ?more_label:None ?value:None ?values:None
            ?param_init:None "also_sized") ~output_dims:[n]) () in
      let* n = sized
      and* other = also_sized in
      (+) ?label:(Some ["letop_outer_scope"]) n other
let () =
  let choose = under_try () in
  p "try body parameter is shared" ((choose false) == (choose false));
  p "try handler parameter is shared" ((choose true) == (choose true));
  p "try body and handler parameters differ"
    ((choose false) != (choose true));
  (let t = under_lazy () in
   p "lazy parameter is shared" ((Lazy.force t) == (Lazy.force t));
   (let wrapped = under_constructor () in
    p "constructor contains a tensor"
      (match wrapped with
       | Some t -> List.mem "wrapped" (t.Tensor.value).label
       | None -> false);
    p "variant contains a tensor"
      (match under_variant () with
       | `Tensor t -> List.mem "tagged" (t.Tensor.value).label);
    p "let operators construct a tensor"
      (List.mem "under_letop" ((under_letop ()).Tensor.value).label)))
let () =
  p "let operator right-hand sides retain outer scope"
    (List.mem "letop_outer_scope"
       ((letop_outer_scope ~n:3 ()).Tensor.value).label)
let capture_callbacks xs =
  let ( let* ) xs f = List.map f xs in
  let open! TDSL.O in
    let* x = xs
     in
    let a = Shape.get_variable_ref "a" in
    let y = einsum1 ?label:(Some ["y"]) ~capture_dims:[a] "a => a" x in
    (y, a)
let () =
  let outputs = capture_callbacks [TDSL.range 3; TDSL.range 5] in
  p "binding operator callbacks get fresh dimension captures"
    (match outputs with
     | (_, a)::(_, b)::[] -> a.Shape.uid <> b.Shape.uid
     | _ -> false)
