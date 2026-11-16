(* gh-ocannl-2200: annotations preserve classification, tuple grouping and schemes. CPU-only: this
   tests the DSL translation, introducing no backend operation. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

let%op tail_plain () = fun x -> x + 1.0
let%op tail_typed () = (fun x -> x + 1.0 : Tensor.t -> Tensor.t)
let head_plain x = [%op add x x]
let head_typed x = [%op (add : _) x x]
let pipeline_plain x = [%op x |> add x]
let pipeline_typed x = [%op x |> (add : _) x]
let grouping_plain x = [%op Some (x + 1.0, x + 2.0)]
let grouping_typed x = [%op Some ((x + 1.0, x + 2.0) : Tensor.t * Tensor.t)]
let binary_plain x = [%op add (x, 2.0)]
let binary_typed x = [%op add ((x, 2.0) : Tensor.t * Tensor.t)]
let ternary_plain x = [%op fma (x, 2.0, 3.0)]

let ternary_typed x =
  [%op fma (((x, 2.0, 3.0) : Tensor.t * Tensor.t * Tensor.t) : Tensor.t * Tensor.t * Tensor.t)]

let argument_plain x = [%op fst (x + 1.0, x + 2.0)]
let argument_typed x = [%op fst ((x + 1.0, x + 2.0) : Tensor.t * Tensor.t)]

let events f =
  let log = ref [] in
  let result = Tensor.For_testing.observe_schemes f ~on_event:(fun e -> log := e :: !log) in
  (result, List.rev !log)

let () =
  Tensor.For_testing.set_op_shape_schemes (Some `Check);
  let plain, plain_trace = events tail_plain in
  let typed, typed_trace = events tail_typed in
  let traced = function Tensor.Scheme.Traced _ -> true | _ -> false in
  p_exists "plain tail traces a scheme" plain_trace ~f:traced;
  p_exists "constrained tail traces a scheme" typed_trace ~f:traced;
  let x = Tensor.term_init [| 2.; 5.; -3. |] ~output_dims:[ 3 ] ~grad_spec:Prohibit_grad () in
  let a, plain_checks = events (fun () -> plain x) in
  let b, typed_checks = events (fun () -> typed x) in
  let checked = function Tensor.Scheme.Checked _ -> true | _ -> false in
  p_exists "plain tail checks its scheme" plain_checks ~f:checked;
  p_exists "constrained tail checks its scheme" typed_checks ~f:checked;
  let pair = function Some (a, b) -> [ a; b ] | None -> assert false in
  let cases =
    [
      ("tail", [ a ], [ b ]);
      ("head", [ head_plain x ], [ head_typed x ]);
      ("pipeline head", [ pipeline_plain x ], [ pipeline_typed x ]);
      ("constructor grouping", pair (grouping_plain x), pair (grouping_typed x));
      ("argument grouping", [ argument_plain x ], [ argument_typed x ]);
      ("uncurried binary", [ binary_plain x ], [ binary_typed x ]);
      ("uncurried ternary", [ ternary_plain x ], [ ternary_typed x ]);
    ]
  in
  let outputs = List.concat_map cases ~f:(fun (_, a, b) -> a @ b) in
  List.iter outputs ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let combined = List.reduce_exn outputs ~f:TDSL.O.add in
  let ctx = Train.forward_once (Context.cpu ()) combined in
  List.iter cases ~f:(fun (name, a, b) ->
      let pairs = List.zip_exn a b in
      p_all (name ^ " values agree") pairs ~f:(fun (a, b) ->
          Array.equal Float.equal
            (Context.get_values ctx a.Tensor.value)
            (Context.get_values ctx b.Tensor.value)));
  Tensor.For_testing.set_op_shape_schemes None
