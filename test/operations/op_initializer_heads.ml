(* gh-ocannl-2237: operator initialization and annotated uncurried heads use the DSL signatures. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* A power the user wrote with its qualifier keeps it: the initializer is this module's operation,
   not [NTDSL.pointpow]. *)
module Custom_power = struct
  module TDSL = struct
    include TDSL

    module O = struct
      include TDSL.O

      let ( **. ) base _exponent = NTDSL.relu base
    end
  end

  let explicit_power source e = [%op { w = TDSL.O.( **. ) source e }]
end

let () =
  let source = Tensor.term_init [| 2.; -4.; 8.; -16. |] ~grad_spec:Prohibit_grad () in
  let other = Tensor.term_init [| 1.; 2.; -3.; 4. |] ~grad_spec:Prohibit_grad () in
  List.iter [ source; other ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  (* Initializers share these computed sources with the ordinary calls below. Establish them before
     parameter initialization reads them from the context. *)
  let ctx = Train.forward_once (Context.cpu ()) (TDSL.O.add source other) in
  (* A non-literal exponent: the power arm's other route. *)
  let e = 3.0 in
  let cases =
    [
      ("tensor initializer", [%op { w = source + other }], [| 3.; -2.; 5.; -12. |]);
      ("literal initializer", [%op { w = source + 0.0 }], [| 2.; -4.; 8.; -16. |]);
      ("subtraction initializer", [%op { w = source - 2.0 }], [| 0.; -6.; 6.; -18. |]);
      ("multiplication initializer", [%op { w = source *. 2.0 }], [| 4.; -8.; 16.; -32. |]);
      ("division initializer", [%op { w = source /. 2.0 }], [| 1.; -2.; 4.; -8. |]);
      ("integer power initializer", [%op { w = source **. 2 }], [| 4.; 16.; 64.; 256. |]);
      ("float power initializer", [%op { w = source **. 2.0 }], [| 4.; 16.; 64.; 256. |]);
      ("non-literal power initializer", [%op { w = source **. e }], [| 8.; -64.; 512.; -4096. |]);
      ( "effectful-base power initializer",
        (let n = ref 2 in
         [%op
           {
             w =
               [%oc
                 Int.incr n;
                 source]
               **. Float.of_int !n;
           }]),
        [| 4.; 16.; 64.; 256. |] );
      ("explicit power head", Custom_power.explicit_power source e, [| 2.; 0.; 8.; 0. |]);
      ("named binary initializer", [%op { w = add source other }], [| 3.; -2.; 5.; -12. |]);
      ("named unary initializer", [%op { w = relu source }], [| 2.; 0.; 8.; 0. |]);
      ("named ternary initializer", [%op { w = fma source 2.0 3.0 }], [| 7.; -5.; 19.; -29. |]);
      ("binary head", [%op (add : _) (source, other)], [| 3.; -2.; 5.; -12. |]);
      ( "binary head and tuple",
        [%op (add : _) ((source, other) : Tensor.t * Tensor.t)],
        [| 3.; -2.; 5.; -12. |] );
      ("ternary head", [%op (fma : _) (source, 2.0, 3.0)], [| 7.; -5.; 19.; -29. |]);
      ( "ternary head and tuple",
        [%op ((fma : _) : _) (((source, 2.0, 3.0) : Tensor.t * Tensor.t * Tensor.t) : _)],
        [| 7.; -5.; 19.; -29. |] );
    ]
  in
  List.iter cases ~f:(fun (_, t, _) -> Train.set_materialized t.Tensor.value);
  let combined = List.reduce_exn (List.map cases ~f:(fun (_, t, _) -> t)) ~f:TDSL.O.add in
  let ctx = Train.forward_once ctx combined in
  List.iter cases ~f:(fun (name, t, expected) ->
      pf "%s computes its expected values" name
        (Array.equal Float.equal (Context.get_values ctx t.Tensor.value) expected))
