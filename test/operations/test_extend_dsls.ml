open! Base
open Ocannl.Nn_blocks.DSL_modules

(* Read by [counted_power]'s exponent and bumped by its base, to show which runs first. *)
let counter = ref 2

(* An exponent the helper reads through an escape. *)
let exponent = ref 2.0

(* Test module-level expansion *)
[%%extend_dsls
let my_tensor x = TDSL.O.( + ) x !.2.0 + !.1.0

let my_complex_op a b c =
  let d = a * b in
  let e = d + c in
  relu e

let my_random_op () = { u1 = uniform () } - { u2 = uniform () }

(* A bare [**.] tail is the named [pointpow], exponent first (gh-ocannl-2320); the exponent stays a
   float, an integer literal converted. *)
let squared x = x **. 2
let float_squared x = x **. 2.0
let escaped_squared x = x **. [%oc !exponent]

(* Return the intermediate unchanged so its own gradient state is observable. *)
let non_tail_square x =
  let square = x **. 2 in
  fun ?label:_ () -> square

(* The infix application runs the exponent before the base: the power is the counter's value before
   the base bumps it. *)
let counted_power x =
  [%oc
    Int.incr counter;
    x]
  **. Float.of_int !counter]

(* The above should create TDSL and NTDSL modules, each containing the functions with appropriate
   DSL operators. *)

let () =
  (* Test that the functions are available in each DSL module *)
  let open! DSL_modules.TDSL.O in
  let x = uniform () in
  let result = my_complex_op (my_tensor x) (my_random_op ()) x in
  Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:true `Inline result

let () =
  let open! DSL_modules.NTDSL.O in
  let x = uniform () in
  let result = my_complex_op (my_tensor x) (my_random_op ()) x in
  if Set.exists result.Tensor.params ~f:(fun p -> Option.is_some p.Tensor.diff) then
    failwith "NTDSL %%extend_dsls helper unexpectedly created gradient-bearing inline params";
  Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:true `Inline result

let () =
  let open Verdict.Claims in
  let source = Tensor.term_init [| 2.; -4.; 8.; -16. |] ~grad_spec:Prohibit_grad () in
  let with_grad = Tensor.term_init [| 2.; -4.; 8.; -16. |] ~grad_spec:Require_grad () in
  p "TDSL non-tail power carries a gradient"
    (Option.is_some (DSL_modules.TDSL.O.non_tail_square with_grad).Tensor.diff);
  p "NTDSL non-tail power has no gradient"
    (Option.is_none (DSL_modules.NTDSL.O.non_tail_square with_grad).Tensor.diff);
  (* The counted power starts from 2, which its exponent reads before its base bumps it to 3. Only
     the NTDSL helper shows it: the TDSL one is shape-schemed, and its first application also runs
     the body under a trace: an effectful body is outside gh-ocannl-1751's determinism premise. Its
     expansion is pinned by test/ppx/test_extend_dsls_power. *)
  ignore (DSL_modules.TDSL.O.counted_power : ?label:string list -> Tensor.t -> Tensor.t);
  counter := 2;
  let ntdsl_counted = DSL_modules.NTDSL.O.counted_power source in
  p "the effectful base ran once, after its exponent" (!counter = 3);
  let plain_escaped = [%op source **. [%oc !exponent]] in
  let squares = [| 4.; 16.; 64.; 256. |] in
  let cases =
    [
      ("TDSL integer power", DSL_modules.TDSL.O.squared source);
      ("NTDSL integer power", DSL_modules.NTDSL.O.squared source);
      ("TDSL float power", DSL_modules.TDSL.O.float_squared source);
      ("NTDSL float power", DSL_modules.NTDSL.O.float_squared source);
      ("TDSL escaped power", DSL_modules.TDSL.O.escaped_squared source);
      ("NTDSL escaped power", DSL_modules.NTDSL.O.escaped_squared source);
      ("Plain %op escaped power", plain_escaped);
      ("NTDSL effectful-base power", ntdsl_counted);
    ]
  in
  List.iter cases ~f:(fun (_, t) -> Ocannl.Train.set_materialized t.Tensor.value);
  let combined =
    List.reduce_exn (List.map cases ~f:(fun (_, t) -> t)) ~f:(fun a b -> TDSL.O.(a + b))
  in
  let ctx = Ocannl.Train.forward_once (Context.cpu ()) combined in
  List.iter cases ~f:(fun (name, t) ->
      pf "%s computes the squares" name
        (Array.equal Float.equal (Context.get_values ctx t.Tensor.value) squares))
