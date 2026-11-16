open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

module Extended = struct
  [%%extend_dsls let negative x = -x]
end

let () =
  let init = Tensor.term_init [| 2.; -4.; 8.; -16. |] ~grad_spec:Prohibit_grad in
  let source = init () in
  Train.set_materialized source.Tensor.value;
  let positive = [%op { positive = [%oc init] }] in
  let negative = [%op { negative = -source }] in
  let tdsl = Extended.DSL_modules.TDSL.O.negative source in
  let ntdsl = Extended.DSL_modules.NTDSL.O.negative source in
  List.iter [ positive; negative; tdsl; ntdsl ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let result = TDSL.O.(positive + negative + tdsl + ntdsl) in
  let ctx = Train.forward_once (Context.cpu ()) result in
  let values t = Context.get_values ctx t.Tensor.value in
  p "negated initializer is exactly the negation of the original"
    (Array.equal Float.equal (values negative) (Array.map (values positive) ~f:Float.neg));
  List.iter
    [ ("TDSL", tdsl); ("NTDSL", ntdsl) ]
    ~f:(fun (name, t) ->
      pf "extend_dsls %s negates its argument" name
        (Array.equal Float.equal (values t) [| -2.; 4.; -8.; 16. |]))
