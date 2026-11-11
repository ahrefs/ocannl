(* gh-ocannl-2149: the fixed output index of an extent-1 accumulator must not lose the reduction
   loop when its computation is captured. Keep the extent-2 control beside it. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Tn = Ir.Tnode
module Asgns = Ir.Assignments

let%cd init_then_sum_asn ~t ~t1 ~t2 ~projections:_ =
  t =: id t1 ~logic:".";
  t =+ id t2 ~logic:"ab => a"

let no_grad2 ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp

let init_then_sum =
  Tensor.binop
    ~compose_op:(Shape.Einsum ("a; ab => a", []))
    ~op_asn:init_then_sum_asn ~grad_asn:no_grad2 ~op_label:"init_sum"
    ~grad_spec:Tensor.Prohibit_grad

type placement = Virtual | Materialized | Auto

let name = function Virtual -> "virtual" | Materialized -> "materialized" | Auto -> "auto"

let run n placement =
  Tensor.unsafe_reinitialize ();
  let src =
    NTDSL.init ~prec:Ir.Ops.single ~l:"src" ~o:[ n ]
      ~f:(fun i -> Float.of_int (7 + (10 * i.(0))))
      ()
  in
  let v =
    NTDSL.init ~prec:Ir.Ops.single ~l:"updates" ~o:[ n; 2 ]
      ~f:(fun i -> Float.of_int (1 + (10 * i.(0)) + i.(1)))
      ()
  in
  let s = init_then_sum src v () in
  (match placement with
  | Virtual -> Train.set_virtual s.Tensor.value
  | Materialized -> Train.set_materialized s.Tensor.value
  | Auto -> ());
  let%op y = s + s in
  Train.set_materialized y.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) y in
  let plc = Context.placements ctx in
  p
    (name placement ^ ": accumulator placement")
    (match placement with
    | Virtual | Auto -> Tn.Placements.known_virtual plc s.Tensor.value
    | Materialized -> Tn.Placements.known_non_virtual plc s.Tensor.value);
  let values = Context.get_values ctx y.Tensor.value in
  Context.release ctx;
  values

let extent n =
  Stdio.printf "extent %d\n" n;
  let expected = Array.init n ~f:(fun i -> Float.of_int (2 * (10 + (30 * i)))) in
  let mat = run n Materialized in
  p "materialized values match the closed form" (Array.equal Float.equal mat expected);
  List.iter [ Virtual; Auto ] ~f:(fun placement ->
      let values = run n placement in
      p
        (name placement ^ ": values match the closed form")
        (Array.equal Float.equal values expected);
      p (name placement ^ ": values match materialization") (Array.equal Float.equal values mat))

let () =
  List.iter [ 2; 1 ] ~f:(fun n -> Verdict.case (Printf.sprintf "extent %d" n) (fun () -> extent n))
