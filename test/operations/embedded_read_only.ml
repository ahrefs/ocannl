open Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train

(* Keep graph identities identical: random draws also depend on node ids. The first draw absorbs the
   singleton seed's forward code, but is not a parameter of the second graph. *)
let draw ~seed ~initialize =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ~seed ();
  let rs = Tensor.get_random_seed () in
  let unused = NTDSL.uniform () () in
  let w = TDSL.param "probe_w" ~output_dims:[ 8 ] () in
  let base = Context.auto () in
  let base =
    if initialize then
      let c, r = Context.compile ~name:"probe_seed" base rs.Tensor.forward Train.IDX.empty in
      Context.run c r
    else base
  in
  let ctx = Train.init_params base Train.IDX.empty w in
  let values = Context.get_values ctx w.Tensor.value in
  ignore (Stdlib.Sys.opaque_identity unused : Tensor.t);
  Context.release ctx;
  values

let refused ~node f =
  match f () with
  | _ -> false
  | exception Utils.User_error msg ->
      String.is_substring msg ~substring:"required input nodes not initialized"
      && String.is_substring msg ~substring:node

let () =
  Verdict.p "an initializer whose seed owner did not run is refused"
    (refused ~node:"random_seed_42" (fun () -> draw ~seed:42 ~initialize:false));
  let seeded = draw ~seed:42 ~initialize:true in
  let zero = draw ~seed:0 ~initialize:true in
  Verdict.p "explicitly initialized seeds produce distinct draws" (not (Poly.equal seeded zero))

let input_case ~zero_valid =
  Tensor.unsafe_reinitialize ();
  let x =
    Tensor.term ~label:[ "embedded_input" ] ~grad_spec:Tensor.Prohibit_grad ~output_dims:[ 4 ] ()
  in
  if zero_valid then Ir.Tnode.mark_zero_valid x.value;
  let y = NTDSL.pointmul x (NTDSL.number 2.) () in
  Train.set_materialized x.value;
  Train.set_materialized y.value;
  let comp = Tensor.consume_forward_code y in
  let ctx, routine =
    Context.compile ~name:"embedded_input" (Context.auto ()) comp Train.IDX.empty
  in
  Verdict.p "the input is embedded and read-only"
    (Set.mem comp.embedded_nodes x.value && not (Set.mem routine.outputs x.value));
  if zero_valid then (
    let ctx = Context.run ctx routine in
    Verdict.p "declared zero-valid embedded input runs from zero"
      (Poly.equal (Context.get_values ctx y.value) [| 0.; 0.; 0.; 0. |]);
    Context.release ctx)
  else (
    Verdict.p "unwritten embedded read-only input is refused"
      (refused ~node:"embedded_input" (fun () -> Context.run ctx routine));
    let ctx = Context.run (Context.set_values ctx x.value [| 1.; 3.; 7.; 11. |]) routine in
    Verdict.p "uploading the embedded input permits the correct result"
      (Poly.equal (Context.get_values ctx y.value) [| 2.; 6.; 14.; 22. |]);
    Context.release ctx)

let () =
  input_case ~zero_valid:false;
  input_case ~zero_valid:true;
  Tensor.unsafe_reinitialize ();
  (* Larger than the literal-inline limit, hence registered host data. *)
  let values = Array.init 32 ~f:(fun i -> Float.of_int (i + 1)) in
  let x = Tensor.term_init values ~grad_spec:Tensor.Prohibit_grad () in
  let y = NTDSL.pointmul x (NTDSL.number 2.) () in
  Train.set_materialized y.value;
  let comp = Tensor.consume_forward_code y in
  Verdict.p "the literal is embedded and host-initialized"
    (Set.mem comp.embedded_nodes x.value && Ir.Host_inits.mem x.value);
  let ctx, routine =
    Context.compile ~name:"embedded_literal" (Context.auto ()) comp Train.IDX.empty
  in
  let ctx = Context.run ctx routine in
  Verdict.p "embedded host literal initializes at link and runs"
    (Poly.equal (Context.get_values ctx y.value) (Array.map values ~f:(fun x -> 2. *. x)));
  Context.release ctx
