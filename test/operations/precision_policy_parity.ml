(* Executable coverage for Ocannl.Precision_policy (gh-ocannl-492 task 1): a two-layer MLP run at
   the default f32 is the oracle for the same model re-run under a bf16 policy with identical
   parameter values (copied via set_values, which converts to the storage precision). A structural
   check on settled precisions alone would miss arithmetic bugs (see the repo's testing notes), so
   the parity of executed outputs is the load-bearing assertion; the precision printout pins the
   class assignments: params and activations bf16, gradients left at single (grad_prec = None), the
   [except]-ed param left at single, and the explicitly-single input left alone (user [Specified]
   wins over policy).

   The uniform() default init of w1/w2 routes through uint4x32 threefry chains; the policy must
   leave those precisions untouched or compilation itself fails (threefry ops require a uint4x32
   target), so the bf16 leg compiling and running is the regression guard for the integer/uint4x32
   protection.

   Two more legs (gh-ocannl-1865) are the mixed-precision recipe's: f32 masters behind half cast
   twins ([Mixed_prec.with_master_weights]), activations reduced to half, under [param_prec = None]
   and under a uniform policy. A master's initializer takes the master's precision, so its executed
   initial values are f32 draws -- not all f16-representable, as they were when the walk narrowed
   the initializer as an activation. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train
module IDX = Train.IDX
module PP = Ocannl.Precision_policy
module MP = Ocannl.Mixed_prec
module Tn = Ir.Tnode

let hid = 6
let out = 2
let x_vals = [| 0.5; -0.3; 0.9; 0.1 |]

let make_model () =
  let%op f x =
    ({ w2 } * relu (({ w1 } * x) + { b1 = 0.; o = [ hid ] })) + { b2 = 0.; o = [ out ] }
  in
  f

let find_param root name =
  List.find_exn (Set.to_list root.Tensor.params) ~f:(fun t ->
      String.equal (Tn.label_name t.Tensor.value) name)

let prec_str tn = Ir.Ops.prec_string (Lazy.force tn.Tn.storage_prec)

(* Whether [x] is a finite IEEE binary16 value: within its range, and a multiple of the half spacing
   at its binade (11 significant bits; subnormals below 2^-14 share the spacing 2^-24). *)
let f16_representable x =
  Float.(x = 0.)
  || Float.(abs x <= 65504.)
     &&
     let _, ex = Float.frexp x in
     let scaled = Float.ldexp x (10 - Int.max (ex - 1) (-14)) in
     Float.(round_towards_zero scaled = scaled)

(* One leg: build the model, let [prepare] see the loss root (the policy application site), compile
   the gradient update, initialize params, optionally overwrite w1/w2 with given values, run, and
   return what the comparisons need. *)
let run_leg ?(build = make_model) base_ctx ~input_l ~prepare ~w_vals =
  (* Each model is compiled from the base, not from the previous seed owner. *)
  Tensor.set_random_seed ();
  let f = build () in
  let x =
    NTDSL.init ~l:input_l ~prec:Ir.Ops.single
      ~o:[ Array.length x_vals ]
      ~f:(function [| i |] -> x_vals.(i) | _ -> assert false)
      ()
  in
  let y = f x in
  let%op loss = y ++ "i=>0" in
  prepare loss;
  Train.set_materialized y.Tensor.value;
  let ctx = Train.init_params base_ctx IDX.empty loss in
  let ctx, routine = Train.to_routine ctx IDX.empty (Train.grad_update loss) in
  let w1 = find_param loss "w1" and w2 = find_param loss "w2" in
  let ctx =
    match w_vals with
    | Some (v1, v2) ->
        let ctx = Context.set_values ctx w1.Tensor.value v1 in
        Context.set_values ctx w2.Tensor.value v2
    | None -> ctx
  in
  let ctx = Context.run ctx routine in
  (ctx, loss, y, x)

let () =
  Tensor.unsafe_reinitialize ();
  let base_ctx = Context.auto () in

  (* Reference leg: default single precision everywhere. *)
  let ctx_a, loss_a, y_a, _x_a =
    run_leg base_ctx ~input_l:"xa" ~prepare:(fun _ -> ()) ~w_vals:None
  in
  let w1_a = find_param loss_a "w1" and w2_a = find_param loss_a "w2" in
  let w_vals =
    (Context.get_values ctx_a w1_a.Tensor.value, Context.get_values ctx_a w2_a.Tensor.value)
  in
  let y_vals_a = Context.get_values ctx_a y_a.Tensor.value in
  let loss_a_v = (Context.get_values ctx_a loss_a.Tensor.value).(0) in

  (* Policy leg: params and activations bf16, grads left alone, b2 excepted. *)
  let policy =
    {
      PP.param_prec = Some Ir.Ops.bfloat16;
      activation_prec = Some Ir.Ops.bfloat16;
      grad_prec = None;
    }
  in
  let except tn = String.equal (Tn.label_name tn) "b2" in
  let ctx_b, loss_b, y_b, x_b =
    run_leg base_ctx ~input_l:"xb"
      ~prepare:(fun loss -> PP.apply ~except policy loss)
      ~w_vals:(Some w_vals)
  in
  let w1_b = find_param loss_b "w1" and b2_b = find_param loss_b "b2" in
  let y_vals_b = Context.get_values ctx_b y_b.Tensor.value in
  let loss_b_v = (Context.get_values ctx_b loss_b.Tensor.value).(0) in

  (* Structural pins on settled precisions. *)
  Stdio.printf "w1 param (policy leg): %s\n" (prec_str w1_b.Tensor.value);
  Stdio.printf "b2 param (excepted): %s\n" (prec_str b2_b.Tensor.value);
  Stdio.printf "y activation (policy leg): %s\n" (prec_str y_b.Tensor.value);
  Stdio.printf "y activation (reference leg): %s\n" (prec_str y_a.Tensor.value);
  Stdio.printf "w1 gradient (grad_prec = None): %s\n"
    (prec_str (Option.value_exn w1_b.Tensor.diff).Tensor.grad);
  Stdio.printf "input x (explicitly single): %s\n" (prec_str x_b.Tensor.value);

  (* Executed parity: bf16 has ~2-3 significant decimal digits; the tolerances are loose bounds on
     accumulated rounding over two matmuls, not tight envelopes. *)
  let max_err =
    Array.foldi y_vals_a ~init:0. ~f:(fun i acc va ->
        Float.max acc (Float.abs (va -. y_vals_b.(i))))
  in
  (* The parity tolerances cannot reject an input-independent forward: if both legs collapsed to one
     constant (or to zeros) every check below would read [true]. The reference's own spread must
     exceed the tolerance it gates. *)
  let ref_spread =
    Array.fold y_vals_a ~init:0. ~f:(fun acc v -> Float.max acc (Float.abs (v -. y_vals_a.(0))))
  in
  Verdict.p "reference forward varies past the parity tolerance" Float.(ref_spread > 0.05);
  Verdict.p "forward parity within 0.05" Float.(max_err < 0.05);
  Verdict.p "loss parity within 0.05" Float.(Float.abs (loss_a_v -. loss_b_v) < 0.05);

  (* Mixed-precision legs: f32 masters behind half twins, with the params left alone and with a
     uniform policy, whose [param_prec] the masters' explicit precision outranks. The step computes
     gradients only, so the masters still hold their initial values. *)
  let master_leg ~name ~input_l policy =
    let ctx, loss, y, _x =
      run_leg base_ctx ~input_l
        ~build:(fun () -> MP.with_master_weights ~prec:Ir.Ops.half make_model)
        ~prepare:(fun loss -> PP.apply policy loss)
        ~w_vals:None
    in
    let w1 = find_param loss "w1" and w2 = find_param loss "w2" in
    Stdio.printf "w1 master (%s): %s\n" name (prec_str w1.Tensor.value);
    Stdio.printf "y activation (%s): %s\n" name (prec_str y.Tensor.value);
    let initial =
      Array.to_list
        (Array.append
           (Context.get_values ctx w1.Tensor.value)
           (Context.get_values ctx w2.Tensor.value))
    in
    Verdict.p_all
      (name ^ ": the f32 masters' initial values lie in the default init's [-0.25, 0.25) range")
      initial ~f:(fun v -> Float.(v >= -0.25 && v < 0.25));
    Verdict.p_exists (name ^ ": some f32 master's initial value is not f16-representable") initial
      ~f:(fun v -> not (f16_representable v))
  in
  master_leg ~name:"param_prec = None" ~input_l:"xc"
    { PP.param_prec = None; activation_prec = Some Ir.Ops.half; grad_prec = None };
  master_leg ~name:"uniform half" ~input_l:"xd" (PP.uniform Ir.Ops.half);

  (* A node both an initializer and the model read is an activation, although the walk reaches the
     initializer first ([w] precedes [shared] in [y]). [param_prec] is a value here: a node the
     policy left alone would follow its consumers into the same answer. *)
  let shared = NTDSL.number 3. in
  let w = TDSL.param ~param_init:(NTDSL.pointmul shared (NTDSL.number 2.)) "shared_w" () in
  let%op y = w *. shared in
  PP.apply
    { PP.param_prec = Some Ir.Ops.bfloat16; activation_prec = Some Ir.Ops.half; grad_prec = None }
    y;
  Verdict.p "a parameter reading a node the model reads keeps param_prec"
    (Ir.Ops.equal_prec (Lazy.force w.Tensor.value.storage_prec) Ir.Ops.bfloat16);
  Verdict.p "a node both an initializer and the model read is an activation"
    (Ir.Ops.equal_prec (Lazy.force shared.Tensor.value.storage_prec) Ir.Ops.half);
  (* A plain parameter under [param_prec = None] is no f32 master: lib/precision_policy.mli sends a
     reader who wants one to [with_master_weights], because the parameter follows its consumers. The
     same graph with its consumers left alone is what makes the half the policy's doing. *)
  let plain_param_prec activation_prec =
    let input = NTDSL.number 5. in
    let w = TDSL.param "plain_w" () in
    let%op y = w *. input in
    PP.apply { PP.param_prec = None; activation_prec; grad_prec = None } y;
    Lazy.force w.Tensor.value.storage_prec
  in
  Verdict.p "under param_prec = None a plain parameter follows its half consumers"
    (Ir.Ops.equal_prec (plain_param_prec (Some Ir.Ops.half)) Ir.Ops.half);
  Verdict.p "a plain parameter whose consumers stay at the default stays there"
    (Ir.Ops.equal_prec (plain_param_prec None) Ir.Ops.single)
