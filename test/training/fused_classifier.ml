(* Fused cross-entropy classifier (gh-464): checks that Nn_blocks.cross_entropy_loss 1. is
   numerically stable on extreme logits (log-sum-exp; naive log(softmax) is non-finite), 2. produces
   dlogits = softmax(logits) - targets directly against the logits gradient, 3. keeps all
   softmax-probabilities intermediates Virtual: no [batch, vocab]-sized node other than the logits
   value, the logits gradient, and the targets is materialized in the forward+backprop routine, 4.
   splits into few kernel-fission segments (structural analysis, backend-independent). *)

open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule

let () =
  Utils.settings.fixed_state_for_init <- Some 42;
  Tensor.unsafe_reinitialize ();
  (* Sizes large enough that the fission analysis annotates the nests as parallel (the segment
     census below would degenerate to a single coalesced serial segment on tiny tensors). *)
  let batch = 2048 and vocab = 512 in
  (* Row 1: huge positive outlier with the target on a tiny-probability class (naive log(softmax)
     gives 0 * -inf = nan here). Row 2: all large negative. Row 3: mixed extremes. All other rows:
     moderate pseudo-random logits. *)
  let logit b v =
    if b = 1 then if v = 0 then 500. else Float.of_int v /. 100.
    else if b = 2 then -500. -. Float.of_int (v * 13 % 7)
    else if b = 3 then (if v % 2 = 0 then 300. else -300.) +. Float.of_int (v % 5)
    else (Float.of_int (((b * 31) + (v * 17)) % 23) /. 3.7) -. 3.0
  in
  let target_idx b = if b = 1 then 1 else ((b * 7) + 3) % vocab in
  (* Double-precision oracle: per-row loss = lse(x) - x[target], probs = exp(x - lse). *)
  let lse row =
    let m = Array.fold row ~init:Float.neg_infinity ~f:Float.max in
    m +. Float.log (Array.fold row ~init:0. ~f:(fun acc x -> acc +. Float.exp (x -. m)))
  in
  let row b = Array.init vocab ~f:(logit b) in
  let row_loss b = lse (row b) -. logit b (target_idx b) in
  let prob b v = Float.exp (logit b v -. lse (row b)) in
  let expected_loss = Array.init batch ~f:row_loss |> Array.fold ~init:0. ~f:( +. ) in

  let logits =
    Operation.init ~l:"logits" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ vocab ]
      ~f:(function [| b; v |] -> logit b v | _ -> assert false)
      ~grad_spec:Tensor.Require_grad ()
  in
  let targets =
    NTDSL.init ~l:"targets" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ vocab ]
      ~f:(function [| b; v |] -> if v = target_idx b then 1. else 0. | _ -> assert false)
      ()
  in
  let loss = Nn_blocks.cross_entropy_loss ~spec:"... | v" () ~logits ~targets in
  Train.set_materialized (Option.value_exn ~here:[%here] logits.Tensor.diff).grad;
  let update = Train.grad_update loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let stash = ref None in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        [ opt ])
      ctx update IDX.empty
  in
  let ctx = Context.run ctx routine in
  let open Operation.At in
  (* 1. Numerical stability and correctness of the loss value. *)
  let fused_loss = (ctx, loss).@[0] in
  let rel_err = Float.(abs (fused_loss - expected_loss) / abs expected_loss) in
  Verdict.p "fused loss is finite" (Float.is_finite fused_loss);
  Verdict.p "fused loss matches log-sum-exp oracle (rel err < 1e-4)" Float.(rel_err < 1e-4);

  (* 2. dlogits = softmax(logits) - targets, sampled on the extreme rows and a few others. *)
  let sample f =
    Array.concat_map [| 0; 1; 2; 3; 17; 1000 |] ~f:(fun b -> Array.init vocab ~f:(f b))
  in
  let actual = sample (fun b v -> (ctx, logits).@%{[| b; v |]}) in
  let expected = sample (fun b v -> prob b v -. if v = target_idx b then 1. else 0.) in
  Verdict.p_close ~difference_only:true "dlogits = softmax - targets (max abs err < 1e-4)" actual
    expected ~tol:(fun _ -> Stdlib.Float.pred 1e-4);

  (* 3. Virtualness census over the whole forward+backprop routine: every [batch, vocab]-sized node
     except the logits value, the logits gradient, and the targets must stay Virtual. *)
  let opt = Option.value_exn ~here:[%here] !stash in
  let plc = Context.placements ctx in
  let allowed =
    [
      logits.Tensor.value;
      (Option.value_exn ~here:[%here] logits.Tensor.diff).grad;
      targets.Tensor.value;
    ]
  in
  let big_nodes =
    Hashtbl.keys opt.LL.traced_store |> List.filter ~f:(fun tn -> Tn.num_elems tn >= batch * vocab)
  in
  let big_non_virtual =
    List.filter big_nodes ~f:(fun tn -> not (Tn.Placements.known_virtual plc tn))
  in
  let unexpected =
    List.filter big_non_virtual ~f:(fun tn ->
        not (List.mem allowed tn ~equal:(fun a b -> Tn.equal a b)))
  in
  Verdict.p_empty "probabilities and other [batch, vocab] intermediates stay virtual"
    ~over:big_nodes unexpected;
  List.iter unexpected ~f:(fun tn -> printf "  unexpectedly materialized: %s\n" (Tn.debug_name tn));
  Verdict.p_all "logits value and gradient are materialized" allowed ~f:(fun tn ->
      not (Tn.Placements.known_virtual plc tn));

  (* 4. Kernel-fission segment census (structural, using the metal analysis on the captured lowered
     code; runs identically under every configured backend). *)
  let segs =
    Sched.maybe_default_schedules ~facts:Ir.Backend_intf.metal_facts ~static_indices:[] opt
  in
  printf "fission segments of forward+backprop: %d\n" (List.length segs);

  (* 5. Masked and normalized variant: positions with mask=0 are excluded, sum divided by the
     kept-position count. *)
  let keep b = b % 4 <> 2 in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 1 ]
      ~f:(function [| b; _ |] -> if keep b then 1. else 0. | _ -> assert false)
      ()
  in
  let kept = List.count (List.init batch ~f:Fn.id) ~f:keep in
  let normalize_by = TDSL.O.(!.(Float.of_int kept)) in
  let masked_loss_t =
    Nn_blocks.cross_entropy_loss ~spec:"... | v" ~mask ~normalize_by () ~logits ~targets
  in
  let ctx2 = Train.forward_once (Context.auto ()) masked_loss_t in
  let masked_loss = (ctx2, masked_loss_t).@[0] in
  let expected_masked =
    (List.init batch ~f:Fn.id |> List.filter ~f:keep
    |> List.fold ~init:0. ~f:(fun acc b -> acc +. row_loss b))
    /. Float.of_int kept
  in
  let rel_err2 = Float.(abs (masked_loss - expected_masked) / abs expected_masked) in
  Verdict.p "masked+normalized loss matches oracle (rel err < 1e-4)" Float.(rel_err2 < 1e-4);

  (* 6. Class axes in the input row (the attention-style spec convention, e.g. "... | t -> ..."):
     the final reduction must still produce the scalar loss. *)
  let batch_in = 8 and vocab_in = 16 in
  let target_in b = if b = 1 then 1 else ((b * 7) + 3) % vocab_in in
  let row_loss_in b = lse (Array.init vocab_in ~f:(logit b)) -. logit b (target_in b) in
  let logits_in =
    NTDSL.init ~l:"logits_in" ~prec:Ir.Ops.single ~b:[ batch_in ] ~i:[ vocab_in ] ~o:[]
      ~f:(function [| b; v |] -> logit b v | _ -> assert false)
      ()
  in
  let targets_in =
    NTDSL.init ~l:"targets_in" ~prec:Ir.Ops.single ~b:[ batch_in ] ~i:[ vocab_in ] ~o:[]
      ~f:(function [| b; v |] -> if v = target_in b then 1. else 0. | _ -> assert false)
      ()
  in
  let loss_in =
    Nn_blocks.cross_entropy_loss ~spec:"... | v -> ..." () ~logits:logits_in ~targets:targets_in
  in
  let ctx4 = Train.forward_once (Context.auto ()) loss_in in
  let input_axis_loss = (ctx4, loss_in).@[0] in
  let expected_in = List.init batch_in ~f:row_loss_in |> List.fold ~init:0. ~f:( +. ) in
  let rel_err3 = Float.(abs (input_axis_loss - expected_in) / abs expected_in) in
  Verdict.p "input-axis spec (attention convention) matches oracle (rel err < 1e-4)"
    Float.(rel_err3 < 1e-4);

  (* Contrast: the naive log(softmax) formulation is non-finite on the same input. *)
  let logits_ng =
    NTDSL.init ~l:"logits_ng" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ vocab ]
      ~f:(function [| b; v |] -> logit b v | _ -> assert false)
      ()
  in
  let%op naive_loss =
    neg ((targets *. log (Nn_blocks.softmax ~spec:"... | v" () logits_ng)) ++ "...|... => |->0")
  in
  let ctx3 = Train.forward_once (Context.auto ()) naive_loss in
  let naive = (ctx3, naive_loss).@[0] in
  Verdict.p "naive log(softmax) loss is non-finite" (not (Float.is_finite naive))
