open Base
module Ops = Ir.Ops
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
open Ocannl_tensor.Operation.DSL_modules

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TRAIN=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TRAIN"]

module CDSL = struct
  let half = Ir.Ops.half
  let single = Ir.Ops.single
  let double = Ir.Ops.double
  let virtualize_settings = Ir.Low_level.virtualize_settings

  let enable_all_debugs ?(debug_logs = false) ?(hosted_only = true) () =
    Utils.set_log_level @@ max 2 @@ Utils.settings.log_level;
    Utils.settings.output_debug_files_in_build_directory <- true;
    if hosted_only then virtualize_settings.enable_device_only <- false;
    if debug_logs then Utils.settings.debug_log_from_routines <- true

  let disable_all_debugs ?(restore_defaults = false) () =
    Utils.settings.debug_log_from_routines <- false;
    Utils.set_log_level 0;
    Utils.settings.output_debug_files_in_build_directory <- false;
    if restore_defaults then virtualize_settings.enable_device_only <- true
end

module IDX = struct
  let empty = Idx.Empty
  let get_static_symbol = Idx.get_static_symbol
  let find_exn = Idx.find_exn
end

let run ctx routine = ignore (Context.run ctx routine)

(* Parameter persistence now lives in {!Persistence} (gh-ocannl-373) and is context-mediated; the
   old hosted-array-based save/restore helpers were removed with the hosted memory mode
   (gh-ocannl-333). *)

let set_materialized (a : Tn.t) = Tn.update_memory_mode a On_device (Site "28:set-materialized")

let forward t =
  let fwd = Tensor.consume_forward_code t in
  set_materialized t.Tensor.value;
  let label = Tn.debug_name t.value in
  { fwd with asgns = Asgns.Block_comment (label ^ " fwd", fwd.asgns) }

let loss_accumulator ?(label = "loss_accum") () =
  let t = NTDSL.init ~l:label ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized t.Tensor.value;
  t

let trainable_params loss =
  match loss.Tensor.diff with
  | None -> Set.empty (module Tensor)
  | Some diff ->
      let written = Asgns.collect_written diff.Tensor.backprop.Asgns.asgns in
      Set.filter loss.Tensor.params ~f:(fun p ->
          match p.Tensor.diff with None -> false | Some d -> Set.mem written d.Tensor.grad)

(* The parameter set an optimizer-side helper operates on: [?params] when given (the escape hatch
   for exotic flows, e.g. a gradient written outside this loss's backprop), the derived
   {!trainable_params} otherwise. An empty derived set fails loudly: a differentiable loss that
   trains no registered parameters has no legitimate use for these helpers, and the failure is
   otherwise silent at every stage — the helpers compile empty or vacuous routines that run as
   no-ops (gh-ocannl-670). *)
let params_for ~fn_name ?params loss =
  match params with
  | Some ps -> ps
  | None ->
      if Option.is_none loss.Tensor.diff then
        raise @@ Tensor.Session_error (fn_name ^ ": loss is not differentiable", Some loss);
      let ps = trainable_params loss in
      if Set.is_empty ps then
        raise
        @@ Tensor.Session_error
             ( fn_name
               ^ ": the loss trains no parameters -- no [loss.params] member's gradient is written \
                  by the backprop code. Note that only [Tensor.param]-registered tensors join \
                  [params] ([Operation.init ~grad_spec:Require_grad] leaves do not), and a \
                  parameter behind [stop_gradient] is not trained",
               Some loss );
      ps

(** Replaces the zeroing of the given gradient nodes with [Noop] inside a [zero_grads] computation
    (each per-tensor zeroing is a [Fetch] of zeros — see [Tensor.fetch_zeros]). Used by the
    gradient-accumulation variant of {!grad_update}, which must keep zeroing the {e intermediate}
    gradients every micro-step (their backprop contributions are plain [=+] accumulations relying on
    a same-routine reset) while the {e parameter} gradients accumulate across micro-steps.

    The match is by [Fetch] constructor and tnode identity, so a change to how zeroing is emitted (a
    different constructor, or a parameter gradient dropping out of the tree) would silently keep
    zeroing parameter gradients — corrupting the accumulation. A gradient that the backprop writes
    is accumulated into, so it is zeroed in the tree: we raise when such a [grads] member had
    nothing removed rather than let the drift through. Pass only the gradients backprop reaches — a
    parameter detached from the loss (behind {!Operation.stop_gradient}, say) stays in [loss.params]
    while its gradient is neither zeroed nor accumulated. *)
let filter_out_grad_zeroing ~grads (comp : Asgns.comp) =
  let removed = Hash_set.create (module Tn) in
  let rec loop = function
    | Asgns.Noop -> Asgns.Noop
    | Asgns.Seq (t1, t2) -> Asgns.Seq (loop t1, loop t2)
    | Asgns.Block_comment (s, t) -> Asgns.Block_comment (s, loop t)
    | Asgns.Fetch { array; _ } when Set.mem grads array ->
        Hash_set.add removed array;
        Asgns.Noop
    | (Asgns.Accum_op _ | Asgns.Set_vec_unop _ | Asgns.Fetch _) as t -> t
  in
  let result = loop comp.asgns in
  let missing = Set.filter grads ~f:(fun g -> not (Hash_set.mem removed g)) in
  if not (Set.is_empty missing) then
    invalid_arg @@ "Train.filter_out_grad_zeroing: no zeroing found for accumulated gradient(s): "
    ^ String.concat ~sep:", " (List.map (Set.to_list missing) ~f:Tn.debug_name)
    ^ " -- the shape of the zero_grads code changed, gradient accumulation would be corrupted";
  (* The parameter gradients stay embedded: the surrounding {!grad_update} backprop accumulates into
     them regardless, and they are materialized so they persist across micro-steps. *)
  { comp with Asgns.asgns = result }

let grad_update ?(setup_for_parallel = false) ?accum_steps ?accum_loss ?loss_scale loss =
  Option.iter accum_steps ~f:(fun k ->
      if k <= 0 then invalid_arg "Train.grad_update: accum_steps must be positive");
  set_materialized loss.Tensor.value;
  (* Training loops read the loss from the host; declare the intent so the liveness memory planner
     (config [buffer_aliasing], gh-ocannl-489) never aliases the loss buffer -- like param
     gradients' observation intent declared in [Tensor.param]. *)
  Tn.set_observable loss.Tensor.value;
  if setup_for_parallel || Option.is_some accum_steps then
    (* Only the trained parameters' gradients: a frozen parameter's gradient is never written, so
       declaring it materialized would demand a buffer nothing computes (gh-ocannl-673). *)
    Set.iter (trainable_params loss) ~f:(fun p ->
        set_materialized (Option.value_exn ~here:[%here] p.diff).grad);
  let zero_grads =
    match loss.Tensor.diff with
    | None ->
        raise @@ Tensor.Session_error ("Train.grad_update: loss is not differentiable", Some loss)
    | Some diff -> (
        match accum_steps with
        | None -> diff.zero_grads
        | Some _ ->
            (* Only the parameters the backprop reaches ({!trainable_params}): one detached behind
               {!Operation.stop_gradient} (a frozen backbone) is still in [loss.params], but its
               gradient is neither zeroed here nor accumulated into, so it is not ours to strip --
               and demanding a zeroing for it would reject the freezing flow outright. *)
            let grads =
              Set.filter_map
                (module Tn)
                (trainable_params loss)
                ~f:(fun p -> Option.map p.Tensor.diff ~f:(fun d -> d.Tensor.grad))
            in
            filter_out_grad_zeroing ~grads diff.zero_grads)
  in
  let inv_accum = 1. /. Float.of_int (Option.value accum_steps ~default:1) in
  (* Note: the %cd syntax for [loss.grad] does not modify roots. *)
  [%cd
    ~~(loss "forward and gradient update";
       (* In the accumulating branch, referencing [loss] embeds its forward code (the single
          consumption of it), so the one statement computes the loss and accumulates it. *)
       (match accum_loss with
       | Some acc -> acc =+ loss
       | None -> loss.forward);
       ~~(loss "zero grads and backprop";
          zero_grads;
          (match (loss_scale, accum_steps) with
          | Some scale, None -> loss.grad =: scale
          | Some scale, Some _ -> loss.grad =: scale * !.inv_accum ~logic:"."
          | None, Some _ -> loss.grad =: !.inv_accum
          | None, None -> loss.grad =: 1);
          loss.backprop))]

let zero_params_grads ?params loss =
  let one_param p =
    match p.Tensor.diff with
    | None -> raise @@ Tensor.Session_error ("Train.zero_params_grads: not differentiable", Some p)
    | Some diff -> diff.zero_grads
  in
  let comp =
    Set.to_list (params_for ~fn_name:"Train.zero_params_grads" ?params loss)
    |> List.map ~f:one_param |> Asgns.sequence
  in
  { comp with asgns = Asgns.Block_comment ("zero_params_grads", comp.asgns) }

let grad_checksum ?params loss =
  let params = params_for ~fn_name:"Train.grad_checksum" ?params loss in
  let flag = NTDSL.init ~l:"grad_checksum" ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized flag.Tensor.value;
  Tn.set_observable flag.Tensor.value;
  (* Settle shape inference for the parameters first: the total-reduce einsum below unifies each
     parameter's rows with the spec's row variables, and a parameter row still unsolved at
     settlement would then be refused the close-to-empty guess (row.ml's "You forgot to specify the
     hidden dimension(s)" — a row variable used in an einsum spec is no longer safe to guess).
     Forcing dims here closes e.g. a bias's inferred-empty input row before the spec touches it —
     which is also why [grad_checksum] must be called only after the model and the loss are fully
     constructed. *)
  Set.iter params ~f:(fun p -> ignore (Lazy.force p.Tensor.value.Tn.dims : int array));
  let one_param p =
    if Option.is_none p.Tensor.diff then
      raise @@ Tensor.Session_error ("Train.grad_checksum: not differentiable", Some p);
    [%cd flag =+ id p.grad ~logic:"...|...->... => |->0"]
  in
  let comps = Set.to_list params |> List.map ~f:one_param in
  let reset = [%cd flag =: 0] in
  let comp = Asgns.sequence (reset :: comps) in
  (flag, { comp with asgns = Asgns.Block_comment ("grad_checksum", comp.asgns) })

let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) ?grad_unscale
    ?grad_scale ?update_gate p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error ("Train.sgd_one: not differentiable", Some p);
  (* The [%cd] payload is written once and the [update_gate] arms differ only where the gating
     actually changes the emitted assignment. Inline declarations ([{ sgd_delta }], [{ sgd_momentum
     }]) are hoisted by the ppx above the enclosing [match]/[if], so an arm that declares a tensor
     and an arm that only reads it name the same tensor. *)
  [%cd
    ~~(p "param sgd step";
       (match grad_unscale with
       (* The binary form: a unary [p.grad =* unscale] would be a Pointwise_un, which does not
          broadcast the scalar's closed rows against parameters with input axes. *)
       | Some unscale -> p.grad =: p.grad * unscale ~logic:"."
       | None -> Asgns.empty_comp);
       (* Gradients can only be read as direct operands (not inside subexpressions), hence the
          scaled read is its own statement. *)
       (match grad_scale with
       | Some scale ->
           { sgd_delta } =: p.grad * scale ~logic:".";
           sgd_delta =+ !.weight_decay *. p
       | None -> sgd_delta =: p.grad + (!.weight_decay *. p));
       if Float.(momentum > 0.0) then (
         (match update_gate with
         | None -> { sgd_momentum } =: (!.momentum *. sgd_momentum) + sgd_delta
         | Some gate ->
             sgd_momentum =: where gate ((!.momentum *. sgd_momentum) + sgd_delta) sgd_momentum);
         if nesterov then sgd_delta =+ !.momentum *. sgd_momentum else sgd_delta =: sgd_momentum);
       (* The final selection covers every path: without momentum it discards the possibly
          non-finite delta; with momentum the buffer kept its old (finite) value above, and this
          still zeroes the step so [p] is untouched. *)
       (match update_gate with
       | Some gate -> sgd_delta =: where gate sgd_delta 0
       | None -> Asgns.empty_comp);
       p =- learning_rate * sgd_delta ~logic:".")]
  |> fun comp ->
  (* The momentum buffer is optimizer state: it is read before it is written, and the value it
     carries into the next step is the whole point of it. Left undetermined it is a virtualization
     candidate, and inlining a node whose defining computation is in a PREVIOUS invocation of the
     routine has no meaning -- lowering fails outright ("No computations found for #N:
     sgd_momentum_..."), which is why [~momentum] was unusable until gh-ocannl-772 covered it. The
     label's head is the inline declaration's identifier, so this names exactly the node the [%cd]
     payload above declared. *)
  if Float.(momentum > 0.0) then
    Set.iter comp.Asgns.embedded_nodes ~f:(fun tn ->
        match tn.Tn.label with "sgd_momentum" :: _ -> set_materialized tn | _ -> ());
  comp

let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale ?grad_scale
    ?update_gate ?params loss =
  let f =
    sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale ?grad_scale ?update_gate
  in
  let comp =
    Set.to_list (params_for ~fn_name:"Train.sgd_update" ?params loss)
    |> List.map ~f |> Asgns.sequence
  in
  { comp with asgns = Asgns.Block_comment ("sgd_update", comp.asgns) }

let%track3_sexp sequential_loop ~f lowered_bindings =
  let rec loop = function
    | [] -> f ()
    | ({ Idx.static_range = None; static_symbol = _; _ }, _) :: more -> loop more
    | ({ Idx.used_as_extent = true; _ }, _) :: more -> loop more
    | ({ Idx.static_range = Some range; static_symbol = _; _ }, idx) :: more ->
        let old_idx = !idx in
        for i = 0 to range - 1 do
          idx := i;
          loop more
        done;
        idx := old_idx
  in
  loop lowered_bindings

module Lr_schedule = struct
  type kind = Constant | Cosine | Linear | Wsd of { decay_frac : float }

  type t = {
    kind : kind;
    base_lr : float;
    warmup_steps : int;
    total_steps : int;
    final_frac : float;
  }

  let learning_rate t ~step =
    let clamped_frac ~from =
      let denom = t.total_steps - from in
      if denom <= 0 then 1.0
      else Float.max 0.0 @@ Float.min 1.0 @@ (Float.of_int (step - from) /. Float.of_int denom)
    in
    let min_lr = t.base_lr *. t.final_frac in
    (* The [total_steps] clamp outranks warmup: a warmup longer than the horizon (a degenerate but
       constructible config — the record is transparent and unvalidated) must not keep ramping past
       the documented endpoint, so past [total_steps] the decay branch answers (its clamped ratio is
       1 there, i.e. the final value). *)
    let with_warmup decayed =
      if step < t.warmup_steps && step < t.total_steps then
        t.base_lr *. Float.of_int (step + 1) /. Float.of_int t.warmup_steps
      else decayed ()
    in
    match t.kind with
    | Constant -> with_warmup (fun () -> t.base_lr)
    | Cosine ->
        with_warmup (fun () ->
            let ratio = clamped_frac ~from:t.warmup_steps in
            min_lr +. (0.5 *. (1.0 +. Float.cos (Float.pi *. ratio)) *. (t.base_lr -. min_lr)))
    | Linear ->
        with_warmup (fun () ->
            let ratio = clamped_frac ~from:t.warmup_steps in
            min_lr +. ((1.0 -. ratio) *. (t.base_lr -. min_lr)))
    | Wsd { decay_frac } ->
        with_warmup (fun () ->
            let decay_point = Float.to_int ((1.0 -. decay_frac) *. Float.of_int t.total_steps) in
            if step < decay_point then t.base_lr
            else
              let ratio = clamped_frac ~from:decay_point in
              min_lr +. ((1.0 -. Float.sqrt ratio) *. (t.base_lr -. min_lr)))
end

let host_scalar ~l v =
  let ndarray =
    Ir.Ndarray.init_array ~debug:l Ir.Ops.single ~dims:[| 1 |] ~padding:None ~f:(fun _ -> v)
  in
  let t =
    (* Reshape rather than Keep_shape_no_padding: the latter pins the data axis at the default
       basis, which conflicts with the [bcast_if_1] tag (bases are incompatible atoms). *)
    Tensor.term ~grad_spec:Tensor.Prohibit_grad ~init_data:(Asgns.Reshape ndarray) ~label:[ l ]
      ~batch_dims:[] ~input_dims:[]
      ~output_axes:[ (Row.bcast_if_1, 1) ]
      ()
  in
  set_materialized t.Tensor.value;
  Tn.set_observable t.Tensor.value;
  t

let scheduled_learning_rate ?(label = "learning_rate") schedule =
  let lr = host_scalar ~l:label (Lr_schedule.learning_rate schedule ~step:0) in
  let set_step ctx ~step =
    Context.set_values ctx lr.Tensor.value [| Lr_schedule.learning_rate schedule ~step |]
  in
  (lr, set_step)

let grad_l2_norm ?grad_unscale ?(label = "grad_norm") ?params loss =
  let params = params_for ~fn_name:"Train.grad_l2_norm" ?params loss in
  let norm = NTDSL.init ~l:label ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized norm.Tensor.value;
  Tn.set_observable norm.Tensor.value;
  (* Settle shape inference for the parameters first — same reason as in {!grad_checksum}: a
     parameter row still unsolved at settlement would be refused the close-to-empty guess once the
     einsum spec's row variables touch it. *)
  Set.iter params ~f:(fun p -> ignore (Lazy.force p.Tensor.value.Tn.dims : int array));
  let one_param p =
    if Option.is_none p.Tensor.diff then
      raise @@ Tensor.Session_error ("Train.grad_l2_norm: not differentiable", Some p);
    [%cd norm =+ p.grad * p.grad ~logic:"...|...->...; ...|...->... => |->0"]
  in
  let comps = Set.to_list params |> List.map ~f:one_param in
  let reset = [%cd norm =: 0] in
  let root =
    match grad_unscale with
    | Some unscale ->
        [%cd
          norm =: sqrt norm;
          norm =: norm * unscale ~logic:"."]
    | None -> [%cd norm =: sqrt norm]
  in
  let comp = Asgns.sequence ((reset :: comps) @ [ root ]) in
  (norm, { comp with asgns = Asgns.Block_comment (label, comp.asgns) })

type grad_clipping = { grad_norm : Tensor.t; grad_scale : Tensor.t; clip_comp : Asgns.comp }

let clip_by_global_norm ?grad_unscale ?(label = "grad_clip") ?params ~max_norm loss =
  (* Validated because this is where configuration-derived floats arrive: a negative threshold would
     REVERSE gradients (negative ratio), and a nan one fails the ordered comparison and silently
     disables clipping. 0 stays legal — "clip to zero" freezes the gradient term. *)
  if (not (Float.is_finite max_norm)) || Float.(max_norm < 0.) then
    invalid_arg "Train.clip_by_global_norm: max_norm must be finite and nonnegative";
  let grad_norm, norm_comp = grad_l2_norm ?grad_unscale ~label:(label ^ "_norm") ?params loss in
  let grad_scale = host_scalar ~l:(label ^ "_scale") 1. in
  let scale_comp =
    (* Ordered comparison and selection: norm = 0 selects 1 (the division's [inf] is discarded by
       the [Where] selection). The [i => j] copy bridges the axis bases — [clip_sel] inherits the
       norm's default-basis axis while [grad_scale] carries the broadcastable [bcast_if_1] axis
       (distinct spec variables keep the two from unifying — incompatible basis atoms). *)
    [%cd
      { clip_ratio } =: !.max_norm /. grad_norm;
      { clip_sel } =: where (!.max_norm < grad_norm) clip_ratio 1;
      grad_scale =: id clip_sel ~logic:"i => j"]
  in
  let comp = Asgns.sequence [ norm_comp; scale_comp ] in
  {
    grad_norm;
    grad_scale;
    clip_comp = { comp with asgns = Asgns.Block_comment (label, comp.asgns) };
  }

module Outlier_detector = struct
  type t = {
    window : float array;
    mutable count : int; (* Number of recorded values while the window is still filling. *)
    mutable index : int; (* Replacement position once the window is full. *)
  }

  let create ?(window_size = 128) () =
    if window_size <= 0 then
      invalid_arg "Train.Outlier_detector.create: window_size must be positive";
    { window = Array.create ~len:window_size 0.; count = 0; index = 0 }

  let update t v =
    if not (Float.is_finite v) then Float.infinity
    else
      let n = Array.length t.window in
      if t.count < n then (
        t.window.(t.count) <- v;
        t.count <- t.count + 1;
        Float.nan)
      else
        let nf = Float.of_int n in
        let mag = Array.fold t.window ~init:0. ~f:(fun acc x -> Float.max acc (Float.abs x)) in
        let z =
          if Float.(mag = 0.) then
            (* All-zero window: any nonzero [v] is infinitely surprising. *)
            if Float.(v = 0.) then 0. else Float.copysign Float.infinity v
          else
            let smean = Array.fold t.window ~init:0. ~f:(fun acc x -> acc +. (x /. mag /. nf)) in
            let s =
              Array.fold t.window ~init:0. ~f:(fun acc x ->
                  let d = (x /. mag) -. smean in
                  acc +. (d *. d))
            in
            let rstd = Float.sqrt (s /. nf) in
            let num = (v /. mag) -. smean in
            if Float.(rstd = 0.) then
              if Float.(num = 0.) then 0. else Float.copysign Float.infinity num
            else num /. rstd
        in
        t.window.(t.index) <- v;
        t.index <- (t.index + 1) % n;
        z
end

let set_virtual (a : Tn.t) = Tn.update_memory_mode a Virtual (Site "29:set-virtual")

let every_non_literal_materialized =
  Tensor.iter_embedded ~f:(fun a ->
      if Tn.mode_is_unspecified a && not (Tn.known_constant a) then set_materialized a)

type placement_arm = Measured_winner | Force_arm_a | Force_arm_b

let placement_arm_of_string ~source s =
  match String.lowercase (String.strip s) with
  | "auto" | "measured" -> Measured_winner
  | "a" | "default" -> Force_arm_a
  | "b" | "materialize-all" | "materialize_all" -> Force_arm_b
  | other ->
      invalid_arg
        (source ^ " should be auto | a | b (aliases: measured, default, materialize-all); found: "
       ^ other)

let placement_arm_name = function
  | Measured_winner -> "the measured winner"
  | Force_arm_a -> "A (default placements)"
  | Force_arm_b -> "B (materialize-all)"

let embedded_nodes loss =
  let embedded = ref [] in
  Tensor.iter_embedded ~f:(fun tn -> embedded := tn :: !embedded) loss;
  !embedded

let placement_problem ?name ?timing_ctx ctx loss comp bindings =
  let embedded = Set.of_list (module Tn) (embedded_nodes loss) in
  let static_indices = Ir.Indexing.bound_symbols bindings in
  let lowered = Context.lowered_for_decisions ?name ctx comp bindings in
  let timing =
    Option.map timing_ctx ~f:(fun tctx ->
        let tlowered = Context.lowered_for_decisions ?name tctx comp bindings in
        ( Context.placements tctx,
          tlowered.Ir.Low_level.optimize_ctx,
          (* gh-ocannl-1421: how each lineage's lowered program accesses the node, which the tags
             above do not pin (an inherited computation changes the program, not a placement tag).
             Whether a search is [Lineage_refused] depends on the two lowerings, so a decision
             recorded beside such a search should apply only where they agree. Per node, so the
             canonicalizer numbers the nodes and the key is process-independent; accesses rather
             than code, since this key is taken before any compile has settled value bounds and the
             code drifts with them (review round 6). *)
          Autotune.lineage_accesses tlowered,
          Autotune.lineage_accesses lowered ))
  in
  Ir.Schedule_cache.canonicalize_source ~static_indices
    ~node_tag:(fun tn ->
      (if Set.mem embedded tn then ";e" else "")
      ^ Option.value_map timing ~default:"" ~f:(fun (plc, octx, t_access, c_access) ->
          ";t"
          ^ Ir.Schedule_cache.lineage_tag plc octx tn
          ^ ";ta=" ^ t_access tn ^ ";ca=" ^ c_access tn))
    ~lineage:(Context.placements ctx) lowered

(** A recorded placement decision as the three lists the context API takes — materialized, inlined,
    footprint-scoped — resolved against [problem]'s numbering ({!Ir.Schedule_cache.tn_of_ref} raises
    on a node the problem lacks). [embedded] is the loss's embedded set, what [Materialize_all]
    decides. *)
let placement_decision_lists ~embedded problem = function
  | Ir.Schedule_cache.Default -> ([], [], [])
  | Materialize_all -> (embedded, [], [])
  | Refined flips ->
      Ir.Low_level.partition_readings
        (List.map flips ~f:(fun { Ir.Schedule_cache.node; flip } ->
             (Ir.Schedule_cache.tn_of_ref problem node, flip)))

(** The placement-aware digest ({!Ir.Schedule_cache.canonicalize}) of the lowering the decision
    lists produce in the lineage the arms are searched in — [timing_ctx] when given, else [ctx] —
    recomputed analysis-only through {!Context.lowered_for_decisions}. The placement store's replay
    guard (gh-ocannl-786): it must equal the recorded [outcome_digest], which since gh-ocannl-1022
    is the shipped search's own {!Autotune.report.source_digest}. The search lineage, not [ctx],
    because that is where the base lowering every candidate and the shipped winner derive from is
    taken. *)
let decision_lowering_digest ?name ?timing_ctx ctx comp bindings (mat, inl, fp) =
  let module SC = Ir.Schedule_cache in
  SC.digest
    (SC.canonicalize
       ~static_indices:(Ir.Indexing.bound_symbols bindings)
       (Context.lowered_for_decisions ?name ~materialized:mat ~inline:inl ~footprint:fp
          (Option.value timing_ctx ~default:ctx)
          comp bindings))

let placement_outcome_digest ?name ?timing_ctx ctx loss comp bindings decision =
  decision_lowering_digest ?name ?timing_ctx ctx comp bindings
    (placement_decision_lists ~embedded:(embedded_nodes loss)
       (placement_problem ?name ?timing_ctx ctx loss comp bindings)
       decision)

let placement_evidence_clean (r : Autotune.report) =
  match r.Autotune.outcome with
  | Autotune.Lineage_refused _ -> true
  | _ ->
      r.Autotune.timings_contended = 0 && (not r.Autotune.budget_cut)
      && Option.is_none (Autotune.terminal_failure r)
      && Float.is_finite r.Autotune.best_ms

(* gh-ocannl-1424: what the device holds at a placement boundary, as progress-line fields.
   [working_bytes] is the census's live working-pool bytes (process-wide, deterministic: counted at
   the shared allocator seam on every backend), so equal readings at two boundaries mean nothing of
   the work between them stayed in OCANNL's pools. [device_free] is the driver's own free bytes,
   [none] where the backend cannot read them: it also sees what no pool table does (other processes,
   contexts, loaded code, kernel local memory, instantiated graphs). Read only when the progress
   stream is on, so a run without it makes no extra driver call. *)
let placement_memory_fields ?(with_total = false) ctx =
  let working = (Ir.Alloc_census.snapshot ()).Ir.Alloc_census.live_working_bytes in
  let reading = try Context.get_free_and_total_memory ctx with _ -> None in
  let field f = Option.value_map reading ~default:"none" ~f:(fun r -> Int.to_string (f r)) in
  Printf.sprintf " working_bytes=%d device_free=%s%s" working (field fst)
    (if with_total then " device_total=" ^ field snd else "")

type shipped = {
  ship_arm : string;
  ship_replayed : bool;
  ship_routine : string;
  ship_report : Autotune.report option;
  ship_compiles : int;
  ship_substituted : bool;
}

let tune_placements ?name ?beam_width ?rounds ?repeats ?cache_dir ?timing_ctx ?report ?flip_report
    ?inline_flips ?ship_arm ?placement_store ?call_budget ?log ?progress ?on_ship ctx loss comp
    bindings =
  (* [?log] and [?progress] scope the two streams over the whole call: the arm lines below and every
     nested {!Autotune.tune} read them through {!Autotune.log_enabled} and
     {!Autotune.progress_enabled}. One search at a time (gh-ocannl-1352): the whole call is one
     search, so another domain's [tune] cannot slip in between two arms, and a call made while
     another domain searches raises {!Autotune.Concurrent_search} before it does anything. *)
  Autotune.exclusive_search @@ fun () ->
  Autotune.with_streams ?log ?progress @@ fun () ->
  (* Arm attribution on the same stderr trace as Autotune's [autotune: ] lines — winner-arm
     ambiguity misdirected the CUDA benchmark debugging on PR #140. *)
  let logf fmt =
    Stdlib.Printf.ksprintf
      (fun s -> if Autotune.log_enabled () then Stdio.eprintf "tune_placements: %s\n%!" s)
      fmt
  in
  (* gh-ocannl-638. Resolved before the first search, and announced there rather than only at the
     decision it changes: a search is minutes to hours, and a measurement that set this on the wrong
     cell should learn it from the head of the log, not from the summary of a run it has finished
     paying for. Unconditional on stderr, not through [logf]: this overrides the optimizer's
     shipping path, which is not something a run should have to have enabled [autotune_log] to find
     out about. *)
  let ship_arm =
    match ship_arm with
    | Some a -> a
    | None ->
        placement_arm_of_string ~source:"ocannl_tune_ship_arm"
          (Utils.get_global_arg ~arg_name:"tune_ship_arm" ~default:"auto")
  in
  let forced = match ship_arm with Measured_winner -> false | Force_arm_a | Force_arm_b -> true in
  if forced then
    Stdio.eprintf
      "Train.tune_placements: tune_ship_arm selects arm %s, which will ship whatever the timings \
       say. This is a measurement-only setting; a normal run ships the measured winner.\n\
       %!"
      (placement_arm_name ship_arm);
  (* gh-ocannl-1440: one wall-clock budget for the whole call, on one clock started here -- before
     the placement-store replay, whose search spends it too. Each search the call starts gets a
     share: what is left, divided among the searches still planned ([~left], the current one
     included), so the time a cache hit or a short search does not use goes to the searches after
     it. The search is cut at its share by {!Autotune.tune}'s own budget rule. *)
  let call_budget =
    match call_budget with
    | Some b -> if Float.(b > 0.) then Some b else None
    | None -> Autotune.call_budget_setting ()
  in
  let call_clock = Autotune.progress_stopwatch () in
  let call_elapsed () = Autotune.Seam.get Autotune.call_budget_elapsed (call_clock ()) in
  let call_share ~left =
    Option.map call_budget ~f:(fun total ->
        {
          Autotune.cb_total_s = total;
          cb_share_s = (total -. call_elapsed ()) /. Float.of_int left;
        })
  in
  let call_budget_spent () =
    Option.exists call_budget ~f:(fun total -> Float.(call_elapsed () >= total))
  in
  (* Set once the call's budget has stopped the flip refinement: a decision reached with flips left
     unsearched is this call's, like one over a cut search, and is not recorded. *)
  let call_budget_stopped_flips = ref false in
  (* gh-ocannl-1440: the backend compiles this call performs outside its searches -- whose own each
     search's report counts ([Autotune.report.compiles]) -- i.e. compiling what ships, a shipping
     compile that failed or was substituted before it included. [shipped.ship_compiles]. *)
  let shipping_compiles = ref 0 in
  let counted f =
    let before = Context.backend_compiles () in
    Exn.protect ~f ~finally:(fun () ->
        shipping_compiles := !shipping_compiles + (Context.backend_compiles () - before))
  in
  let compile_deferred d = counted (fun () -> Autotune.compile_deferred d) in
  (* gh-ocannl-1424: the baseline the arm lines' memory fields compare against. *)
  if Autotune.progress_enabled () then
    Autotune.progressf "event=placements_start%s" (placement_memory_fields ~with_total:true ctx);
  let last = ref None in
  (* Every report this call observes, arms and flips alike: the evidence a recorded placement
     decision rests on (gh-ocannl-786). *)
  let observed = ref [] in
  (* The public [?report] contract is positional — arm A's report then arm B's, which consumers
     (e.g. the benchmark harness) attribute by arrival order — so flip-refinement searches report
     through the separate [?flip_report] instead ([~to_report] selects the callback).

     gh-ocannl-550: the arms are independent experiments, so one arm's terminal failure is no
     evidence about another arm's completed result — and must not destroy it. Per-candidate
     containment inside a search is {!Autotune.tune}'s job (gh-ocannl-533/536) and it holds: on the
     reproduction that motivated this (benchmarks/report-gh528-gpt2-cuda.md §3, five of five tf32
     gpt2_mini runs) the OOMing candidates were absorbed as ordinary [Backend_link] declines and the
     search ran to its end. What escaped is the aftermath: with the device exhausted, the arm's
     winner replay could not compile and neither could its untuned-default fallback, so
     [Autotune.tune] raised — and with no handler here, arm B's failure took arm A's already
     finished, already cached winner (106.389 ms, the best arm-A result of that whole leg) out of
     the process with it. Catching per arm is the whole fix: a failed search returns [Error] and
     ranks at [infinity], which is not the same as a completed search that timed nothing (which
     returns the untuned default compile and also ranks at [infinity] — hence [Result.t] rather than
     a sentinel time). *)
  (* The arms are contained; two classes of failure are not the arm's to absorb, and they are the
     same two {!Ir.Schedule_outcome.classify_raw} refuses to classify one level down — containing
     them here would re-absorb exactly what that policy exists to refuse.

     - Process-level ([Out_of_memory] on the host heap, [Sys.Break]): no evidence about this arm and
       nothing to fall back to. Swallowing an interrupt to go run the other arm would make Ctrl-C
       during a half-hour search do nothing.
     - Compiler invariant violations ([Assert_failure], [Stack_overflow]): a tuner bug, not a
       schedule that lost. Demoting one to "that arm lost" would let a process exit 0 with a shipped
       winner and the bug unmentioned outside the report. (A stale cache entry tripping an assert is
       already turned into a classified decline inside {!Autotune.tune} under [Cache_replay]
       provenance, so it never reaches here as an exception.)

     The device OOM this containment exists for stays contained: a typed device-memory refusal
     surfaces as the [Utils.User_error] its [Resource_exceeded] cause renders to (gh-ocannl-1326),
     and any other driver failure as an ordinary [Invalid_argument]. *)
  let must_propagate = function
    | Out_of_memory | Stdlib.Sys.Break | Stack_overflow | Assert_failure _ -> true
    (* A malformed registered setting (gh-ocannl-1332) is the configuration's failure, not an arm's
       or a flip's: contained, it would ship whichever placement happens not to read the key, as
       though the setting had been honored. *)
    | Utils.Invalid_config_value _ -> true
    | _ -> false
  in
  (* A [report] callback exception is the caller's failure, not the search's — {!Autotune.tune}
     propagates it deliberately on the completion path — so it is re-raised rather than reclassified
     as an arm failure (which would hide it, and could ship the other arm even though this search
     completed). It is marked by WRAPPING it at the raise site: a nullary exception ([Exit],
     [End_of_file]) is a singleton value, so physical identity cannot tell a callback's [Exit] from
     an arm failure's, and the tuner's fatal path deliberately swallows the callback's exception and
     raises the arm's — the very case a same-value collision would misread as "the callback failed".
     A wrapper the tuner never constructs cannot collide: whatever comes out unwrapped is the
     search's. The rendered message is carried alongside because the tuner prints the wrapper with
     the default exception printer when it swallows it, and that printer shows string fields
     only. *)
  let exception Report_callback_failed of string * exn * Stdlib.Printexc.raw_backtrace in
  (* gh-ocannl-1408: a search's result is its DEFERRED compile ({!Autotune.tune_deferred}): nothing
     an arm or a flip compiled stays resident while the next one searches, and only the placement
     that ships is compiled ([ship]). [eager] compiles at once, inside the failure containment: the
     replayed decision's one search uses it, since a compile failure there is a failed replay and
     sends the call to the arms. *)
  let tune ?to_report ?(progress_note = "") ?abandon ?(eager = false) ~left arm ctx timing_ctx =
    let to_report = Option.value to_report ~default:report in
    let capture r =
      last := Some r;
      observed := r :: !observed;
      Option.iter to_report ~f:(fun f ->
          match f r with
          | () -> ()
          (* Wrapped so it can be told apart from the search's own failures — except when it is
             process-fatal, where that distinction is pointless and the wrapper would hide it from
             the tuner's own guard on the fatal path (which re-raises such exceptions rather than
             swallowing them). *)
          | exception exn when must_propagate exn -> raise exn
          | exception exn ->
              raise
                (Report_callback_failed
                   (Exn.to_string exn, exn, Stdlib.Printexc.get_raw_backtrace ())))
    in
    let call_budget = call_share ~left in
    logf "arm %s search:%s" arm
      (Option.value_map call_budget ~default:"" ~f:(fun cb ->
           Printf.sprintf " a %.4g s share of the call's %g s budget (%d search(es) planned)"
             cb.Autotune.cb_share_s cb.Autotune.cb_total_s left));
    (* gh-ocannl-1061: the arm-level cost record, on {!Autotune.progressf}'s stream. *)
    let stopwatch = Autotune.progress_stopwatch () in
    let memory_fields () =
      if Autotune.progress_enabled () then placement_memory_fields ctx else ""
    in
    Autotune.progressf "event=arm_start arm=%S%s%s" arm progress_note (memory_fields ());
    last := None;
    let result =
      match
        let deferred =
          Autotune.tune_deferred ?name ?beam_width ?rounds ?repeats ?cache_dir ?timing_ctx ?abandon
            ?call_budget ~report:capture ctx comp bindings
        in
        if eager then ignore (compile_deferred deferred : Context.t * Context.routine);
        deferred
      with
      | deferred -> Ok deferred
      (* Unwrapped, so the caller sees its own exception with its own backtrace: the wrapper is an
         internal marker, not part of this function's contract. *)
      | exception Report_callback_failed (_, callback_exn, callback_backtrace) ->
          Stdlib.Printexc.raise_with_backtrace callback_exn callback_backtrace
      | exception exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          if must_propagate exn then Stdlib.Printexc.raise_with_backtrace exn backtrace
          else Error (exn, backtrace)
    in
    (* [?report] is positional and consumers name arms by arrival order, so a failing arm must still
       occupy its slot. It does: {!Autotune.tune} reports exactly once per call on every path that
       does any work, each pre-search failure carrying the phase it died at. *)
    let r = !last in
    let best_ms =
      match result with
      (* Not [r.best_ms]: a [Search_died] report's best is a measurement of the search context, and
         no routine was compiled from [ctx] for it. There is nothing to ship at that time. *)
      | Error _ -> Float.infinity
      | Ok _ -> Option.value_map r ~default:Float.infinity ~f:(fun r -> r.Autotune.best_ms)
    in
    (match result with
    | Error (Autotune.Search_abandoned ab, _) ->
        (* gh-ocannl-1110: the flip's best at the verdict's depth, not the [infinity] it ships
           at. *)
        Autotune.progressf
          "event=arm_done arm=%S%s result=abandoned best_ms=%s after=%d incumbent_ms=%s \
           elapsed_s=%.1f%s"
          arm progress_note
          (Autotune.progress_ms ab.Autotune.ab_best_ms)
          ab.Autotune.ab_timed
          (Autotune.progress_ms ab.Autotune.ab_incumbent_ms)
          (stopwatch ()) (memory_fields ())
    | _ ->
        Autotune.progressf "event=arm_done arm=%S%s result=%s best_ms=%s elapsed_s=%.1f%s" arm
          progress_note
          (match result with Ok _ -> "ok" | Error _ -> "failed")
          (Autotune.progress_ms best_ms) (stopwatch ()) (memory_fields ()));
    (match result with
    | Error (Autotune.Search_abandoned ab, _) ->
        let timing_source =
          match r with
          | Some { Autotune.outcome = Autotune.Abandonment_replay _; _ } -> "cached timings"
          | _ -> "timed candidates"
        in
        logf
          "arm %s ABANDONED after %d %s: its best %.4f ms trails the incumbent's %.4f ms at the \
           same depth by more than %.4gx"
          arm ab.Autotune.ab_timed timing_source ab.Autotune.ab_best_ms ab.Autotune.ab_incumbent_ms
          ab.Autotune.ab_ratio
    | Error (exn, _) ->
        logf "arm %s FAILED, it loses the comparison (%s): %s" arm
          (Option.value_map r ~default:"it reported nothing" ~f:(fun r ->
               if Float.is_inf r.Autotune.best_ms then "it had timed nothing"
               else
                 Printf.sprintf "its pre-failure best of %.4f ms is not shippable"
                   r.Autotune.best_ms))
          (Exn.to_string exn)
    | Ok _ ->
        logf "arm %s best: %.4f ms (%s)" arm best_ms
          (Option.value_map r ~default:"no report" ~f:(fun r ->
               Printf.sprintf "%s%s, best tensorized %s"
                 (if String.is_empty r.Autotune.best_label then "nothing timed"
                  else r.Autotune.best_label)
                 (* What the schedule asked for, and next to it what the emission delivered
                    (gh-ocannl-626): "[tensorized]" alone has read as a tensor-core claim over a
                    kernel whose every [Tile_mma] fell back to the lane-0 scalar loop. *)
                 (if r.Autotune.best_tensorized then
                    Printf.sprintf " [tensorized/%s]"
                      (Option.value_map r.Autotune.best_tensorization ~default:"no census"
                         ~f:Ir.C_syntax.tensorization_name)
                  else "")
                 (if Float.is_inf r.Autotune.mma_best_ms then "none"
                  else Printf.sprintf "%.4f ms" r.Autotune.mma_best_ms))));
    (result, best_ms, r)
  in
  (* gh-ocannl-550 asked what becomes of every arm and flip result that does not ship: dropping the
     OCaml value frees nothing (the backend's pool table roots the slabs, see {!Context.release}).
     Since gh-ocannl-1408 there is nothing to drop: a result is a deferred compile, holding no device
     resources, and only the one that ships is compiled. So arm A's routine is not resident while
     arm B searches -- on a step over half the device, that residency refused arm B's base compile
     and made every placement A/B single-arm -- and no routine is compiled for a losing arm or
     flip. *)
  (* The arms are independent experiments but they are not independent lineages: the B arm searches
     in a {!Context.decide_materialized} sibling, which shares the execution ledger. A failure whose
     damage the backend could not bound ([Writes_may_have_occurred], or an unattributed launch/sync
     failure) poisons that ledger, and every timing run in the sibling then refuses to execute — so
     the next arm cannot produce a result, only burn a search proving it. Propagate the failure that
     poisoned it instead, which is also the honest error: the second arm's failures would all be
     consequences of the first (gh-ocannl-550). Giving each arm its own root lineage is the other
     way out, at the cost of re-initializing the caller's parameters per arm. *)
  let search_lineage = Option.value timing_ctx ~default:ctx in
  let lineage_poisoned () = Option.is_some (Context.poisoned_failure search_lineage) in
  let propagate_if_poisoned who = function
    | Error (exn, backtrace) when lineage_poisoned () ->
        logf "arm %s poisoned the shared %s lineage, so no sibling arm can run: propagating" who
          (if Option.is_some timing_ctx then "timing" else "caller's");
        Stdlib.Printexc.raise_with_backtrace exn backtrace
    | _ -> ()
  in
  let embedded = embedded_nodes loss in
  (* [decide_materialized] skips the nodes constrained away from materialization (constants,
     declared-virtual), mirroring [every_non_literal_materialized]'s guards at the decision
     level. *)
  let materialize c = Context.decide_materialized c embedded in
  (* gh-ocannl-786: the placement-decision store. The schedule cache persists each arm's and each
     flip's crowned schedule with its timings, so on a warm cache the arms and the chain replay
     their measurements -- but the DECISION (which arm shipped, which flips the chain accepted) was
     re-derived every process: candidate enumeration, per-arm and per-flip specialization, a [tune]
     per candidate. It is recorded here instead, keyed by the decision problem's structural identity
     ({!Ir.Schedule_cache.canonicalize_source}: the raw program plus what the lineage already
     decided -- the decision itself is what the entry holds, so it is outside the key), in the same
     directory under the same regime as the schedule entries it complements.

     Opened by the same rule as the schedule cache ({!Autotune.resolve_cache_dir}), and never under
     a forced arm: forcing ships a chosen artifact whatever the evidence says, which is neither a
     decision to replay nor one to record. Nor under [placement_store = false] (gh-ocannl-1020),
     which asks for the arms' comparison itself, schedule cache warm or not. Every failure of the
     store -- an unreadable entry, a lowering that declined, a device whose limits cannot be read --
     degrades to the cold path, which reports it in position; only the two process-level classes
     propagate from here. *)
  let module SC = Ir.Schedule_cache in
  let cache_dir = Autotune.resolve_cache_dir ?cache_dir ~search:(Autotune.search_setting ()) () in
  let placement_store =
    match placement_store with
    | Some b -> b
    | None -> Utils.get_global_flag ~default:true ~arg_name:"tune_placement_store"
  in
  if (not placement_store) && not forced then
    logf "placement store bypassed (tune_placement_store=false): the arms are compared afresh";
  let store =
    if forced || (not placement_store) || String.is_empty cache_dir then None
    else (
      (* gh-ocannl-1061: the store's problem is a lowering, and a replay check another -- steps that
         can block, so a line names them first, as {!Autotune.tune} does its own. *)
      Autotune.progressf "event=stage stage=placement_store";
      match
        let problem = placement_problem ?name ?timing_ctx ctx loss comp bindings in
        if SC.complete problem then
          let limits = Context.hardware_limits ctx in
          let capabilities = Context.codegen_capabilities ctx in
          let backend = Context.backend_name ctx in
          Option.map
            (SC.placement_key ~timing_identity:(Context.timing_identity ctx) ~limits ~capabilities
               problem ~backend)
            ~f:(fun key -> (problem, key, limits, capabilities, backend))
        else None
      with
      | store -> store
      | exception exn when not (must_propagate exn) ->
          logf "placement store not consulted: %s" (Exn.to_string exn);
          None)
  in
  (* A decision as the three lists the context API takes ({!placement_decision_lists}); the context
     it yields; and, at replay, the lowering it produces in the search lineage, whose
     placement-aware digest is the entry's guard ({!decision_lowering_digest}). *)
  let decisions_of problem = placement_decision_lists ~embedded problem in
  let apply decisions c = Context.decide_readings c decisions in
  let outcome_digest decisions =
    decision_lowering_digest ?name ?timing_ctx ctx comp bindings decisions
  in
  (* The refined vector's flips by structural position, [None] when a flipped node has none (every
     flip candidate is a node the raw code sets or reads, so this is defensive). *)
  let refined_decision accepted =
    match store with
    | None -> Some (SC.Refined [])
    | Some (problem, _, _, _, _) ->
        let registry = SC.base_registry problem in
        Option.map
          (List.map accepted ~f:(fun (tn, flip) ->
               Option.map (SC.resolve_tn registry tn) ~f:(fun node -> { SC.node; flip }))
          |> Option.all)
          ~f:(fun flips -> SC.Refined flips)
  in
  (* Recorded only when the evidence is the kind the schedule cache itself keeps (gh-ocannl-855's
     rule, one level up): every search this call observed completed with no contention-refused
     window and MEASURED something -- a competitor that timed nothing (a search that lost every
     candidate, or [autotune_search=false] over a schedule cache holding only the other arm) was
     never compared, and recording the winner over it would let a later search-enabled process skip
     the unmeasured competitor for good. A decision reached because an arm failed, or over refused
     windows, is likewise this process's to make and not the next one's to inherit. So is one whose
     crown came from a search its wall-clock budget cut (gh-ocannl-1409): the schedule cache lets
     only a budgeted call replay such a crown, and a decision recorded from it would reach every
     later call, budgeted or not.

     gh-ocannl-1022: the entry's [outcome_digest] is the SHIPPED search's own
     {!Autotune.report.source_digest} -- the digest of the base lowering it tuned, and the key its
     schedule entry lives under -- not a re-lowering of the decision after the fact. The replay
     guard still recomputes it ({!decision_lowering_digest}); that the two agree is lowering
     determinism, which [test/operations/placement_store.ml] asserts directly. A shipped search that
     reached no base lowering has no digest to record, and records nothing. *)
  let persist ~decision ~(shipped : Autotune.report option) ~shipped_ms ~a_ms ~b_ms =
    match store with
    | None -> ()
    | Some (problem, key, limits, capabilities, backend) ->
        let unclean =
          !call_budget_stopped_flips || not (List.for_all !observed ~f:placement_evidence_clean)
        in
        let outcome_digest =
          Option.value_map shipped ~default:"" ~f:(fun r -> r.Autotune.source_digest)
        in
        if unclean || not (Float.is_finite shipped_ms) then
          logf
            "placement store: decision %s not recorded (a compared search failed, timed nothing, \
             had contention-refused windows, or was cut by its budget, or the call's budget \
             stopped the flip refinement)"
            (SC.shipped_label decision)
        else if String.is_empty outcome_digest then
          logf "placement store: decision %s not recorded (its search reported no source digest)"
            (SC.shipped_label decision)
        else (
          SC.store_placements ~dir:cache_dir ~key:(Some key)
            {
              SC.version = SC.placement_entry_version;
              backend;
              numerics = SC.numerics_tag ();
              codegen = SC.codegen_tag ~backend ~limits ~capabilities ();
              objective = SC.objective_tag ();
              problem_digest = SC.digest problem;
              decision;
              outcome_digest;
              shipped_ms;
              arm_a_ms = a_ms;
              arm_b_ms = b_ms;
            };
          logf "placement store: recorded decision %s (%.4f ms)" (SC.shipped_label decision)
            shipped_ms)
  in
  let replay =
    match store with
    | None -> None
    | Some (problem, key, limits, capabilities, backend) -> (
        match SC.lookup_placements ~dir:cache_dir ~key:(Some key) with
        | None ->
            logf "placement store: no decision recorded for this problem";
            None
        | Some e
          when not
                 (String.equal e.SC.problem_digest (SC.digest problem)
                 && String.equal e.SC.backend backend
                 && String.equal e.SC.numerics (SC.numerics_tag ())
                 && String.equal e.SC.codegen (SC.codegen_tag ~backend ~limits ~capabilities ())
                 && String.equal e.SC.objective (SC.objective_tag ())) ->
            (* Belt-and-braces like the schedule cache's: the key carries every one of these, so
               only a hand-moved file fails here -- and every one is checked, since a decision
               crowned under isolated timing accepted under a queued filename would skip exactly the
               comparison the objective component keeps apart (gh-ocannl-755). *)
            logf "placement store: the entry's self-description does not match its key; ignored";
            None
        | Some e -> (
            match
              let decisions = decisions_of problem e.SC.decision in
              (decisions, outcome_digest decisions)
            with
            | decisions, outcome when String.equal outcome e.SC.outcome_digest -> Some (e, decisions)
            | _ ->
                logf
                  "placement store: the recorded decision %s no longer reproduces the program it \
                   was measured on; re-tuning"
                  (SC.shipped_label e.SC.decision);
                None
            | exception exn when not (must_propagate exn) ->
                logf "placement store: the recorded decision cannot be applied (%s); re-tuning"
                  (Exn.to_string exn);
                None))
  in
  (* gh-ocannl-638: the one place that knows what shipped, on the one path that ships. *)
  let deliver ?(replayed = false) ?(substituted = false) ~what ~report ~commit compiled =
    (* gh-ocannl-638: the one place that knows what shipped, on the one path that ships.

       A raising callback is the caller's failure and propagates, like [report]'s — but it
       propagates INSTEAD of returning [compiled]. Dropping the OCaml value frees nothing (the
       backend's pool table roots the slabs), and the caller never received a handle, so the routine
       would be permanently unreachable and unreleasable: a repeatedly-tuning process with a flaky
       callback accumulates one routine footprint per call. Release it here, then re-raise the
       caller's own exception with its original backtrace. *)
    (match on_ship with
    | None -> ()
    | Some f -> (
        try
          f
            {
              ship_arm = what;
              ship_replayed = replayed;
              ship_routine = (snd compiled).Context.name;
              ship_report = report;
              ship_compiles = !shipping_compiles;
              ship_substituted = substituted;
            }
        with exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          (try Context.release (fst compiled)
           with exn2 when not (must_propagate exn2) ->
             logf "release after a failing on_ship callback failed: %s" (Exn.to_string exn2));
          Stdlib.Printexc.raise_with_backtrace exn backtrace));
    (* gh-ocannl-786: recorded once the artifact has actually shipped -- after the callback, whose
       failure means nothing shipped -- and protected like it: a process-level failure while the
       entry is written would otherwise leave the shipped routine unreachable. *)
    (try commit ()
     with exn ->
       let backtrace = Stdlib.Printexc.get_raw_backtrace () in
       (try Context.release (fst compiled)
        with exn2 when not (must_propagate exn2) ->
          logf "release after a failing placement recording failed: %s" (Exn.to_string exn2));
       Stdlib.Printexc.raise_with_backtrace exn backtrace);
    compiled
  in
  (* The unwrap is the type-level statement of an invariant established on every path that reaches
     it: on the cold path both arms [Error] propagated, [a_wins] never picks a failed arm, and the
     flip chain only ever replaces its incumbent with a strictly faster {e completed} search
     ([infinity] ranks a failed one); on the replay path the one search's failure propagates like a
     forced arm's.

     gh-ocannl-1408: shipping is where the one compile from [ctx] happens. A compile that fails is
     that placement failing after all, which the arms' containment (gh-ocannl-550) answers by
     shipping the next completed result: [fallbacks], in order, each shipped under its own name and
     never recorded -- the evidence the recording would rest on includes a placement that did not
     compile. A forced arm passes none, so its failure propagates. *)
  let rec ship ?(fallbacks = []) ?(last_resort = false) ?replayed ~what ~report ~commit = function
    | Ok deferred -> (
        (* A later arm can poison the lineage after this one succeeded. With a [timing_ctx] that is
           the scratch lineage and the winner — compiled from [ctx] — is unaffected. WITHOUT one the
           arms search in the caller's own lineage, so the winner's first [Context.run] is
           guaranteed to raise: handing it back would report success for a routine that cannot
           execute, and blame it on whichever routine poisoned the ledger. The poisoning failure is
           the honest answer, at the point where it is known -- and before the compile, which would
           only be wasted. *)
        match if Option.is_none timing_ctx then Context.poisoned_failure ctx else None with
        | Some poisoned ->
            logf "the winner cannot ship: a later arm poisoned the caller's lineage";
            raise poisoned
        | None -> (
            match compile_deferred deferred with
            | exception exn
              when (not (must_propagate exn)) && List.is_empty fallbacks && last_resort ->
                (* Every completed result failed to compile after an earlier one shipped a
                   substitute and was released for them. Rather than failing a call that had a
                   usable routine, the caller's own untuned compile ships: from [ctx], the default
                   placements, so it ships as arm A whichever placement the substitute was. *)
                logf
                  "%s's compile failed (%s): shipping the untuned default of the default \
                   placements (arm A) after all, unrecorded"
                  what (Exn.to_string exn);
                deliver ~substituted:true ~what:"A" ~report:None
                  ~commit:(fun () ->
                    logf "placement store: nothing recorded, no measured placement compiled")
                  (counted (fun () -> Autotune.untuned_compile ?name ctx comp bindings))
            | exception exn when (not (must_propagate exn)) && not (List.is_empty fallbacks) ->
                let fallback_what, fallback, fallback_report = List.hd_exn fallbacks in
                logf "%s's compile failed (%s): shipping %s instead, unrecorded" what
                  (Exn.to_string exn) fallback_what;
                ship ~fallbacks:(List.tl_exn fallbacks) ~last_resort ~what:fallback_what
                  ~report:fallback_report
                  ~commit:(fun () ->
                    logf "placement store: nothing recorded, the measured winner did not compile")
                  fallback
            | compiled when Autotune.shipped_substitute deferred && not (List.is_empty fallbacks) ->
                (* The compile shipped the untuned default in place of the schedule the search
                   reported, so this placement won the ranking on a time nothing it ships was
                   measured at: the next completed result ships instead, as for a failed compile. *)
                (try Context.release (fst compiled)
                 with exn when not (must_propagate exn) ->
                   logf "release of a substituted compile failed: %s" (Exn.to_string exn));
                let fallback_what, fallback, fallback_report = List.hd_exn fallbacks in
                logf
                  "%s's compile shipped the untuned default in place of its measured schedule: \
                   shipping %s instead, unrecorded"
                  what fallback_what;
                ship ~fallbacks:(List.tl_exn fallbacks) ~last_resort:true ~what:fallback_what
                  ~report:fallback_report
                  ~commit:(fun () ->
                    logf "placement store: nothing recorded, the measured winner did not compile")
                  fallback
            | compiled ->
                (* Nothing to fall back to: it ships, but its decision is not recorded -- the
                   recorded time would be a schedule's that this routine is not. *)
                let commit =
                  if Autotune.shipped_substitute deferred then fun () ->
                    logf
                      "placement store: nothing recorded, %s shipped the untuned default in place \
                       of its measured schedule"
                      what
                  else commit
                in
                deliver ?replayed
                  ~substituted:(Autotune.shipped_substitute deferred)
                  ~what ~report ~commit compiled))
    | Error (exn, backtrace) -> Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  (* The completed results among [candidates], as [ship]'s fallbacks. *)
  let completed candidates = List.filter candidates ~f:(fun (_, r, _) -> Result.is_ok r) in
  let cold () =
    (* Resolved before the arms (gh-ocannl-1440): the call budget's split plans the flip slots the
       arms share it with. *)
    let inline_flips =
      match inline_flips with
      | Some n -> n
      | None -> Int.of_string (Utils.get_global_arg ~arg_name:"tune_inline_flips" ~default:"0")
    in
    let planned_flips = if forced then 0 else max 0 inline_flips in
    let a, a_ms, a_report =
      tune ~left:(2 + planned_flips) "A (default placements)" ctx timing_ctx
    in
    propagate_if_poisoned "A" a;
    let b, b_ms, b_report =
      tune ~left:(1 + planned_flips) "B (materialize-all)" (materialize ctx)
        (Option.map timing_ctx ~f:materialize)
    in
    (* Both arms gone: there is no winner to ship, and the first failure is the one that has not
       been cascaded from — a device the other arm's failure exhausted or a lineage it poisoned
       would otherwise be reported as the cause. *)
    (match (a, b) with
    | Error (a_exn, a_backtrace), Error (b_exn, b_backtrace) ->
        (* gh-ocannl-638: which failure to propagate is the selector's question too. With an arm
           forced, the caller asked for THAT artifact and its failure is the answer — handing back
           the other arm's exception would report a search the caller did not select, and contradict
           the documented promise that a forced arm's failure propagates rather than being replaced.
           With no arm forced the first failure still wins, for the original reason: it is the one
           that has not been cascaded from (a device the other arm exhausted, or a lineage it
           poisoned, would otherwise be reported as the cause). *)
        let exn, backtrace =
          match ship_arm with
          | Measured_winner | Force_arm_a -> (a_exn, a_backtrace)
          | Force_arm_b -> (b_exn, b_backtrace)
        in
        logf "both arms failed, nothing to ship (A: %s; B: %s); propagating %s"
          (Exn.to_string a_exn) (Exn.to_string b_exn)
          (match ship_arm with
          | Measured_winner -> "arm A's, the failure that has not been cascaded from"
          | Force_arm_a -> "arm A's, the forced arm"
          | Force_arm_b -> "arm B's, the forced arm");
        Stdlib.Printexc.raise_with_backtrace exn backtrace
    | _ -> ());
    let measured_a_wins =
      match (a, b) with
      (* A failed arm never wins, whatever the other arm's time is — including [infinity], which a
         completed search that timed nothing legitimately reports. *)
      | Ok _, Error _ -> true
      | Error _, Ok _ -> false
      | Ok _, Ok _ | Error _, Error _ -> Float.( <= ) a_ms b_ms
    in
    (* gh-ocannl-638: the measured comparison is still computed and still logged under a forced arm
       — it is the number the measurement reports — but it no longer decides. *)
    let a_wins =
      match ship_arm with
      | Measured_winner -> measured_a_wins
      | Force_arm_a -> true
      | Force_arm_b -> false
    in
    let arm_ms r ms = match r with Error _ -> "FAILED" | Ok _ -> Printf.sprintf "%.4f ms" ms in
    logf "winner: arm %s (A %s vs B %s)"
      (if measured_a_wins then "A" else "B")
      (arm_ms a a_ms) (arm_ms b b_ms);
    if forced then (
      Stdio.eprintf
        "Train.tune_placements: shipping arm %s by tune_ship_arm; the measured winner is arm %s (A \
         %s vs B %s)%s.\n\
         %!"
        (if a_wins then "A" else "B")
        (if measured_a_wins then "A" else "B")
        (arm_ms a a_ms) (arm_ms b b_ms)
        (if Bool.equal a_wins measured_a_wins then ", so the override changed nothing"
         else ", so the override changed what ships");
      (* A forced arm has no fallback: shipping the other one would return an artifact the caller
         did not ask for, under a setting whose whole purpose is that the returned routine IS the
         profiled one. Said here because the propagation below is otherwise indistinguishable from
         an ordinary both-arms-failed run. *)
      match if a_wins then a else b with
      | Ok _ -> ()
      | Error (exn, _) ->
          Stdio.eprintf
            "Train.tune_placements: the arm tune_ship_arm selected failed, so its failure \
             propagates rather than the other arm shipping in its place: %s\n\
             %!"
            (Exn.to_string exn));
    (* gh-ocannl-546: a tensorized winner of the arm that is then discarded reaches no artifact and
       no end-to-end number, so the placement A/B is where it has to be said. Stated as the margin
       it lost by, not as a bare flag: on a small routine the arms can be separated by less than the
       candidate-level timing spread. *)
    let shipped, dropped = if a_wins then (a_report, b_report) else (b_report, a_report) in
    Option.iter dropped ~f:(fun d ->
        if d.Autotune.best_tensorized then
          logf
            "NOTE arm %s crowned a tensorized candidate (%s at %.4f ms%s) and did NOT ship: arm %s \
             %s at %.4f ms%s"
            (if a_wins then "B" else "A")
            d.Autotune.best_label d.Autotune.best_ms
            (* A failed arm's crown is mid-search: it lost the A/B by failing, not by its time. *)
            (if Option.is_some (Autotune.terminal_failure d) then ", before that arm failed" else "")
            (if a_wins then "A" else "B")
            (* Under a forced arm the shipped one need not have won anything (gh-ocannl-638). *)
            (if forced then "ships by tune_ship_arm" else "wins the placement A/B")
            (if a_wins then a_ms else b_ms)
            (Option.value_map shipped ~default:"" ~f:(fun s ->
                 if s.Autotune.best_tensorized then " (which is tensorized too)"
                 else if Float.is_inf s.Autotune.mma_best_ms then
                   " (no tensorized candidate was timed in the shipping arm)"
                 else
                   Printf.sprintf " (its own best tensorized candidate: %.4f ms)"
                     s.Autotune.mma_best_ms)));
    let winner, winner_ms, winner_report =
      if a_wins then (a, a_ms, a_report) else (b, b_ms, b_report)
    in
    (* gh-ocannl-638: the chain walks from arm A toward arm B one node at a time, so a refined
       result is neither arm — which is exactly what a forced arm asks not to ship. Skipped rather
       than rejected: the combination arises from configuration (a config file's flip budget plus a
       commandline arm), so it is a request to resolve, not a caller error to fail. *)
    let inline_flips =
      if forced && inline_flips > 0 then (
        Stdio.eprintf
          "Train.tune_placements: tune_ship_arm forces an arm, so the %d-flip inline refinement is \
           skipped -- a refined placement vector is neither arm.\n\
           %!"
          inline_flips;
        0)
      else inline_flips
    in
    let winner_arm = if a_wins then "A" else "B" in
    let winner_decision = if a_wins then SC.Default else SC.Materialize_all in
    let commit_winner () =
      persist ~decision:winner_decision ~shipped:winner_report ~shipped_ms:winner_ms ~a_ms ~b_ms
    in
    (* gh-ocannl-1408: the other arm, should the winner fail to compile -- never under a forced arm,
       whose failure propagates. *)
    let winner_fallbacks =
      if forced then []
      else completed [ (if a_wins then ("B", b, b_report) else ("A", a, a_report)) ]
    in
    let ship_winner () =
      ship ~fallbacks:winner_fallbacks ~what:winner_arm ~report:winner_report ~commit:commit_winner
        winner
    in
    if inline_flips <= 0 then ship_winner ()
    else if call_budget_spent () then (
      (* gh-ocannl-1440 review round 1: no flip can start, so the decision surface -- a lowering and
         its analyses, which can block for long on a large step -- is not read either. *)
      call_budget_stopped_flips := true;
      logf "flip refinement skipped: the call budget of %g s is spent after the arms"
        (Option.value call_budget ~default:0.);
      Autotune.progressf "event=call_budget_spent budget_s=%g elapsed_s=%.1f measured=0"
        (Option.value call_budget ~default:0.)
        (call_clock ());
      ship_winner ())
    else
      let (* gh-555: greedy per-node refinement over the inlining decision vector. The vector lives
             on the default-policy arm (arm B's placements are caller-seeded wholesale, so its
             compile reports no policy decisions to flip), so the chain refines from arm A's context
             — a Materialize chain walks from A toward B one node at a time — and the refined result
             ships only if it beats the A/B winner. The decision surface is read analyze-only
             (gh-560; the arms' compiles already populated the analysis cache, so this costs
             specialization replays).

             gh-514, the placement-space search over this chain: the surface arrives ranked
             enablement-first ({!Autotune.placement_surface} — the gh-558 ordering lesson: a flip's
             value includes which sketch families become expressible under it, so family-unlocking
             Materialize flips outrank cost), and under config [autotune_bound_pruning] a
             [`Materialize] flip whose partial-vector roofline floor
             ({!Ir.Cost_model.completion_floor}, monotone in the chain's accumulated commitments)
             already meets the chain's best measured time is fathomed without spending budget — the
             admissible direction, exactly phase 4b's rule one level up. The budget counts {e
             measured} flips, so a fathomed candidate lets the next one in. *)
        module
        LL =
        Ir.Low_level
      in
      let
          (* gh-514 follow-up gh-ocannl-579: the enablement prior prices which sketch families a
             flip makes EXPRESSIBLE, and nothing about whether they pay. The arms just searched
             settle that, for free: arm B is the all-materialized specialization the prior derives
             its enablement set from, so its report's best tensorized time against its best time
             says what the promoted flips' family is worth on this device, on this computation, in
             this session. Under [tune_flip_ordering=profitable] (the default) a family measured to
             lose here voids the prior and the surface ranks by cost — on gh-514's metal/f16 cell
             the promotion displaced the winning cheap inline flip out of a budget-5 chain. Both
             arms' reports are handed over, including a failing arm's: its timings are measurements
             of the family even though its [best_ms] is not shippable. The surface derives the
             verdict, and only when the ordering in force consults one. *)
          evidence =
        List.filter_opt [ a_report; b_report ]
      in
      let surface =
        (* Outside the tuner's failure containment; a lowering failure (the A/B searches above can
           still have crowned a winner) must skip the refinement, not fail the tune. *)
        Autotune.progressf "event=stage stage=flip_surface";
        match Autotune.placement_surface ?name ~evidence ctx comp bindings with
        | s -> Some s
        (* This containment is for a lowering that declined, and for nothing else. A malformed
           [tune_flip_profit_margin] raises {!Utils.User_error} from inside it (gh-ocannl-579): the
           configuration asked for a refinement it also made impossible, and swallowing that would
           silently skip the refinement and ship the A/B winner as though the setting had been
           honored. The two process-level classes are not this containment's either, for the same
           reasons they are not the arms'; nor is a malformed registered setting, which the surface
           can be the first to read ([must_propagate]). *)
        | exception exn when match exn with Utils.User_error _ -> true | _ -> must_propagate exn ->
            Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
        | exception exn ->
            logf "flip refinement skipped: the decision-surface lowering failed: %s"
              (Exn.to_string exn);
            None
      in
      match surface with
      | None -> ship_winner ()
      | Some surface ->
          let bound_pruning = Utils.registered_flag ~arg_name:"autotune_bound_pruning" in
          (* gh-ocannl-1110: a flip is abandoned once its best after its first [beam_width] timed
             candidates trails the incumbent's best at the same depth by more than this ratio
             ({!Autotune.flip_abandon_ratio}). Read once, here, so a malformed margin fails the
             chain before any flip rather than midway. It propagates, like the surface's own reading
             of it above. *)
          let abandon_ratio = Autotune.flip_abandon_ratio () in
          (* Against the incumbent's own timed record, which a report without one (an entry cached
             before the field) leaves empty: no comparison, so the flip runs its full search. *)
          let abandon_against (incumbent : Autotune.report option) =
            Option.map incumbent ~f:(fun r ->
                { Autotune.incumbent_steps = r.Autotune.best_steps; trailing_ratio = abandon_ratio })
          in
          let candidates = surface.Autotune.ps_candidates in
          logf "flip refinement: %d candidate(s), %d enablement-promoted, ranked by %s, budget %d%s"
            (List.length candidates)
            (Set.length surface.Autotune.ps_enablement)
            (let name =
               match surface.Autotune.ps_ordering with
               | `Cost -> "cost"
               | `Enablement -> "enablement"
             in
             match surface.Autotune.ps_profit with
             | Some profit -> name ^ " (" ^ Autotune.family_profit_summary profit ^ ")"
             | None -> name ^ " (configured unconditionally, so no profitability evidence was read)")
            inline_flips
            (if bound_pruning then ", bound pruning on" else "");
          Autotune.progressf "event=flips_start candidates=%d budget=%d" (List.length candidates)
            inline_flips;
          (* The incumbent: its result, time, contexts, and the report of the search that produced
             it -- what a refined decision's recording reads its outcome digest from
             (gh-ocannl-1022). *)
          let chain = ref (a, a_ms, ctx, timing_ctx, a_report) in
          (* The accepted flips in chain order, the placement vector a refined result ships under
             and what the placement store records for it (gh-ocannl-786). *)
          let accepted = ref [] in
          (* The chain's accumulated placement commitments, for the floor: an accepted
             [`Materialize] flip and a rejected [`Inline] flip both leave the node certainly
             materialized in every completion the chain can still reach; the other two outcomes
             leave it open (inline commitments never tighten the floor). *)
          let certain_mat = ref [] in
          (* A candidate's alternatives are its node's mutually exclusive readings (gh-ocannl-616),
             so they are measured as ONE group, each from the chain's own context against the same
             incumbent, and the best of them is committed when it beats the incumbent — applied one
             after another, the second would be measured on top of the first's acceptance. Only a
             node whose DEFAULT reading is [`Materialize] (the candidate's own field,
             gh-ocannl-1017) is left certainly materialized by a group whose every alternative
             lost. *)
          let measured = ref 0 and pruned = ref 0 and abandoned = ref 0 and twinned = ref 0 in
          (* gh-ocannl-1274: the structural-twin rule. The identity of every alternative measured
             and lost in this chain -- abandoned, or searched and not accepted -- with the label of
             the first flip that lost under it. An alternative whose identity is already here is a
             canonically identical local reading ({!Autotune.placement_surface}'s
             [ps_twin_identity]) of a flip that lost, so it is skipped as [twin-of] that flip
             without a budget slot: on rog's approximate CUDA gpt2_mini cell the per-layer q/k/v
             flips took the first 11 slots of the chain and each set's members lost by the same
             figure, so a budget of 2 never reached past layer 1's k and v. Deterministic in what
             the chain measured, so a warm chain, whose anchors replay their abandonments or their
             crowns, skips the same twins. *)
          let lost = Hashtbl.create (module String) in
          let chain_reading n = List.Assoc.find !accepted n ~equal:Tn.equal in
          (* The flips are measured in the timing lineage, so a twin must agree on what THAT lineage
             brings to each neighbour too -- the tag {!placement_problem} gives it -- and on the
             placement the timing lowering DECIDED from it, which an inherited computation or a
             virtualization cap can make differ from the surface's for the same code. A timing
             lineage whose decision surface cannot be read disables the rule rather than twinning on
             half the evidence. *)
          let twin_tag =
            match timing_ctx with
            | None -> Some (fun _ -> Some "")
            | Some tctx -> (
                match
                  ( Context.placements tctx,
                    (Context.lowered_for_decisions ?name tctx comp bindings).LL.optimize_ctx )
                with
                | plc, octx ->
                    Some
                      (fun n ->
                        (* A computation an earlier routine of THAT lineage stored is replayed where
                           the flips are measured, and the source does not show it. *)
                        if Context.inherits_computation tctx n then None
                        else
                          Some
                            (";t"
                            ^ Ir.Schedule_cache.lineage_tag plc octx n
                            ^
                            match Tn.Placements.get octx.LL.placements n with
                            | None -> ";td=u"
                            | Some (m, _) -> ";td=" ^ Sexp.to_string (Tn.sexp_of_memory_mode m)))
                | exception exn when must_propagate exn ->
                    (* A process-level class propagates. *)
                    Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
                | exception exn ->
                    logf "structural-twin rule off: the timing lineage's lowering failed: %s"
                      (Exn.to_string exn);
                    None)
          in
          (* gh-ocannl-1440: the flip searches the call budget's split still plans -- the flip
             budget left, or the alternatives not yet considered if fewer (a pruned or twinned
             alternative is considered without a search; review round 1) -- and the stop once the
             call's budget is spent, checked before every alternative, a group's included: each
             alternative measured is compared against the same incumbent, and a group whose
             alternatives were not all measured leaves its node open, as a pruned alternative
             does. *)
          let unconsidered =
            ref (List.sum (module Int) candidates ~f:(fun fc -> List.length fc.LL.fc_alternatives))
          in
          let flips_left () = max 1 (min (inline_flips - !measured) !unconsidered) in
          let call_stopped = ref false in
          let stop_for_call_budget () =
            if (not !call_stopped) && call_budget_spent () then (
              call_stopped := true;
              call_budget_stopped_flips := true;
              logf "flip refinement stopped: the call budget of %g s is spent (%d flip(s) measured)"
                (Option.value call_budget ~default:0.)
                !measured;
              Autotune.progressf "event=call_budget_spent budget_s=%g elapsed_s=%.1f measured=%d"
                (Option.value call_budget ~default:0.)
                (call_clock ()) !measured);
            !call_stopped
          in
          let rec walk = function
            | [] -> ()
            | _ when !measured >= inline_flips -> ()
            | _ when stop_for_call_budget () -> ()
            (* Same lineage, same rule as the A/B above: a poisoned one refuses every timing run, so
               a further search can only fail. Unlike the arms, there is nothing to propagate here —
               the A/B winner is already in hand — so the refinement just stops. *)
            | _ when lineage_poisoned () ->
                logf "flip refinement stopped: the shared lineage is poisoned"
            | (fc : LL.flip_candidate) :: rest ->
                let tn = fc.LL.fc_tn in
                let chain_result, chain_ms, base_ctx, base_timing, chain_report = !chain in
                (* Only against a shippable incumbent: a failed arm A still starts the chain, with
                   its partial record, and a flip may then be the only result that ships. *)
                let abandon =
                  match chain_result with Ok _ -> abandon_against chain_report | Error _ -> None
                in
                (* A group started is a group finished: the budget is checked between candidates,
                   not between a node's alternatives, or the comparison against the same incumbent
                   that the group exists for would be cut short by an exhausted budget. *)
                let consider (fa : LL.flip_alternative) =
                  let arm =
                    Printf.sprintf "flip %s %s (cost %d%s)"
                      (LL.reading_to_string fa.LL.fa_flip)
                      (Tn.debug_name tn) fa.LL.fa_recompute_cost
                      (if Set.mem surface.Autotune.ps_enablement tn then ", enablement" else "")
                  in
                  let identity =
                    Option.bind twin_tag ~f:(fun node_tag ->
                        surface.Autotune.ps_twin_identity ~node_tag ~chain:chain_reading fc fa)
                  in
                  match Option.bind identity ~f:(Hashtbl.find lost) with
                  | Some anchor ->
                      (* Lost by its twin: rejected, like a measured loss, at no budget. *)
                      Int.incr twinned;
                      logf "%s skipped: twin-of %s, which lost (no budget slot)" arm anchor;
                      Autotune.progressf "event=flip_twin arm=%S twin_of=%S" arm anchor;
                      Some `Twin
                  | None -> (
                      let floor =
                        match fa.LL.fa_flip with
                        | `Materialize when bound_pruning ->
                            surface.Autotune.ps_floor_ms ~materialized:(tn :: !certain_mat)
                        | `Materialize | `Inline | `Footprint -> None
                      in
                      match floor with
                      | Some fl when Float.(fl >= chain_ms) ->
                          (* Fathomed: the floor lower-bounds every completion with this node
                             materialized, so no nested search from here can beat the incumbent. The
                             node's placement stays open — nothing to commit. *)
                          Int.incr pruned;
                          logf "%s bound-pruned: floor %.4f ms >= incumbent %.4f ms" arm fl chain_ms;
                          None
                      | _ when stop_for_call_budget () -> Some `Cut
                      | _ ->
                          let apply c = Context.decide_reading c fa.LL.fa_flip [ tn ] in
                          let ctx' = apply base_ctx in
                          let timing' = Option.map base_timing ~f:apply in
                          let r, ms, rep =
                            tune ~to_report:flip_report ~left:(flips_left ())
                              ~progress_note:
                                (Printf.sprintf " flip=%d/%d" (!measured + 1) inline_flips)
                              ?abandon arm ctx' timing'
                          in
                          (* An abandoned flip spent a budget slot: it was measured, briefly, and
                             lost. *)
                          Int.incr measured;
                          (match r with
                          | Error (Autotune.Search_abandoned _, _) -> Int.incr abandoned
                          | Ok _ | Error _ -> ());
                          Some (`Measured (fa, r, ms, ctx', timing', rep, (arm, identity))))
                in
                (* The alternative leaves [unconsidered] once its own share is taken (review round
                   3): its search is among those still planned when it starts. *)
                let try_alternative fa =
                  Exn.protect ~f:(fun () -> consider fa) ~finally:(fun () -> Int.decr unconsidered)
                in
                let outcomes = List.filter_map fc.LL.fc_alternatives ~f:try_alternative in
                let results =
                  List.filter_map outcomes ~f:(function
                    | `Measured m -> Some m
                    | `Twin | `Cut -> None)
                in
                (* gh-ocannl-1440 review round 2: a group the call's budget cut is a comparison not
                   made -- its unmeasured alternatives might have beaten the measured ones -- so
                   nothing of it is committed, and its node stays open. *)
                let group_cut = List.exists outcomes ~f:(function `Cut -> true | _ -> false) in
                let best =
                  List.min_elt results ~compare:(fun (_, _, a, _, _, _, _) (_, _, b, _, _, _, _) ->
                      Float.compare a b)
                in
                (* Every alternative that lost to the incumbent on clean evidence becomes a twin
                   anchor: abandoned, or completed at a finite time no faster than the incumbent. A
                   failed search measured nothing about the flip, and a sibling that beat the
                   incumbent but not the group's best did not lose. *)
                List.iter results ~f:(fun (_, r, ms, _, _, rep, (arm, identity)) ->
                    (* Only clean evidence anchors: a search with a refused (contended) window
                       ranked an incomplete candidate set, and one that timed nothing measured
                       nothing -- its twin gets its own chance. The search-disabled and
                       lineage-refused reports are the exception: nothing is ever timed there,
                       nothing can ship from such a flip, and anchoring is what makes a search-off
                       chain list its twin classes -- and what keeps the timing tag's refusal the
                       thing that separates two flips of a lineage-refused chain
                       (gh-ocannl-1421). *)
                    let search_off, clean =
                      match rep with
                      | Some
                          {
                            Autotune.outcome = Autotune.Search_disabled | Autotune.Lineage_refused _;
                            _;
                          } ->
                          (true, true)
                      | Some rep -> (false, rep.Autotune.timings_contended = 0)
                      | None -> (false, false)
                    in
                    let lost_to_incumbent =
                      match r with
                      | Error (Autotune.Search_abandoned _, _) -> clean
                      | Ok _ when search_off -> true
                      | Ok _ -> clean && Float.is_finite ms && Float.(ms >= chain_ms)
                      | Error _ -> false
                    in
                    match identity with
                    | Some id when lost_to_incumbent -> Hashtbl.add lost ~key:id ~data:arm |> ignore
                    | _ -> ());
                (match best with
                | Some _ when group_cut ->
                    logf
                      "flip group %s: cut by the call budget after %d of its %d alternative(s); \
                       nothing of it is committed"
                      (Tn.debug_name tn) (List.length results)
                      (List.length fc.LL.fc_alternatives)
                | _ when group_cut -> ()
                | Some (bfa, r, ms, ctx', timing', rep, _) when Float.(ms < chain_ms) -> (
                    chain := (r, ms, ctx', timing', rep);
                    accepted := (tn, bfa.LL.fa_flip) :: !accepted;
                    if List.length fc.LL.fc_alternatives > 1 then
                      logf "flip group %s: %s wins at %.4f ms" (Tn.debug_name tn)
                        (LL.reading_to_string bfa.LL.fa_flip)
                        ms;
                    match bfa.LL.fa_flip with
                    | `Materialize -> certain_mat := tn :: !certain_mat
                    | `Inline | `Footprint -> ())
                | _ ->
                    (* Every alternative measured and rejected: a default-materialized node stays
                       materialized in every completion the chain can still reach; a virtual or
                       footprint-scoped one keeps its reading and stays open. A pruned alternative
                       leaves the node open too. *)
                    if
                      LL.equal_reading fc.LL.fc_default `Materialize
                      && List.length outcomes = List.length fc.LL.fc_alternatives
                    then certain_mat := tn :: !certain_mat);
                walk rest
          in
          walk candidates;
          if !pruned > 0 then
            logf "flip refinement: %d flip(s) bound-pruned, %d measured" !pruned !measured;
          if !abandoned > 0 then
            logf "flip refinement: %d of %d measured flip(s) abandoned early" !abandoned !measured;
          if !twinned > 0 then
            logf "flip refinement: %d flip(s) skipped as structural twins of a lost flip" !twinned;
          let chain_result, chain_ms, _, _, chain_report = !chain in
          Autotune.progressf
            "event=flips_done candidates=%d budget=%d measured=%d pruned=%d abandoned=%d \
             twinned=%d improved=%b best_ms=%s"
            (List.length candidates) inline_flips !measured !pruned !abandoned !twinned
            Float.(chain_ms < winner_ms)
            (Autotune.progress_ms (Float.min chain_ms winner_ms));
          if Float.(chain_ms < winner_ms) then (
            logf "flip refinement ships: %.4f ms (the placement A/B winner was %.4f ms)" chain_ms
              winner_ms;
            let commit () =
              match refined_decision (List.rev !accepted) with
              | Some decision ->
                  persist ~decision ~shipped:chain_report ~shipped_ms:chain_ms ~a_ms ~b_ms
              | None ->
                  logf
                    "placement store: the refined vector names a node outside the decision \
                     problem, so it is not recorded"
            in
            ship
              ~fallbacks:(completed [ (winner_arm, winner, winner_report) ] @ winner_fallbacks)
              ~what:"flip" ~report:chain_report ~commit chain_result)
          else (
            logf "flip refinement did not improve on the A/B winner (%.4f ms vs %.4f ms)" chain_ms
              winner_ms;
            ship_winner ())
  in
  match replay with
  | Some (e, decisions) -> (
      (* The recorded decision, applied as the search applied it, then the one search the decision
         produces -- normally a schedule-cache replay too, and a search when the schedule entry is
         cold (contention kept it so, or the directory was swept). Exactly one report reaches
         [report]: the shipped placement's -- delivered once the search has succeeded, so a replay
         that fails and falls back to the arms below never occupies an arm slot (the arms then
         report in position, as on a cold run, and a positional consumer names them right). *)
      let what = SC.shipped_label e.SC.decision in
      logf "placement store: replaying decision %s (recorded at %.4f ms; A %.4f ms vs B %.4f ms)"
        what e.SC.shipped_ms e.SC.arm_a_ms e.SC.arm_b_ms;
      let deferred = ref None in
      (* Compiled eagerly: a replayed decision that does not compile is a failed replay, which the
         arms below answer. *)
      let r, _ms, replay_report =
        tune ~eager:true ~left:1
          ~to_report:(Some (fun r -> deferred := Some r))
          (what ^ " (replayed placement)") (apply decisions ctx)
          (Option.map timing_ctx ~f:(apply decisions))
      in
      propagate_if_poisoned what r;
      match r with
      | Ok compiled when Autotune.shipped_substitute compiled ->
          (* The replay validated, but its compile shipped the untuned default in place of the
             schedule the decision was recorded for: a failed replay like the one below, so the
             decision is treated as stale and the arms are searched. *)
          (try Context.release (fst (compile_deferred compiled))
           with exn when not (must_propagate exn) ->
             logf "release of a substituted replay failed: %s" (Exn.to_string exn));
          logf
            "placement store: the replayed decision %s shipped the untuned default in place of its \
             schedule; treating it as stale and searching the arms"
            what;
          observed := [];
          cold ()
      | Ok compiled ->
          (* The caller's callback, with the caller's exception: it propagates, and nothing ships
             (gh-ocannl-550), so the routine compiled for it is released first. *)
          (match (report, !deferred) with
          | Some f, Some rep -> (
              try f rep
              with exn ->
                let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                (try Context.release (fst (compile_deferred compiled))
                 with exn2 when not (must_propagate exn2) ->
                   logf "release after a failing report callback failed: %s" (Exn.to_string exn2));
                Stdlib.Printexc.raise_with_backtrace exn backtrace)
          | _ -> ());
          ship ~replayed:true ~what ~report:replay_report ~commit:(fun () -> ()) r
      | Error (exn, _) ->
          (* A replayed search that fails without poisoning the lineage is the cold path's losing
             arm, one level up (gh-ocannl-550): a cached materialize-all decision can meet a
             recoverable device OOM under today's memory pressure while the default arm would still
             compile, and the cold path ships that sibling. So the recorded decision is treated as
             stale and the arms are searched; the failed replay's report is no evidence about the
             decision they reach, so it is dropped from what a recording rests on, and it was never
             delivered to [report] (it ships nothing, and the arms take the two slots). *)
          logf
            "placement store: the replayed decision %s failed (%s); treating it as stale and \
             searching the arms"
            what (Exn.to_string exn);
          observed := [];
          cold ())
  | None -> cold ()

module Lazy = Utils.Lazy

(* The untuned compile of the recipes ({!Autotune.untuned_compile}): with
   [model_default_schedule=true] (gh-ocannl-491), the default schedule is picked by the analytic
   cost model ({!Autotune.model_default} — zero timing runs, advisory, falls back to the ordinary
   default pipeline); otherwise the default pipeline with its large GPU contractions seeded with the
   matmul family's reuse-tiled sketch ({!Autotune.seeded_default}, gh-ocannl-1324 — untimed, results
   unchanged). *)
let compile_with_model_gate ?(budgeted = false) ctx comp bindings =
  (* gh-ocannl-498: a memory budget is scored against the DEFAULT schedule pipeline
     ({!Ir.Schedule.maybe_default_schedules}, what [Backends.score_footprint] runs). The model gate
     picks a different segmentation, and alias spans and arena size are functions of the final
     segmentation — so honoring the gate on a budgeted compile would link a layout nobody scored,
     and a plan reported within budget could still exhaust the device. The budget is a constraint
     and the model pick is an advisory optimization, so the constraint wins: a budgeted compile uses
     the pipeline that was scored. The seeded default (gh-ocannl-1324) keeps the default
     segmentation but stages tiles and moves accumulators the scorer did not see, so it yields to
     the budget too. Documented at [memory_budget] in ocannl_config.reference. *)
  if budgeted then (
    if Lazy.force Autotune.model_default_enabled then
      Stdio.eprintf
        "Train: memory_budget is set, so this compile uses the default schedule pipeline that the \
         budget was scored against, not model_default_schedule's pick.\n\
         %!";
    Context.compile ctx comp bindings)
  else Autotune.untuned_compile ctx comp bindings

let memory_budget_setting () =
  let raw = String.strip (Utils.get_global_arg ~arg_name:"memory_budget" ~default:"0") in
  let bad () =
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Train: ocannl_memory_budget should be a byte count (optionally suffixed K, M or G), \
             the word \"minimize\", or 0 to disable; found: %s"
            raw)
  in
  match String.lowercase raw with
  | "" | "0" | "off" | "false" | "none" -> None
  | "minimize" -> Some Memory_budget.Minimize
  | lower ->
      let lower = Option.value (String.chop_suffix lower ~suffix:"b") ~default:lower in
      let mult, digits =
        match String.chop_suffix lower ~suffix:"k" with
        | Some d -> (1024, d)
        | None -> (
            match String.chop_suffix lower ~suffix:"m" with
            | Some d -> (1024 * 1024, d)
            | None -> (
                match String.chop_suffix lower ~suffix:"g" with
                | Some d -> (1024 * 1024 * 1024, d)
                | None -> (1, lower)))
      in
      let n = match Int.of_string_opt (String.strip digits) with Some n -> n | None -> bad () in
      (* The suffix scaling is where a syntactically fine setting turns into nonsense: "5000000000G"
         parses, then wraps to a negative or tiny target that the planner would honor as an
         unreachably tight budget and rematerialize hard against. Reject instead. *)
      if n <= 0 || n > Int.max_value / mult then bad () else Some (Memory_budget.Bytes (n * mult))

(** gh-ocannl-498: plan [comp]'s inlining decision vector against a device-memory budget and return
    the context to compile it from. [budget] overrides the config key [memory_budget]; with neither,
    this is the identity on [ctx] and returns [None] — the default-off path does not lower, score or
    decide anything. See {!Memory_budget.fit} for what the planner does and what it requires (config
    [buffer_aliasing]). *)
let fit_memory_budget ?budget ?max_candidates ?name ctx comp bindings =
  match match budget with Some _ as b -> b | None -> memory_budget_setting () with
  | None -> (ctx, None)
  | Some budget ->
      let ctx, plan = Memory_budget.fit ?name ?max_candidates ~budget ctx comp bindings in
      (ctx, Some plan)

(** Dumps [comp] as a [.cd] file in the build directory, for the [?output_cd_file] argument of
    {!to_routine} and {!run_once}. [caller] names the calling function in the error raised when the
    global setting [output_debug_files_in_build_directory] is false. *)
let dump_cd_file ~caller bindings (comp : Asgns.comp) =
  let name = Asgns.get_name_exn comp.Asgns.asgns in
  if not Utils.settings.output_debug_files_in_build_directory then
    raise
    @@ Utils.User_error
         (caller ^ ": output_cd_file is true, but output_debug_files_in_build_directory is false");
  let cd_source = Utils.output_to_build_file ~fname:(name ^ "-debug.cd") in
  let static_indices = Idx.bound_symbols bindings in
  match cd_source with
  | None -> ()
  | Some callback -> callback (Asgns.to_doc ~name ~static_indices () comp.Asgns.asgns)

(** The gh-ocannl-498 rematerialization seam shared by {!to_routine} and {!run_once}: plan [comp]'s
    inlining decision vector against the budget, compile from the planned context, and only then let
    [budget_report] observe the plan.

    The report fires AFTER the compile so it observes the plan that SHIPPED — a compile or link
    failure must not have announced one. It also keeps a callback from reaching the compile it is
    reporting on: the config gates the scoring depends on ([buffer_aliasing]) are re-read at each
    compile, so a callback that flipped one would make the routine use a layout other than the one
    just scored. *)
let compile_within_budget ?budget ?max_candidates ?budget_report ctx comp bindings =
  let ctx, budget_plan = fit_memory_budget ?budget ?max_candidates ctx comp bindings in
  let budgeted = Option.is_some budget_plan in
  let ctx, routine = compile_with_model_gate ~budgeted ctx comp bindings in
  Option.iter budget_report ~f:(fun f -> Option.iter budget_plan ~f);
  (ctx, routine)

let%track7_sexp to_routine (ctx : Context.t) ?(output_cd_file = false) ?budget ?max_candidates
    ?budget_report bindings comp =
  if output_cd_file then dump_cd_file ~caller:"Train.to_routine" bindings comp;
  (* Materialize the guessed output nodes so they persist across calls and are inspectable on demand
     via the context (gh-ocannl-333). *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  (* gh-ocannl-498: budget planning goes AFTER the output-materialization intent above (those nodes
     must not be flip candidates) and BEFORE the compile whose placements it steers. Off by default:
     with no budget this is the identity and the compile below is unchanged. *)
  compile_within_budget ?budget ?max_candidates ?budget_report ctx comp bindings

let bind_all_to_zero (routine : Context.routine) =
  List.iter routine.Context.bindings ~f:(fun (_, r) -> r := 0)

let init_params ?(reinit_all = false) ctx bindings t =
  let skip =
    if reinit_all then None
    else
      (* Check which params are already initialized. *)
      Some
        (Set.fold t.Tensor.params
           ~init:(Map.empty (module Tn))
           ~f:(fun skip p ->
             if Context.is_initialized ctx p.Tensor.value then
               Map.set skip ~key:p.Tensor.value ~data:()
             else skip))
  in
  let comp = Tensor.init_params ?skip t in
  (* Mirror the selected parameter initialization graph, including parameters read by another
     parameter's initializer. Those must be restored before executing the dependent initializer. *)
  let rec collect_host_params nodes (t : Tensor.t) =
    Set.fold t.params ~init:nodes ~f:(fun nodes p ->
        if Option.exists skip ~f:(fun skip -> Map.mem skip p.Tensor.value) then nodes
        else
          let nodes =
            if Ir.Host_inits.mem p.Tensor.value then Set.add nodes p.Tensor.value else nodes
          in
          collect_host_params nodes p)
  in
  let host_params = collect_host_params (Set.empty (module Tn)) t in
  (* Materialize the parameters being initialized so they persist and are inspectable on demand. *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  (* Linking initializes newly allocated literal buffers. Reinitialization must also restore
     existing host-backed parameters (gh-ocannl-641), which have no executable forward initializer.
     Upload after compilation finalizes shapes, before running computed initializers that may read
     these parameters. A newly linked buffer needs no second upload during ordinary
     initialization. *)
  let ctx, routine = Context.compile ctx comp bindings in
  (* Initialization runs once, not per launch-parameter value: it binds every parameter to 0,
     explicitly, since linking leaves them unassigned (gh-ocannl-940). *)
  bind_all_to_zero routine;
  let ctx =
    Set.fold host_params ~init:ctx ~f:(fun ctx tn ->
        if reinit_all || not (Context.is_initialized ctx tn) then
          Context.from_host ctx tn (Lazy.force (Option.value_exn (Ir.Host_inits.find tn)))
        else ctx)
  in
  Context.run ctx routine

let%track3_sexp run_once ?(output_cd_file = false) ?(skip_init = false) ?reinit_all
    ?(bindings = IDX.empty) ?budget ?max_candidates ?budget_report ~f ctx (t : Tensor.t) : Context.t
    =
  set_materialized t.Tensor.value;
  (* Compute the update early, to ensure the shape inference is done. *)
  let update = f t in
  if output_cd_file then dump_cd_file ~caller:"Train.run_once" bindings update;
  let ctx =
    if skip_init || Set.is_empty t.params then ctx else init_params ?reinit_all ctx bindings t
  in
  let ctx, routine =
    compile_within_budget ?budget ?max_candidates ?budget_report ctx update bindings
  in
  (* A one-off run: every launch parameter at 0, written explicitly (gh-ocannl-940). *)
  bind_all_to_zero routine;
  Context.run ctx routine

let forward_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ?budget
    ?max_candidates ?budget_report ctx t =
  let ctx =
    run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ?budget ?max_candidates ?budget_report
      ~f:forward ctx t
  in
  (* gh-ocannl-777's recompute-on-read path should retire this forward-root consumption cleanup. *)
  Tensor.discard_backprop_code t;
  ctx

let update_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ?budget
    ?max_candidates ?budget_report ctx t =
  run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ?budget ?max_candidates ?budget_report
    ~f:grad_update ctx t

(* For-print materialization (gh-ocannl-333 AC 5): the [%cd "for_print" =: t] trick. When a tensor's
   value is not already materialized in the printing context, recompile a copy of it ([for_print = t
   + 0]) into a fresh device-resident node and register that node as a for-print proxy, so the
   printer reads the tensor's value through it.

   This is best-effort: it works for recomputable (e.g. virtual / fetch-defined) tensors. For a
   tensor that is materialized elsewhere but simply absent from this context, the copy cannot be
   linked (its operand has no value here) — in that case we fall back to the metadata placeholder
   rather than crash. A fresh copy is built each call because [forward_once] consumes the copy's
   forward root; the for-print node is registered as the source's proxy for subsequent reads. *)
let ensure_printable (ctx : Context.t) (t : Tensor.t) : Context.t =
  if Context.mem ctx t.Tensor.value then ctx
  else
    try
      let for_print =
        let%op for_print = t + 0 in
        for_print
      in
      let ctx = forward_once ctx for_print in
      Context.register_for_print ~src:t.Tensor.value ~proxy:for_print.Tensor.value;
      ctx
    with _ -> ctx

let%debug7_sexp printf ?here ?(with_grad = true) ?(with_code = false) ?(with_low_level = false)
    ?(style = `Default) (ctx : Context.t) (t : Tensor.t) : unit =
  let ctx = ensure_printable ctx t in
  Tensor.print ?here ~force:true ~ctx ~with_grad ~with_code ~with_low_level style t

let printf_tree ?here ?with_value ?(with_grad = true) ?(depth = 9) (ctx : Context.t) t =
  let ctx = ensure_printable ctx t in
  Tensor.print_tree ?here ~force:true ~ctx ?with_value ~with_grad ~depth t
