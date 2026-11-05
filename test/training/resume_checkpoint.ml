(* Resumable training checkpoints (gh-ocannl-96): a run stopped mid-epoch, checkpointed with
   [Checkpoint.save], and resumed in a FRESH PROCESS by [Checkpoint.restore] must continue exactly
   as the uninterrupted run -- the same per-step losses and the same final parameters, bit for bit,
   on cc.

   The loop carries every kind of state a resume has to bring back: randomly initialized parameters
   (the counter-based device RNG), an SGD momentum buffer (optimizer state the compiled routine
   carries between runs), a cosine learning-rate schedule fed from the step counter, a data position
   inside an epoch, a per-epoch shuffle drawn from a host [Stdlib.Random.State], and the current
   epoch's batch order, kept in [progress.extra].

   The controls show the bit-for-bit claim discriminates on each piece: a resume that restores the
   parameters but not the momentum, and one that restores everything but the host generator, both
   end on different parameters. [Checkpoint.restore] refuses a state set missing the momentum
   (rather than restoring it partially) and refuses a plain [Persistence] checkpoint, which records
   no progress.

   The random seeds (gh-ocannl-1727): the checkpoint records them, a program built under the same
   seeds restores -- the default one above, and a graph's scoped seed below -- and one built under
   others is refused, naming both sides' seeds, a different scoped seed under the same global one
   included. A format-1 checkpoint, which predates the recorded seed, restores under any seed with a
   warning; a format-2 one missing its seed is refused as malformed.

   The mixed-precision leg (gh-ocannl-1726): master weights with f16 cast twins, activations and
   gradients, dynamic loss scaling ([Mixed_prec.scaled_grad_update] / [scaled_sgd_update], the
   checksum read on the host between the two routines), and a [Train.Outlier_detector] on the loss
   whose verdict skips the optimizer step. The scaler's scale and good-step count and the detector's
   window are host state no device checkpoint holds; they travel in [progress.extra] through their
   [save_state] / [restore_state] pairs, and the fresh-process resume continues bit for bit --
   losses, z-scores, skips, scales and final master weights. The controls: a resume re-creating the
   scaler, and one re-creating the detector, both end on other parameters; and both restores refuse
   a state saved under other schedule parameters. *)

open Base
open Ocannl
open Stdio
module IDX = Train.IDX
module Tn = Ir.Tnode
module MP = Mixed_prec
module Outliers = Train.Outlier_detector
open Nn_blocks.DSL_modules

let n_batches = 4
let batch_size = 8
let hidden = 6
let epochs = 3
let total_steps = n_batches * epochs

(* Mid-epoch: epoch 1, position 2. *)
let stop_step = 6
let rng_seed = 7
let x_val i d = (Float.of_int (((i * 13) + (d * 7)) % 17) /. 8.5) -. 1.0
let y_val i = (Float.of_int (i * 5 % 9) /. 4.5) -. 1.0

type program = {
  ctx : Context.t;
  routine : Context.routine;
  loss : Tensor.t;
  set_lr : Context.t -> step:int -> Context.t;
  batch : int ref;
  params : Tn.t list;  (** In id order. *)
  state : Tn.t_set;
}

let dataset () =
  let xs =
    NTDSL.init ~l:"xs" ~prec:Ir.Ops.single ~b:[ n_batches; batch_size ] ~o:[ 2 ]
      ~f:(function [| b; i; d |] -> x_val ((b * batch_size) + i) d | _ -> assert false)
      ()
  in
  let ys =
    NTDSL.init ~l:"ys" ~prec:Ir.Ops.single ~b:[ n_batches; batch_size ] ~o:[ 1 ]
      ~f:(function [| b; i; _ |] -> y_val ((b * batch_size) + i) | _ -> assert false)
      ()
  in
  (xs, ys)

let make_mlp () =
  let%op mlp x = { w2 } * relu ({ b1; o = [ hidden ] } + ({ w1 } * x)) in
  mlp

(* The regression loss of [mlp] on the batch [batch_n] selects. *)
let batch_loss ~mlp ~xs ~ys batch_n =
  let%op input = xs @| batch_n in
  let%op target = ys @| batch_n in
  let%op diff = mlp input - target in
  let%op loss = ((diff *. diff) ++ "...|... => |->0") /. !..batch_size in
  loss

let cosine_learning_rate ~total_steps =
  Train.scheduled_learning_rate
    {
      Train.Lr_schedule.kind = Cosine;
      base_lr = 0.1;
      warmup_steps = 2;
      total_steps;
      final_frac = 0.1;
    }

(* Built identically by every run: the same construction order gives the same node ids, which is how
   a checkpoint matches its nodes. *)
let build_graph () =
  let xs, ys = dataset () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let loss = batch_loss ~mlp:(make_mlp ()) ~xs ~ys batch_n in
  let learning_rate, set_lr = cosine_learning_rate ~total_steps in
  let update = Train.grad_update loss in
  let sgd = Train.sgd_update ~learning_rate ~momentum:0.9 ~weight_decay:1e-3 loss in
  let ctx = Train.init_params (Context.cpu ()) bindings loss in
  let ctx, routine = Train.to_routine ctx bindings (Ir.Assignments.sequence [ update; sgd ]) in
  let params =
    Set.to_list loss.Tensor.params
    |> List.map ~f:(fun p -> p.Tensor.value)
    |> List.sort ~compare:(fun a b -> Int.compare a.Tn.id b.Tn.id)
  in
  {
    ctx;
    routine;
    loss;
    set_lr;
    batch = IDX.find_exn routine.Context.bindings batch_n;
    params;
    state = Checkpoint.training_state ~routines:[ routine ] loss;
  }

(* The seed is set up front, under every seed alike, so its node takes the same id in every build. A
   [scoped] seed builds the graph under [Tensor.with_saved_random_seed], as [Parallel.data_parallel]
   builds its shards: the graph keeps drawing from it after the global singleton is restored. *)
let build ?seed ?scoped () =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ?seed ();
  match scoped with
  | None -> build_graph ()
  | Some seed ->
      Tensor.with_saved_random_seed (fun () ->
          Tensor.set_random_seed ~seed ();
          build_graph ())

let shuffle rng =
  let a = Array.init n_batches ~f:Fn.id in
  for i = n_batches - 1 downto 1 do
    let j = Stdlib.Random.State.int rng (i + 1) in
    Array.swap a i j
  done;
  a

let order_key = "epoch_order"
let show_order order = String.concat_array ~sep:" " (Array.map order ~f:Int.to_string)

let order_of_progress (progress : Checkpoint.progress) =
  match List.Assoc.find progress.extra order_key ~equal:String.equal with
  | None -> [||]
  | Some s -> String.split s ~on:' ' |> List.map ~f:Int.of_string |> Array.of_list

(* A function: the generator is mutable, and each run starts its own. *)
let start () =
  {
    Checkpoint.step = 0;
    epoch = 0;
    position = 0;
    host_rng = Some (Stdlib.Random.State.make [| rng_seed |]);
    extra = [];
  }

(* Runs the loop from [progress] until [until] steps are done. Returns the context, the progress to
   checkpoint, and the per-step losses. The batch order of the epoch in progress travels in [extra]:
   the generator has already moved past the draw that made it. *)
let run_steps prog (progress : Checkpoint.progress) ~until =
  let rng = Option.value_exn ~here:[%here] progress.host_rng in
  let order = ref (order_of_progress progress) in
  let step = ref progress.step and epoch = ref progress.epoch in
  let position = ref progress.position in
  let ctx = ref prog.ctx and losses = ref [] in
  while !step < until do
    if !position = 0 then order := shuffle rng;
    prog.batch := !order.(!position);
    ctx := prog.set_lr !ctx ~step:!step;
    ctx := Context.run !ctx prog.routine;
    losses := (Context.get_values !ctx prog.loss.Tensor.value).(0) :: !losses;
    Int.incr step;
    Int.incr position;
    if !position = n_batches then (
      position := 0;
      Int.incr epoch)
  done;
  let progress =
    {
      Checkpoint.step = !step;
      epoch = !epoch;
      position = !position;
      host_rng = Some rng;
      extra = [ (order_key, show_order !order) ];
    }
  in
  (!ctx, progress, List.rev !losses)

let bits v = Printf.sprintf "%Lx" (Int64.bits_of_float v)

(* One line per parameter (its values' bit patterns), then one per loss. Single-precision values
   convert to doubles exactly, so equal lines are equal device bits. *)
let fingerprint ctx prog losses =
  List.map prog.params ~f:(fun tn ->
      "param " ^ Int.to_string tn.Tn.id ^ " "
      ^ String.concat_array ~sep:"," (Array.map (Context.get_values ctx tn) ~f:bits))
  @ List.map losses ~f:(fun l -> "loss " ^ bits l)

let params_only fp = List.filter fp ~f:(String.is_prefix ~prefix:"param ")

(* The resumed half, as a fresh process runs it. *)
let resume ?(restore_state = fun prog path -> Checkpoint.restore ~ctx:prog.ctx prog.state path)
    ?(host_rng = Fn.id) path =
  let prog = build () in
  let ctx, progress = restore_state prog path in
  let progress = { progress with host_rng = host_rng progress.Checkpoint.host_rng } in
  let ctx, _, losses = run_steps { prog with ctx } progress ~until:total_steps in
  (progress, fingerprint ctx prog losses)

let child path =
  let progress, fp = resume path in
  printf "progress step=%d epoch=%d position=%d %s=%s\n" progress.step progress.epoch
    progress.position order_key
    (show_order (order_of_progress progress));
  List.iter fp ~f:print_endline

let raises_failure ?(also = "") ~substring f =
  match f () with
  | _ -> false
  | exception Failure msg ->
      eprintf "(refusal, not part of the golden) %s\n" msg;
      String.is_substring msg ~substring && String.is_substring msg ~substring:also

let parent () =
  let open Verdict.Claims in
  let path = Stdlib.Filename.temp_file "ocannl-resume-" ".ckpt" in
  let plain_path = Stdlib.Filename.temp_file "ocannl-resume-plain-" ".ckpt" in
  let seeded_path = Stdlib.Filename.temp_file "ocannl-resume-seeded-" ".ckpt" in
  let malformed_path = Stdlib.Filename.temp_file "ocannl-resume-malformed-" ".ckpt" in
  Exn.protect
    ~finally:(fun () ->
      (* Best effort: a restore may still hold a payload mapped (the default
         [checkpoint_load_mmap]), and Windows refuses to delete a mapped file. The files live in the
         temp directory. *)
      List.iter [ path; plain_path; seeded_path; malformed_path ] ~f:(fun p ->
          try Stdlib.Sys.remove p with Sys_error _ -> ()))
    ~f:(fun () ->
      (* The uninterrupted run. *)
      let prog = build () in
      let ctx, _, losses = run_steps prog (start ()) ~until:total_steps in
      let uninterrupted = fingerprint ctx prog losses in
      List.iteri losses ~f:(fun i l -> eprintf "(not part of the golden) step %d loss %.9g\n" i l);
      let n_params = List.length prog.params in
      let extra_state = Set.diff prog.state (Set.of_list (module Tn) prog.params) |> Set.to_list in
      List.iter extra_state ~f:(fun tn ->
          eprintf "(not part of the golden) carried beyond the parameters: %s\n" (Tn.debug_name tn));
      (* Each parameter is trained with a momentum buffer the update carries between runs. *)
      p "the training state carries a momentum buffer for every parameter beyond the parameters"
        (List.length extra_state >= n_params);
      (* The interrupted run: stopped mid-epoch and checkpointed. *)
      let prog = build () in
      let ctx, progress, first_losses = run_steps prog (start ()) ~until:stop_step in
      List.iteri first_losses ~f:(fun i l ->
          eprintf "(not part of the golden) interrupted step %d loss %.9g\n" i l);
      let rng_at_save = Stdlib.Random.State.copy (Option.value_exn progress.host_rng) in
      Checkpoint.save ~ctx ~progress prog.state path;
      let saved_seeds = Tensor.random_seeds () in
      p "the interrupted run stops mid-epoch" (progress.epoch = 1 && progress.position = 2);
      p "the interrupted run's losses are the uninterrupted run's first ones"
        (List.equal String.equal
           (List.map first_losses ~f:(fun l -> "loss " ^ bits l))
           (List.take (List.drop uninterrupted n_params) stop_step));
      (* The resume, in a fresh process. *)
      let ((status, out, _) as result) = Fresh_process.run [ "resume"; path ] in
      let child_ok = Poly.equal status (Unix.WEXITED 0) in
      if not child_ok then Fresh_process.report ~label:"resume child" result;
      let lines = String.split_lines out in

      let expected_progress =
        Printf.sprintf "progress step=%d epoch=%d position=%d %s=%s" stop_step 1 2 order_key
          (show_order (order_of_progress progress))
      in
      p "the fresh process restores the step, epoch, data position and batch order"
        (child_ok && List.mem lines expected_progress ~equal:String.equal);
      let resumed = List.filter lines ~f:(Fn.non (String.is_prefix ~prefix:"progress ")) in
      let expected_resumed =
        params_only uninterrupted @ List.drop uninterrupted (n_params + stop_step)
      in
      p "the resumed run's final parameters equal the uninterrupted run's, bit for bit"
        (child_ok && List.equal String.equal (params_only resumed) (params_only uninterrupted));
      let same_run = List.equal String.equal resumed expected_resumed in
      p "the resumed run's per-step losses equal the uninterrupted run's, bit for bit"
        (child_ok && same_run);
      if child_ok && not same_run then (
        List.iter lines ~f:(eprintf "resumed (fresh process): %s\n");
        List.iter uninterrupted ~f:(eprintf "uninterrupted: %s\n"));
      (* The restored host generator continues the saved one's stream. *)
      let _, restored =
        let prog = build () in
        Checkpoint.restore ~ctx:prog.ctx prog.state path
      in
      let draws rng = List.init 8 ~f:(fun _ -> Stdlib.Random.State.bits rng) in
      p "the restored host generator draws what the saved one would have"
        (List.equal Int.equal (draws (Option.value_exn restored.host_rng)) (draws rng_at_save));
      (* Controls: each piece of the state matters to the bit-for-bit claim. *)
      let params_set prog = Set.of_list (module Tn) prog.params in
      let _, without_momentum =
        resume path ~restore_state:(fun prog path ->
            (Persistence.restore ~ctx:prog.ctx (params_set prog) path, progress))
      in
      p "a resume restoring the parameters but not the momentum ends elsewhere"
        (not (List.equal String.equal (params_only without_momentum) (params_only uninterrupted)));
      let _, fresh_rng =
        resume path ~host_rng:(fun _ -> Some (Stdlib.Random.State.make [| rng_seed |]))
      in
      p "a resume restarting the host generator ends elsewhere"
        (not (List.equal String.equal (params_only fresh_rng) (params_only uninterrupted)));
      (* Refusals. *)
      let prog = build () in
      p "restoring a state set without the momentum is refused, naming the difference"
        (raises_failure ~substring:"differs from the one saved" (fun () ->
             Checkpoint.restore ~ctx:prog.ctx (params_set prog) path));
      Persistence.save ~ctx:prog.ctx ~appending:false (params_set prog) plain_path;
      p "a plain Persistence checkpoint is refused as recording no progress"
        (raises_failure ~substring:"is not a training checkpoint" (fun () ->
             Checkpoint.restore ~ctx:prog.ctx prog.state plain_path));
      (* The random seed. *)
      let seeds_key = "ocannl.random_seeds" in
      let metadata = Persistence.metadata path in
      let show_int = Int.to_string in
      p "the checkpoint records the one random seed it was saved under"
        (List.length saved_seeds = 1
        && Option.equal String.equal
             (List.Assoc.find metadata seeds_key ~equal:String.equal)
             (Some (String.concat ~sep:" " (List.map saved_seeds ~f:show_int))));
      let saved_seed = List.hd_exn saved_seeds in
      let other_seed = saved_seed + 1 in
      let prog = build ~seed:other_seed () in
      p "a resume under a different random seed is refused, naming both seeds"
        (raises_failure
           ~substring:("saved under random seed " ^ show_int saved_seed)
           ~also:("built under random seed " ^ show_int other_seed)
           (fun () -> Checkpoint.restore ~ctx:prog.ctx prog.state path));
      (* A graph built under a scoped seed: the global singleton is the default one throughout. *)
      let scoped_seed = saved_seed + 2 in
      let prog = build ~scoped:scoped_seed () in
      let ctx, scoped_at_save, _ = run_steps prog (start ()) ~until:stop_step in
      Checkpoint.save ~ctx ~progress:scoped_at_save prog.state seeded_path;
      let prog = build ~scoped:scoped_seed () in
      let _, scoped_progress = Checkpoint.restore ~ctx:prog.ctx prog.state seeded_path in
      p "a checkpoint of a graph built under a scoped seed restores under that scoped seed"
        (scoped_progress.step = stop_step);
      let prog = build ~scoped:other_seed () in
      p "a resume whose graph was built under another scoped seed is refused, naming both"
        (raises_failure
           ~substring:
             ("saved under random seeds " ^ show_int saved_seed ^ ", " ^ show_int scoped_seed)
           ~also:("built under random seeds " ^ show_int saved_seed ^ ", " ^ show_int other_seed)
           (fun () -> Checkpoint.restore ~ctx:prog.ctx prog.state seeded_path));
      (* Rewritten from the checkpoint's own header and values: the state, without the seed. *)
      let prog = build () in
      let ctx, _ = Checkpoint.restore ~ctx:prog.ctx prog.state path in
      let without_seed ~version =
        List.filter_map metadata ~f:(fun (k, v) ->
            if String.equal k seeds_key then None
            else if String.equal k "ocannl.training_checkpoint" then Some (k, version)
            else Some (k, v))
      in
      Persistence.save ~ctx ~appending:false ~metadata:(without_seed ~version:"1") prog.state
        seeded_path;
      Persistence.save ~ctx ~appending:false ~metadata:(without_seed ~version:"2") prog.state
        malformed_path;
      let prog = build ~seed:other_seed () in
      let (_, legacy_progress), warning =
        Stderr_capture.capture (fun () -> Checkpoint.restore ~ctx:prog.ctx prog.state seeded_path)
      in
      p "a format-1 checkpoint, recording no seed, restores under another seed with a warning"
        (legacy_progress.step = stop_step
        && String.is_substring warning ~substring:"records no random seed"
        && String.is_substring warning
             ~substring:("this program's (random seed " ^ show_int other_seed ^ ")"));
      p "a format-2 checkpoint missing its seed is refused as malformed"
        (raises_failure ~substring:("missing or malformed " ^ seeds_key) (fun () ->
             Checkpoint.restore ~ctx:prog.ctx prog.state malformed_path));
      (* The header metadata an appending save merges: the named key replaced, the others kept. *)
      Persistence.save ~ctx:prog.ctx ~appending:true
        ~metadata:[ ("ocannl.step", "99") ]
        (params_set prog) path;
      let metadata = Persistence.metadata path in
      let find k = List.Assoc.find metadata k ~equal:String.equal in
      p "an appending save replaces the metadata key it names and keeps the others"
        (Option.equal String.equal (find "ocannl.step") (Some "99")
        && Option.equal String.equal (find "ocannl.epoch") (Some "1")
        && Option.equal String.equal
             (find ("extra." ^ order_key))
             (Some (show_order (order_of_progress progress)))))

(* ----- The mixed-precision leg (gh-ocannl-1726) ----- *)

let mp_total_steps = 16
let mp_stop_step = 8

(* Regrown every two good steps, the scale climbs until the f16 gradients overflow (here at 2^19):
   the step is skipped and the scale backs off, again and again through the run, so a scaler
   re-created at the resume overflows on other steps. *)
let mp_init_scale = 65536.
let mp_growth_interval = 2
let mp_window = 3

(* Low enough that the loss's batch-to-batch spread trips it now and then. *)
let mp_outlier_threshold = 1.0

type mp_program = {
  mp_ctx : Context.t;
  grad_routine : Context.routine;
  sgd_routine : Context.routine;
  checksum : Tensor.t;
  scaler : MP.Loss_scaler.t;
  mp_loss : Tensor.t;
  mp_set_lr : Context.t -> step:int -> Context.t;
  mp_batch : int ref;
  mp_params : Tn.t list;  (** In id order. *)
  mp_state : Tn.t_set;
}

let build_mp () =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ();
  let xs, ys = dataset () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let mlp = MP.with_master_weights ~prec:Ir.Ops.half make_mlp in
  let loss = batch_loss ~mlp ~xs ~ys batch_n in
  (* f16 activations and gradients, where loss scaling matters. The masters' initializers are in the
     masters' class, so with [param_prec = None] their draws stay f32 (gh-ocannl-1865). *)
  Precision_policy.apply
    {
      Precision_policy.param_prec = None;
      activation_prec = Some Ir.Ops.half;
      grad_prec = Some Ir.Ops.half;
    }
    loss;
  let scaler =
    MP.Loss_scaler.create ~init_scale:mp_init_scale ~growth_interval:mp_growth_interval ()
  in
  let learning_rate, set_lr = cosine_learning_rate ~total_steps:mp_total_steps in
  let checksum, grad_comp = MP.scaled_grad_update scaler loss in
  let sgd_comp = MP.scaled_sgd_update scaler ~learning_rate ~momentum:0.9 loss in
  let ctx = Train.init_params (Context.cpu ()) bindings loss in
  let ctx, grad_routine = Train.to_routine ctx bindings grad_comp in
  let ctx, sgd_routine = Train.to_routine ctx IDX.empty sgd_comp in
  {
    mp_ctx = ctx;
    grad_routine;
    sgd_routine;
    checksum;
    scaler;
    mp_loss = loss;
    mp_set_lr = set_lr;
    mp_batch = IDX.find_exn grad_routine.Context.bindings batch_n;
    mp_params =
      Set.to_list loss.Tensor.params
      |> List.map ~f:(fun p -> p.Tensor.value)
      |> List.sort ~compare:(fun a b -> Int.compare a.Tn.id b.Tn.id);
    mp_state = Checkpoint.training_state ~routines:[ grad_routine; sgd_routine ] loss;
  }

let scaler_key = "loss_scaler"
let detector_key = "loss_outliers"

type mp_step = { loss : float; z : float; finite : bool; outlier : bool; scale : float }

let show_step i s =
  Printf.sprintf "step %d loss %s z %s finite %b outlier %b scale %h" i (bits s.loss) (bits s.z)
    s.finite s.outlier s.scale

(* The batch is the step's, so the data needs no position of its own here: the first leg covers
   that. A step runs the optimizer only when the gradients are finite and the loss is no outlier. *)
let mp_run_steps prog detector (progress : Checkpoint.progress) ~until =
  let ctx = ref prog.mp_ctx and steps = ref [] in
  for step = progress.step to until - 1 do
    prog.mp_batch := step % n_batches;
    ctx := prog.mp_set_lr !ctx ~step;
    ctx := Context.run !ctx prog.grad_routine;
    let finite = Float.is_finite (Context.get_values !ctx prog.checksum.Tensor.value).(0) in
    let loss = (Context.get_values !ctx prog.mp_loss.Tensor.value).(0) in
    let z = Outliers.update detector loss in
    let outlier = Float.(z > mp_outlier_threshold) in
    if finite && not outlier then ctx := Context.run !ctx prog.sgd_routine;
    ctx := MP.Loss_scaler.update prog.scaler !ctx ~grads_finite:finite;
    steps :=
      (step, { loss; z; finite; outlier; scale = MP.Loss_scaler.scale_value prog.scaler }) :: !steps
  done;
  let progress =
    {
      Checkpoint.step = until;
      epoch = 0;
      position = 0;
      host_rng = None;
      extra =
        [
          (scaler_key, MP.Loss_scaler.save_state prog.scaler);
          (detector_key, Outliers.save_state detector);
        ];
    }
  in
  (!ctx, progress, List.rev !steps)

let mp_fingerprint ctx prog steps =
  List.map prog.mp_params ~f:(fun tn ->
      "param " ^ Int.to_string tn.Tn.id ^ " "
      ^ String.concat_array ~sep:"," (Array.map (Context.get_values ctx tn) ~f:bits))
  @ List.map steps ~f:(fun (i, s) -> show_step i s)

let extra_exn (progress : Checkpoint.progress) key =
  Option.value_exn ~here:[%here] (List.Assoc.find progress.extra key ~equal:String.equal)

(* The resumed half: rebuild, restore the device state, then the scaler's and the detector's. *)
let mp_resume ?(restore_scaler = true) ?(restore_detector = true) path =
  let prog = build_mp () in
  let ctx, progress = Checkpoint.restore ~ctx:prog.mp_ctx prog.mp_state path in
  let ctx =
    if restore_scaler then
      MP.Loss_scaler.restore_state prog.scaler ctx (extra_exn progress scaler_key)
    else ctx
  in
  let detector = Outliers.create ~window_size:mp_window () in
  if restore_detector then Outliers.restore_state detector (extra_exn progress detector_key);
  let ctx, _, steps =
    mp_run_steps { prog with mp_ctx = ctx } detector progress ~until:mp_total_steps
  in
  mp_fingerprint ctx prog steps

let mp_child path = List.iter (mp_resume path) ~f:print_endline

let mp_parent () =
  let open Verdict.Claims in
  let path = Stdlib.Filename.temp_file "ocannl-resume-mp-" ".ckpt" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove path with Sys_error _ -> ())
    ~f:(fun () ->
      let prog = build_mp () in
      let detector = Outliers.create ~window_size:mp_window () in
      let ctx, _, steps = mp_run_steps prog detector (start ()) ~until:mp_total_steps in
      let uninterrupted = mp_fingerprint ctx prog steps in
      List.iter steps ~f:(fun (i, s) ->
          eprintf "(not part of the golden) mixed-precision %s (z %g, loss %g)\n" (show_step i s)
            s.z s.loss);
      let n_params = List.length prog.mp_params in
      let resumed_part = List.filter steps ~f:(fun (i, _) -> i >= mp_stop_step) in
      (* Non-vacuity: the state the resume must carry decides something after it. *)
      p_exists "the mixed-precision run backs the scale off after the stop step" resumed_part
        ~f:(fun (_, s) -> not s.finite);
      p_exists "the mixed-precision run skips an outlier step after the stop step" resumed_part
        ~f:(fun (_, s) -> s.outlier);
      p_exists "the mixed-precision run takes an optimizer step after the stop step" resumed_part
        ~f:(fun (_, s) -> s.finite && not s.outlier);
      p_all "the mixed-precision run's losses are finite" steps ~f:(fun (_, s) ->
          Float.is_finite s.loss);
      (* What makes the finiteness claim bite: the loss is an f16 activation (gh-ocannl-1769). *)
      p "the mixed-precision loss is an f16 activation"
        (Ir.Ops.equal_prec (Lazy.force prog.mp_loss.Tensor.value.Tn.storage_prec) Ir.Ops.half);
      p_all "the mixed-precision masters stay single" prog.mp_params ~f:(fun tn ->
          Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single);
      (* The interrupted run. *)
      let prog = build_mp () in
      let detector = Outliers.create ~window_size:mp_window () in
      let ctx, progress, _ = mp_run_steps prog detector (start ()) ~until:mp_stop_step in
      Checkpoint.save ~ctx ~progress prog.mp_state path;
      let ((status, out, _) as result) = Fresh_process.run [ "resume-mp"; path ] in
      let child_ok = Poly.equal status (Unix.WEXITED 0) in
      if not child_ok then Fresh_process.report ~label:"mixed-precision resume child" result;
      let resumed = String.split_lines out in
      let expected =
        List.take uninterrupted n_params @ List.drop uninterrupted (n_params + mp_stop_step)
      in
      let same_run = child_ok && List.equal String.equal resumed expected in
      p
        "the resumed mixed-precision run's losses, z-scores, skips and scales equal the \
         uninterrupted run's, bit for bit"
        same_run;
      p
        "the resumed mixed-precision run's final master weights equal the uninterrupted run's, bit \
         for bit"
        (child_ok && List.equal String.equal (params_only resumed) (params_only uninterrupted));
      if child_ok && not same_run then (
        List.iter resumed ~f:(eprintf "resumed (fresh process): %s\n");
        List.iter expected ~f:(eprintf "uninterrupted: %s\n"));
      (* Controls: each helper's host state matters to the bit-for-bit claim. *)
      p "a resume re-creating the loss scaler ends on other master weights"
        (not
           (List.equal String.equal
              (params_only (mp_resume ~restore_scaler:false path))
              (params_only uninterrupted)));
      p "a resume re-creating the outlier detector ends on other master weights"
        (not
           (List.equal String.equal
              (params_only (mp_resume ~restore_detector:false path))
              (params_only uninterrupted)));
      (* Refusals: the schedule belongs to the program. *)
      let _, saved = Checkpoint.restore ~ctx:prog.mp_ctx prog.mp_state path in
      let prog = build_mp () in
      let other_scaler =
        MP.Loss_scaler.create ~init_scale:mp_init_scale ~growth_interval:(mp_growth_interval + 1) ()
      in
      p "restoring the scaler's state into one with another growth interval is refused, naming both"
        (raises_failure
           ~substring:("saved under growth_interval " ^ Int.to_string mp_growth_interval)
           ~also:("created with " ^ Int.to_string (mp_growth_interval + 1))
           (fun () ->
             MP.Loss_scaler.restore_state other_scaler prog.mp_ctx (extra_exn saved scaler_key)));
      let other_detector = Outliers.create ~window_size:(mp_window + 1) () in
      p "restoring the detector's state into one over another window is refused, naming both"
        (raises_failure
           ~substring:("window of " ^ Int.to_string mp_window ^ " samples")
           ~also:("window holds " ^ Int.to_string (mp_window + 1))
           (fun () -> Outliers.restore_state other_detector (extra_exn saved detector_key))))

let () =
  match Stdlib.Sys.argv with
  | [| _; "resume"; path |] -> child path
  | [| _; "resume-mp"; path |] -> mp_child path
  | _ ->
      parent ();
      mp_parent ()
