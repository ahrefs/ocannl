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
   no progress. *)

open Base
open Ocannl
open Stdio
module IDX = Train.IDX
module Tn = Ir.Tnode
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

(* Built identically by every run: the same construction order gives the same node ids, which is how
   a checkpoint matches its nodes. *)
let build () =
  Tensor.unsafe_reinitialize ();
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
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op mlp x = { w2 } * relu ({ b1; o = [ hidden ] } + ({ w1 } * x)) in
  let%op input = xs @| batch_n in
  let%op target = ys @| batch_n in
  let%op diff = mlp input - target in
  let%op loss = ((diff *. diff) ++ "...|... => |->0") /. !..batch_size in
  let learning_rate, set_lr =
    Train.scheduled_learning_rate
      {
        Train.Lr_schedule.kind = Cosine;
        base_lr = 0.1;
        warmup_steps = 2;
        total_steps;
        final_frac = 0.1;
      }
  in
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

let raises_failure ~substring f =
  match f () with
  | _ -> false
  | exception Failure msg ->
      eprintf "(refusal, not part of the golden) %s\n" msg;
      String.is_substring msg ~substring

let parent () =
  let open Verdict.Claims in
  let path = Stdlib.Filename.temp_file "ocannl-resume-" ".ckpt" in
  let plain_path = Stdlib.Filename.temp_file "ocannl-resume-plain-" ".ckpt" in
  Exn.protect
    ~finally:(fun () ->
      (* Best effort: a restore may still hold a payload mapped (the default
         [checkpoint_load_mmap]), and Windows refuses to delete a mapped file. The files live in the
         temp directory. *)
      List.iter [ path; plain_path ] ~f:(fun p -> try Stdlib.Sys.remove p with Sys_error _ -> ()))
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

let () = match Stdlib.Sys.argv with [| _; "resume"; path |] -> child path | _ -> parent ()
