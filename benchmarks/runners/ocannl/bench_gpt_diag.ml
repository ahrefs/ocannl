(* GPT segment diagnostic. Fixture mode selects forward or the real training step (backprop +
   optimizer); model construction, parameter injection and step compilation are shared with
   bench_gpt. *)
open Base
open Ocannl
module IDX = Train.IDX
module H = Bench_harness

let () =
  let st = Safetensors.read (Stdlib.Sys.getenv "BENCH_FIXTURE") in
  let { Bench_gpt_model.ctx; batch_loss; step_shape; bindings; batch_n; n_batches; _ } =
    Bench_gpt_model.prepare ~materialize:(H.env_flag "BENCH_MATERIALIZE") ~debug:false st
  in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let promote_locals =
    match Stdlib.Sys.getenv_opt "BENCH_PROMOTE" with Some "0" -> Some false | _ -> None
  in
  let forward_opt =
    match step_shape with
    | `Forward fwd ->
        let opt = H.capture_lowering ctx fwd bindings in
        H.print_census ?promote_locals ~backend ~limits ~static_indices:[ batch_n ] opt;
        Some (fwd, opt)
    | `Train _ ->
        if Option.equal Bool.equal promote_locals (Some false) then
          failwith "BENCH_PROMOTE=0 is forward-only; training diagnostics use the shipped pipeline";
        None
  in
  let t0 = Unix.gettimeofday () in
  let ctx, routines =
    match step_shape with
    | `Train parts ->
        H.compile_train_step ~tune:false
          ~tuned:(fun _ _ -> failwith "bench_gpt_diag does not autotune")
          ctx bindings parts
    | `Forward fwd ->
        let ctx, routine = Context.compile ctx fwd bindings in
        (ctx, H.Plain routine)
  in
  Stdio.printf "mode: %s backend: %s compile_s: %.3f\n%!"
    (if H.is_training st then "train" else "infer")
    backend
    (Unix.gettimeofday () -. t0);
  let ctx_ref = ref ctx in
  let batch_ref = IDX.find_exn (H.train_step_bindings routines) batch_n in
  let run step =
    batch_ref := step % n_batches;
    H.run_train_step routines ctx_ref ~step;
    Context.sync !ctx_ref
  in
  (* Full-step controls precede isolated timing, which mutates gradients and parameters. *)
  if H.env_flag "BENCH_STEPS" then
    for step = 0 to 2 do
      let t0 = Unix.gettimeofday () in
      run step;
      let open Operation.At in
      Stdio.printf "step %d: %.1f ms loss: %.7f\n%!" step
        ((Unix.gettimeofday () -. t0) *. 1000.)
        (!ctx_ref, batch_loss).@[0]
    done
  else if H.env_flag "BENCH_SEG_TIMES" then run 0;
  let shipped = H.step_routines routines in
  if H.is_training st then H.print_shipped_census shipped;
  if H.env_flag "BENCH_SEG_TIMES" then
    match forward_opt with
    | Some (fwd, opt) ->
        H.time_segments ?promote_locals ~backend ~limits ~static_indices:[ batch_n ] ~ctx:!ctx_ref
          ~comp:fwd ~bindings
          ~bind:(fun r -> IDX.find_exn r.Context.bindings batch_n := !batch_ref)
          opt
    | None ->
        ignore
          (H.time_shipped_segments ~ctx:!ctx_ref ~bindings shipped
            : (int * (float, string) Result.t) list)
