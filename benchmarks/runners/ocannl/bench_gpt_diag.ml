(* GPT segment diagnostic. Fixture mode selects forward or the real training step (backprop +
   optimizer); model construction, parameter injection and step compilation are shared with
   bench_gpt. *)
open Base
open Ocannl
module IDX = Train.IDX
module H = Bench_harness

(* Opt-in replay data: float32 preserves every observed half/single input exactly. The caller
   creates the destination directory and selects node IDs; ordinary diagnostics write no dumps. *)
let dump_replay_values phase tn values =
  match (Stdlib.Sys.getenv_opt "BENCH_DUMP_DIR", Stdlib.Sys.getenv_opt "BENCH_DUMP_NODES") with
  | Some dir, Some ids when String.equal phase "step0" ->
      let selected = String.split ids ~on:',' |> List.map ~f:Int.of_string in
      if List.mem selected tn.Ir.Tnode.id ~equal:Int.equal then
        let path = Stdlib.Filename.concat dir (Printf.sprintf "node%d.f32" tn.Ir.Tnode.id) in
        let ch = Stdlib.open_out_bin path in
        Exn.protect
          ~f:(fun () ->
            Array.iter values ~f:(fun value ->
                let bits = Stdlib.Int32.bits_of_float value in
                for byte = 0 to 3 do
                  Stdlib.output_byte ch
                    (Stdlib.Int32.to_int
                       (Stdlib.Int32.logand (Stdlib.Int32.shift_right_logical bits (8 * byte)) 255l))
                done))
          ~finally:(fun () -> Stdlib.close_out ch)
  | _ -> ()

(* Investigation-only observer: reads the compiled routine's actual buffers without changing graph
   construction or placement. Buffer aliasing must be disabled for an after-step snapshot. *)
let snapshot ctx phase nodes =
  Set.iter nodes ~f:(fun tn ->
      let values = Context.get_values ctx tn in
      let finite = ref 0 and lo = ref Float.infinity and hi = ref Float.neg_infinity in
      let sum = ref 0. and squares = ref 0. and hash = ref 0L in
      Array.iter values ~f:(fun v ->
          hash :=
            Stdlib.Int64.add (Stdlib.Int64.mul !hash 1099511628211L) (Stdlib.Int64.bits_of_float v);
          if Float.is_finite v then (
            Int.incr finite;
            lo := Float.min !lo v;
            hi := Float.max !hi v;
            sum := !sum +. v;
            squares := !squares +. (v *. v)));
      Stdio.printf
        "snapshot %s %d %s count=%d finite=%d hash=%Lx min=%h max=%h sum=%h squares=%h\n%!" phase
        tn.Ir.Tnode.id (Ir.Tnode.debug_name tn) (Array.length values) !finite !hash !lo !hi !sum
        !squares;
      dump_replay_values phase tn values)

let () =
  let st = Safetensors.read (Stdlib.Sys.getenv "BENCH_FIXTURE") in
  let { Bench_gpt_model.ctx; batch_loss; step_shape; bindings; batch_n; n_batches; mapping; _ } =
    Bench_gpt_model.prepare ~materialize:(H.env_flag "BENCH_MATERIALIZE") st
  in
  let ctx = H.inject ctx st batch_loss mapping in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  (* The reconstruction exists only for the explicit no-promotion inference experiment. *)
  let forward_opt =
    match (Stdlib.Sys.getenv_opt "BENCH_PROMOTE", step_shape) with
    | Some "0", `Forward fwd ->
        let opt = H.capture_lowering ctx fwd bindings in
        H.print_census ~promote_locals:false ~backend ~limits ~static_indices:[ batch_n ] opt;
        Some (fwd, opt)
    | Some "0", `Train _ ->
        failwith "BENCH_PROMOTE=0 is forward-only; training diagnostics use the shipped pipeline"
    | _ -> None
  in
  let t0 = Unix.gettimeofday () in
  let ctx, routines =
    H.compile_step ~tune:false
      ~tuned:(fun _ _ -> failwith "bench_gpt_diag does not autotune")
      ctx bindings step_shape
  in
  Stdio.printf "mode: %s backend: %s compile_s: %.3f\n%!"
    (if H.is_training st then "train" else "infer")
    backend
    (Unix.gettimeofday () -. t0);
  let ctx_ref = ref ctx in
  let snapshots = H.env_flag "BENCH_SNAPSHOT" in
  if snapshots && Utils.get_global_flag ~default:false ~arg_name:"buffer_aliasing" then
    failwith "BENCH_SNAPSHOT requires buffer_aliasing=false";
  let shipped = H.compiled_step_routines routines in
  let inputs =
    List.fold shipped
      ~init:(Set.empty (module Ir.Tnode))
      ~f:(fun nodes r -> Set.union nodes r.Context.inputs)
  in
  let outputs =
    List.fold shipped
      ~init:(Set.empty (module Ir.Tnode))
      ~f:(fun nodes r -> Set.union nodes r.Context.outputs)
  in
  let params =
    Set.fold batch_loss.Tensor.params
      ~init:(Set.empty (module Ir.Tnode))
      ~f:(fun nodes p -> Set.add nodes p.Tensor.value)
  in
  if snapshots then snapshot !ctx_ref "inputs" (Set.diff inputs outputs);
  if snapshots then snapshot !ctx_ref "parameters-before" params;
  let batch_ref = IDX.find_exn (H.train_step_bindings routines) batch_n in
  let run step =
    batch_ref := step % n_batches;
    H.run_train_step routines ctx_ref ~step;
    Context.sync !ctx_ref
  in
  (* Full-step controls precede isolated timing, which mutates gradients and parameters. *)
  if
    H.env_flag "BENCH_STEPS"
    || Option.equal String.equal (Stdlib.Sys.getenv_opt "BENCH_STEPS") (Some "parity")
  then
    let steps =
      match Stdlib.Sys.getenv_opt "BENCH_STEPS" with
      | Some "parity" -> (H.protocol_of_st st).H.parity_steps
      | _ -> 3
    in
    for step = 0 to steps - 1 do
      let t0 = Unix.gettimeofday () in
      run step;
      let open Operation.At in
      Stdio.printf "step %d: %.1f ms loss: %.7f\n%!" step
        ((Unix.gettimeofday () -. t0) *. 1000.)
        (!ctx_ref, batch_loss).@[0];
      if snapshots then (
        (match routines with
        | H.Host_gate (scaler, checksum, _, _) ->
            Stdio.printf "gate step=%d optimizer_runs=%d scale=%h checksum=%h\n%!" step
              !H.host_gated_optimizer_runs
              (Mixed_prec.Loss_scaler.scale_value scaler)
              (!ctx_ref, checksum).@[0]
        | _ -> ());
        snapshot !ctx_ref ("parameters-step" ^ Int.to_string step) params;
        if step = 0 then snapshot !ctx_ref "step0" outputs)
    done
  else if H.env_flag "BENCH_SEG_TIMES" then run 0;
  if Option.is_none forward_opt then H.print_shipped_census shipped;
  if H.env_flag "BENCH_SEG_TIMES" then
    match forward_opt with
    | Some (fwd, opt) ->
        H.time_segments ~promote_locals:false ~backend ~limits ~static_indices:[ batch_n ]
          ~ctx:!ctx_ref ~comp:fwd ~bindings
          ~bind:(fun r -> IDX.find_exn r.Context.bindings batch_n := !batch_ref)
          opt
    | None ->
        ignore
          (H.time_shipped_segments ~ctx:!ctx_ref ~bindings shipped : (float, string) Result.t list)
