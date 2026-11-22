(* gh-ocannl-1854: CUDA automatic page-locking of recurring host arrays ([cuMemHostRegister_v2] /
   [cuMemHostUnregister]).

   Exercises the full [Context.from_host] -> [Context.run] -> [Context.get_values] path across every
   backend: - Executed parity for small (4 KiB) and threshold-sized (4 MiB) recurring host arrays,
   including immediately overwriting the host array with a poison sentinel right after
   [Context.from_host] returns and before [Context.run], pinning the stream-synchronization
   completion contract on page-locked transfers. - On eligible discrete CUDA devices (gated via
   [Verdict.leg]), pins the driver-level counters from [Context.global_debug_info]: below-threshold
   arrays never register, first upload sights without registering, second upload registers once with
   exact [live_host_registered_bytes], third upload reuses the registration, and dropping a
   registered host array triggers GC finaliser unregistration once with zero failures. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX

let small_elems = 1024 (* 4 KiB < 4 MiB default threshold *)
let large_elems = 1_048_576 (* 4 MiB = 4_194_304 B default threshold *)
let large_bytes = large_elems * 4

let debug_counter ctx field =
  match Context.global_debug_info ctx with
  | Sexp.List (_ :: fields) ->
      List.find_map fields ~f:(function
        | Sexp.List [ Sexp.Atom k; Sexp.Atom v ] when String.equal k field -> Int.of_string_opt v
        | _ -> None)
  | _ -> None

let count ctx field = Option.value ~default:0 (debug_counter ctx field)

let make_f32_host_nd n =
  let ba = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| n |] in
  Ir.Ndarray.as_array Ir.Ops.Single ba

let fill_step nd ~step ~n =
  let f (type a b) (arr : (a, b) Ir.Ndarray.bigarray) =
    match Bigarray.Genarray.kind arr with
    | Bigarray.Float32 ->
        let flat = Bigarray.reshape_1 arr n in
        for i = 0 to n - 1 do
          Bigarray.Array1.set flat i (Float.of_int ((((step * 37) + i) % 101) - 50))
        done
    | _ -> assert false
  in
  Ir.Ndarray.apply { f } nd

let fill_poison nd ~n =
  let f (type a b) (arr : (a, b) Ir.Ndarray.bigarray) =
    match Bigarray.Genarray.kind arr with
    | Bigarray.Float32 ->
        let flat = Bigarray.reshape_1 arr n in
        Bigarray.Array1.fill flat (-999.0)
    | _ -> assert false
  in
  Ir.Ndarray.apply { f } nd

let matches_step ~step ~n got =
  Array.length got = n
  &&
  let ok = ref true in
  let i = ref 0 in
  while !ok && !i < n do
    let x = Float.of_int ((((step * 37) + !i) % 101) - 50) in
    let expected = (2. *. x) +. 1. in
    if not (Float.equal got.(!i) expected) then ok := false;
    Int.incr i
  done;
  !ok

let compile_scale_add ctx ~name ~n =
  let x =
    Operation.init ~mutable_:true ~l:(name ^ "_x") ~prec:Ir.Ops.single ~o:[ n ]
      ~f:(fun _ -> 0.)
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let%op y = (x *. 2.) + 1. in
  let ctx, routine = Train.to_routine ctx IDX.empty (Train.forward y) in
  (ctx, routine, x.value, y.value)

let upload_poison_and_check ctx routine x_tn y_tn nd ~step ~n =
  fill_step nd ~step ~n;
  let ctx = Context.from_host ctx x_tn nd in
  (* Immediately overwrite the caller's host buffer before launching the routine: if a page-locked
     upload returned before DMA finished reading [nd], the device would see [-999.0]. *)
  fill_poison nd ~n;
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y_tn in
  (ctx, matches_step ~step ~n got)

let[@inline never] run_droppable_large_upload ctx routine x_tn y_tn =
  let nd = make_f32_host_nd large_elems in
  let ctx, ok1 = upload_poison_and_check ctx routine x_tn y_tn nd ~step:10 ~n:large_elems in
  let ctx, ok2 = upload_poison_and_check ctx routine x_tn y_tn nd ~step:11 ~n:large_elems in
  let reg_during = count ctx "host_registrations" in
  let live_during = count ctx "live_host_registered_bytes" in
  let w = Gc_witness.witness nd in
  (ctx, w, ok1 && ok2, reg_during, live_during)

let () =
  let ctx = Context.auto () in
  let backend_name = Context.backend_name ctx in
  let on_cuda = String.equal backend_name "cuda" in
  let eligible = on_cuda && count ctx "host_register_eligible_devices" > 0 in
  let cuda = Verdict.leg ~when_:eligible ~on:backend_name () in
  Stdio.eprintf
    "cuda_host_registration: backend=%s eligible_devices=%d (not part of the golden)\n%!"
    backend_name
    (count ctx "host_register_eligible_devices");

  (* 1. Small recurring array (4 KiB < 4 MiB default threshold). *)
  let ctx, small_routine, small_x, small_y =
    compile_scale_add ctx ~name:"chr_small" ~n:small_elems
  in
  let reg0 = count ctx "host_registrations" in
  let live0 = count ctx "live_host_registered_bytes" in
  let small_nd = make_f32_host_nd small_elems in
  let ctx, sok1 =
    upload_poison_and_check ctx small_routine small_x small_y small_nd ~step:1 ~n:small_elems
  in
  let ctx, sok2 =
    upload_poison_and_check ctx small_routine small_x small_y small_nd ~step:2 ~n:small_elems
  in
  let ctx, sok3 =
    upload_poison_and_check ctx small_routine small_x small_y small_nd ~step:3 ~n:small_elems
  in
  ignore (Stdlib.Sys.opaque_identity small_nd);
  p "small array (4 KiB): repeated uploads compute exact values after immediate host overwrite"
    (sok1 && sok2 && sok3);
  in_leg ~leg:cuda "small array (4 KiB): repeated uploads below threshold register nothing"
    (fun label ->
      p label
        (count ctx "host_registrations" = reg0 && count ctx "live_host_registered_bytes" = live0));

  (* 2. Large recurring array (4 MiB = default threshold): step-by-step sighting, registration,
     reuse, and kept-witness control. *)
  let ctx, large_routine, large_x, large_y =
    compile_scale_add ctx ~name:"chr_large" ~n:large_elems
  in
  let live_nd = make_f32_host_nd large_elems in
  let live_w = Gc_witness.witness live_nd in
  p "kept control: live 4 MiB host array survives collection rounds" (Gc_witness.kept live_w);

  let reg_b1 = count ctx "host_registrations" in
  let live_b1 = count ctx "live_host_registered_bytes" in
  let ctx, lok1 =
    upload_poison_and_check ctx large_routine large_x large_y live_nd ~step:1 ~n:large_elems
  in
  p "large array (4 MiB): first upload computes exact values after immediate host overwrite" lok1;
  in_leg ~leg:cuda "large array (4 MiB): first upload sights without registering" (fun label ->
      p label
        (count ctx "host_registrations" = reg_b1 && count ctx "live_host_registered_bytes" = live_b1));

  let reg_b2 = count ctx "host_registrations" in
  let dec_b2 = count ctx "host_register_declined" in
  let live_b2 = count ctx "live_host_registered_bytes" in
  let ctx, lok2 =
    upload_poison_and_check ctx large_routine large_x large_y live_nd ~step:2 ~n:large_elems
  in
  p "large array (4 MiB): second upload computes exact values after immediate host overwrite" lok2;
  in_leg ~leg:cuda "large array (4 MiB): second upload registers once and accounts live bytes"
    (fun label ->
      p label
        (count ctx "host_registrations" = reg_b2 + 1
        && count ctx "host_register_declined" = dec_b2
        && count ctx "live_host_registered_bytes" = live_b2 + large_bytes));

  let reg_b3 = count ctx "host_registrations" in
  let live_b3 = count ctx "live_host_registered_bytes" in
  let ctx, lok3 =
    upload_poison_and_check ctx large_routine large_x large_y live_nd ~step:3 ~n:large_elems
  in
  p "large array (4 MiB): third upload computes exact values after immediate host overwrite" lok3;
  in_leg ~leg:cuda "large array (4 MiB): third upload reuses registration without re-registering"
    (fun label ->
      p label
        (count ctx "host_registrations" = reg_b3 && count ctx "live_host_registered_bytes" = live_b3));

  (* 3. Droppable large array: register on second upload, drop, and verify GC finaliser calls
     cuMemHostUnregister cleanly. *)
  let reg_before_drop = count ctx "host_registrations" in
  let unreg_before_drop = count ctx "host_unregistrations" in
  let unreg_fail_before_drop = count ctx "host_unregister_failures" in
  let live_before_drop = count ctx "live_host_registered_bytes" in
  let ctx, w_drop, drop_parity_ok, reg_during_drop, live_during_drop =
    run_droppable_large_upload ctx large_routine large_x large_y
  in
  let drop_reclaimed = Gc_witness.reclaimed w_drop in
  p "dropped large array: uploads computed exact values and host array is reclaimed by GC"
    (drop_parity_ok && drop_reclaimed);
  in_leg ~leg:cuda
    "dropped large array: registered while live and unregistered once by GC finaliser" (fun label ->
      p label
        (reg_during_drop = reg_before_drop + 1
        && live_during_drop = live_before_drop + large_bytes
        && count ctx "host_unregistrations" = unreg_before_drop + 1
        && count ctx "host_unregister_failures" = unreg_fail_before_drop
        && count ctx "live_host_registered_bytes" = live_before_drop));

  Stdio.eprintf
    "cuda_host_registration: reg=%d unreg=%d declined=%d unreg_fail=%d live_bytes=%d (not part of \
     the golden)\n\
     %!"
    (count ctx "host_registrations")
    (count ctx "host_unregistrations")
    (count ctx "host_register_declined")
    (count ctx "host_unregister_failures")
    (count ctx "live_host_registered_bytes");
  ignore (Stdlib.Sys.opaque_identity live_nd);
  Context.release ctx
