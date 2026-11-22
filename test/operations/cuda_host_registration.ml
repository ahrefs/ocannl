(* gh-ocannl-1854: CUDA automatic page-locking of recurring host arrays ([cuMemHostRegister_v2] /
   [cuMemHostUnregister]).

   Exercises recurring host-to-device uploads across every backend: - Explicitly states
   [cuda_host_register_min_bytes] via [Utils.with_stated_config] so ambient environment or config
   settings cannot mask or falsify the threshold checks, plus an explicit disabled ([0]) check. -
   Executed parity for small (4 KiB) and threshold-sized (4 MiB) recurring host arrays. - Link-time
   [Backends.allocate_delta] completion-contract check: links a registered 16 MiB [Ir.Host_inits]
   buffer into a fresh context (calling [Device.from_host] without any [Context.from_host]-level
   [await]), immediately overwrites the host buffer with a poison sentinel before [Context.run], and
   verifies every output element matches the pre-poison data. - Overlapping Bigarray views: uploads
   a 4 MiB sub-slice twice (registering it on CUDA), then uploads its 8 MiB parent twice (evicting
   the slice so CUDA never sees a mixed registered / unregistered copy range, then registering the
   parent), then uploads the sub-slice again (covered by the registered parent without
   re-registering), checking exact parity at each step. - On eligible discrete CUDA devices (gated
   via [Verdict.leg]), pins the driver-level counters from [Context.global_debug_info], including GC
   finaliser unregistration with zero failures. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX

let small_elems = 1024 (* 4 KiB < 4 MiB stated threshold *)
let large_elems = 1_048_576 (* 4 MiB = 4_194_304 B stated threshold *)
let large_bytes = large_elems * 4
let delta_elems = 4_194_304 (* 16 MiB for link-time allocate_delta overwrite check *)

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

let upload_and_check ctx routine x_tn y_tn nd ~step ~n =
  fill_step nd ~step ~n;
  let ctx = Context.from_host ctx x_tn nd in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y_tn in
  (ctx, matches_step ~step ~n got)

let[@inline never] run_droppable_large_upload ctx routine x_tn y_tn =
  let nd = make_f32_host_nd large_elems in
  let ctx, ok1 = upload_and_check ctx routine x_tn y_tn nd ~step:10 ~n:large_elems in
  let ctx, ok2 = upload_and_check ctx routine x_tn y_tn nd ~step:11 ~n:large_elems in
  let reg_during = count ctx "host_registrations" in
  let live_during = count ctx "live_host_registered_bytes" in
  let w = Gc_witness.witness nd in
  (ctx, w, ok1 && ok2, reg_during, live_during)

let () =
  Utils.with_stated_config
    [ ("cuda_host_register_min_bytes", Int.to_string large_bytes) ]
    (fun () ->
      let ctx = Context.auto () in
      let backend_name = Context.backend_name ctx in
      let on_cuda = String.equal backend_name "cuda" in
      let cuda_backend = Verdict.leg ~when_:on_cuda ~on:backend_name () in
      let cuda =
        Verdict.sub_leg cuda_backend ~aggregation:`Environment
          ~when_:(lazy (count ctx "host_register_eligible_devices" > 0))
          ~on:"cuda (integrated or no host-register support)"
      in
      Stdio.eprintf
        "cuda_host_registration: backend=%s eligible_devices=%d (not part of the golden)\n%!"
        backend_name
        (count ctx "host_register_eligible_devices");
      in_leg ~leg:cuda_backend "cuda global debug info reports host_register_eligible_devices"
        (fun label -> p label (Option.is_some (debug_counter ctx "host_register_eligible_devices")));

      (* 1. Small recurring array (4 KiB < 4 MiB stated threshold). *)
      let ctx, small_routine, small_x, small_y =
        compile_scale_add ctx ~name:"chr_small" ~n:small_elems
      in
      let reg0 = count ctx "host_registrations" in
      let live0 = count ctx "live_host_registered_bytes" in
      let small_nd = make_f32_host_nd small_elems in
      let ctx, sok1 =
        upload_and_check ctx small_routine small_x small_y small_nd ~step:1 ~n:small_elems
      in
      let ctx, sok2 =
        upload_and_check ctx small_routine small_x small_y small_nd ~step:2 ~n:small_elems
      in
      let ctx, sok3 =
        upload_and_check ctx small_routine small_x small_y small_nd ~step:3 ~n:small_elems
      in
      ignore (Stdlib.Sys.opaque_identity small_nd);
      p "small array (4 KiB): repeated uploads compute exact values" (sok1 && sok2 && sok3);
      in_leg ~leg:cuda "small array (4 KiB): repeated uploads below threshold register nothing"
        (fun label ->
          p label
            (count ctx "host_registrations" = reg0 && count ctx "live_host_registered_bytes" = live0));

      (* 2. Large recurring array (4 MiB = stated threshold): step-by-step sighting, registration,
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
        upload_and_check ctx large_routine large_x large_y live_nd ~step:1 ~n:large_elems
      in
      p "large array (4 MiB): first upload computes exact values" lok1;
      in_leg ~leg:cuda "large array (4 MiB): first upload sights without registering" (fun label ->
          p label
            (count ctx "host_registrations" = reg_b1
            && count ctx "live_host_registered_bytes" = live_b1));

      let reg_b2 = count ctx "host_registrations" in
      let dec_b2 = count ctx "host_register_declined" in
      let live_b2 = count ctx "live_host_registered_bytes" in
      let ctx, lok2 =
        upload_and_check ctx large_routine large_x large_y live_nd ~step:2 ~n:large_elems
      in
      p "large array (4 MiB): second upload computes exact values" lok2;
      in_leg ~leg:cuda "large array (4 MiB): second upload registers once and accounts live bytes"
        (fun label ->
          p label
            (count ctx "host_registrations" = reg_b2 + 1
            && count ctx "host_register_declined" = dec_b2
            && count ctx "live_host_registered_bytes" = live_b2 + large_bytes));

      let reg_b3 = count ctx "host_registrations" in
      let live_b3 = count ctx "live_host_registered_bytes" in
      let ctx, lok3 =
        upload_and_check ctx large_routine large_x large_y live_nd ~step:3 ~n:large_elems
      in
      p "large array (4 MiB): third upload computes exact values" lok3;
      in_leg ~leg:cuda
        "large array (4 MiB): third upload reuses registration without re-registering" (fun label ->
          p label
            (count ctx "host_registrations" = reg_b3
            && count ctx "live_host_registered_bytes" = live_b3));

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
        "dropped large array: registered while live and unregistered once by GC finaliser"
        (fun label ->
          p label
            (reg_during_drop = reg_before_drop + 1
            && live_during_drop = live_before_drop + large_bytes
            && count ctx "host_unregistrations" = unreg_before_drop + 1
            && count ctx "host_unregister_failures" = unreg_fail_before_drop
            && count ctx "live_host_registered_bytes" = live_before_drop));

      (* 4. Overlapping Bigarray views: upload a 4 MiB sub-slice twice (registers slice), then
         upload its 8 MiB parent twice (evicts slice before copy so CUDA never sees a mixed
         registered/unregistered range, then registers parent), then upload the sub-slice again
         (covered by the registered parent without re-registering). *)
      let parent_elems = 2 * large_elems in
      let parent_bytes = parent_elems * 4 in
      let parent_ba =
        Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 2; large_elems |]
      in
      let parent_flat_ba =
        Bigarray.reshape_1 parent_ba parent_elems |> Bigarray.genarray_of_array1
      in
      let slice_ba = Bigarray.Genarray.sub_left parent_ba 1 1 in
      let slice_flat_ba = Bigarray.reshape_1 slice_ba large_elems |> Bigarray.genarray_of_array1 in
      let parent_nd = Ir.Ndarray.as_array Ir.Ops.Single parent_flat_ba in
      let slice_nd = Ir.Ndarray.as_array Ir.Ops.Single slice_flat_ba in
      let ctx, parent_routine, parent_x, parent_y =
        compile_scale_add ctx ~name:"chr_parent" ~n:parent_elems
      in
      let reg_ov0 = count ctx "host_registrations" in
      let unreg_ov0 = count ctx "host_unregistrations" in
      let live_ov0 = count ctx "live_host_registered_bytes" in
      let ctx, ov_sl1 =
        upload_and_check ctx large_routine large_x large_y slice_nd ~step:20 ~n:large_elems
      in
      let ctx, ov_sl2 =
        upload_and_check ctx large_routine large_x large_y slice_nd ~step:21 ~n:large_elems
      in
      let reg_after_sl2 = count ctx "host_registrations" in
      let live_after_sl2 = count ctx "live_host_registered_bytes" in
      let ctx, ov_p1 =
        upload_and_check ctx parent_routine parent_x parent_y parent_nd ~step:22 ~n:parent_elems
      in
      let unreg_after_p1 = count ctx "host_unregistrations" in
      let live_after_p1 = count ctx "live_host_registered_bytes" in
      let ctx, ov_p2 =
        upload_and_check ctx parent_routine parent_x parent_y parent_nd ~step:23 ~n:parent_elems
      in
      let reg_after_p2 = count ctx "host_registrations" in
      let live_after_p2 = count ctx "live_host_registered_bytes" in
      let ctx, ov_sl3 =
        upload_and_check ctx large_routine large_x large_y slice_nd ~step:24 ~n:large_elems
      in
      let reg_after_sl3 = count ctx "host_registrations" in
      ignore (Stdlib.Sys.opaque_identity (parent_nd, slice_nd));
      p "overlapping slice then parent: all uploads compute exact values without mixed-range errors"
        (ov_sl1 && ov_sl2 && ov_p1 && ov_p2 && ov_sl3);
      in_leg ~leg:cuda
        "overlapping slice then parent: slice evicted on parent upload, parent registers, slice \
         reuses parent pin" (fun label ->
          p label
            (reg_after_sl2 = reg_ov0 + 1
            && live_after_sl2 = live_ov0 + large_bytes
            && unreg_after_p1 = unreg_ov0 + 1
            && live_after_p1 = live_ov0
            && reg_after_p2 = reg_ov0 + 2
            && live_after_p2 = live_ov0 + parent_bytes
            && reg_after_sl3 = reg_ov0 + 2));

      (* 5. Link-time [Backends.allocate_delta] completion-contract check: no [Context.from_host]
         await stands between [Device.from_host] and returning from [Train.to_routine]. Register a
         16 MiB [Host_inits] buffer on two link passes, then link it into a third fresh context and
         immediately poison-overwrite the host array before [Context.run]; without stream
         synchronisation inside [Cuda_backend.from_host], the device reads [-999.0]. *)
      let delta_nd = make_f32_host_nd delta_elems in
      fill_step delta_nd ~step:30 ~n:delta_elems;
      let link_delta_once ~label_suffix =
        let x =
          Operation.init ~mutable_:true ~l:("chr_delta_x_" ^ label_suffix) ~prec:Ir.Ops.single
            ~o:[ delta_elems ]
            ~f:(fun _ -> 0.)
            ~grad_spec:Tensor.Prohibit_grad ()
        in
        Ir.Host_inits.register x.value (lazy delta_nd);
        let%op y = (x *. 2.) + 1. in
        let fresh = Context.auto () in
        let fresh, routine = Train.to_routine fresh IDX.empty (Train.forward y) in
        (fresh, routine, y.value)
      in
      let d_ctx1, d_r1, d_y1 = link_delta_once ~label_suffix:"1" in
      let d_ctx1 = Context.run d_ctx1 d_r1 in
      let d_ok1 = matches_step ~step:30 ~n:delta_elems (Context.get_values d_ctx1 d_y1) in
      Context.release d_ctx1;
      let d_ctx2, d_r2, d_y2 = link_delta_once ~label_suffix:"2" in
      let d_ctx2 = Context.run d_ctx2 d_r2 in
      let d_ok2 = matches_step ~step:30 ~n:delta_elems (Context.get_values d_ctx2 d_y2) in
      Context.release d_ctx2;
      (* Third link hits the kernel cache and uploads the now-registered 16 MiB [delta_nd] via
         [allocate_delta] -> [Device.from_host] with no [Context]-level [await]. *)
      fill_step delta_nd ~step:31 ~n:delta_elems;
      let d_ctx3, d_r3, d_y3 = link_delta_once ~label_suffix:"3" in
      fill_poison delta_nd ~n:delta_elems;
      let d_ctx3 = Context.run d_ctx3 d_r3 in
      let d_ok3 = matches_step ~step:31 ~n:delta_elems (Context.get_values d_ctx3 d_y3) in
      Context.release d_ctx3;
      ignore (Stdlib.Sys.opaque_identity delta_nd);
      p
        "link-time allocate_delta upload (16 MiB): immediate host poison overwrite after compile \
         does not corrupt device values"
        (d_ok1 && d_ok2 && d_ok3);

      (* 6. Explicit disable via [cuda_host_register_min_bytes=0]: fresh 4 MiB array uploaded twice
         registers nothing on CUDA while computing exact values. *)
      let reg_dis0 = count ctx "host_registrations" in
      let ctx, dis_ok =
        Utils.with_stated_config
          [ ("cuda_host_register_min_bytes", "0") ]
          (fun () ->
            let dis_nd = make_f32_host_nd large_elems in
            let ctx, ok1 =
              upload_and_check ctx large_routine large_x large_y dis_nd ~step:40 ~n:large_elems
            in
            let ctx, ok2 =
              upload_and_check ctx large_routine large_x large_y dis_nd ~step:41 ~n:large_elems
            in
            ignore (Stdlib.Sys.opaque_identity dis_nd);
            (ctx, ok1 && ok2))
      in
      p "disabled (cuda_host_register_min_bytes=0): repeated 4 MiB uploads compute exact values"
        dis_ok;
      in_leg ~leg:cuda
        "disabled (cuda_host_register_min_bytes=0): repeated 4 MiB uploads register nothing"
        (fun label -> p label (count ctx "host_registrations" = reg_dis0));

      Stdio.eprintf
        "cuda_host_registration: reg=%d unreg=%d declined=%d unreg_fail=%d live_bytes=%d (not part \
         of the golden)\n\
         %!"
        (count ctx "host_registrations")
        (count ctx "host_unregistrations")
        (count ctx "host_register_declined")
        (count ctx "host_unregister_failures")
        (count ctx "live_host_registered_bytes");
      ignore (Stdlib.Sys.opaque_identity live_nd);
      Context.release ctx)
