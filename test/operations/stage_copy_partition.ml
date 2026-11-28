(* gh-ocannl-1661: a shared Stage partitions its copy across the workgroup axis its tile does not
   use, rather than restricting the copy to one representative thread of that axis.

   The staged matmul of schedule_smem_matmul ([mc = ma * mb], i and j split into Grid x Workgroup by
   [bm], k into Serial x Serial by [bk], both operands staged at k_o), at three geometries, one per
   way [Schedule.apply_stage] folds the unused axis (extent [w = bm]) into the copy's serial k loop
   (extent [e = bk]): - divide: [w] divides [e] (8 into 16): the loop runs [e / w] steps over [w * s
   + t], no guard; - drop: [e <= w] (8 against 16): the axis replaces the loop under [t < e]; -
   restrict: neither (24 against 16): one representative thread ([t == 0]) copies, as before.

   On GPU backends each kernel executes against the serial twin, its accumulator prefilled off the
   result so a dropped zero shows. Everywhere, the scheduled IR is read on the host
   ([Ll_test.captured_lowering]): which guards survive, and whether every tile store names both
   Workgroup loops -- each thread copying cells of its own. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let skipped = Verdict.skipped ~backend:backend_name
let has_shared = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  Test_utils.Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let geometry ~tag ~n ~bm ~bk ~expect_guard ~partitioned =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ tag ^ "_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ tag ^ "_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  (* The serial twin: [i, j] sums [ma[i, k] * mb[k, j]] over k, on the host. *)
  let want =
    Array.init (n * n) ~f:(fun c ->
        let i = c / n and j = c % n in
        List.sum
          (module Float)
          (List.range 0 n)
          ~f:(fun k -> mav.((i * n) + k) *. mbv.((k * n) + j)))
  in
  p_exists (tag ^ ": the serial twin has a nonzero cell") (Array.to_list want) ~f:(fun x ->
      Float.(x <> 0.));
  let%op mc = ma * mb in
  let schedule (opt : LL.optimized) : Sched.schedule =
    let paths = Ll_test.nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:mc.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    [
      ez;
      sp_zi;
      sp_zj;
      sp_i;
      sp_j;
      sp_k;
      stage ma.Tensor.value [ i_i; k_i ];
      stage mb.Tensor.value [ k_i; j_i ];
    ]
  in
  let comp = named ("scp_" ^ tag) (Train.forward mc) in
  let transform opt = Sched.apply (schedule opt) opt in
  let parity = tag ^ ": staged matmul matches the serial twin" in
  if has_shared then
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    let prefill = Array.init (n * n) ~f:(fun c -> Float.of_int (1000 + c)) in
    let ctx = Context.set_values ctx mc.Tensor.value prefill in
    let ctx = Context.run ctx routine in
    p_close parity (Context.get_values ctx mc.Tensor.value) want ~tol:(fun _ -> 1e-3)
  else skipped parity;
  let post = transform (Ll_test.captured_lowering ~name:("scp_capture_" ^ tag) comp) in
  let guards = ref [] in
  Ll_test.walk post.LL.llc ~on_if:(fun c -> guards := c :: !guards);
  let is_guard op = function LL.Binop (op', _, _) -> Ir.Ops.equal_binop op op' | _ -> false in
  let stores = Ll_test.tile_store_partitions ~tiles:post.LL.workgroup_shared post.LL.llc in
  (match expect_guard with
  | None -> p_empty (tag ^ ": no guard survives over the tile copies") ~over:stores !guards
  | Some (what, op) ->
      p (tag ^ ": exactly the two copies are guarded") (List.length !guards = 2);
      p_all (tag ^ ": each guard is " ^ what) !guards ~f:(is_guard op));
  p_all
    (tag
    ^
    if partitioned then ": every tile store is partitioned across both Workgroup loops"
    else ": every tile store sits under both Workgroup loops, indexed by one")
    stores
    ~f:(fun (n_wg, named_all) -> n_wg = 2 && Bool.equal named_all partitioned)

let () =
  geometry ~tag:"divide" ~n:32 ~bm:8 ~bk:16 ~expect_guard:None ~partitioned:true;
  geometry ~tag:"drop" ~n:32 ~bm:16 ~bk:8
    ~expect_guard:(Some ("the thread bound t < e", Ir.Ops.Cmplt))
    ~partitioned:true;
  geometry ~tag:"restrict" ~n:48 ~bm:16 ~bk:24
    ~expect_guard:(Some ("the representative t == 0", Ir.Ops.Cmpeq))
    ~partitioned:false

(* gh-ocannl-1842: helpers inspecting whether every read of a staged source maps hardware Workgroup
   slot 0 onto the source's minor axis with unit stride. *)
let unit_coeff_on sym = function
  | Ir.Indexing.Iterator s -> Ir.Indexing.equal_symbol sym s
  | Ir.Indexing.Affine { symbols; _ } ->
      List.exists symbols ~f:(fun (c, s) -> c = 1 && Ir.Indexing.equal_symbol sym s)
  | Ir.Indexing.Fixed_idx _ | Ir.Indexing.Sub_axis | Ir.Indexing.Concat _ -> false

let slot0_unit_on_src_minor_reads ~(source : Ir.Tnode.t) (post : LL.optimized) : bool list =
  let slot0_syms =
    LL.hardware_axes post.LL.llc
    |> List.filter_map ~f:(fun a ->
        if Poly.equal a.LL.ha_kind `Workgroup && a.LL.ha_slot = 0 then Some a.LL.ha_index else None)
  in
  LL.affine_accesses post.LL.llc
  |> List.filter_map ~f:(fun (a : Ir.Tnode.t Ir.Affine.access) ->
      if (not a.a_write) && (not a.a_whole) && Ir.Tnode.equal a.a_tn source then
        let minor = a.a_map.(Array.length a.a_map - 1) in
        Some (List.exists slot0_syms ~f:(fun s -> unit_coeff_on s minor))
      else None)

let transposed_matmul_ref ~m ~n ~k mav mwv =
  Array.init (m * n) ~f:(fun c ->
      let i = c / n and j = c % n in
      List.sum
        (module Float)
        (List.range 0 k)
        ~f:(fun kk -> mav.((i * k) + kk) *. mwv.((j * k) + kk)))

(* Section 1 (gh-ocannl-1842): transposed-B matmul [d[i, j] += a[i, k] * w[j, k]]. Unpadded 32x32
   f32 shared tiles (128B row, multiple of 32B) must decline source-minor remapping on transposed
   [w] to avoid 8-way shared-store bank conflicts; 16-byte bank padding ([pad_stride = Some 36])
   unlocks it, keeps zero guards, emits [aligned(16)] on GPU, and matches the serial twin. *)
let () =
  let m = 64 and n = 64 and k = 32 in
  let mav =
    Array.init (m * k) ~f:(Ll_test.cycle_flat ~dims:[| m; k |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mwv =
    Array.init (n * k) ~f:(Ll_test.cycle_flat ~dims:[| n; k |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let want = transposed_matmul_ref ~m ~n ~k mav mwv in
  p_exists "trans: the serial twin has a nonzero cell" (Array.to_list want) ~f:(fun x ->
      Float.(x <> 0.));
  let run_trans ~tag ~pad_stride ~expect_w_slot0_minor ~check_gpu_align =
    let ma = TDSL.ndarray mav ~label:[ tag ^ "_a" ] ~output_dims:[ m; k ] () in
    let mw = TDSL.ndarray mwv ~label:[ tag ^ "_w" ] ~output_dims:[ n; k ] () in
    let%op md = ma +* "ik; jk => ij" mw in
    let schedule (opt : LL.optimized) : Sched.schedule =
      let paths = Ll_test.nest_paths opt.LL.llc in
      let i, j, kk =
        match List.find_exn paths ~f:(fun p -> List.length p = 3) with
        | [ i; j; kk ] -> (i, j, kk)
        | _ -> assert false
      in
      let ez, zsyms = Sched.expand_zero ~tn:md.Tensor.value in
      let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
      let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:32 ~outer:LL.Grid ~inner:LL.Serial in
      let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:32 ~outer:LL.Grid ~inner:LL.Serial in
      let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:32 ~outer:LL.Grid ~inner:LL.Serial in
      let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:32 ~outer:LL.Grid ~inner:LL.Serial in
      let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_k, k_o, k_i = Sched.split ~axis:kk ~factor:32 ~outer:LL.Serial ~inner:LL.Serial in
      let sink s targets = List.map targets ~f:(fun t -> Sched.Swap { outer = s; inner = t }) in
      [ ez; sp_zi; sp_zi2; sp_zj; sp_zj2; sp_i; sp_i2; sp_j; sp_j2; sp_k ]
      @ sink i_t [ j_o; j_w; j_t; k_o; k_i ]
      @ sink j_t [ k_o; k_i ]
      @ [
          Sched.Stage
            {
              source = ma.Tensor.value;
              tile_loops = [ i_w; i_t; k_i ];
              shared = true;
              cooperative = None;
              hoisted = false;
              swizzle = None;
              pad_stride;
              pipeline_depth = 1;
              tile_prec = None;
            };
          Sched.Stage
            {
              source = mw.Tensor.value;
              tile_loops = [ k_i; j_w; j_t ];
              shared = true;
              cooperative = None;
              hoisted = false;
              swizzle = None;
              pad_stride;
              pipeline_depth = 1;
              tile_prec = None;
            };
          Sched.privatize ~accum_prec:Fn.id ~target:md.Tensor.value ~over:k_o;
        ]
    in
    let routine_name = "scp_" ^ tag in
    let comp = named routine_name (Train.forward md) in
    let transform opt = Sched.apply (schedule opt) opt in
    let post = transform (Ll_test.captured_lowering ~name:("scp_cap_" ^ tag) comp) in
    let stores = Ll_test.tile_store_partitions ~tiles:post.LL.workgroup_shared post.LL.llc in
    p_all (tag ^ ": every tile store is partitioned across both Workgroup loops") stores
      ~f:(fun (n_wg, named_all) -> n_wg = 2 && named_all);
    p_all
      (tag ^ ": non-transposed A maps slot 0 with unit stride on its source minor axis")
      (slot0_unit_on_src_minor_reads ~source:ma.Tensor.value post)
      ~f:Fn.id;
    p_all
      (tag
      ^
      if expect_w_slot0_minor then
        ": bank-padded transposed W maps slot 0 with unit stride on its source minor axis"
      else ": unpadded transposed W declines source-minor remapping")
      (slot0_unit_on_src_minor_reads ~source:mw.Tensor.value post)
      ~f:(fun unit_minor -> Bool.equal unit_minor expect_w_slot0_minor);
    let gpu_leg = Verdict.leg ~when_:has_shared ~on:backend_name () in
    let got =
      lazy
        (let ctx, routine =
           Context.compile ~name:routine_name
             ~lowered_transform:(fun o -> [ transform o ])
             (Context.auto ()) comp Ir.Indexing.Empty
         in
         let prefill = Array.init (m * n) ~f:(fun c -> Float.of_int (1000 + c)) in
         let ctx = Context.set_values ctx md.Tensor.value prefill in
         let ctx = Context.run ctx routine in
         Context.get_values ctx md.Tensor.value)
    in
    in_leg ~leg:gpu_leg (tag ^ ": executed output matches the serial twin") (fun label ->
        p_close label (Lazy.force got) want ~tol:(fun _ -> 1e-3));
    if check_gpu_align then
      in_leg ~leg:gpu_leg (tag ^ ": emitted padded shared tile carries aligned(16)") (fun label ->
          ignore (Lazy.force got : float array);
          Test_utils.Generated.assert_emits ~routine:routine_name
            ~contains:"__attribute__((aligned(16)))" label)
  in
  run_trans ~tag:"trans_unpadded" ~pad_stride:None ~expect_w_slot0_minor:false
    ~check_gpu_align:false;
  run_trans ~tag:"trans_padded" ~pad_stride:(Some 36) ~expect_w_slot0_minor:true
    ~check_gpu_align:true

(* Section 2 (gh-ocannl-1842): 64x64x8/4x4 scalar GPU sketch splits its 16x16 workgroup into a 3D
   (8, 2, 16) workgroup so slot 0 matches bk=8, eliminates idle-lane copy guards, pads both shared
   tiles by +16 bytes (12 and 68 f32), maps slot 0 with unit stride on k for both operands, and
   matches the serial twin on GPU. *)
let () =
  let m = 64 and n = 64 and k = 32 in
  let mav =
    Array.init (m * k) ~f:(Ll_test.cycle_flat ~dims:[| m; k |] ~modulus:11 ~offset:0. ~stride:0.25)
  in
  let mwv =
    Array.init (n * k)
      ~f:(Ll_test.cycle_flat ~dims:[| n; k |] ~modulus:13 ~offset:(-6.) ~stride:0.5)
  in
  let want = transposed_matmul_ref ~m ~n ~k mav mwv in
  let ma = TDSL.ndarray mav ~label:[ "wg3d_a" ] ~output_dims:[ m; k ] () in
  let mw = TDSL.ndarray mwv ~label:[ "wg3d_w" ] ~output_dims:[ n; k ] () in
  let%op md = ma +* "ik; jk => ij" mw in
  let routine_name = "scp_wg3d_64x64x8" in
  let comp = named routine_name (Train.forward md) in
  let base_opt = Ll_test.captured_lowering ~name:"scp_cap_wg3d" comp in
  let cuda_like =
    {
      Ir.Backend_intf.no_hardware_limits with
      max_threads_per_workgroup = Some 1024;
      max_workgroup_memory_bytes = Some 49152;
      max_workgroup_dims = Some (1024, 1024, 64);
    }
  in
  let seeds = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:cuda_like base_opt in
  let p_64 =
    List.find_exn seeds ~f:(fun (q : Autotune.sketch_params) ->
        q.sk_gpu && (not q.sk_mma) && (not q.sk_epilogue) && q.sk_bm = 64 && q.sk_bn = 64
        && q.sk_bk = 8 && q.sk_tm = 4 && q.sk_tn = 4)
  in
  let transform opt =
    let sched = Autotune.sketch_schedule ~accum_prec:Fn.id ~p:p_64 opt in
    Sched.apply sched opt
  in
  let post = transform base_opt in
  let dims = LL.launch_dims post.LL.llc in
  p "wg3d: 64x64x8/4x4 launches a 3D (8, 2, 16) workgroup"
    (Array.equal Int.equal dims.LL.block [| 8; 2; 16 |]);
  let guards = ref [] in
  Ll_test.walk post.LL.llc ~on_if:(fun c -> guards := c :: !guards);
  let stores = Ll_test.tile_store_partitions ~tiles:post.LL.workgroup_shared post.LL.llc in
  p_empty "wg3d: no idle-lane guard survives over the tile copies" ~over:stores !guards;
  p_all "wg3d: every tile store is partitioned across all three Workgroup loops" stores
    ~f:(fun (n_wg, named_all) -> n_wg = 3 && named_all);
  p_all "wg3d: A maps slot 0 with unit stride on its source minor axis"
    (slot0_unit_on_src_minor_reads ~source:ma.Tensor.value post)
    ~f:Fn.id;
  p_all "wg3d: transposed W maps slot 0 with unit stride on its source minor axis"
    (slot0_unit_on_src_minor_reads ~source:mw.Tensor.value post)
    ~f:Fn.id;
  let tile_minors =
    Set.to_list post.LL.workgroup_shared
    |> List.map ~f:(fun tn ->
        let d = Lazy.force tn.Ir.Tnode.dims in
        d.(Array.length d - 1))
    |> List.sort ~compare:Int.compare
  in
  p "wg3d: staged tiles carry +4 f32 bank-padded minor dims (12 and 68)"
    (List.equal Int.equal tile_minors [ 12; 68 ]);
  let gpu_leg = Verdict.leg ~when_:has_shared ~on:backend_name () in
  let got =
    lazy
      (let ctx, routine =
         Context.compile ~name:routine_name
           ~lowered_transform:(fun o -> [ transform o ])
           (Context.auto ()) comp Ir.Indexing.Empty
       in
       let prefill = Array.init (m * n) ~f:(fun c -> Float.of_int (1000 + c)) in
       let ctx = Context.set_values ctx md.Tensor.value prefill in
       let ctx = Context.run ctx routine in
       Context.get_values ctx md.Tensor.value)
  in
  in_leg ~leg:gpu_leg "wg3d: executed 64x64x8/4x4 matches the serial twin" (fun label ->
      p_close label (Lazy.force got) want ~tol:(fun _ -> 1e-3));
  in_leg ~leg:gpu_leg "wg3d: emitted padded shared tiles carry aligned(16)" (fun label ->
      ignore (Lazy.force got : float array);
      Test_utils.Generated.assert_emits ~routine:routine_name
        ~contains:"__attribute__((aligned(16)))" label)

(* Section 3 (gh-ocannl-1842): under a 32 KiB workgroup-memory cap, 64x64x64/4x4 f32 falls back to
   [sk_no_bank_pad = true] (32768 B, unpadded transposed W keeps base copy); under 48 KiB it keeps
   [sk_no_bank_pad = false] (34816 B, both operands map slot 0 with unit stride on K). *)
let () =
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let w = NTDSL.init ~l:"bp_wout" ~prec:Ir.Ops.single ~o:[ 64 ] ~i:[ 2; 32 ] ~f:cell () in
  let x = NTDSL.init ~l:"bp_xout" ~prec:Ir.Ops.single ~b:[ 2; 64 ] ~o:[ 2; 32 ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let base_opt = Ll_test.captured_lowering ~name:"scp_cap_bp_fallback" (Train.forward d) in
  let find_64 cap =
    let limits =
      {
        Ir.Backend_intf.no_hardware_limits with
        max_threads_per_workgroup = Some 1024;
        max_workgroup_memory_bytes = Some cap;
        max_workgroup_dims = Some (1024, 1024, 64);
      }
    in
    Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits base_opt
    |> List.find_exn ~f:(fun (q : Autotune.sketch_params) ->
        q.sk_gpu && (not q.sk_mma) && (not q.sk_epilogue) && q.sk_bm = 64 && q.sk_bn = 64
        && q.sk_bk = 64 && q.sk_tm = 4 && q.sk_tn = 4 && (not q.sk_batch_grid)
        && Poly.equal q.sk_coalesce (Some Autotune.Coalesce_contraction))
  in
  let p_32k = find_64 32768 and p_48k = find_64 49152 in
  p "fallback: 32 KiB cap sets sk_no_bank_pad on 64x64x64 f32" p_32k.sk_no_bank_pad;
  p "fallback: 48 KiB cap enables bank padding on 64x64x64 f32" (not p_48k.sk_no_bank_pad);
  let opt_32k = Sched.apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:p_32k base_opt) base_opt
  and opt_48k =
    Sched.apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:p_48k base_opt) base_opt
  in
  p "fallback: 32 KiB schedule fits in 32768 B"
    (Sched.workgroup_memory_bytes ~capabilities:Ir.Backend_intf.no_codegen_capabilities opt_32k
    = 32768);
  p "fallback: 48 KiB schedule allocates 34816 B padded tiles"
    (Sched.workgroup_memory_bytes ~capabilities:Ir.Backend_intf.no_codegen_capabilities opt_48k
    = 34816);
  p_none "fallback: 32 KiB unpadded transposed W declines source-minor remapping"
    (slot0_unit_on_src_minor_reads ~source:w.Tensor.value opt_32k)
    ~f:Fn.id;
  p_all "fallback: 48 KiB padded transposed W maps slot 0 with unit stride on K"
    (slot0_unit_on_src_minor_reads ~source:w.Tensor.value opt_48k)
    ~f:Fn.id

(* Section 4 (gh-ocannl-1842 local-review regression): when dense remapping is declined on an
   unpadded transposed tile and hardware slot 0 (width 16) is unused while the remaining serial copy
   loops are outer [k_i = 32] and inner [i_t = 4], slot 0 must divide the 32-element loop with zero
   idle-lane guards rather than dropping onto the 4-element loop under [t < 4]. *)
let () =
  let m = 64 and n = 64 and k = 32 in
  let mav =
    Array.init (m * k) ~f:(Ll_test.cycle_flat ~dims:[| m; k |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mwv =
    Array.init (n * k) ~f:(Ll_test.cycle_flat ~dims:[| n; k |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "divfirst_a" ] ~output_dims:[ m; k ] () in
  let mw = TDSL.ndarray mwv ~label:[ "divfirst_w" ] ~output_dims:[ n; k ] () in
  let%op md = ma +* "ik; jk => ij" mw in
  let comp = named "scp_divfirst" (Train.forward md) in
  let base_opt = Ll_test.captured_lowering ~name:"scp_cap_divfirst" comp in
  let paths = Ll_test.nest_paths base_opt.LL.llc in
  let i, j, kk =
    match List.find_exn paths ~f:(fun p -> List.length p = 3) with
    | [ i; j; kk ] -> (i, j, kk)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:md.Tensor.value in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:64 ~outer:LL.Grid ~inner:LL.Serial in
  let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:64 ~outer:LL.Grid ~inner:LL.Serial in
  let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:64 ~outer:LL.Grid ~inner:LL.Serial in
  let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:64 ~outer:LL.Grid ~inner:LL.Serial in
  let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:4 ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:kk ~factor:32 ~outer:LL.Serial ~inner:LL.Serial in
  let sink s targets = List.map targets ~f:(fun t -> Sched.Swap { outer = s; inner = t }) in
  let sched =
    [ ez; sp_zi; sp_zi2; sp_zj; sp_zj2; sp_i; sp_i2; sp_j; sp_j2; sp_k ]
    @ sink i_t [ j_o; j_w; j_t; k_o; k_i ]
    @ sink j_t [ k_o; k_i ]
    @ [
        (* Unpadded transposed tile of [ma]: [k_i=32] outer, [i_w=16; i_t=4] inner (minor dim 64 f32
           = 256B, multiple of 32B -> dense_copy declined). Unused slot 0 is [j_w=16]; serial copy
           loops are [k_i=32; i_t=4]. *)
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ k_i; i_w; i_t ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.privatize ~accum_prec:Fn.id ~target:md.Tensor.value ~over:k_o;
      ]
  in
  let post = Sched.apply sched base_opt in
  let guards = ref [] in
  Ll_test.walk post.LL.llc ~on_if:(fun c -> guards := c :: !guards);
  let stores = Ll_test.tile_store_partitions ~tiles:post.LL.workgroup_shared post.LL.llc in
  p_empty "divfirst: fallback slot 0 divides outer 32-trip loop with zero idle-lane guards"
    ~over:stores !guards;
  p_all "divfirst: tile store is partitioned across both Workgroup loops" stores
    ~f:(fun (n_wg, named_all) -> n_wg = 2 && named_all)
