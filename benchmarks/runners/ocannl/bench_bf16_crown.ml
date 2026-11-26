(* gh-ocannl-2372: in-process mirrored re-timing of the n = 1024 (or configured n) bf16 GEMM
   finalists beside a byte-identical f32 drift control.

   gh-ocannl-2321 timed wide vs narrow n = 1024 bf16 GEMMs across processes via [schedule_bench]
   (where each process compiles all five variants) and observed a wide/narrow ratio of 1.128
   (0.898-1.717) against an f32 control of 1.007 (0.900-1.183). This runner compiles the wide
   ([Bf16_wide]) and narrow ([Bf16_narrow]) finalists together with their byte-identical f32 drift
   controls in ONE process, verifies: - generated-source identity for the f32 controls
   (byte-identical across [bf16_arithmetic] arms) and reach for the bf16 finalists (sources differ),
   - MMA intrinsic census on every tensorized finalist (zero scalar fallbacks), - whole-output
   checksums (exact f32 equality, structural bound on wide bf16, non-degeneracy on narrow bf16, and
   pre/post-timing determinism), and then interleaves all four routines in {!Bench_retime.retime}
   ({!Bench_design}'s spread mirrored rotation) so session drift divides out call by call. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let p fmt = Bench_out.p fmt

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let all_variants = [ "parallel"; "smem"; "regtile"; "mma_pd1"; "mma_pd2" ]

let build_schedule ~variant ~ma ~mb ~mc ~accum_prec opt =
  let accum_syms opt =
    let paths = nest_paths opt.LL.llc in
    match List.find paths ~f:(fun p -> List.length p = 3) with
    | Some [ i; j; k ] -> (i, j, k)
    | _ -> failwith "bench_bf16_crown: expected a 3-deep i/j/k loop nest"
  in
  let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
  match variant with
  | "parallel" ->
      let b = 16 in
      let i, j, _k = accum_syms opt in
      let ez, zsyms = Sched.expand_zero ~tn:mc in
      let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
      let sp_zi, _, _ = Sched.split ~axis:zi ~factor:b ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_zj, _, _ = Sched.split ~axis:zj ~factor:b ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_i, _, _ = Sched.split ~axis:i ~factor:b ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_j, _, _ = Sched.split ~axis:j ~factor:b ~outer:LL.Grid ~inner:LL.Workgroup in
      [ ez; sp_zi; sp_zj; sp_i; sp_j ]
  | "smem" ->
      let bm, bn, bk = (16, 16, 8) in
      let i, j, k = accum_syms opt in
      let ez, zsyms = Sched.expand_zero ~tn:mc in
      let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
      let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_j, _, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [
        ez;
        sp_zi;
        sp_zj;
        sp_i;
        sp_j;
        sp_k;
        Sched.Stage
          {
            source = ma;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Stage
          {
            source = mb;
            tile_loops = [ k_i; j_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.privatize ~accum_prec ~target:mc ~over:k_o;
      ]
  | "regtile" ->
      let bm, bn, bk, tm, tn = (64, 64, 8, 8, 8) in
      let i, j, k = accum_syms opt in
      let ez, zsyms = Sched.expand_zero ~tn:mc in
      let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
      let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
      let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
      let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
      let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [ ez; sp_zi; sp_zi2; sp_zj; sp_zj2; sp_i; sp_i2; sp_j; sp_j2; sp_k ]
      @ sink i_t [ j_o; j_w; j_t; k_o; k_i ]
      @ sink j_t [ k_o; k_i ]
      @ [
          Sched.Stage
            {
              source = ma;
              tile_loops = [ i_w; i_t; k_i ];
              shared = true;
              cooperative = None;
              hoisted = false;
              swizzle = None;
              pad_stride = None;
              pipeline_depth = 1;
              tile_prec = None;
            };
          Sched.Stage
            {
              source = mb;
              tile_loops = [ k_i; j_w; j_t ];
              shared = true;
              cooperative = None;
              hoisted = false;
              swizzle = None;
              pad_stride = None;
              pipeline_depth = 1;
              tile_prec = None;
            };
          Sched.privatize ~accum_prec ~target:mc ~over:k_o;
          Sched.Unroll { axis = i_t; materialize = true };
          Sched.Unroll { axis = j_t; materialize = true };
        ]
  | ("mma_pd1" | "mma_pd2") as v ->
      let pipeline_depth = if String.equal v "mma_pd2" then 2 else 1 in
      let bm, bn, bk, w = (32, 32, 32, 32) in
      let i, j, k = accum_syms opt in
      let ez, zsyms = Sched.expand_zero ~tn:mc in
      let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
      let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      let sp_zj, _, _ = Sched.split ~axis:zj ~factor:w ~outer:LL.Grid ~inner:LL.Workgroup in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
      let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:k_i ~simd_width:w () in
      let stage source tile_loops =
        Sched.Stage
          {
            source;
            tile_loops;
            shared = true;
            cooperative = Some w;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth;
            tile_prec = None;
          }
      in
      [ ez; sp_zi; sp_zj; sp_i; sp_j; sp_k ]
      @ sink i_i [ j_o ] @ sink j_i [ k_o ] @ sink i_i [ k_o ]
      @ [ stage ma [ i_i; k_i ]; stage mb [ k_i; j_i ]; tz ]
  | other -> invalid_arg ("bench_bf16_crown: unknown variant " ^ other)

let clear_sources ~dir ~ext =
  match Stdlib.Sys.readdir dir with
  | entries ->
      Array.iter entries ~f:(fun name ->
          if String.is_suffix name ~suffix:ext then
            try Stdlib.Sys.remove (Stdlib.Filename.concat dir name) with Stdlib.Sys_error _ -> ())
  | exception Stdlib.Sys_error _ -> ()

let read_sources ~dir ~ext =
  let entries = Stdlib.Sys.readdir dir |> Array.to_list in
  List.filter entries ~f:(fun name -> String.is_suffix name ~suffix:ext)
  |> List.sort ~compare:String.compare
  |> List.map ~f:(fun name ->
      let path = Stdlib.Filename.concat dir name in
      (name, Stdio.In_channel.read_all path))

let sources_equal = List.equal (fun (a1, b1) (a2, b2) -> String.equal a1 a2 && String.equal b1 b2)

let rel_err a b =
  List.fold2_exn a b ~init:0.0 ~f:(fun acc x y ->
      Float.max acc (Float.abs (x -. y) /. Float.max (Float.abs y) 1e-300))

let non_degenerate c exact =
  List.for_all2_exn c exact ~f:(fun x y ->
      Float.(x *. y > 0.0 && Float.abs x >= 0.1 *. Float.abs y))

let is_tensorized (mma : Ir.C_syntax.mma_summary) =
  mma.statements > 0 && mma.scalar_fallbacks = 0
  && List.exists mma.renderings ~f:(fun (_, r) -> Ir.C_syntax.is_tensorized_rendering r)

let () =
  let args = Bench_args.create "bench_bf16_crown" in
  let dry_run = Bench_args.flag args ~name:"dry-run" in
  let default_n = if dry_run then 128 else 1024 in
  let default_calls = if dry_run then Bench_retime.min_readings else 16 in
  let n = Bench_args.int args 0 ~flag:"n" ~name:"n" ~default:default_n in
  let calls = Bench_args.int args 1 ~flag:"calls" ~name:"calls" ~default:default_calls in
  if n % 64 <> 0 then
    invalid_arg (Printf.sprintf "bench_bf16_crown: n=%d must be a multiple of 64" n);
  if calls < Bench_retime.min_readings || calls % 2 <> 0 then
    invalid_arg
      (Printf.sprintf "bench_bf16_crown: calls=%d must be even and >= %d" calls
         Bench_retime.min_readings);
  let wide_spec = Option.value (Bench_args.flag_value args ~flag:"wide") ~default:"auto" in
  let narrow_spec = Option.value (Bench_args.flag_value args ~flag:"narrow") ~default:"auto" in
  let control_spec = Bench_args.flag_value args ~flag:"control" in
  let check_spec ~allow_auto flag s =
    if (allow_auto && String.equal s "auto") || List.mem all_variants s ~equal:String.equal then s
    else
      invalid_arg
        (Printf.sprintf "bench_bf16_crown: --%s=%s must be one of %s" flag s
           (String.concat ~sep:"|" (if allow_auto then "auto" :: all_variants else all_variants)))
  in
  let control_variants = [ "regtile"; "mma_pd1"; "mma_pd2" ] in
  let wide_spec = check_spec ~allow_auto:true "wide" wide_spec in
  let narrow_spec = check_spec ~allow_auto:true "narrow" narrow_spec in
  let control_spec =
    Option.map control_spec ~f:(fun s ->
        if List.mem control_variants s ~equal:String.equal then s
        else
          invalid_arg
            (Printf.sprintf "bench_bf16_crown: --control=%s must be one of %s" s
               (String.concat ~sep:"|" control_variants)))
  in

  let probe_ctx = Context.auto () in
  let backend = String.lowercase (Context.backend_name probe_ctx) in
  let ext =
    if String.is_substring backend ~substring:"metal" then ".metal"
    else if String.is_substring backend ~substring:"cuda" then ".cu"
    else if String.is_substring backend ~substring:"hip" then ".hip"
    else
      failwith
        (Printf.sprintf "bench_bf16_crown: requires a GPU backend (metal/cuda/hip), got %s" backend)
  in

  (* Identical operand generator to bin/schedule_bench.ml. *)
  let a_levels, a_step = (48, 16) in
  let b_max = 8 in
  let mav =
    Array.init (n * n)
      ~f:
        (Bench_checksum.positive_level ~salt:0x5A17 ~row_stride:n ~levels:a_levels
           ~scale:(1. /. Float.of_int a_step))
  in
  let mbv =
    Array.init (n * n) ~f:(fun t ->
        Float.of_int
          (Bench_checksum.residue ~salt:0x3C6E ~row_stride:n ~modulus:((2 * b_max) + 1) t - b_max))
  in

  let base_policy = { (Numerics.get ()) with tf32_matmuls = true } in
  let set_arm ~mode ~prec =
    Numerics.set_policy { base_policy with bf16_arithmetic = mode };
    Tensor.default_value_prec := prec
  in
  let build_problem ~name ~prec =
    Tensor.default_value_prec := prec;
    let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
    let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
    let%op mc = ma * mb in
    let comp = named name (Train.forward mc) in
    (ma.Tensor.value, mb.Tensor.value, mc.Tensor.value, comp)
  in
  let compile_with_sources ~mode ~prec ~variant ~ma ~mb ~mc ~comp ?prebuilt ~capture () =
    set_arm ~mode ~prec;
    let ctx = Context.auto () in
    let accum_prec p = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec p in
    let built = ref None in
    let transform opt =
      match prebuilt with
      | Some scheduled -> scheduled
      | None ->
          let s = Sched.apply (build_schedule ~variant ~ma ~mb ~mc ~accum_prec opt) opt in
          built := Some s;
          s
    in
    let dir = Utils.build_files_dir () in
    let prev = Utils.settings.output_debug_files_in_build_directory in
    let sources = ref [] in
    let ctx, routine =
      Exn.protect
        ~f:(fun () ->
          if capture then (
            clear_sources ~dir ~ext;
            Utils.settings.output_debug_files_in_build_directory <- true);
          let res =
            Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
          in
          if capture then sources := read_sources ~dir ~ext;
          res)
        ~finally:(fun () -> Utils.settings.output_debug_files_in_build_directory <- prev)
    in
    (ctx, routine, !sources, !built)
  in
  let screen_fastest ~label ~mode =
    let timing = Bench_retime.configured_timing () in
    p "screening %s bf16 variants at n=%d:\n" label n;
    let results =
      List.map all_variants ~f:(fun variant ->
          let ma, mb, mc, comp =
            build_problem ~name:("screen_" ^ label ^ "_" ^ variant) ~prec:Ir.Ops.bfloat16
          in
          let ctx, routine, _, _ =
            compile_with_sources ~mode ~prec:Ir.Ops.bfloat16 ~variant ~ma ~mb ~mc ~comp
              ~capture:false ()
          in
          let reading, refused = Bench_retime.read_screen ~timing (ctx, routine) in
          let mma_s = Ir.C_syntax.mma_summary_string routine.Context.mma in
          p "  %-8s %s%s  [%s]\n" variant
            (match reading with Some ms -> Printf.sprintf "%.4f ms" ms | None -> "no-reading")
            (if refused then " (refused)" else "")
            mma_s;
          (variant, reading, refused, routine.Context.mma))
    in
    let valid =
      List.filter_map results ~f:(fun (v, reading, refused, mma) ->
          if String.is_prefix v ~prefix:"mma" && not (is_tensorized mma) then None
          else Option.map reading ~f:(fun ms -> (v, ms, refused)))
    in
    let admitted = List.filter valid ~f:(fun (_, _, refused) -> not refused) in
    let pool = if List.is_empty admitted then valid else admitted in
    match List.min_elt pool ~compare:(fun (_, a, _) (_, b, _) -> Float.compare a b) with
    | Some (v, ms, _) ->
        p "crowned %s finalist: %s (screen %.4f ms)\n" label v ms;
        v
    | None -> failwith ("bench_bf16_crown: no valid variant screened for " ^ label)
  in

  let wide_variant =
    if String.equal wide_spec "auto" then screen_fastest ~label:"wide" ~mode:Numerics.Bf16_wide
    else wide_spec
  in
  let narrow_variant =
    if String.equal narrow_spec "auto" then
      screen_fastest ~label:"narrow" ~mode:Numerics.Bf16_narrow
    else narrow_spec
  in
  let default_control =
    if List.mem control_variants wide_variant ~equal:String.equal then wide_variant else "mma_pd1"
  in
  let control_variant = Option.value control_spec ~default:default_control in

  p "bench_bf16_crown: backend=%s n=%d calls=%d timing=%s wide=%s narrow=%s control=%s dry_run=%b\n"
    backend n calls
    (Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued")
    wide_variant narrow_variant control_variant dry_run;

  (* Verify f32 capabilities are invariant across bf16 policies, then compile the identical
     scheduled f32 IR under BOTH policies to prove byte-identical generated source without UID drift
     from repeating [Sched.apply] in one process. *)
  let precs_for mode prec =
    set_arm ~mode ~prec;
    let caps = Context.codegen_capabilities (Context.auto ()) in
    (caps.accum_prec prec, caps.compute_prec prec)
  in
  let wide_f32_acc, wide_f32_cmp = precs_for Numerics.Bf16_wide Ir.Ops.single in
  let narrow_f32_acc, narrow_f32_cmp = precs_for Numerics.Bf16_narrow Ir.Ops.single in
  if
    not
      (Ir.Ops.equal_prec wide_f32_acc narrow_f32_acc
      && Ir.Ops.equal_prec wide_f32_cmp narrow_f32_cmp)
  then (
    p "INVALID: f32 compute/accum precision changed across bf16 policies\n";
    Stdlib.exit 1);
  let wide_bf16_acc, _ = precs_for Numerics.Bf16_wide Ir.Ops.bfloat16 in
  let narrow_bf16_acc, _ = precs_for Numerics.Bf16_narrow Ir.Ops.bfloat16 in
  let ma_f32, mb_f32, mc_f32, comp_f32 = build_problem ~name:"crown_ctrl_f32" ~prec:Ir.Ops.single in
  let ctx_wf32, r_wf32, src_wf32, sched_f32 =
    compile_with_sources ~mode:Numerics.Bf16_wide ~prec:Ir.Ops.single ~variant:control_variant
      ~ma:ma_f32 ~mb:mb_f32 ~mc:mc_f32 ~comp:comp_f32 ~capture:true ()
  in
  let ctx_nf32, r_nf32, src_nf32, _ =
    compile_with_sources ~mode:Numerics.Bf16_narrow ~prec:Ir.Ops.single ~variant:control_variant
      ~ma:ma_f32 ~mb:mb_f32 ~mc:mc_f32 ~comp:comp_f32 ?prebuilt:sched_f32 ~capture:true ()
  in
  if List.is_empty src_wf32 || not (sources_equal src_wf32 src_nf32) then (
    p "INVALID: f32 drift control sources are not byte-identical (%d vs %d files)\n"
      (List.length src_wf32) (List.length src_nf32);
    (match (src_wf32, src_nf32) with
    | (n1, s1) :: _, (n2, s2) :: _ ->
        let l1 = String.split_lines s1 and l2 = String.split_lines s2 in
        let rec show i a b =
          match (a, b) with
          | x :: xs, y :: ys ->
              if String.equal x y then show (i + 1) xs ys
              else p "  first diff (%s vs %s) line %d:\n  - %s\n  + %s\n" n1 n2 i x y
          | x :: _, [] -> p "  extra line %d in wide: %s\n" i x
          | [], y :: _ -> p "  extra line %d in narrow: %s\n" i y
          | [], [] -> ()
        in
        show 1 l1 l2
    | _ -> ());
    Stdlib.exit 1);
  p "drift control proof: %d generated f32 source(s) byte-identical across bf16 policies\n"
    (List.length src_wf32);

  (* Compile both bf16 finalists under their policies. When both policies pick the same
     non-privatized variant (parallel/mma_pd1/mma_pd2), reuse the scheduled IR so source differences
     prove backend bf16_arithmetic reach without UID noise. *)
  let ma_bf16, mb_bf16, mc_bf16, comp_bf16 =
    build_problem ~name:"crown_trt_bf16" ~prec:Ir.Ops.bfloat16
  in
  let ctx_wbf16, r_wbf16, src_wbf16, sched_wbf16 =
    compile_with_sources ~mode:Numerics.Bf16_wide ~prec:Ir.Ops.bfloat16 ~variant:wide_variant
      ~ma:ma_bf16 ~mb:mb_bf16 ~mc:mc_bf16 ~comp:comp_bf16 ~capture:true ()
  in
  let uses_privatize v = String.equal v "smem" || String.equal v "regtile" in
  let share_bf16_sched =
    String.equal wide_variant narrow_variant && not (uses_privatize wide_variant)
  in
  let ctx_nbf16, r_nbf16, src_nbf16, _ =
    compile_with_sources ~mode:Numerics.Bf16_narrow ~prec:Ir.Ops.bfloat16 ~variant:narrow_variant
      ~ma:ma_bf16 ~mb:mb_bf16 ~mc:mc_bf16 ~comp:comp_bf16
      ?prebuilt:(if share_bf16_sched then sched_wbf16 else None)
      ~capture:true ()
  in
  if
    List.is_empty src_wbf16 || List.is_empty src_nbf16 || sources_equal src_wbf16 src_nbf16
    || String.equal wide_variant narrow_variant
       && uses_privatize wide_variant
       && Ir.Ops.equal_prec wide_bf16_acc narrow_bf16_acc
  then (
    p "INVALID: bf16 treatment sources did not differ across policies (%d vs %d files)\n"
      (List.length src_wbf16) (List.length src_nbf16);
    Stdlib.exit 1);
  p "treatment reach proof: bf16 sources differ across policies (%d wide, %d narrow file(s))\n"
    (List.length src_wbf16) (List.length src_nbf16);

  (* Tensorization census gate on every mma_* finalist. *)
  let check_census ~arm ~variant (routine : Context.routine) =
    let mma = routine.Context.mma in
    p "census %-12s (%s): [%s]\n" arm variant (Ir.C_syntax.mma_summary_string mma);
    if String.is_prefix variant ~prefix:"mma" && not (is_tensorized mma) then (
      p "INVALID: %s (%s) did not tensorize [%s]\n" arm variant (Ir.C_syntax.mma_summary_string mma);
      Stdlib.exit 1)
  in
  check_census ~arm:"wide-bf16" ~variant:wide_variant r_wbf16;
  check_census ~arm:"narrow-bf16" ~variant:narrow_variant r_nbf16;
  check_census ~arm:"wide-f32" ~variant:control_variant r_wf32;
  check_census ~arm:"narrow-f32" ~variant:control_variant r_nf32;

  (* Pre-timing output validation. *)
  let eval_arm ctx routine mc =
    let ctx = Context.run ctx routine in
    let vals = Context.get_values ctx mc in
    (ctx, vals, Bench_checksum.whole_output ~row_stride:n vals)
  in
  let ctx_wf32, vals_wf32, chk_wf32_pre = eval_arm ctx_wf32 r_wf32 mc_f32 in
  let ctx_nf32, vals_nf32, chk_nf32_pre = eval_arm ctx_nf32 r_nf32 mc_f32 in
  (match Bench_checksum.first_difference ~reference:vals_wf32 vals_nf32 with
  | None -> ()
  | Some d ->
      p "INVALID: f32 drift controls disagree: %s\n"
        (Bench_checksum.render_agreement ~name:"wide-f32" (Some d));
      Stdlib.exit 1);
  let ctx_wbf16, _vals_wbf16, chk_wbf16_pre = eval_arm ctx_wbf16 r_wbf16 mc_bf16 in
  let ctx_nbf16, _vals_nbf16, chk_nbf16_pre = eval_arm ctx_nbf16 r_nbf16 mc_bf16 in
  let rel_bound = 0.25 in
  let err_wbf16 = rel_err chk_wbf16_pre chk_wf32_pre in
  let err_nbf16 = rel_err chk_nbf16_pre chk_wf32_pre in
  if Float.(err_wbf16 > rel_bound) then (
    p "INVALID: wide-bf16 checksum off exact f32 by %.4g (bound %.2f)\n" err_wbf16 rel_bound;
    Stdlib.exit 1);
  if not (non_degenerate chk_nbf16_pre chk_wf32_pre) then (
    p "INVALID: narrow-bf16 checksum %s degenerate against exact %s\n"
      (Bench_checksum.render chk_nbf16_pre)
      (Bench_checksum.render chk_wf32_pre);
    Stdlib.exit 1);
  p "checksums (pre): exact-f32=%s wide-bf16=%s (err=%.3g) narrow-bf16=%s (err=%.3g)\n"
    (Bench_checksum.render chk_wf32_pre)
    (Bench_checksum.render chk_wbf16_pre)
    err_wbf16
    (Bench_checksum.render chk_nbf16_pre)
    err_nbf16;

  (* In-process spread mirrored rotation over all 4 routines. *)
  let _timing, readings =
    Bench_retime.retime ~calls
      [ (ctx_wbf16, r_wbf16); (ctx_nbf16, r_nbf16); (ctx_wf32, r_wf32); (ctx_nf32, r_nf32) ]
  in
  let r_wbf16_ms, r_nbf16_ms, r_wf32_ms, r_nf32_ms =
    match readings with [ a; b; c; d ] -> (a, b, c, d) | _ -> assert false
  in

  (* Post-timing determinism check. *)
  let _, _, chk_wf32_post = eval_arm ctx_wf32 r_wf32 mc_f32 in
  let _, _, chk_nf32_post = eval_arm ctx_nf32 r_nf32 mc_f32 in
  let _, _, chk_wbf16_post = eval_arm ctx_wbf16 r_wbf16 mc_bf16 in
  let _, _, chk_nbf16_post = eval_arm ctx_nbf16 r_nbf16 mc_bf16 in
  let chk_eq = List.equal Float.equal in
  if
    not
      (chk_eq chk_wf32_pre chk_wf32_post && chk_eq chk_nf32_pre chk_nf32_post
      && chk_eq chk_wbf16_pre chk_wbf16_post
      && chk_eq chk_nbf16_pre chk_nbf16_post)
  then (
    p "INVALID: post-timing checksums changed (non-deterministic output)\n";
    Stdlib.exit 1);
  p "checksums (post): deterministic across all 4 arms\n";

  let s_wbf16 = Bench_retime.spread_of r_wbf16_ms in
  let s_nbf16 = Bench_retime.spread_of r_nbf16_ms in
  let s_wf32 = Bench_retime.spread_of r_wf32_ms in
  let s_nf32 = Bench_retime.spread_of r_nf32_ms in
  let ratio_bf16 = Bench_retime.paired_ratio ~num:r_wbf16_ms ~den:r_nbf16_ms in
  let ratio_f32 = Bench_retime.paired_ratio ~num:r_wf32_ms ~den:r_nf32_ms in
  p "arm %-12s (%s): %s\n" "wide-bf16" wide_variant (Bench_retime.spread_fields ~name:"ms" s_wbf16);
  p "arm %-12s (%s): %s\n" "narrow-bf16" narrow_variant
    (Bench_retime.spread_fields ~name:"ms" s_nbf16);
  p "arm %-12s (%s): %s\n" "wide-f32" control_variant (Bench_retime.spread_fields ~name:"ms" s_wf32);
  p "arm %-12s (%s): %s\n" "narrow-f32" control_variant
    (Bench_retime.spread_fields ~name:"ms" s_nf32);
  p "paired bf16 wide/narrow (%s/%s, pairs=%d): %s\n" wide_variant narrow_variant ratio_bf16.pairs
    (Bench_retime.ratio_fields ~name:"ratio" ratio_bf16);
  p "paired f32 control wide/narrow (%s/%s, pairs=%d): %s\n" control_variant control_variant
    ratio_f32.pairs
    (Bench_retime.ratio_fields ~name:"ratio" ratio_f32);
  if
    s_wbf16.admitted < Bench_retime.min_readings
    || s_nbf16.admitted < Bench_retime.min_readings
    || s_wf32.admitted < Bench_retime.min_readings
    || s_nf32.admitted < Bench_retime.min_readings
    || ratio_bf16.pairs < Bench_retime.min_readings
    || ratio_f32.pairs < Bench_retime.min_readings
  then (
    p "INVALID: fewer than %d admitted readings/pairs\n" Bench_retime.min_readings;
    Stdlib.exit 1)
