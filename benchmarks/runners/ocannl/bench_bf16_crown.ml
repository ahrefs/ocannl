(* gh-ocannl-2372: in-process mirrored re-timing of the n = 1024 (or configured n) bf16 GEMM
   finalists beside a byte-identical f32 drift control. Launch through
   [benchmarks/gh2372_bf16_crown.py] (or [benchmarks/cells.py cell]) so the OCaml process starts in
   a private working directory before [Utils] runs its startup cleanup.

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

let clear_sources ~dir ~prefix ~ext =
  match Stdlib.Sys.readdir dir with
  | entries ->
      Array.iter entries ~f:(fun name ->
          if String.is_prefix name ~prefix && String.is_suffix name ~suffix:ext then
            try Stdlib.Sys.remove (Stdlib.Filename.concat dir name) with Stdlib.Sys_error _ -> ())
  | exception Stdlib.Sys_error _ -> ()

let read_sources ~dir ~prefix ~ext =
  let entries = Stdlib.Sys.readdir dir |> Array.to_list in
  List.filter entries ~f:(fun name ->
      String.is_prefix name ~prefix && String.is_suffix name ~suffix:ext)
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
  if String.is_substring backend ~substring:"cuda" then
    failwith
      "bench_bf16_crown: CUDA's bf16 accumulators are structurally f32 under both policies, so \
       bf16_arithmetic cannot differentiate wide vs narrow on CUDA";
  let ext =
    if String.is_substring backend ~substring:"metal" then ".metal"
    else if String.is_substring backend ~substring:"hip" then ".hip"
    else failwith (Printf.sprintf "bench_bf16_crown: requires metal or hip backend, got %s" backend)
  in
  if not (String.is_empty (Utils.get_global_arg ~default:"" ~arg_name:"build_files_prefix")) then
    failwith
      "bench_bf16_crown: leave build_files_prefix at its default (configured prefixes cannot be \
       assumed process-private for generated-source capture)";
  let f32_supports_mma =
    Option.is_some
      ((Context.codegen_capabilities probe_ctx).Ir.Backend_intf.mma_arm ~a_prec:Ir.Ops.single
         ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single ~scope:Ir.Backend_intf.Mma_fragment_scope)
  in
  (match control_spec with
  | Some v when String.is_prefix v ~prefix:"mma" && not f32_supports_mma ->
      invalid_arg
        (Printf.sprintf
           "bench_bf16_crown: --control=%s requires f32 tensor-unit support, which backend %s does \
            not advertise (use --control=regtile)"
           v backend)
  | _ -> ());
  Context.release probe_ctx;

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
  let run_prefix = Printf.sprintf "crown_%d_" (Utils.os_process_id ()) in
  let compile_with_sources ~mode ~prec ~variant ~ma ~mb ~mc ~comp ?prebuilt ~source_prefix ~capture
      () =
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
            clear_sources ~dir ~prefix:source_prefix ~ext;
            Utils.settings.output_debug_files_in_build_directory <- true);
          let res =
            Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
          in
          if capture then sources := read_sources ~dir ~prefix:source_prefix ~ext;
          res)
        ~finally:(fun () -> Utils.settings.output_debug_files_in_build_directory <- prev)
    in
    (ctx, routine, !sources, !built)
  in
  let screen_fastest ~label ~mode =
    let timing = Bench_retime.configured_timing () in
    p "screening %s bf16 variants at n=%d:\n" label n;
    let screened =
      List.map all_variants ~f:(fun variant ->
          let rname = run_prefix ^ "screen_" ^ label ^ "_" ^ variant in
          let ma, mb, mc, comp = build_problem ~name:rname ~prec:Ir.Ops.bfloat16 in
          let ctx, routine, _, _ =
            compile_with_sources ~mode ~prec:Ir.Ops.bfloat16 ~variant ~ma ~mb ~mc ~comp
              ~source_prefix:rname ~capture:false ()
          in
          let reading, refused = Bench_retime.read_screen ~timing (ctx, routine) in
          let mma_s = Ir.C_syntax.mma_summary_string routine.Context.mma in
          p "  %-8s %s%s  [%s]\n" variant
            (match reading with Some ms -> Printf.sprintf "%.4f ms" ms | None -> "no-reading")
            (if refused then " (refused)" else "")
            mma_s;
          (variant, ctx, routine, reading, refused))
    in
    let valid, invalid =
      List.partition_tf screened ~f:(fun (v, _, routine, reading, _) ->
          Option.is_some reading
          && ((not (String.is_prefix v ~prefix:"mma")) || is_tensorized routine.Context.mma))
    in
    List.iter invalid ~f:(fun (_, ctx, _, _, _) -> Context.release ctx);
    let fastest_ms =
      List.fold valid ~init:Float.infinity ~f:(fun acc (_, _, _, reading, refused) ->
          match reading with Some ms when not refused -> Float.min acc ms | _ -> acc)
    in
    let fastest_ms =
      if Float.is_finite fastest_ms then fastest_ms
      else
        List.fold valid ~init:Float.infinity ~f:(fun acc (_, _, _, reading, _) ->
            match reading with Some ms -> Float.min acc ms | None -> acc)
    in
    let ranked =
      List.filter valid ~f:(fun (_, _, _, reading, _) ->
          match reading with Some ms -> Float.(ms <= fastest_ms *. 2.0) | None -> false)
      |> List.stable_sort ~compare:(fun (_, _, _, a, _) (_, _, _, b, _) ->
          Float.compare
            (Option.value a ~default:Float.infinity)
            (Option.value b ~default:Float.infinity))
    in
    let shortlist, dropped = List.split_n ranked 3 in
    let shortlist_names = List.map shortlist ~f:(fun (v, _, _, _, _) -> v) in
    List.iter valid ~f:(fun (v, ctx, _, _, _) ->
        if not (List.mem shortlist_names v ~equal:String.equal) then Context.release ctx);
    ignore dropped;
    if List.is_empty shortlist then
      failwith ("bench_bf16_crown: no valid variant screened for " ^ label);
    let _timing, readings =
      Exn.protect
        ~f:(fun () ->
          Bench_retime.retime ~calls:Bench_retime.min_readings
            (List.map shortlist ~f:(fun (_, ctx, routine, _, _) -> (ctx, routine))))
        ~finally:(fun () -> List.iter shortlist ~f:(fun (_, ctx, _, _, _) -> Context.release ctx))
    in
    let retimed =
      List.map2_exn shortlist readings ~f:(fun (v, _, _, _, _) rs ->
          let sp = Bench_retime.spread_of rs in
          p "  shortlist %-8s %s\n" v (Bench_retime.spread_fields ~name:"ms" sp);
          (v, sp))
    in
    let admitted =
      List.filter retimed ~f:(fun (_, sp) -> sp.admitted >= Bench_retime.min_readings)
    in
    match List.min_elt admitted ~compare:(fun (_, a) (_, b) -> Float.compare a.ms b.ms) with
    | Some (v, sp) ->
        p "crowned %s finalist: %s (interleaved median %.4f ms, %d/%d admitted)\n" label v sp.ms
          sp.admitted (sp.admitted + sp.refused);
        v
    | None ->
        failwith
          (Printf.sprintf "bench_bf16_crown: no %s shortlist candidate reached %d admitted readings"
             label Bench_retime.min_readings)
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
    if
      List.mem control_variants wide_variant ~equal:String.equal
      && ((not (String.is_prefix wide_variant ~prefix:"mma")) || f32_supports_mma)
    then wide_variant
    else if f32_supports_mma then "mma_pd1"
    else "regtile"
  in
  let control_variant = Option.value control_spec ~default:default_control in

  p "bench_bf16_crown: backend=%s n=%d calls=%d timing=%s wide=%s narrow=%s control=%s dry_run=%b\n"
    backend n calls
    (Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued")
    wide_variant narrow_variant control_variant dry_run;

  (* Verify f32 capabilities are invariant across bf16 policies, then run dedicated source-capture
     proof compiles (released afterwards) so all four timed compiles share identical default debug
     and compiler flags. *)
  let precs_for mode prec =
    set_arm ~mode ~prec;
    let ctx = Context.auto () in
    let caps = Context.codegen_capabilities ctx in
    let res = (caps.accum_prec prec, caps.compute_prec prec) in
    Context.release ctx;
    res
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

  (* Proof 1: f32 drift control byte-identity across bf16 policies (proof-only compiles). *)
  let proof_ctrl_name = run_prefix ^ "proof_ctrl_f32" in
  let ma_pf32, mb_pf32, mc_pf32, comp_pf32 =
    build_problem ~name:proof_ctrl_name ~prec:Ir.Ops.single
  in
  let ctx_pwf32, _, src_wf32, sched_pf32 =
    compile_with_sources ~mode:Numerics.Bf16_wide ~prec:Ir.Ops.single ~variant:control_variant
      ~ma:ma_pf32 ~mb:mb_pf32 ~mc:mc_pf32 ~comp:comp_pf32 ~source_prefix:proof_ctrl_name
      ~capture:true ()
  in
  Context.release ctx_pwf32;
  let ctx_pnf32, _, src_nf32, _ =
    compile_with_sources ~mode:Numerics.Bf16_narrow ~prec:Ir.Ops.single ~variant:control_variant
      ~ma:ma_pf32 ~mb:mb_pf32 ~mc:mc_pf32 ~comp:comp_pf32 ?prebuilt:sched_pf32
      ~source_prefix:proof_ctrl_name ~capture:true ()
  in
  Context.release ctx_pnf32;
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

  (* Proof 2: BF16 policy reach holding the schedule fixed at [wide_variant] (proof-only
     compiles). *)
  let proof_trt_name = run_prefix ^ "proof_trt_bf16" in
  let ma_pbf16, mb_pbf16, mc_pbf16, comp_pbf16 =
    build_problem ~name:proof_trt_name ~prec:Ir.Ops.bfloat16
  in
  let uses_privatize v = String.equal v "smem" || String.equal v "regtile" in
  let share_wide_sched = not (uses_privatize wide_variant) in
  let ctx_pwbf16, _, src_wbf16, sched_pbf16 =
    compile_with_sources ~mode:Numerics.Bf16_wide ~prec:Ir.Ops.bfloat16 ~variant:wide_variant
      ~ma:ma_pbf16 ~mb:mb_pbf16 ~mc:mc_pbf16 ~comp:comp_pbf16 ~source_prefix:proof_trt_name
      ~capture:true ()
  in
  Context.release ctx_pwbf16;
  let ctx_pnbf16, _, src_reach_narrow, _ =
    compile_with_sources ~mode:Numerics.Bf16_narrow ~prec:Ir.Ops.bfloat16 ~variant:wide_variant
      ~ma:ma_pbf16 ~mb:mb_pbf16 ~mc:mc_pbf16 ~comp:comp_pbf16
      ?prebuilt:(if share_wide_sched then sched_pbf16 else None)
      ~source_prefix:proof_trt_name ~capture:true ()
  in
  Context.release ctx_pnbf16;
  if
    List.is_empty src_wbf16 || List.is_empty src_reach_narrow
    || sources_equal src_wbf16 src_reach_narrow
    || (uses_privatize wide_variant && Ir.Ops.equal_prec wide_bf16_acc narrow_bf16_acc)
  then (
    p "INVALID: bf16 policy did not reach codegen on fixed schedule %s (%d vs %d files)\n"
      wide_variant (List.length src_wbf16) (List.length src_reach_narrow);
    Stdlib.exit 1);
  p "treatment reach proof: bf16 sources differ on fixed schedule %s (%d wide, %d narrow file(s))\n"
    wide_variant (List.length src_wbf16) (List.length src_reach_narrow);

  (* Clean timed compiles: all four arms compile with identical default debug/compiler settings. *)
  let ctrl_name = run_prefix ^ "ctrl_f32" in
  let ma_f32, mb_f32, mc_f32, comp_f32 = build_problem ~name:ctrl_name ~prec:Ir.Ops.single in
  let ctx_wf32, r_wf32, _, sched_f32 =
    compile_with_sources ~mode:Numerics.Bf16_wide ~prec:Ir.Ops.single ~variant:control_variant
      ~ma:ma_f32 ~mb:mb_f32 ~mc:mc_f32 ~comp:comp_f32 ~source_prefix:ctrl_name ~capture:false ()
  in
  let ctx_nf32, r_nf32, _, _ =
    compile_with_sources ~mode:Numerics.Bf16_narrow ~prec:Ir.Ops.single ~variant:control_variant
      ~ma:ma_f32 ~mb:mb_f32 ~mc:mc_f32 ~comp:comp_f32 ?prebuilt:sched_f32 ~source_prefix:ctrl_name
      ~capture:false ()
  in
  let trt_name = run_prefix ^ "trt_bf16" in
  let ma_bf16, mb_bf16, mc_bf16, comp_bf16 = build_problem ~name:trt_name ~prec:Ir.Ops.bfloat16 in
  let ctx_wbf16, r_wbf16, _, sched_wbf16 =
    compile_with_sources ~mode:Numerics.Bf16_wide ~prec:Ir.Ops.bfloat16 ~variant:wide_variant
      ~ma:ma_bf16 ~mb:mb_bf16 ~mc:mc_bf16 ~comp:comp_bf16 ~source_prefix:trt_name ~capture:false ()
  in
  let ctx_nbf16, r_nbf16, _, _ =
    compile_with_sources ~mode:Numerics.Bf16_narrow ~prec:Ir.Ops.bfloat16 ~variant:narrow_variant
      ~ma:ma_bf16 ~mb:mb_bf16 ~mc:mc_bf16 ~comp:comp_bf16
      ?prebuilt:
        (if String.equal wide_variant narrow_variant && share_wide_sched then sched_wbf16 else None)
      ~source_prefix:trt_name ~capture:false ()
  in

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

  (* Independent unscheduled f32 GEMM oracle outside timing: proves both f32 drift controls computed
     the true exact GEMM answer rather than merely matching each other. *)
  let oracle_vals, exact_chk =
    set_arm ~mode:Numerics.Bf16_wide ~prec:Ir.Ops.single;
    let _, _, mc_ref, comp_ref =
      build_problem ~name:(run_prefix ^ "oracle_f32") ~prec:Ir.Ops.single
    in
    let ctx_ref = Context.auto () in
    let ctx_ref, r_ref = Context.compile ctx_ref comp_ref Ir.Indexing.Empty in
    let ctx_ref = Context.run ctx_ref r_ref in
    let vals = Context.get_values ctx_ref mc_ref in
    Context.release ctx_ref;
    (vals, Bench_checksum.whole_output ~row_stride:n vals)
  in

  (* Pre-timing cell-by-cell output validation + checksum fingerprints. *)
  let eval_arm ctx routine mc =
    let ctx = Context.run ctx routine in
    let vals = Context.get_values ctx mc in
    (ctx, vals, Bench_checksum.whole_output ~row_stride:n vals)
  in
  let ctx_wf32, vals_wf32_pre, _chk_wf32_pre = eval_arm ctx_wf32 r_wf32 mc_f32 in
  let ctx_nf32, vals_nf32_pre, _chk_nf32_pre = eval_arm ctx_nf32 r_nf32 mc_f32 in
  (match Bench_checksum.first_difference ~reference:oracle_vals vals_wf32_pre with
  | None -> ()
  | Some d ->
      p "INVALID: wide-f32 control disagrees with unscheduled oracle: %s\n"
        (Bench_checksum.render_agreement ~name:"oracle-f32" (Some d));
      Stdlib.exit 1);
  (match Bench_checksum.first_difference ~reference:oracle_vals vals_nf32_pre with
  | None -> ()
  | Some d ->
      p "INVALID: narrow-f32 control disagrees with unscheduled oracle: %s\n"
        (Bench_checksum.render_agreement ~name:"oracle-f32" (Some d));
      Stdlib.exit 1);
  let ctx_wbf16, vals_wbf16_pre, chk_wbf16_pre = eval_arm ctx_wbf16 r_wbf16 mc_bf16 in
  let ctx_nbf16, vals_nbf16_pre, chk_nbf16_pre = eval_arm ctx_nbf16 r_nbf16 mc_bf16 in
  let check_wide_cells ~reference vals =
    if Array.length reference <> Array.length vals then (
      p "INVALID: wide-bf16 length %d <> oracle %d\n" (Array.length vals) (Array.length reference);
      Stdlib.exit 1);
    (* Wide bf16 accumulates exact products in f32 and rounds once to bf16 (1 ulp <= 2^-7 ~ 0.0078);
       allow 2.5 ulps relative (or 0.05 absolute near zero) to catch any corrupted cell. *)
    let worst_rel = ref 0.0 in
    Array.iteri vals ~f:(fun t v ->
        let r = reference.(t) in
        let diff = Float.abs (v -. r) in
        let limit = Float.max (0.02 *. Float.abs r) 0.05 in
        if (not (Float.is_finite v)) || Float.(diff > limit) then (
          p
            "INVALID: wide-bf16 cell [%d] = %.10g outside bound vs oracle %.10g (|diff|=%.4g > %.4g)\n"
            t v r diff limit;
          Stdlib.exit 1);
        if Float.(Float.abs r > 0.0) then worst_rel := Float.max !worst_rel (diff /. Float.abs r));
    !worst_rel
  in
  let check_narrow_cells ~reference vals =
    if Array.length reference <> Array.length vals then (
      p "INVALID: narrow-bf16 length %d <> oracle %d\n" (Array.length vals) (Array.length reference);
      Stdlib.exit 1);
    (* Exact rounding-aware spot reference on 64 spread cells across the whole matrix: - Metal mma_*
       (simdgroup_bfloat8x8): accumulates each 8-deep k-block in f32 and rounds to bf16 once per
       block (gh-ocannl-2029). - Serial (parallel/smem/regtile): per-step bf16 FMA. *)
    let round_bf16 = Ir.Ops.round_to_prec Ir.Ops.bfloat16 in
    let spot_check ~block_size =
      let stride = Int.max 1 (n / 8) in
      for ii = 0 to 7 do
        let i = ((ii * stride) + (ii * 7)) % n in
        for jj = 0 to 7 do
          let j = ((jj * stride) + (jj * 11)) % n in
          let expected =
            if block_size = 1 then (
              let acc = ref 0.0 in
              for k = 0 to n - 1 do
                let a = round_bf16 mav.((i * n) + k) in
                let b = round_bf16 mbv.((k * n) + j) in
                acc := round_bf16 (!acc +. (a *. b))
              done;
              !acc)
            else
              let acc = ref 0.0 in
              let k0 = ref 0 in
              while !k0 < n do
                let blk = ref 0.0 in
                for dk = 0 to block_size - 1 do
                  let k = !k0 + dk in
                  let a = round_bf16 mav.((i * n) + k) in
                  let b = round_bf16 mbv.((k * n) + j) in
                  blk := !blk +. (a *. b)
                done;
                acc := round_bf16 (!acc +. !blk);
                k0 := !k0 + block_size
              done;
              !acc
          in
          let got = vals.((i * n) + j) in
          let diff = Float.abs (got -. expected) in
          (* Allow 2.5 bf16 ulps for FMA/block-order variations; a 2x-scaled or corrupted cell fails
             immediately. *)
          let limit = Float.max (0.02 *. Float.abs expected) 0.25 in
          if (not (Float.is_finite got)) || Float.(diff > limit) then (
            p
              "INVALID: narrow-bf16 spot cell (%d,%d) = %.10g departs from rounding-aware model \
               %.10g (|diff|=%.4g > %.4g)\n"
              i j got expected diff limit;
            Stdlib.exit 1)
        done
      done
    in
    if not (String.is_prefix narrow_variant ~prefix:"mma") then spot_check ~block_size:1
    else if String.is_substring backend ~substring:"metal" then spot_check ~block_size:8;
    (* Whole-output two-sided bounds across every cell: catches both zeroed/sign-flipped outputs and
       upward-scaled (e.g. 2x) wrong kernels. Error at cancelled cells scales with the matrix's
       partial-sum scale [max_ref], not just the cancelled cell's own magnitude. *)
    let max_ref = Array.fold reference ~init:0.0 ~f:(fun m r -> Float.max m (Float.abs r)) in
    let max_got = ref 0.0 and big = ref 0 and same_sign = ref 0 in
    let cell_rel_cap =
      if
        String.is_substring backend ~substring:"metal"
        && String.is_prefix narrow_variant ~prefix:"mma"
      then 0.15
      else 0.35
    in
    let upper_abs = Float.max (1.25 *. max_ref) 1.0 +. 16.0 in
    Array.iteri vals ~f:(fun t v ->
        if not (Float.is_finite v) then (
          p "INVALID: narrow-bf16 cell [%d] is non-finite (%g)\n" t v;
          Stdlib.exit 1);
        let r = reference.(t) in
        if Float.(Float.abs v > upper_abs) then (
          p
            "INVALID: narrow-bf16 cell [%d] = %.10g exceeds two-sided upper bound %.10g (ref=%.10g)\n"
            t v upper_abs r;
          Stdlib.exit 1);
        (if
           (not (String.is_prefix narrow_variant ~prefix:"mma"))
           || String.is_substring backend ~substring:"metal"
         then
           let diff = Float.abs (v -. r) in
           let scale = Float.max (Float.abs r) (0.25 *. max_ref) in
           let lim = Float.max (cell_rel_cap *. scale) 16.0 in
           if Float.(diff > lim) then (
             p
               "INVALID: narrow-bf16 cell [%d] = %.10g departs from oracle %.10g (|diff|=%.4g > \
                %.4g)\n"
               t v r diff lim;
             Stdlib.exit 1));
        max_got := Float.max !max_got (Float.abs v);
        if Float.(Float.abs r > 1.0) then (
          Int.incr big;
          if Float.(v *. r > 0.0) then Int.incr same_sign));
    if
      Float.(!max_got < 0.1 *. max_ref || !max_got > 1.25 *. max_ref)
      || (!big > 0 && Float.(Float.of_int !same_sign < 0.5 *. Float.of_int !big))
    then (
      p
        "INVALID: narrow-bf16 output outside two-sided envelope (max |got|=%.4g vs max |ref|=%.4g, \
         sign-match %d/%d)\n"
        !max_got max_ref !same_sign !big;
      Stdlib.exit 1)
  in
  let wide_max_cell_rel = check_wide_cells ~reference:oracle_vals vals_wbf16_pre in
  check_narrow_cells ~reference:oracle_vals vals_nbf16_pre;
  let rel_bound = 0.25 in
  let err_wbf16 = rel_err chk_wbf16_pre exact_chk in
  let err_nbf16 = rel_err chk_nbf16_pre exact_chk in
  if Float.(err_wbf16 > rel_bound) then (
    p "INVALID: wide-bf16 checksum off exact f32 by %.4g (bound %.2f)\n" err_wbf16 rel_bound;
    Stdlib.exit 1);
  if not (non_degenerate chk_nbf16_pre exact_chk) then (
    p "INVALID: narrow-bf16 checksum %s degenerate against exact %s\n"
      (Bench_checksum.render chk_nbf16_pre)
      (Bench_checksum.render exact_chk);
    Stdlib.exit 1);
  p
    "checksums (pre): exact-f32=%s wide-bf16=%s (err=%.3g, max-cell-rel=%.3g) narrow-bf16=%s \
     (err=%.3g)\n"
    (Bench_checksum.render exact_chk)
    (Bench_checksum.render chk_wbf16_pre)
    err_wbf16 wide_max_cell_rel
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

  (* Post-timing cell-by-cell determinism check across all 4 arms. *)
  let ctx_wf32, vals_wf32_post, _ = eval_arm ctx_wf32 r_wf32 mc_f32 in
  let ctx_nf32, vals_nf32_post, _ = eval_arm ctx_nf32 r_nf32 mc_f32 in
  let ctx_wbf16, vals_wbf16_post, _ = eval_arm ctx_wbf16 r_wbf16 mc_bf16 in
  let ctx_nbf16, vals_nbf16_post, _ = eval_arm ctx_nbf16 r_nbf16 mc_bf16 in
  let check_deterministic ~arm pre post =
    match Bench_checksum.first_difference ~reference:pre post with
    | None -> ()
    | Some d ->
        p "INVALID: %s non-deterministic post-timing: %s\n" arm
          (Bench_checksum.render_agreement ~name:"pre-timing" (Some d));
        Stdlib.exit 1
  in
  check_deterministic ~arm:"wide-f32" vals_wf32_pre vals_wf32_post;
  check_deterministic ~arm:"narrow-f32" vals_nf32_pre vals_nf32_post;
  check_deterministic ~arm:"wide-bf16" vals_wbf16_pre vals_wbf16_post;
  check_deterministic ~arm:"narrow-bf16" vals_nbf16_pre vals_nbf16_post;
  Context.release ctx_wf32;
  Context.release ctx_nf32;
  Context.release ctx_wbf16;
  Context.release ctx_nbf16;
  p "outputs (post): cell-for-cell deterministic across all 4 arms\n";

  let s_wbf16 = Bench_retime.spread_of r_wbf16_ms in
  let s_nbf16 = Bench_retime.spread_of r_nbf16_ms in
  let s_wf32 = Bench_retime.spread_of r_wf32_ms in
  let s_nf32 = Bench_retime.spread_of r_nf32_ms in
  (* Compute treatment and f32 control ratios from the EXACT same complete mirrored round pairs (2k,
     2k+1) where all four arms were admitted, so the control observes every round that enters the
     treatment ratio and position balance is preserved. *)
  let complete_round i =
    Option.is_some r_wbf16_ms.(i)
    && Option.is_some r_nbf16_ms.(i)
    && Option.is_some r_wf32_ms.(i)
    && Option.is_some r_nf32_ms.(i)
  in
  let keep_round = Array.create ~len:calls false in
  for pair = 0 to (calls / 2) - 1 do
    let a = 2 * pair and b = (2 * pair) + 1 in
    if complete_round a && complete_round b then (
      keep_round.(a) <- true;
      keep_round.(b) <- true)
  done;
  let mask arr = Array.mapi arr ~f:(fun i v -> if keep_round.(i) then v else None) in
  let ratio_bf16 = Bench_retime.paired_ratio ~num:(mask r_wbf16_ms) ~den:(mask r_nbf16_ms) in
  let ratio_f32 = Bench_retime.paired_ratio ~num:(mask r_wf32_ms) ~den:(mask r_nf32_ms) in
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
