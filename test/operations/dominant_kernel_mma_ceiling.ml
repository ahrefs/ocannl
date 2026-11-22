(* gh-ocannl-1092: per-format advisory MMA peak ceilings for the benchmark report's [kernel %peak]
   column, with provenance in [hardware_limits], Metal's shader-ALU scalar-ceiling rule, config
   overrides ([model_peak_mma_*]), and an end-to-end GPU execution leg. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module BI = Ir.Backend_intf
module CS = Ir.C_syntax
module H = Bench_harness
module U = Yojson.Safe.Util

let override_probe_arg = "--override-probe"

let cuda_limits ~cc =
  let peaks = Ir.Cuda_mma.mma_peaks ~cc in
  {
    BI.no_hardware_limits with
    mma = Ir.Cuda_mma.capability ~cc;
    peak_flops = Some 1.5e13;
    peak_memory_bandwidth = Some 8.0e12;
    mma_peak_f16 = peaks.f16;
    mma_peak_bf16 = peaks.bf16;
    mma_peak_tf32 = peaks.tf32;
    mma_peak_fp8 = peaks.fp8;
    mma_uses_scalar_peak = false;
  }

let hip_rdna3_limits =
  {
    BI.no_hardware_limits with
    mma = Some Ir.Hip_mma.capability;
    peak_flops = Some 1.0e13;
    peak_memory_bandwidth = Some 6.0e12;
    mma_peak_f16 = Some Ir.Hip_mma.mma_peak_f16;
    mma_peak_bf16 = Some Ir.Hip_mma.mma_peak_bf16;
    mma_uses_scalar_peak = false;
  }

let metal_limits =
  {
    BI.no_hardware_limits with
    mma = Some Ir.Metal_mma.capability;
    peak_flops = Some 5.0e12;
    peak_memory_bandwidth = Some 1.0e12;
    mma_uses_scalar_peak = true;
  }

let summary_of emitted_formats = CS.summarize_census ~emitted_formats [ ("k0", CS.Mma_intrinsics) ]

let emit ?(tf32 = false) a b d =
  { CS.mma_a_prec = a; mma_b_prec = b; mma_d_prec = d; mma_tf32 = tf32 }

(* Child-process mode: launched with explicit [--ocannl_model_peak_mma_f16] and
   [--ocannl_model_peak_flops] flags so the lazy config readers resolve those overrides in a clean
   process without touching any GPU. *)
let () =
  if Array.exists Stdlib.Sys.argv ~f:(String.equal override_probe_arg) then (
    let cuda = cuda_limits ~cc:120 in
    let ok_cuda_override =
      match Autotune.mma_peak_leg ~limits:cuda BI.Mma_f16 with
      | Some (v, Autotune.Mma_peak_config "model_peak_mma_f16") -> Float.equal v 8.79e13
      | _ -> false
    in
    let ok_cuda_untouched =
      match Autotune.mma_peak_leg ~limits:cuda BI.Mma_fp8_e5m2 with
      | Some (v, Autotune.Mma_peak_backend (Datasheet_class _)) -> Float.equal v 6.0e13
      | _ -> false
    in
    let ok_metal_explicit =
      match Autotune.mma_peak_leg ~limits:metal_limits BI.Mma_f16 with
      | Some (v, Autotune.Mma_peak_config "model_peak_mma_f16") -> Float.equal v 8.79e13
      | _ -> false
    in
    let ok_metal_scalar_override =
      match Autotune.mma_peak_leg ~limits:metal_limits BI.Mma_f32 with
      | Some (v, Autotune.Mma_peak_scalar_alu `Config) -> Float.equal v 1.7e13
      | _ -> false
    in
    let ok_harness_provenance =
      match
        ( H.mma_ceiling_of ~backend:"cuda" ~limits:cuda
            (summary_of [ emit Ir.Ops.half Ir.Ops.half Ir.Ops.single ]),
          H.mma_ceiling_of ~backend:"metal" ~limits:metal_limits
            (summary_of [ emit Ir.Ops.single Ir.Ops.single Ir.Ops.single ]) )
      with
      | ( Bench_json.Mma_peak { format = "f16"; flops = f1; source = "model_peak_mma_f16 config" },
          Bench_json.Mma_peak
            {
              format = "f32";
              flops = f2;
              source = "model_peak_flops config (mma shares shader ALU)";
            } ) ->
          Float.equal f1 8.79e13 && Float.equal f2 1.7e13
      | _ -> false
    in
    if
      ok_cuda_override && ok_cuda_untouched && ok_metal_explicit && ok_metal_scalar_override
      && ok_harness_provenance
    then Stdlib.exit 0
    else
      let failed =
        List.filter_map
          [
            ("cuda_override", ok_cuda_override);
            ("cuda_untouched", ok_cuda_untouched);
            ("metal_explicit", ok_metal_explicit);
            ("metal_scalar_override", ok_metal_scalar_override);
            ("harness_provenance", ok_harness_provenance);
          ]
          ~f:(fun (name, ok) -> if ok then None else Some name)
      in
      Stdio.eprintf "override probe failed (%s)\n%!" (String.concat ~sep:", " failed);
      Stdlib.exit 1)

(* Part 1: pure backend tables and ceiling resolution, no GPU needed. *)
let () =
  let check_datasheet = function
    | Some { BI.mma_peak_flops; mma_peak_provenance = Datasheet_class cite } ->
        Float.(mma_peak_flops > 0.) && not (String.is_empty cite)
    | _ -> false
  in
  let sm60 = Ir.Cuda_mma.mma_peaks ~cc:60
  and sm70 = Ir.Cuda_mma.mma_peaks ~cc:70
  and sm80 = Ir.Cuda_mma.mma_peaks ~cc:80
  and sm89 = Ir.Cuda_mma.mma_peaks ~cc:89
  and sm120 = Ir.Cuda_mma.mma_peaks ~cc:120 in
  p "CUDA pre-Volta (sm_60) advertises no tensor-core peaks"
    (Option.is_none sm60.f16 && Option.is_none sm60.bf16 && Option.is_none sm60.tf32
   && Option.is_none sm60.fp8);
  p "CUDA Volta (sm_70) advertises only f16 with datasheet provenance"
    (check_datasheet sm70.f16 && Option.is_none sm70.bf16 && Option.is_none sm70.tf32
   && Option.is_none sm70.fp8);
  p "CUDA Ampere (sm_80) adds bf16 and tf32 with datasheet provenance, not fp8"
    (check_datasheet sm80.f16 && check_datasheet sm80.bf16 && check_datasheet sm80.tf32
   && Option.is_none sm80.fp8);
  p_all "CUDA Ada/Blackwell (sm_89, sm_120) advertise all four formats with datasheet provenance"
    [ sm89; sm120 ] ~f:(fun pk ->
      check_datasheet pk.f16 && check_datasheet pk.bf16 && check_datasheet pk.tf32
      && check_datasheet pk.fp8);
  p "HIP RDNA3 advertises f16 and bf16 WMMA peaks with datasheet provenance"
    (check_datasheet (Some Ir.Hip_mma.mma_peak_f16)
    && check_datasheet (Some Ir.Hip_mma.mma_peak_bf16));

  let cuda120 = cuda_limits ~cc:120 in
  let cuda_cases =
    [
      ("f16", emit Ir.Ops.half Ir.Ops.half Ir.Ops.single, 3.0e13);
      ("bf16", emit Ir.Ops.bfloat16 Ir.Ops.bfloat16 Ir.Ops.single, 3.0e13);
      ("tf32", emit ~tf32:true Ir.Ops.single Ir.Ops.single Ir.Ops.single, 1.5e13);
      ("fp8", emit Ir.Ops.fp8 Ir.Ops.fp8 Ir.Ops.single, 6.0e13);
    ]
  in
  p_all "CUDA sm_120 resolves each emitted MMA format to its datasheet peak" cuda_cases
    ~f:(fun (expected_fmt, emitted, expected_flops) ->
      match H.mma_ceiling_of ~backend:"cuda" ~limits:cuda120 (summary_of [ emitted ]) with
      | Bench_json.Mma_peak { format; flops; source } ->
          String.equal format expected_fmt && Float.equal flops expected_flops
          && String.is_substring source ~substring:"cuda datasheet ("
      | Bench_json.Mma_no_peak _ -> false);

  let cuda75 = cuda_limits ~cc:75 in
  p "CUDA sm_75 refuses fp8 (no peak for that architecture) while keeping f16"
    ((match
        H.mma_ceiling_of ~backend:"cuda" ~limits:cuda75
          (summary_of [ emit Ir.Ops.half Ir.Ops.half Ir.Ops.single ])
      with
       | Bench_json.Mma_peak { format = "f16"; flops; _ } -> Float.equal flops 3.0e13
       | _ -> false)
    &&
    match
      H.mma_ceiling_of ~backend:"cuda" ~limits:cuda75
        (summary_of [ emit Ir.Ops.fp8 Ir.Ops.fp8 Ir.Ops.single ])
    with
    | Bench_json.Mma_no_peak _ -> true
    | Bench_json.Mma_peak _ -> false);

  let hip_cases =
    [
      ("f16", emit Ir.Ops.half Ir.Ops.half Ir.Ops.single, 4.0e13);
      ("bf16", emit Ir.Ops.bfloat16 Ir.Ops.bfloat16 Ir.Ops.single, 4.0e13);
    ]
  in
  p_all "HIP RDNA3 resolves f16 and bf16 WMMA to their datasheet peaks" hip_cases
    ~f:(fun (expected_fmt, emitted, expected_flops) ->
      match H.mma_ceiling_of ~backend:"hip" ~limits:hip_rdna3_limits (summary_of [ emitted ]) with
      | Bench_json.Mma_peak { format; flops; source } ->
          String.equal format expected_fmt && Float.equal flops expected_flops
          && String.is_substring source ~substring:"hip datasheet ("
      | Bench_json.Mma_no_peak _ -> false);

  p "HIP without mma capability (CDNA) fails closed to no-ceiling"
    (match
       H.mma_ceiling_of ~backend:"hip" ~limits:BI.no_hardware_limits
         (summary_of [ emit Ir.Ops.half Ir.Ops.half Ir.Ops.single ])
     with
    | Bench_json.Mma_no_peak _ -> true
    | Bench_json.Mma_peak _ -> false);

  let metal_cases =
    [
      ("f32", emit Ir.Ops.single Ir.Ops.single Ir.Ops.single);
      ("f16", emit Ir.Ops.half Ir.Ops.half Ir.Ops.half);
      ("bf16", emit Ir.Ops.bfloat16 Ir.Ops.bfloat16 Ir.Ops.single);
    ]
  in
  p_all "Metal maps f32, f16 and bf16 simdgroup_matrix kernels to the scalar ceiling" metal_cases
    ~f:(fun (expected_fmt, emitted) ->
      match H.mma_ceiling_of ~backend:"metal" ~limits:metal_limits (summary_of [ emitted ]) with
      | Bench_json.Mma_peak { format; flops; source } ->
          String.equal format expected_fmt && Float.equal flops 5.0e12
          && String.equal source "metal class constant (mma shares shader ALU)"
      | Bench_json.Mma_no_peak _ -> false);

  p "a kernel mixing two distinct MMA formats fails closed to no-ceiling"
    (match
       H.mma_ceiling_of ~backend:"cuda" ~limits:cuda120
         (summary_of
            [
              emit Ir.Ops.half Ir.Ops.half Ir.Ops.single;
              emit Ir.Ops.bfloat16 Ir.Ops.bfloat16 Ir.Ops.single;
            ])
     with
    | Bench_json.Mma_no_peak reason -> String.is_substring reason ~substring:"mixes MMA formats"
    | Bench_json.Mma_peak _ -> false);

  p "a tensorized summary with no emitted formats fails closed to no-ceiling"
    (match H.mma_ceiling_of ~backend:"cuda" ~limits:cuda120 (summary_of []) with
    | Bench_json.Mma_no_peak reason -> String.is_substring reason ~substring:"no emitted MMA format"
    | Bench_json.Mma_peak _ -> false);

  p "choose_ceiling still refuses a GPU tensor-core kernel when bandwidth is absent"
    (match
       Bench_json.choose_ceiling
         ~mma_ceiling:
           (Some (Bench_json.Mma_peak { format = "f16"; flops = 3.0e13; source = "cuda datasheet" }))
         ~gpu:true ~tensorization:"tensorized" ~narrow_native:false
         ~peak_flops:(Some (1.5e13, "cuda class constant"))
         ~peak_memory_bandwidth:None
     with
    | Error reason -> String.is_substring reason ~substring:"peak_memory_bandwidth"
    | Ok _ -> false)

(* Part 2: config overrides in a clean subprocess. *)
let () =
  let exe = Stdlib.Sys.executable_name in
  let peak_env_prefixes =
    [
      "OCANNL_MODEL_PEAK_FLOPS=";
      "OCANNL_MODEL_PEAK_MMA_F16=";
      "OCANNL_MODEL_PEAK_MMA_BF16=";
      "OCANNL_MODEL_PEAK_MMA_TF32=";
      "OCANNL_MODEL_PEAK_MMA_FP8=";
    ]
  in
  let env =
    Array.filter (Unix.environment ()) ~f:(fun binding ->
        not (List.exists peak_env_prefixes ~f:(fun prefix -> String.is_prefix binding ~prefix)))
  in
  let pid =
    Unix.create_process_env exe
      [|
        exe;
        "--ocannl_model_peak_mma_f16=8.79e13";
        "--ocannl_model_peak_flops=1.7e13";
        override_probe_arg;
      |]
      env Unix.stdin Unix.stdout Unix.stderr
  in
  let _, status = Unix.waitpid [] pid in
  p "model_peak_mma_* overrides backend constants and Metal scalar-ALU follows model_peak_flops"
    (match status with Unix.WEXITED 0 -> true | _ -> false)

(* Part 3: live GPU execution leg — compiles a real tensorized matmul on the active GPU backend
   (Metal on mac-studio, CUDA/HIP on GPU boxes) and runs [Bench_harness.dominant_kernel] on it. *)
let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let is_gpu = BI.binds_hardware_grid (Context.backend_facts_of_name backend) in
  let gpu_with_mma = is_gpu && Option.is_some limits.BI.mma in
  let leg =
    Verdict.leg ~aggregation:`Backend ~when_:gpu_with_mma
      ~on:(backend ^ " (no GPU tensor-core unit)")
      ()
  in
  let run_tensorized_matmul ~prec_name ~a_prec ~d_prec =
    (* 256x256x256 puts arithmetic intensity above the ridge point on Metal, CUDA and HIP so compute
       binds (flops_exact = true) even when the lower-level byte count is an upper bound. *)
    let m = 256 and n = 256 and k = 256 in
    let a =
      Operation.init ~l:("dk_mma_a_" ^ prec_name) ~prec:a_prec ~b:[] ~i:[] ~o:[ m; k ]
        ~grad_spec:Tensor.Prohibit_grad
        ~f:(Ll_test.cycle ~radix:3 ~dims:[| m; k |] ~modulus:7 ~offset:1.0 ~stride:0.25)
        ()
    in
    let b =
      Operation.init ~l:("dk_mma_b_" ^ prec_name) ~prec:a_prec ~b:[] ~i:[] ~o:[ k; n ]
        ~grad_spec:Tensor.Prohibit_grad
        ~f:(Ll_test.cycle ~radix:3 ~dims:[| k; n |] ~modulus:5 ~offset:1.0 ~stride:0.25)
        ()
    in
    let c =
      Operation.einsum
        ~label:[ "dk_mma_c_" ^ prec_name ]
        ~grad_spec:Tensor.Prohibit_grad "mk;kn=>mn" a b ()
    in
    Ir.Tnode.update_prec c.Tensor.value d_prec;
    let comp = Train.forward c in
    let sched_applied = ref false in
    let lowered_transform (opt : Ir.Low_level.optimized) =
      let seeds =
        Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt
        |> List.filter ~f:(fun (s : Autotune.sketch_params) -> s.sk_mma)
      in
      let try_seed (s : Autotune.sketch_params) =
        let copy =
          {
            opt with
            Ir.Low_level.traced_store = Hashtbl.copy opt.Ir.Low_level.traced_store;
            optimize_ctx = Ir.Low_level.copy_optimize_ctx opt.Ir.Low_level.optimize_ctx;
          }
        in
        try
          let sched = Autotune.sketch_schedule ~accum_prec:Fn.id ~p:s copy in
          let out = Ir.Schedule.apply sched copy in
          Some out
        with _ -> None
      in
      match List.find_map seeds ~f:try_seed with
      | Some scheduled ->
          sched_applied := true;
          [ scheduled ]
      | None -> [ opt ]
    in
    let ctx, routine = Context.compile ~lowered_transform ctx comp Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    let dk_str = H.dominant_kernel ~calls:2 ~ctx ~bindings:Ir.Indexing.Empty [ routine ] in
    Stdio.eprintf "dominant_kernel_mma_ceiling (%s %s, not part of the golden): %s\n%!" backend
      prec_name dk_str;
    let dk = Yojson.Safe.from_string dk_str in
    (!sched_applied, routine.Context.mma, dk)
  in
  let live =
    lazy
      (let applied, mma, dk =
         run_tensorized_matmul ~prec_name:"f16" ~a_prec:Ir.Ops.half ~d_prec:Ir.Ops.single
       in
       (applied, mma, dk))
  in
  in_leg ~leg "a live GPU f16 matmul compiles tensorized with emitted MMA format recorded"
    (fun label ->
      let applied, mma, _ = Lazy.force live in
      p label
        (applied
        && CS.equal_tensorization mma.CS.tensorization CS.Tensorized
        && not (List.is_empty mma.CS.emitted_formats)));
  in_leg ~leg "Bench_harness.mma_ceiling_of resolves the live GPU f16 matmul to mma-f16"
    (fun label ->
      let _, mma, _ = Lazy.force live in
      p label
        (match H.mma_ceiling_of ~backend ~limits mma with
        | Bench_json.Mma_peak { format = "f16"; flops; source } ->
            Float.(flops > 0.) && not (String.is_empty source)
        | _ -> false));
  in_leg ~leg "dominant_kernel scores the live GPU f16 tensorized matmul as exact mma-f16"
    (fun label ->
      let _, _, dk = Lazy.force live in
      let verdict = U.member "verdict" dk in
      let all_refused =
        Yojson.Safe.equal verdict (`String "no-kernel")
        &&
        match U.member "note" dk with
        | `String note -> String.is_substring note ~substring:": 0 declined to compile alone"
        | _ -> false
      in
      if all_refused then skipped ~aggregation:`Environment ~backend label
      else
        let pct_positive =
          match U.member "pct_of_peak" dk with
          | `Float f -> Float.(is_finite f && f > 0.)
          | `Int i -> i > 0
          | _ -> false
        in
        p label
          (Yojson.Safe.equal verdict (`String "exact")
          && Yojson.Safe.equal (U.member "ceiling" dk) (`String "mma-f16")
          && pct_positive));
  let metal_f32_leg =
    Verdict.leg ~aggregation:`Backend
      ~when_:(String.equal backend "metal" && gpu_with_mma)
      ~on:(backend ^ " (not Metal)") ()
  in
  in_leg ~leg:metal_f32_leg
    "Metal f32 simdgroup_matrix matmul scores as exact mma-f32 against the scalar ALU ceiling"
    (fun label ->
      let applied, mma, dk =
        run_tensorized_matmul ~prec_name:"f32" ~a_prec:Ir.Ops.single ~d_prec:Ir.Ops.single
      in
      let verdict = U.member "verdict" dk in
      let all_refused =
        Yojson.Safe.equal verdict (`String "no-kernel")
        &&
        match U.member "note" dk with
        | `String note -> String.is_substring note ~substring:": 0 declined to compile alone"
        | _ -> false
      in
      if all_refused then skipped ~aggregation:`Environment ~backend label
      else
        let pct_positive =
          match U.member "pct_of_peak" dk with
          | `Float f -> Float.(is_finite f && f > 0.)
          | `Int i -> i > 0
          | _ -> false
        in
        let source_ok =
          match U.member "ceiling_source" dk with
          | `String s -> String.is_substring s ~substring:"mma shares shader ALU"
          | _ -> false
        in
        p label
          (applied
          && CS.equal_tensorization mma.CS.tensorization CS.Tensorized
          && Yojson.Safe.equal verdict (`String "exact")
          && Yojson.Safe.equal (U.member "ceiling" dk) (`String "mma-f32")
          && source_ok && pct_positive))
