(* gh-ocannl-2251: compile-only witness for the rendered uniform-f16 fragment scope, including its
   converted d boundary. Context.compile renders on the attached CUDA device, but no routine is run.
   Recompile that fresh source explicitly for compute_75, and compute_70 only when NVRTC accepts
   that target. This is instruction/toolchain evidence, not execution on Volta or Turing. The bf16
   control uses the arm's shared emitter without the attached device's capability gate, so an
   sm_70-79 device can still compile the positive compute_80 control without loading it. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module S = Ir.Schedule
module M = Ir.Cuda_mma
module G = Test_utils.Generated
module Claims = Cuda_wmma_floor_claims

let render ~name ~ab_prec ~d_prec =
  let a = NTDSL.init ~l:"floor_a" ~prec:ab_prec ~i:[ 32 ] ~o:[ 32 ] ~f:(fun _ -> 1.) () in
  let b = NTDSL.init ~l:"floor_b" ~prec:ab_prec ~i:[ 32 ] ~o:[ 32 ] ~f:(fun _ -> 1.) () in
  let%op y = a * b in
  Ir.Tnode.update_prec y.Tensor.value d_prec;
  let transform (opt : LL.optimized) =
    let i, j, k =
      match List.find_exn (Ll_test.nest_paths opt.llc) ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let split, ko, ki = S.split ~axis:k ~factor:16 ~outer:LL.Serial ~inner:LL.Serial in
    let rows, _, ii = S.split ~axis:i ~factor:16 ~outer:LL.Grid ~inner:LL.Serial in
    let tensorize, _ = S.tensorize ~i:ii ~j ~k:ki ~simd_width:32 () in
    let zero, zs = S.expand_zero ~tn:y.Tensor.value in
    let zi, zj = match zs with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let zr, _, _ = S.split ~axis:zi ~factor:16 ~outer:LL.Grid ~inner:LL.Serial in
    let zc, _, _ = S.split ~axis:zj ~factor:32 ~outer:LL.Serial ~inner:LL.Workgroup in
    [
      S.apply
        [
          zero;
          zr;
          zc;
          rows;
          split;
          S.Swap { outer = j; inner = ko };
          S.Swap { outer = ii; inner = ko };
          tensorize;
        ]
        opt;
    ]
  in
  ignore
    (Context.compile ~name ~lowered_transform:transform (Context.auto ()) (Train.forward y)
       Ir.Indexing.Empty);
  G.read name

let compile arch source =
  Nvrtc.compile_to_ptx ~cu_src:source ~name:"cuda_wmma_floor.cu"
    ~options:[ "--gpu-architecture=compute_" ^ Int.to_string arch ]
    ~with_debug:true
  |> Nvrtc.string_from_ptx

let has source substring = String.is_substring source ~substring

let check_ptx arch source =
  let ptx = compile arch source in
  (* The compiler release and exact target/instruction go to stderr, keeping the golden portable
     across toolkits. *)
  String.split_lines ptx
  |> List.filter ~f:(fun line ->
      has line "release" || has line ".target" || has line "wmma.mma.sync")
  |> List.iter ~f:(Stdio.eprintf "%s (not part of the golden)\n");
  has ptx (".target sm_" ^ Int.to_string arch)
  && has ptx "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32"

let bf16_refused () =
  let kind = M.Wmma_bf16 in
  let bf16 =
    String.concat ~sep:"\n"
      ([
         "#include <mma.h>";
         "#include <cuda_bf16.h>";
         "extern \"C\" __global__ void bf16_control(const __nv_bfloat16 *__mma_ap, const \
          __nv_bfloat16 *__mma_bp, float *d) {";
         M.wmma_frag_typ kind "accumulator" "float" None ^ " __mma_acc[1][1];";
         "nvcuda::wmma::fill_fragment(__mma_acc[0][0], 0.0f);";
       ]
      @ M.wmma_k_loop_lines kind ~acc:"__mma_acc" ~ta:false ~tb:false ~lda:16 ~ldb:16 ~m:16 ~n:16
          ~k:16
      @ [
          "nvcuda::wmma::store_matrix_sync(d, __mma_acc[0][0], 16, nvcuda::wmma::mem_row_major);";
          "}";
        ])
  in
  ignore (compile 80 bf16);
  match compile 75 bf16 with
  | _ -> false
  | exception Nvrtc.Nvrtc_error { status; message } ->
      Stdio.eprintf "bf16 control refusal: %s (not part of the golden)\n" message;
      String.equal (Sexp.to_string (Nvrtc.sexp_of_result status)) "NVRTC_ERROR_COMPILATION"

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  G.init ~backend_name:"cuda";
  Test_utils.with_policy
    (fun p -> { p with fp16_arithmetic = Ir.Numerics.Fp16_wide })
    (fun () ->
      let source = lazy (render ~name:"wmma_floor_half" ~ab_prec:Ir.Ops.half ~d_prec:Ir.Ops.half) in
      let accepts_70 =
        match compile 70 "extern \"C\" __global__ void target_probe() {}" with
        | _ -> true
        | exception Nvrtc.Nvrtc_error { status; message }
          when String.equal
                 (Sexp.to_string (Nvrtc.sexp_of_result status))
                 "NVRTC_ERROR_INVALID_OPTION"
               && has message "invalid value for --gpu-architecture" ->
            Stdio.eprintf "compute_70 unavailable in installed NVRTC: %s (not part of the golden)\n"
              message;
            false
      in
      let leg = Verdict.leg ~when_:true ~on:"NVRTC bindings unavailable" () in
      let compute_70_leg =
        Verdict.leg ~aggregation:`Environment ~when_:accepts_70
          ~on:"NVRTC without compute_70 support" ()
      in
      Claims.run ~leg ~compute_70_leg (function
        | Claims.Wide_fragment ->
            let logic = G.main_logic (Lazy.force source) in
            has logic "(wmma-f16-wide)"
            && has logic "wmma fragment reduction body begins"
            && has logic "__half2float" && has logic "__float2half"
        | Compute_75 -> check_ptx 75 (Lazy.force source)
        | Compute_70 -> check_ptx 70 (Lazy.force source)
        | Shipping_floor ->
            List.equal String.equal
              (M.gpu_arch_options ~target:M.Floor ~device_cc:120 (Lazy.force source))
              (M.gpu_arch_options ~target:M.Floor ~device_cc:120 "nvcuda::wmma")
        | Async_floor ->
            List.equal String.equal
              (M.gpu_arch_options ~target:M.Floor ~device_cc:120
                 (Lazy.force source ^ "\n/* cp.async. */"))
              (M.gpu_arch_options ~target:M.Floor ~device_cc:120 "cp.async.")
        | Bf16_refusal -> bf16_refused ()))
