(* gh-ocannl-2251: compile-only witness for the rendered uniform-f16 fragment scope, including its
   converted d boundary. Context.compile renders on the attached CUDA device, but no routine is run.
   Recompile that fresh source explicitly for compute_75, and compute_70 only when NVRTC accepts
   that target. This is instruction/toolchain evidence, not execution on Volta or Turing. The bf16
   control uses the arm's shared emitter without the attached device's capability gate, so an
   sm_70-79 device can still compile the positive compute_80 control without loading it. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module S = Ir.Schedule
module M = Ir.Cuda_mma
module G = Test_utils.Generated

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

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  G.init ~backend_name:"cuda";
  Test_utils.with_policy
    (fun p -> { p with fp16_arithmetic = Ir.Numerics.Fp16_wide })
    (fun () ->
      let source = render ~name:"wmma_floor_half" ~ab_prec:Ir.Ops.half ~d_prec:Ir.Ops.half in
      let logic = G.main_logic source in
      p "the witness renders the wide fragment scope and both destination conversions"
        (has logic "(wmma-f16-wide)"
        && has logic "wmma fragment reduction body begins"
        && has logic "__half2float" && has logic "__float2half");
      p "compute_75 PTX targets sm_75 and accumulates WMMA f16 products in f32"
        (check_ptx 75 source);
      p "compute_70 has the same f32 WMMA instruction when NVRTC accepts the target"
        (match compile 70 "extern \"C\" __global__ void target_probe() {}" with
        | _ -> check_ptx 70 source
        | exception Nvrtc.Nvrtc_error { status; message }
          when String.equal
                 (Sexp.to_string (Nvrtc.sexp_of_result status))
                 "NVRTC_ERROR_INVALID_OPTION"
               && has message "invalid value for --gpu-architecture" ->
            Stdio.eprintf "compute_70 unavailable in installed NVRTC: %s (not part of the golden)\n"
              message;
            true);
      let shipping = M.gpu_arch_options ~target:M.Floor ~device_cc:120 source in
      p "the wide fragment alone ships at the toolkit-adjusted f16 floor"
        (List.equal String.equal shipping
           (M.gpu_arch_options ~target:M.Floor ~device_cc:120 "nvcuda::wmma"));
      p "cp.async still raises a wide-f16 kernel to the async-copy floor"
        (List.equal String.equal
           (M.gpu_arch_options ~target:M.Floor ~device_cc:120 (source ^ "\n/* cp.async. */"))
           (M.gpu_arch_options ~target:M.Floor ~device_cc:120 "cp.async."));
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
          @ M.wmma_k_loop_lines kind ~acc:"__mma_acc" ~ta:false ~tb:false ~lda:16 ~ldb:16 ~m:16
              ~n:16 ~k:16
          @ [
              "nvcuda::wmma::store_matrix_sync(d, __mma_acc[0][0], 16, \
               nvcuda::wmma::mem_row_major);";
              "}";
            ])
      in
      ignore (compile 80 bf16);
      p "compute_75 refuses the otherwise compilable WMMA bf16 control"
        (match compile 75 bf16 with
        | _ -> false
        | exception Nvrtc.Nvrtc_error { status; message } ->
            Stdio.eprintf "bf16 control refusal: %s (not part of the golden)\n" message;
            String.equal (Sexp.to_string (Nvrtc.sexp_of_result status)) "NVRTC_ERROR_COMPILATION"))
