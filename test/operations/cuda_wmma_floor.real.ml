(* gh-ocannl-2251: compile-only witness for the rendered uniform-f16 fragment scope, including its
   converted d boundary. Context.compile renders on the attached CUDA device, but no routine is run.
   Recompile that fresh source explicitly for compute_75, and compute_70 only when NVRTC accepts
   that target. This is instruction/toolchain evidence, not execution on Volta or Turing. The bf16
   control uses the arm's shared emitter without the attached device's capability gate, so an
   sm_70-79 device can still compile the positive compute_80 control without loading it.

   gh-ocannl-2284: the per-statement uniform-f16 statement a Turing device renders, m16n8k8, through
   the backend under [Cuda_mma.with_resolution_ceiling] (the hooks resolve as an sm_75 device would;
   acceptance stays the attached device's). NVRTC does not assemble inline [asm]: it emits an
   [mma.sync] of any shape into PTX for any target, so a target's refusal is the PTX assembler's.
   The witness asks the driver's (the JIT behind [cuModuleLoadDataEx], which checks each instruction
   against the module's [.target]) to load the PTX, with no kernel launched: the m16n8k8 witness at
   sm_75 and retargeted to sm_70, and an m16n8k16 f16 statement from the shared emitter -- the arm
   the base rendered per statement at every capability -- at sm_80 and sm_75. A refusal must name
   the instruction's shape in the JIT's log, and the sm_70 refusal is claimed only where the JIT
   loads a kernel with no tensor instruction retargeted the same way. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module S = Ir.Schedule
module M = Ir.Cuda_mma
module G = Test_utils.Generated
module Claims = Cuda_wmma_floor_claims
module Cu = Cuda

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

(* One uniform-f16 [Tile_mma] statement per 16-row block with the whole k extent inside it: the
   per-statement scope. *)
let render_statement ~name =
  let a = NTDSL.init ~l:"k8_a" ~prec:Ir.Ops.half ~i:[ 32 ] ~o:[ 32 ] ~f:(fun _ -> 1.) () in
  let b = NTDSL.init ~l:"k8_b" ~prec:Ir.Ops.half ~i:[ 32 ] ~o:[ 32 ] ~f:(fun _ -> 1.) () in
  let%op y = a * b in
  Ir.Tnode.update_prec y.Tensor.value Ir.Ops.half;
  let transform (opt : LL.optimized) =
    let i, j, k =
      match List.find_exn (Ll_test.nest_paths opt.llc) ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let zero, zs = S.expand_zero ~tn:y.Tensor.value in
    let zi, zj = match zs with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let zr, _, _ = S.split ~axis:zi ~factor:16 ~outer:LL.Grid ~inner:LL.Serial in
    let rows, _, ii = S.split ~axis:i ~factor:16 ~outer:LL.Grid ~inner:LL.Serial in
    let tensorize, _ = S.tensorize ~i:ii ~j ~k ~simd_width:32 () in
    [ S.apply [ zero; zr; S.Retype { axis = zj; ty = LL.Workgroup }; rows; tensorize ] opt ]
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

(* [ptx] with its [.target] directive naming [sm] instead, any [, debug] suffix kept. *)
let retarget ptx ~sm =
  String.split_lines ptx
  |> List.map ~f:(fun line ->
      match String.lsplit2 (String.strip line) ~on:' ' with
      | Some (".target", rest) ->
          Printf.sprintf ".target sm_%d%s" sm
            (Option.value_map (String.lsplit2 rest ~on:',') ~default:"" ~f:(fun (_, tail) ->
                 "," ^ tail))
      | _ -> line)
  |> String.concat ~sep:"\n"

(* Whether the driver's PTX assembler loads [ptx] into a module on the first device: [Ok ()], or
   [Error log] with the JIT's error log. No kernel is launched. *)
let jit_load ptx =
  ignore (Context.auto ());
  Cu.Context.set_current (Cu.Context.get_primary (Cu.Device.get ~ordinal:0));
  let image = Ctypes.CArray.of_string ptx in
  let log = Stdlib.Bigarray.(Array1.create char c_layout 16384) in
  Stdlib.Bigarray.Array1.fill log '\000';
  match
    Cu.Module.load_data_ex
      { Nvrtc.log = None; ptx = Ctypes.CArray.start image; ptx_length = Ctypes.CArray.length image }
      [ Cu.Module.ERROR_LOG_BUFFER log ]
  with
  | (_ : Cu.Module.t) ->
      ignore (Stdlib.Sys.opaque_identity image);
      Ok ()
  | exception Cu.Cuda_error { message; _ } ->
      let text = String.init (Stdlib.Bigarray.Array1.dim log) ~f:(Stdlib.Bigarray.Array1.get log) in
      let text =
        Option.value_map (String.index text '\000') ~default:text ~f:(String.prefix text)
      in
      Stdio.eprintf "driver JIT refusal: %s: %s (not part of the golden)\n" message
        (String.substr_replace_all text ~pattern:"\n" ~with_:" | ");
      Error text

let loads ptx = Result.is_ok (jit_load ptx)
let refused ~naming ptx = match jit_load ptx with Ok () -> false | Error log -> has log naming

(* The m16n8k16 f16 statement the base rendered per statement at every capability, from the shared
   emitter with no capability gate: one 16x8x16 tile over 16-bit operands at [__mma_ap] / [__mma_bp]
   into [__mma_dp]. *)
let k16_control =
  lazy
    (let title, lines =
       M.inline_ptx_statement_lines (M.M16n8k16 M.M16n8k16_f16) ~a_prec:Ir.Ops.half
         ~b_prec:Ir.Ops.half ~resident:None ~ta:false ~tb:false ~lda:16 ~ldb:8 ~ldd:8 ~a_swz:false
         ~b_swz:false ~m:16 ~n:8 ~k:16
     in
     String.concat ~sep:"\n"
       ([
          "#include <cuda_fp16.h>";
          "extern \"C\" __global__ void k16_control(const __half *__mma_ap, const __half \
           *__mma_bp, __half *__mma_dp) {";
          title;
        ]
       @ lines @ [ "}"; "}" ]))

let plain_control =
  "extern \"C\" __global__ void plain_control(float *d) { d[threadIdx.x] = 1.0f; }"

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
      let k8_arm = M.Inline_ptx M.M16n8k8_f16 in
      let k8_floor = M.arm_floor k8_arm in
      let k8_source =
        lazy
          (M.with_resolution_ceiling ~cc:k8_floor (fun () ->
               render_statement ~name:"wmma_floor_k8_statement"))
      in
      let k8_ptx = lazy (compile k8_floor (Lazy.force k8_source)) in
      let k8_instruction = "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in
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
      let jit_70_leg =
        Verdict.sub_leg ~aggregation:`Environment leg
          ~when_:(lazy (loads (retarget (compile k8_floor plain_control) ~sm:70)))
          ~on:"a driver JIT refusing sm_70 PTX of this toolkit's version"
      in
      Claims.run ~leg ~compute_70_leg ~jit_70_leg (function
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
        | Bf16_refusal -> bf16_refused ()
        | K8_statement ->
            let logic = G.main_logic (Lazy.force k8_source) in
            has logic ("(" ^ M.arm_marker k8_arm ^ ")")
            && has logic k8_instruction
            && (not (has logic "m16n8k16"))
            && has logic "__half2float" && has logic "__float2half"
        | K8_compute_75 ->
            let ptx = Lazy.force k8_ptx in
            has ptx (".target sm_" ^ Int.to_string k8_floor) && has ptx k8_instruction
        | K8_shipping_floor ->
            List.equal String.equal
              (M.gpu_arch_options ~target:M.Floor ~device_cc:120 (Lazy.force k8_source))
              [ "--gpu-architecture=compute_" ^ Int.to_string k8_floor ]
        | K8_loads_at_75 -> loads (Lazy.force k8_ptx)
        | K8_refused_at_70 -> refused ~naming:"m16n8k8" (retarget (Lazy.force k8_ptx) ~sm:70)
        | K16_loads_at_80 -> loads (compile 80 (Lazy.force k16_control))
        | K16_refused_at_75 ->
            refused ~naming:"m16n8k16" (compile k8_floor (Lazy.force k16_control))))
