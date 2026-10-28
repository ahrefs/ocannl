(* The nvrtc [--gpu-architecture] policy (gh-ocannl-481 item 3, D4).

   [Ir.Cuda_mma.gpu_arch_options] decides one thing with two very different answers:

   - a FLOOR: the lowest arch whose PTX contains the instructions a kernel uses. The driver
   forward-JIT-compiles floor-targeted PTX on every later GPU, so one compile covers everything
   above it — which is why a triggered floor is deliberately NOT raised to the device arch, only up
   to compute_75 (CUDA 13 dropped offline compilation below that). - a FAMILY target: a
   [compute_120a]-style architecture-specific arch, loadable only by the device family it names.
   Blackwell's block-scaled [kind::mxf8f6f4] forms exist only under such a target, so it is the one
   marker that gives up forward-JIT portability — and therefore the one gated on the attached
   devices' own family, so family PTX is never produced for a device that could not load it.

   No arm emits [(mma-mxfp8)] yet (block scaling is blocked on OCANNL having microscaling storage at
   all), so this pins the MECHANISM: that the marker selects a family target on a family device,
   that it does not on any other device, and that its presence changes nothing for every existing
   floor marker. The function is pure and lives outside the cudajit-gated backend (gh-ocannl-1232),
   so it is checked wherever OCANNL builds, with no toolkit and no device.

   The arm floors are not restated here: every arm's own marker, compiled for a device exactly at
   the arm's own floor, must target that floor -- loadable there, and containing the arm's
   instructions -- so a new arm, or a floor moved in [Cuda_mma.arm_floor], is covered with no edit
   to this file. *)

open Base

let check ?target label ~device_cc src ~expected =
  let got = Ir.Cuda_mma.gpu_arch_options ?target ~device_cc src in
  let render l = if List.is_empty l then "(none)" else String.concat ~sep:" " l in
  if not (List.equal String.equal got expected) then
    failwith (Printf.sprintf "%s: expected %s, got %s" label (render expected) (render got))

let () =
  (* No markers, no arch option at all: the kernel compiles at nvrtc's default target. *)
  check "plain kernel" ~device_cc:120 "__global__ void k() {}" ~expected:[];
  (* Floors, and their forward-JIT contract: the option names the FLOOR, never the device. *)
  check "fp8 mma.sync floor" ~device_cc:120 "/* tile_mma 16x8x32 (mma-fp8) e5m2 */"
    ~expected:[ "--gpu-architecture=compute_89" ];
  check "bf16 mma.sync floor" ~device_cc:120 "/* tile_mma 16x8x16 (mma-bf16) */"
    ~expected:[ "--gpu-architecture=compute_80" ];
  check "wmma floor" ~device_cc:120 "nvcuda::wmma::mma_sync(a, b, c, d);"
    ~expected:[ "--gpu-architecture=compute_75" ];
  check "wmma bf16 floor" ~device_cc:120 "nvcuda::wmma::mma_sync(x); /* (wmma-bf16) */"
    ~expected:[ "--gpu-architecture=compute_80" ];
  (* The [cp.async] staging builtins (gh-ocannl-487 phase 2): the marker is the PTX instruction
     text, which lives in the prepended helper definitions' asm literals — so the floor fires
     exactly when a helper was actually emitted, and a routine merely NAMED like one (which the
     token-scoped builtin filter declines to activate) raises nothing. *)
  check "cp.async floor" ~device_cc:120 {|asm volatile("cp.async.ca.shared.global [d], [s], 4;");|}
    ~expected:[ "--gpu-architecture=compute_80" ];
  check "a helper-like routine name alone raises no floor" ~device_cc:75
    {|extern "C" __global__ void ocannl_cp_async4_probe() {}|} ~expected:[];
  (* Batched sources take the max of the triggered floors, not the first match. *)
  check "batched floors take the max" ~device_cc:120 "nvcuda::wmma::mma_sync(x); /* (mma-fp8) */"
    ~expected:[ "--gpu-architecture=compute_89" ];
  check "cp.async batched with fp8 takes the max" ~device_cc:120
    {|asm volatile("cp.async.wait_all;"); /* (mma-fp8) */|}
    ~expected:[ "--gpu-architecture=compute_89" ];
  (* A device below the raise point keeps the literal floor (it must be paired with an nvrtc that
     still accepts it). *)
  check "half arithmetic on an old device" ~device_cc:70 "__hfma(a, b, c)"
    ~expected:[ "--gpu-architecture=compute_70" ];
  check "half arithmetic on a new device" ~device_cc:120 "__hfma(a, b, c)"
    ~expected:[ "--gpu-architecture=compute_75" ];
  (* The family marker: architecture-specific target on a family device... *)
  check "mxfp8 marker on sm_120" ~device_cc:120 "/* (mma-mxfp8) */"
    ~expected:[ "--gpu-architecture=compute_120a" ];
  check "mxfp8 marker on sm_121" ~device_cc:121 "/* (mma-mxfp8) */"
    ~expected:[ "--gpu-architecture=compute_121a" ];
  (* ...and nothing of the sort anywhere else: an sm_89 device gets the ordinary floor policy, so no
     unloadable PTX is ever produced for it. *)
  check "mxfp8 marker off-family falls back to the floor" ~device_cc:89
    "/* (mma-mxfp8) */ /* (mma-fp8) */"
    ~expected:[ "--gpu-architecture=compute_89" ];
  check "mxfp8 marker off-family with no floor" ~device_cc:89 "/* (mma-mxfp8) */" ~expected:[];
  (* The family target supersedes floors on a family device: family PTX is not forward-JIT PTX, and
     mixing the two targets in one compile is not expressible. *)
  check "mxfp8 marker supersedes a floor on-family" ~device_cc:120
    "/* (mma-mxfp8) */ /* (mma-fp8) */"
    ~expected:[ "--gpu-architecture=compute_120a" ];
  (* Every arm's marker, as its rendering comment carries it, at a device sitting exactly at the
     arm's floor. Only the marker: a wmma rendering also mentions [nvcuda::wmma], whose lowest-floor
     trigger the max would absorb anyway. *)
  List.iter Ir.Cuda_mma.all_of_arm ~f:(fun arm ->
      let floor = Ir.Cuda_mma.arm_floor arm and marker = Ir.Cuda_mma.arm_marker arm in
      check
        (Printf.sprintf "the (%s) arm at its own floor" marker)
        ~device_cc:floor
        (Printf.sprintf "/* tile_mma 16x16x16 (%s) */" marker)
        ~expected:[ Printf.sprintf "--gpu-architecture=compute_%d" floor ]);
  (* gh-ocannl-1674: the native target is the device's own arch for every source -- a plain kernel,
     a floored one and a batch of floors alike -- and the family target on a family device. *)
  let native = Ir.Cuda_mma.Native in
  check ~target:native "native: a plain kernel" ~device_cc:120 "__global__ void k() {}"
    ~expected:[ "--gpu-architecture=compute_120" ];
  check ~target:native "native: a floored kernel" ~device_cc:120
    "nvcuda::wmma::mma_sync(x); /* (wmma-tf32) */"
    ~expected:[ "--gpu-architecture=compute_120" ];
  check ~target:native "native: an Ampere device" ~device_cc:86 "/* (mma-fp8) */ __hfma(a, b, c)"
    ~expected:[ "--gpu-architecture=compute_86" ];
  check ~target:native "native: the family marker" ~device_cc:120 "/* (mma-mxfp8) */"
    ~expected:[ "--gpu-architecture=compute_120a" ];
  check ~target:native "native: no device, the floor policy" ~device_cc:0 "/* (wmma-tf32) */"
    ~expected:[ "--gpu-architecture=compute_80" ];
  (* The [.target] directive the PTX target record reads. *)
  let directive ptx ~expected =
    let got = Ir.Cuda_mma.ptx_target_directive ptx in
    if not (Option.equal String.equal got expected) then
      failwith
        (Printf.sprintf "ptx_target_directive %S: expected %s, got %s" ptx
           (Option.value expected ~default:"(none)")
           (Option.value got ~default:"(none)"))
  in
  directive "//\n.version 8.8\n.target sm_80\n.address_size 64\n" ~expected:(Some "sm_80");
  directive ".version 9.0\n  .target sm_120a, debug\n" ~expected:(Some "sm_120a");
  directive ".version 9.0\n.address_size 64\n" ~expected:None;
  Stdio.print_endline "cuda gpu_arch_options: ok"
