(* GPU-free coverage for CUDA's nvrtc option vector (gh-ocannl-784).

   The CUDA counterpart of [test_hip_compile_options]: it invokes the complete production option
   builder used by Cuda_backend.Impl, with sentinels for the two discovered slots (the CUDA_PATH
   include directory and [gpu_arch_options]' architecture targets), so the exact vector can be
   pinned without cudajit or a device.

   HIP's claim is an ORDERING one -- clang's [-ffast-math] umbrella silently re-enables anything
   spelled before it. CUDA's is a MEMBERSHIP one: nvrtc has no umbrella to override, and its
   [--use_fast_math] was measured not to reassociate reductions (the probe is recorded in
   [Ir.Compiler_options]'s header), so what has to stay true is that the vector never acquires
   nvrtc's undocumented reassociation opt-in -- the one lever that would make CUDA behave the way
   hiprtc did in gh-ocannl-735. [reduction_forms] remains the hardware-backed proof that a CUDA
   compile honors reduction order numerically; this is the half that fails on a laptop. *)

open Base

let build ~arch_options ~fast_math ~with_device_debug =
  Ir.Compiler_options.nvrtc ~cuda_include_options:[ "-I/cuda/include" ] ~arch_options ~fast_math
    ~with_device_debug ()

(* The two shapes [gpu_arch_options] can return: nothing (no arch marker in the source), or exactly
   one [--gpu-architecture] target. *)
let no_arch = []
let one_arch = [ "--gpu-architecture=compute_80" ]

(* gh-ocannl-1710: the precise policy ([cuda_fast_math] off, as [profile=reproducible] sets it) is
   the four switches the umbrella would set, each spelled out at nvrtc's IEEE value with contraction
   off. *)
let precise = [ "--ftz=false"; "--prec-div=true"; "--prec-sqrt=true"; "--fmad=false" ]

let cases =
  [
    ("no arch, no debug", no_arch, true, false, [ "-I/cuda/include"; "--use_fast_math" ]);
    ( "no arch, device debug",
      no_arch,
      true,
      true,
      [ "-I/cuda/include"; "--use_fast_math"; "--device-debug" ] );
    ( "arch floor, no debug",
      one_arch,
      true,
      false,
      [ "-I/cuda/include"; "--gpu-architecture=compute_80"; "--use_fast_math" ] );
    ( "arch floor, device debug",
      one_arch,
      true,
      true,
      [ "-I/cuda/include"; "--gpu-architecture=compute_80"; "--use_fast_math"; "--device-debug" ] );
    ("precise, no arch, no debug", no_arch, false, false, "-I/cuda/include" :: precise);
    ( "precise, arch floor, device debug",
      one_arch,
      false,
      true,
      [ "-I/cuda/include"; "--gpu-architecture=compute_80" ] @ precise @ [ "--device-debug" ] );
  ]

let () =
  (* Both lists on stderr for every case, per the pinning guidance: a bare [false] on a
     list-equality claim says nothing about WHICH element moved, and the option vector is exactly
     the state a numeric mismatch has to be read against. *)
  List.iter cases ~f:(fun (label, arch_options, fast_math, with_device_debug, want) ->
      let got = build ~arch_options ~fast_math ~with_device_debug in
      Stdio.eprintf "%s:\n  got:  %s\n  want: %s\n" label (Ir.Compiler_options.render got)
        (Ir.Compiler_options.render want));
  Verdict.p_all "every nvrtc variant pins the discovered slots, the math policy and debug in order"
    cases ~f:(fun (_label, arch_options, fast_math, with_device_debug, want) ->
      List.equal String.equal (build ~arch_options ~fast_math ~with_device_debug) want);
  (* gh-ocannl-1710: the umbrella is present exactly under the fast policy -- under the precise one
     its library-function substitution would survive any switch spelled beside it. *)
  Verdict.p_all "an nvrtc variant passes --use_fast_math exactly when its policy is fast" cases
    ~f:(fun (_label, arch_options, fast_math, with_device_debug, _want) ->
      Bool.equal fast_math
        (List.mem
           (build ~arch_options ~fast_math ~with_device_debug)
           "--use_fast_math" ~equal:String.equal));
  (* gh-ocannl-1457: the contraction pin [C_syntax_config.compiler_contraction_off] reads off this
     builder holds exactly under the precise policy, the one that spells out [--fmad=false]. *)
  Verdict.p_all "an nvrtc variant pins contraction off exactly when its policy is precise" cases
    ~f:(fun (_label, arch_options, fast_math, with_device_debug, _want) ->
      Bool.equal (not fast_math)
        (Ir.Compiler_options.nvrtc_contraction_off
           (build ~arch_options ~fast_math ~with_device_debug)));
  (* The membership claim. Quantified over the same case matrix so a variant added above is covered
     by it automatically, and phrased over the option NAME the production module exports rather than
     a second copy of the spelling. *)
  Verdict.p_none "no nvrtc variant passes nvrtc's reassociation opt-in" cases
    ~f:(fun (_label, arch_options, fast_math, with_device_debug, _want) ->
      List.mem
        (build ~arch_options ~fast_math ~with_device_debug)
        Ir.Compiler_options.nvrtc_reassociation_opt_in ~equal:String.equal)
