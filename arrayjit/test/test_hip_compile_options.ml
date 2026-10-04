(* GPU-free coverage for HIP's compiler-level reduction-order policy (gh-ocannl-735).

   This invokes the complete production option builder used by Hip_backend.Impl, with sentinels for
   the SDK-discovery results, so it can pin the exact ordering without hipjit or a device.
   [reduction_forms] remains the hardware-backed proof that hiprtc honors it numerically, and
   [half_softmax] the proof that [-fhonor-infinities] keeps the [(-INFINITY)] mask sentinel a usable
   value under the same fast-math umbrella. *)

open Base

let build ~uses_rocwmma ~with_debug =
  Ir.Compiler_options.hiprtc ~target_arch:None ~hip_include_options:[ "-Ihip" ]
    ~rocwmma_include_options:[ "-Irocwmma" ] ~uses_rocwmma ~with_debug

let () =
  let cases =
    [
      (false, false, [ "-Ihip"; "-ffast-math"; "-fno-associative-math"; "-fhonor-infinities" ]);
      (false, true, [ "-Ihip"; "-ffast-math"; "-fno-associative-math"; "-fhonor-infinities"; "-g" ]);
      ( true,
        false,
        [
          "-Ihip";
          "-Irocwmma";
          "-std=c++17";
          "-ffast-math";
          "-fno-associative-math";
          "-fhonor-infinities";
        ] );
      ( true,
        true,
        [
          "-Ihip";
          "-Irocwmma";
          "-std=c++17";
          "-ffast-math";
          "-fno-associative-math";
          "-fhonor-infinities";
          "-g";
        ] );
    ]
  in
  (* Both lists on stderr for all four cases (gh-ocannl-784). The claim below is one boolean over
     the whole matrix, so a failure used to say only that SOMETHING moved -- and the thing that
     moves here is an option's position under an umbrella flag, which is unreadable from a [false].
     stderr rather than stdout because it is diagnostic rather than a verdict, and because a run
     that fails exits nonzero, which is exactly when dune discards the redirected stdout. *)
  List.iter cases ~f:(fun (uses_rocwmma, with_debug, want) ->
      Stdio.eprintf "rocwmma=%b debug=%b:\n  got:  %s\n  want: %s\n" uses_rocwmma with_debug
        (Ir.Compiler_options.render (build ~uses_rocwmma ~with_debug))
        (Ir.Compiler_options.render want));
  Verdict.p_all "every HIPRTC variant keeps both fast-math overrides, in order, after the umbrella"
    cases ~f:(fun (uses_rocwmma, with_debug, want) ->
      List.equal String.equal (build ~uses_rocwmma ~with_debug) want);
  let affected = [ Some "gfx1102"; Some "gfx1102:xnack-" ] in
  let unaffected = [ None; Some "gfx1100"; Some "gfx1151"; Some "gfx1201"; Some "gfx11020" ] in
  Verdict.p_all "only the observed gfx1102 target enables conservative HIP load waits" affected
    ~f:(fun target_arch ->
      List.equal String.equal
        (Ir.Compiler_options.hip_load_wait_options ~target_arch)
        [ "-mllvm"; "-amdgpu-waitcnt-load-forcezero" ]);
  Verdict.p_all "unmeasured HIP targets retain their compiler load scheduling" unaffected
    ~f:(fun target_arch -> List.is_empty (Ir.Compiler_options.hip_load_wait_options ~target_arch));
  Verdict.p_all "the gfx1102 workaround reaches every production HIPRTC variant" cases
    ~f:(fun (uses_rocwmma, with_debug, _) ->
      let options =
        Ir.Compiler_options.hiprtc ~target_arch:(Some "gfx1102") ~hip_include_options:[ "-Ihip" ]
          ~rocwmma_include_options:[ "-Irocwmma" ] ~uses_rocwmma ~with_debug
      in
      List.mem options "-amdgpu-waitcnt-load-forcezero" ~equal:String.equal)
