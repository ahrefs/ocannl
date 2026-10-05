(* GPU-free coverage for HIP's compiler-level reduction-order policy (gh-ocannl-735).

   This invokes the complete production option builder used by Hip_backend.Impl, with sentinels for
   the SDK-discovery results, so it can pin the exact ordering without hipjit or a device.
   [reduction_forms] remains the hardware-backed proof that hiprtc honors it numerically, and
   [half_softmax] the proof that [-fhonor-infinities] keeps the [(-INFINITY)] mask sentinel a usable
   value under the same fast-math umbrella. *)

open Base

let build ~uses_rocwmma ~with_debug =
  Ir.Compiler_options.hiprtc ~target_archs:[] ~hip_include_options:[ "-Ihip" ]
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
  let affected =
    [ [ "gfx1102" ]; [ "gfx1102:xnack-" ]; [ "gfx1151"; "gfx1102" ]; [ "gfx1102"; "gfx1100" ] ]
  in
  let unaffected =
    [ []; [ "gfx1100" ]; [ "gfx1151" ]; [ "gfx1201" ]; [ "gfx11020" ]; [ "gfx1151"; "gfx1201" ] ]
  in
  Verdict.p_all "every target set containing observed gfx1102 disables HIP half-register allocation"
    affected ~f:(fun target_archs ->
      List.equal String.equal
        (Ir.Compiler_options.hip_target_options ~target_archs)
        [ "-Xclang"; "-target-feature"; "-Xclang"; "-real-true16" ]);
  Verdict.p_all "unmeasured HIP targets retain their compiler register allocation" unaffected
    ~f:(fun target_archs -> List.is_empty (Ir.Compiler_options.hip_target_options ~target_archs));
  Verdict.p_all "the gfx1102 workaround reaches every production HIPRTC variant" cases
    ~f:(fun (uses_rocwmma, with_debug, _) ->
      let options =
        Ir.Compiler_options.hiprtc ~target_archs:[ "gfx1102" ] ~hip_include_options:[ "-Ihip" ]
          ~rocwmma_include_options:[ "-Irocwmma" ] ~uses_rocwmma ~with_debug
      in
      List.mem options "-real-true16" ~equal:String.equal);
  (* gh-ocannl-1222: HIPRTC 9.0's [-amdgpu-waitcnt-forcezero] (and its load-only sibling) puts a
     wait between [s_getpc_b64] and the [s_add_u32] carrying a PC-relative offset that assumes the
     two are adjacent, so every [__constant__] global was read 4 bytes early. *)
  Verdict.p_none "no HIPRTC variant forces wait counters, which shifts PC-relative constant loads"
    (List.concat_map (affected @ unaffected) ~f:(fun target_archs ->
         List.map cases ~f:(fun (uses_rocwmma, with_debug, _) ->
             Ir.Compiler_options.hiprtc ~target_archs ~hip_include_options:[ "-Ihip" ]
               ~rocwmma_include_options:[ "-Irocwmma" ] ~uses_rocwmma ~with_debug)))
    ~f:(List.exists ~f:(String.is_prefix ~prefix:"-amdgpu-waitcnt-"))
