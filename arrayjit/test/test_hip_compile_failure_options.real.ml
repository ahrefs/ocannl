(* gh-ocannl-849: a HIPRTC compile failure carries the exact effective option vector beside the
   compiler log. This calls [Hip_backend.Impl.hip_to_code], the production seam that assembles the
   options and invokes hiprtc, rather than restating that call in the test. The deliberately invalid
   source establishes hipjit's real compile-failure constructor; the valid source is the positive
   control that the wrapper still returns a code object. *)

open Base

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* The options hiprtc is expected to report, once the runtime is initialized. *)
let expected_suffix =
  lazy
    (Hip.init ();
     let target_archs =
       List.init (Hip.Device.get_count ()) ~f:(fun ordinal ->
           (Hip.Device.get_attributes (Hip.Device.get ~ordinal)).gcn_arch_name)
     in
     let expected_options =
       Ir.Compiler_options.hiprtc ~target_archs
         ~hip_include_options:(Hip_backend.hip_include_options ())
         ~rocwmma_include_options:[] ~uses_rocwmma:false
         ~fast_math:(Ir.Cuda_like_config.hip_fast_math ())
         ~with_debug:(Utils.with_runtime_debug ())
         ~with_line_info:Utils.settings.output_debug_files_in_build_directory ()
     in
     "\nhiprtc options: " ^ Ir.Compiler_options.render expected_options)

let failure_has_options () =
  let expected_suffix = Lazy.force expected_suffix in
  let failure_message =
    match Hip_backend.Impl.hip_to_code ~name:"hiprtc_options_invalid" "this is not HIP C++" with
    | _ -> None
    | exception Hiprtc.Hiprtc_error { message; _ } -> Some message
  in
  let failure_has_options =
    Option.exists failure_message ~f:(String.is_suffix ~suffix:expected_suffix)
  in
  if not failure_has_options then
    Option.iter failure_message ~f:(Stdio.eprintf "hiprtc failure: %s\n");
  failure_has_options

let valid_compiles () =
  let valid_source = {|extern "C" __global__ void hiprtc_options_valid() {}|} in
  let valid_error =
    try
      ignore
        (Hip_backend.Impl.hip_to_code ~name:"hiprtc_options_valid" valid_source
          : Hiprtc.compile_to_code_result);
      None
    with exn -> Some (Exn.to_string exn)
  in
  Option.iter valid_error ~f:(Stdio.eprintf "valid HIP source failed: %s\n");
  Option.is_none valid_error

let () =
  Hip_compile_failure_options_claims.run
    ~leg:(Verdict.leg ~when_:(String.equal backend_name "hip") ~on:backend_name ())
    ~failure_has_options ~valid_compiles
