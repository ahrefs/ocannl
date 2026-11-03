(* GPU-free coverage for Metal's MTLCompileOptions property sequence (gh-ocannl-848).

   Metal_backend interprets this pure list into the mutable Objective-C options object. Keeping the
   complete property sequences here pin both debug variants and both API capabilities without
   linking Metal or requiring a device. The modern arithmetic and math-function properties are
   separate on purpose: Safe prevents the measured reassociation, while Fast retains the function
   family the old default selected. macOS 14 receives the safe legacy spelling instead. *)

open Base

let selectors = [ "setMathMode:"; "setMathFloatingPointFunctions:" ]

let cases =
  [
    ( false,
      selectors,
      [ Ir.Compiler_options.Language_version_3_1; Math_mode_safe; Math_functions_fast ] );
    ( true,
      selectors,
      [
        Ir.Compiler_options.Language_version_3_2;
        Math_mode_safe;
        Math_functions_fast;
        Enable_logging;
      ] );
    ( false,
      [ "setMathFloatingPointFunctions:" ],
      [ Ir.Compiler_options.Language_version_3_1; Fast_math_enabled false ] );
    ( true,
      [ "setMathMode:" ],
      [ Ir.Compiler_options.Language_version_3_2; Fast_math_enabled false; Enable_logging ] );
    (false, [], [ Ir.Compiler_options.Language_version_3_1; Fast_math_enabled false ]);
  ]

let () =
  List.iter cases ~f:(fun (routine_logging, available_selectors, want) ->
      let math_api =
        Ir.Compiler_options.metal_math_api ~selector_available:(fun selector ->
            List.mem available_selectors selector ~equal:String.equal)
      in
      let got = Ir.Compiler_options.metal ~routine_logging ~math_api in
      Stdio.eprintf "debug=%b selectors=%s api=%s:\n  got:  %s\n  want: %s\n" routine_logging
        (String.concat ~sep:"," available_selectors)
        (match math_api with Modern_split -> "modern" | Legacy -> "legacy")
        (Ir.Compiler_options.render_metal got)
        (Ir.Compiler_options.render_metal want));
  Verdict.p_all "the selector seam drives every Metal API/debug option sequence" cases
    ~f:(fun (routine_logging, available_selectors, want) ->
      let math_api =
        Ir.Compiler_options.metal_math_api ~selector_available:(fun selector ->
            List.mem available_selectors selector ~equal:String.equal)
      in
      List.equal Ir.Compiler_options.equal_metal_option
        (Ir.Compiler_options.metal ~routine_logging ~math_api)
        want);
  (* gh-ocannl-1457: the math regime enters the schedule-cache key from the same builder: nothing on
     the default modern regime (its keys stay valid), and the legacy sequence itself on the
     other. *)
  Verdict.p "the modern Metal math regime adds no schedule-cache component"
    (Option.is_none (Ir.Compiler_options.metal_regime_tag Modern_split));
  Verdict.p "the legacy Metal math regime adds a schedule-cache component"
    (Option.is_some (Ir.Compiler_options.metal_regime_tag Legacy))
