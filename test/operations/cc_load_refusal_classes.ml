(* gh-ocannl-1717: the cc backend's [dlopen] refusals are told apart by [classify_load_refusal]. An
   undefined symbol or a missing dependency is OCANNL's link bug (gh-ocannl-1077); a code-signing or
   system-policy refusal, and a library that is gone by the time it is loaded, are environment
   refusals, rendered as [Schedule_outcome.Environment] rejections rather than a symbol bug. The
   macOS refusal that motivated this was transient (a [schedule_conv_gemm] run on a loaded host,
   passing alone on rerun), so it cannot be provoked on demand: the classes are pinned here over
   canned [dlerror()] texts, one per loader shape. [cc_dlopen_cause]'s [code_signing] mode provokes
   a real signing refusal end to end where the host enforces signatures (arm64 macOS).

   The cases discriminate between reading the text and reading the filesystem: dyld lists a
   candidate path it found no file at inside a signing refusal, and a missing dependency reads like
   a missing file in every loader, so a classifier keyed on "no such file" fails the signing and
   dependency rows. The missing class is decided by whether the library exists after the refusal. *)

open Base
open Verdict.Claims
module Cc = Context.Cc_backend

let lib = "/var/folders/xy/T/ocannl_cc_k_1234.so"

let cases =
  [
    ( "glibc: undefined symbol",
      true,
      lib ^ ": undefined symbol: ocannl_gh1077_undefined_expf",
      Cc.Load_defect );
    ( "dyld: symbol not found in flat namespace",
      true,
      "dlopen(" ^ lib
      ^ ", 0x000A): symbol not found in flat namespace '_ocannl_gh1077_undefined_expf'",
      Cc.Load_defect );
    ( "glibc: a missing dependency, the library present",
      true,
      "libgomp.so.1: cannot open shared object file: No such file or directory",
      Cc.Load_defect );
    ( "dyld: a missing dependency, the library present",
      true,
      "dlopen(" ^ lib
      ^ ", 0x000A): Library not loaded: @rpath/libomp.dylib\n  Referenced from: <5D1B2C3A> " ^ lib
      ^ "\n  Reason: tried: '/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file)",
      Cc.Load_defect );
    ( "dyld: signature not valid, load disallowed by system policy",
      true,
      "dlopen(" ^ lib ^ ", 0x000A): tried: '" ^ lib ^ "' (code signature in <5D1B2C3A> '" ^ lib
      ^ "' not valid for use in process: library load disallowed by system policy), \
         '/System/Volumes/Preboot/Cryptexes/OS" ^ lib ^ "' (no such file), '" ^ lib
      ^ "' (code signature in <5D1B2C3A> '" ^ lib
      ^ "' not valid for use in process: library load disallowed by system policy)",
      Cc.Code_signing_refusal );
    ( "dyld: missing code signature",
      true,
      "dlopen(" ^ lib ^ ", 0x0002): tried: '" ^ lib ^ "' (missing code signature in <24A7D570> '"
      ^ lib ^ "'), '/System/Volumes/Preboot/Cryptexes/OS" ^ lib ^ "' (no such file)",
      Cc.Code_signing_refusal );
    ( "dyld: a signing refusal, the library gone since",
      false,
      "dlopen(" ^ lib ^ ", 0x000A): tried: '" ^ lib ^ "' (code signature in <5D1B2C3A> '" ^ lib
      ^ "' not valid for use in process: library load disallowed by system policy)",
      Cc.Code_signing_refusal );
    ( "glibc: the library itself missing",
      false,
      lib ^ ": cannot open shared object file: No such file or directory",
      Cc.Library_missing );
    ( "dyld: the library itself missing",
      false,
      "dlopen(" ^ lib ^ ", 0x000A): tried: '" ^ lib
      ^ "' (no such file), '/System/Volumes/Preboot/Cryptexes/OS" ^ lib ^ "' (no such file), '"
      ^ lib ^ "' (no such file)",
      Cc.Library_missing );
    ( "LoadLibrary: the library itself missing",
      false,
      "The specified module could not be found.",
      Cc.Library_missing );
  ]

let () =
  List.iter cases ~f:(fun (label, library_present, dlerror, expected) ->
      let got = Cc.classify_load_refusal ~library_present dlerror in
      Stdio.eprintf "%s: %s (not part of the golden)\n" label
        (Sexp.to_string (Cc.sexp_of_load_refusal got));
      p
        (Printf.sprintf "%s is %s" label (Sexp.to_string (Cc.sexp_of_load_refusal expected)))
        (Cc.equal_load_refusal got expected));
  let severity_of refusal =
    Sexp.to_string (Ir.Schedule_outcome.sexp_of_severity (Cc.load_refusal_severity refusal))
  in
  List.iter [ Cc.Load_defect; Cc.Code_signing_refusal; Cc.Library_missing ] ~f:(fun refusal ->
      Stdio.printf "%s raises severity %s\n"
        (Sexp.to_string (Cc.sexp_of_load_refusal refusal))
        (severity_of refusal));
  p_exists "some case is a signing refusal" cases ~f:(fun (_, _, _, e) ->
      Cc.equal_load_refusal e Cc.Code_signing_refusal);
  p_exists "some case is a missing library" cases ~f:(fun (_, _, _, e) ->
      Cc.equal_load_refusal e Cc.Library_missing);
  p_exists "some case is a link defect" cases ~f:(fun (_, _, _, e) ->
      Cc.equal_load_refusal e Cc.Load_defect)
