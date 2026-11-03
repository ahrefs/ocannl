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
   dependency rows. The missing class is decided by whether the library exists after the refusal.

   gh-ocannl-1728 adds the one retry of a signing refusal, [load_retrying_signing_refusal]: its
   policy is pinned over a scripted loader (which refusals are retried, how many loads and pauses
   each costs, which dlerror the report keeps), and the platform fact it rests on -- dyld keeps no
   verdict on a path it refused -- over a real library where the host enforces signatures (arm64
   macOS). There the probe library is built and its signature stripped, so the first load is a real
   refusal; the pause re-signs it ad hoc in place, and the retry must load it. The control leaves it
   unsigned, and the retry must be refused again. *)

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

(* A scripted loader: each load returns the next answer, counting loads and pauses. *)
let scripted answers =
  let loads = ref 0 and pauses = ref 0 and remaining = ref answers in
  let load () =
    Int.incr loads;
    match !remaining with
    | answer :: rest ->
        remaining := rest;
        answer
    | [] -> failwith "the scripted loader ran out of answers"
  in
  (load, (fun () -> Int.incr pauses), loads, pauses)

let signing =
  "dlopen(" ^ lib ^ ", 0x0002): tried: '" ^ lib ^ "' (missing code signature in <24A7D570> '" ^ lib
  ^ "')"

let undefined = lib ^ ": undefined symbol: ocannl_gh1077_undefined_expf"
let gone = lib ^ ": cannot open shared object file: No such file or directory"

let outcome_name = function
  | Cc.Loaded () -> "Loaded"
  | Cc.Loaded_on_retry _ -> "Loaded_on_retry"
  | Cc.Refused { refusal; _ } -> "Refused " ^ Sexp.to_string (Cc.sexp_of_load_refusal refusal)

let () =
  let run ?(present = fun () -> true) answers =
    let load, pause, loads, pauses = scripted answers in
    let outcome = Cc.load_retrying_signing_refusal ~load ~pause ~library_present:present in
    Stdio.eprintf "retry policy: %s after %d loads, %d pauses (not part of the golden)\n"
      (outcome_name outcome) !loads !pauses;
    (outcome, !loads, !pauses)
  in
  let outcome, loads, pauses = run [ Ok () ] in
  p "a library that loads is loaded once, without a pause"
    (match outcome with Cc.Loaded () -> loads = 1 && pauses = 0 | _ -> false);
  let outcome, loads, pauses = run [ Error signing; Ok () ] in
  p "a signing refusal that clears loads on the one retry, after one pause"
    (match outcome with
    | Cc.Loaded_on_retry { first_dlerror; _ } ->
        String.equal first_dlerror signing && loads = 2 && pauses = 1
    | _ -> false);
  let outcome, loads, pauses = run [ Error signing; Error signing; Ok () ] in
  p "a signing refusal that persists is refused after exactly one retry"
    (match outcome with
    | Cc.Refused { refusal = Cc.Code_signing_refusal; dlerror; first_dlerror = Some first } ->
        String.equal dlerror signing && String.equal first signing && loads = 2 && pauses = 1
    | _ -> false);
  let outcome, loads, pauses = run ~present:(fun () -> false) [ Error signing; Error gone ] in
  p "a retry refused for another reason is classified by the retry's own dlerror"
    (match outcome with
    | Cc.Refused { refusal = Cc.Library_missing; dlerror; first_dlerror = Some first } ->
        String.equal dlerror gone && String.equal first signing && loads = 2 && pauses = 1
    | _ -> false);
  let outcome, loads, pauses = run [ Error undefined; Ok () ] in
  p "a link defect is not retried"
    (match outcome with
    | Cc.Refused { refusal = Cc.Load_defect; first_dlerror = None; _ } -> loads = 1 && pauses = 0
    | _ -> false);
  let outcome, loads, pauses = run ~present:(fun () -> false) [ Error gone; Ok () ] in
  p "a missing library is not retried"
    (match outcome with
    | Cc.Refused { refusal = Cc.Library_missing; first_dlerror = None; _ } ->
        loads = 1 && pauses = 0
    | _ -> false)

(* arm64 macOS refuses to map unsigned code; other hosts load an unsigned library. Read off the host
   -- the predicate the cc backend's link flags read (gh-ocannl-1741) -- never off the run under
   test. *)
let enforces_code_signatures = Utils.host_is_darwin_arm64

(* The real leg: a refused library, a pause that either re-signs it in place or does nothing, and
   the retry. [None] where the host enforces no signatures. *)
let real_retry ~heal =
  if not enforces_code_signatures then None
  else
    let src = Stdlib.Filename.temp_file "ocannl_gh1728_" ".c" in
    let dll = Stdlib.Filename.temp_file "ocannl_gh1728_" ".so" in
    Exn.protect
      ~finally:(fun () ->
        List.iter [ src; dll ] ~f:(fun f -> try Stdlib.Sys.remove f with _ -> ()))
      ~f:(fun () ->
        Stdio.Out_channel.write_all src ~data:"int ocannl_gh1728_answer(void) { return 42; }\n";
        let quiet command = Stdlib.Sys.command (command ^ " > /dev/null 2>&1") = 0 in
        let quoted = Stdlib.Filename.quote dll in
        if
          not
            (quiet
               (Printf.sprintf "%s %s -o %s %s" (Cc.compiler_command ())
                  (Lazy.force Cc.kernel_link_flags) quoted (Stdlib.Filename.quote src))
            && quiet ("codesign --remove-signature " ^ quoted))
        then failwith "cannot build the unsigned probe library";
        let load () =
          try Ok (Dl.dlopen ~filename:dll ~flags:[ RTLD_NOW; RTLD_LOCAL ])
          with Dl.DL_error dlerror -> Error dlerror
        in
        let pause () =
          if heal && not (quiet ("codesign -s - " ^ quoted)) then
            failwith "cannot re-sign the probe library"
        in
        let outcome =
          Cc.load_retrying_signing_refusal ~load ~pause ~library_present:(fun () ->
              Stdlib.Sys.file_exists dll)
        in
        (match outcome with
        | Cc.Loaded lib | Cc.Loaded_on_retry { lib; _ } -> Dl.dlclose ~handle:lib
        | Cc.Refused _ -> ());
        Stdio.eprintf "real retry (heal=%b): %s (not part of the golden)\n%!" heal
          (match outcome with
          | Cc.Loaded _ -> "Loaded"
          | Cc.Loaded_on_retry { first_dlerror; _ } -> "Loaded_on_retry after: " ^ first_dlerror
          | Cc.Refused { refusal; dlerror; _ } ->
              Sexp.to_string (Cc.sexp_of_load_refusal refusal) ^ ": " ^ dlerror);
        Some outcome)

let () =
  let claim =
    gated ~aggregation:`Environment ~when_:enforces_code_signatures
      ~on:"a host whose loader enforces no code signatures (only arm64 macOS does)"
  in
  let healed = real_retry ~heal:true in
  claim "a real signing refusal, its library re-signed in place, loads on the retry"
    (match healed with Some (Cc.Loaded_on_retry _) -> true | _ -> false);
  let unhealed = real_retry ~heal:false in
  claim "a real signing refusal that persists is refused again on the retry"
    (match unhealed with
    | Some (Cc.Refused { refusal = Cc.Code_signing_refusal; first_dlerror = Some _; _ }) -> true
    | _ -> false)
