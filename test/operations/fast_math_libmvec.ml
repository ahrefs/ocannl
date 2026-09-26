(* gh-ocannl-1045: a cc kernel compiled under fast-math whose [expf] loop gcc vectorizes into
   glibc's libmvec entry points ([_ZGVbN4v_expf], [_ZGVdN8v_expf], ...) must still load and run.
   Nothing in the host process loads libmvec, so a kernel [.so] that does not name it as a
   dependency dies at dlopen with [undefined symbol: _ZGV...]; the approximate profile's cc cells of
   gh-ocannl-719 did exactly that. [kernel_link_flags] passing [-lm] is the fix: glibc's [libm.so]
   is a linker script that adds libmvec to the kernel's [DT_NEEDED] exactly when it references a
   vector variant.

   Whether gcc vectorizes the call at all is decided by glibc's [<bits/math-vector.h>], which
   declares the vector variants only under [__FAST_MATH__] -- a macro gcc stops defining once
   [cc_backend_fast_math]'s [-fno-finite-math-only] follows [-ffast-math]. That suppression is an
   accident of this flag spelling, not a guard, so the dune rule forces the macro back through the
   compiler command ([-D__FAST_MATH__], the state plain [-ffast-math] leaves the compiler in) and
   this test checks that the forcing took: libmvec must be mapped into the process after the kernel
   loaded, and must not have been before it. That leg is gated on Linux with a glibc libmvec beside
   the process's libm AND a gcc compiler command: clang does not route [expf] to libmvec unless
   asked ([-fveclib=libmvec]), so under clang the kernel rightly keeps its scalar calls. The gate is
   the compiler's identity rather than whether the artifact happened to reference [_ZGV*], so that
   under gcc a loop that stopped vectorizing fails the claim instead of skipping it. Elsewhere the
   load-and-run claims still execute, with nothing to force. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let n = 256

(* The process's own mappings; [None] off Linux. *)
let maps () = try Some (Stdio.In_channel.read_all "/proc/self/maps") with Sys_error _ -> None

let mapped_paths maps =
  String.split_lines maps
  |> List.filter_map ~f:(fun line ->
      match String.lsplit2 line ~on:'/' with Some (_, path) -> Some ("/" ^ path) | None -> None)

let is_basename_prefix ~prefix path = String.is_prefix (Stdlib.Filename.basename path) ~prefix

(* Whether this system has a libmvec to be forced into: one beside the libm the process mapped. *)
let libmvec_available maps =
  List.exists (mapped_paths maps) ~f:(fun path ->
      is_basename_prefix ~prefix:"libm.so" path
      && Stdlib.Sys.file_exists
           (Stdlib.Filename.concat (Stdlib.Filename.dirname path) "libmvec.so.1"))

let is_libmvec = is_basename_prefix ~prefix:"libmvec"
let libmvec_mapped maps = List.exists (mapped_paths maps) ~f:is_libmvec

(* An empty mapping list would pass [not libmvec_mapped] vacuously; the process maps at least its
   own executable, so an empty one is a misread and fails. *)
let libmvec_unmapped maps =
  let paths = mapped_paths maps in
  (not (List.is_empty paths)) && not (List.exists paths ~f:is_libmvec)

(* Whether the configured compiler command is gcc: it predefines [__GNUC__] without [__clang__]
   (which clang, and the compilers built on it, define beside [__GNUC__]). Asked only on Linux,
   where the shell spelling below holds. *)
let compiler_is_gcc () =
  match Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command" with
  | "" -> false
  | command ->
      let defines macro =
        Stdlib.Sys.command
          (Printf.sprintf "%s -dM -E - </dev/null 2>/dev/null | grep -q '#define %s '" command macro)
        = 0
      in
      defines "__GNUC__" && not (defines "__clang__")

let () =
  let before = maps () in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  Stdio.printf "backend: %s\n" backend;
  p "cc_backend_fast_math is on"
    (Utils.get_global_flag ~default:false ~arg_name:"cc_backend_fast_math");
  let x = TDSL.range n in
  let%op y = exp (x /. 64.) in
  (* Raises at dlopen, failing the run, when the kernel references a symbol nothing supplies. *)
  let ctx = Ocannl.Train.forward_once ctx y in
  let after = maps () in
  let got = Context.get_values ctx y.Tensor.value in
  let want = Array.init n ~f:(fun i -> Float.exp (Float.of_int i /. 64.)) in
  (* libmvec's single-precision [expf] variants are within a few ulp; 1e-5 relative is a loose bound
     on that and a tight one on anything other than [exp]. *)
  p_all2 "kernel loaded and computed exp to 1e-5 relative" got want ~f:(fun g w ->
      Float.(abs (g - w) <= 1e-5 * w));
  let gcc_on_glibc =
    (match before with Some m -> libmvec_available m | None -> false) && compiler_is_gcc ()
  in
  gated ~aggregation:`Environment ~when_:gcc_on_glibc ~on:"no glibc libmvec with gcc"
    "libmvec unmapped before the kernel loaded"
    (match before with Some m -> libmvec_unmapped m | None -> false);
  gated ~aggregation:`Environment ~when_:gcc_on_glibc ~on:"no glibc libmvec with gcc"
    "libmvec mapped after the kernel loaded (the vector variant was reached)"
    (match after with Some m -> libmvec_mapped m | None -> false)
