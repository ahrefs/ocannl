(* Regression coverage for the Metal side of gh-343: guarded dynamic gather uses [double] as a
   signed guard precision in the shared lowering, but Metal has no native double. The backend must
   keep true double storage unsupported while rendering this scalar guard as float casts. *)

open! Base
open Ocannl
open Operation.DSL_modules

let build_prefix = "test_metal_guarded_gather_codegen"
let vocab = 4
let embed = 3

let read_metal_sources () =
  let dir = Stdlib.Filename.concat "build_files" build_prefix in
  (try Stdlib.Sys.readdir dir |> Array.to_list with _ -> [])
  |> List.filter ~f:(String.is_suffix ~suffix:".metal")
  |> List.map ~f:(fun file -> Stdio.In_channel.read_all (Stdlib.Filename.concat dir file))

let build_embedding id_values =
  let ids =
    TDSL.ndarray id_values ~label:[ "ids" ]
      ~batch_dims:[ Array.length id_values ]
      ~output_dims:[] ()
  in
  let table =
    TDSL.ndarray
      (Array.init (embed * vocab) ~f:Float.of_int)
      ~label:[ "C" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  let classes = TDSL.range vocab in
  let%op one_hot = classes = ids in
  let%op embedded = table * one_hot in
  embedded

let () =
  Tensor.unsafe_reinitialize ();
  Utils.settings.output_debug_files_in_build_directory <- true;
  Unix.putenv "OCANNL_BUILD_FILES_PREFIX" build_prefix;
  let ctx = Context.metal () in
  let embedded = build_embedding [| 1.; 3.; 0. |] in
  let _ctx = Train.forward_once ctx embedded in
  let sources = read_metal_sources () in
  let has substring = List.exists sources ~f:(String.is_substring ~substring) in
  Stdio.printf "guarded gather source casts signed guard to float = %b\n"
    (has "select((float)(0), C[" && has "(float)(ids[");
  Stdio.printf "guarded gather source contains no double declarations = %b\n" (not (has "double"))
