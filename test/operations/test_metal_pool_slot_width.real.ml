(* Regression test for gh-ocannl-344: under [large_models] the per-pool 4 GB cap is lifted, so a
   pooled byte offset can exceed UINT32_MAX. The Metal pooled slot table -- and the MSL type the
   generated shader declares for it -- must therefore be 64-bit ([ulong]); a 32-bit ([uint]) table
   would silently truncate large-model offsets, defeating the "large_models=true => 64-bit" AC for
   the Metal pool path.

   This compiles a pooled Metal kernel with [large_models = true] and inspects the emitted shader.
   The invariant pinned: the generated source declares [ulong* __pool_slots] and NOT [uint*
   __pool_slots]. If the slot table regressed to [uint] under [large_models], the first line would
   print [false] (and the second [true]). The harness condition that instantiates the AC is
   [large_models = true] set before compilation -- the same kernel under the default setting emits
   [uint], so the setting is what the assertion actually exercises. *)

open! Base
open Ocannl
open Operation.DSL_modules

let make_const label v =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| 2 |] in
  Genarray.set ga [| 0 |] v;
  Genarray.set ga [| 1 |] (v +. 0.5);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()

let build_prefix = "test_metal_pool_slot_width"

let read_metal_sources () =
  let dir = Stdlib.Filename.concat "build_files" build_prefix in
  (try Stdlib.Sys.readdir dir |> Array.to_list with _ -> [])
  |> List.filter ~f:(String.is_suffix ~suffix:".metal")
  |> List.map ~f:(fun f -> Stdio.In_channel.read_all (Stdlib.Filename.concat dir f))

let () =
  Tensor.unsafe_reinitialize ();
  Utils.settings.large_models <- true;
  Utils.settings.output_debug_files_in_build_directory <- true;
  Unix.putenv "OCANNL_BUILD_FILES_PREFIX" build_prefix;
  let ctx = Context.metal () in
  let sum = TDSL.O.(make_const "a" 1. + make_const "b" 2.) in
  let _ctx = Train.forward_once ctx sum in
  let srcs = read_metal_sources () in
  let has sub = List.exists srcs ~f:(String.is_substring ~substring:sub) in
  Stdio.printf "large_models=true: generated slot table is ulong* __pool_slots = %b\n"
    (has "ulong* __pool_slots");
  Stdio.printf "large_models=true: generated slot table is uint* __pool_slots = %b\n"
    (has "uint* __pool_slots")
