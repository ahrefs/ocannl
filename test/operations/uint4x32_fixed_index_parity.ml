open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

(* gh-ocannl-1468: a [uint4x32] counter set from a fixed index ([Range_over_offsets] over an
   all-unit shape lowers to [Embed_index (Fixed_idx 0)]) must convert to its four words the same way
   whether it is inlined beneath the threefry that reads it or materialized and read back. The
   inlined copy renders the index through [int32_to_uint4x32]; a fold [Embed_index (Fixed_idx i)] ->
   [Constant i] in the materialized store's simplify re-keyed the conversion on a double source
   ([double_to_uint4x32]: {0,0,0,0} for index 0, against {0, 0x9E3779B9, 0x6C078965, 0}). Simplify
   now reaches both placements at the counter's [uint4x32] slot (gh-ocannl-1498), and keeps the
   index there.

   Each case builds the packed [uniform] chain by hand, with the counter a tensor of its own, and
   runs it twice: the counter forced [Virtual], then forced materialized. The draws must agree bit
   for bit, and the generated code must show the two placements really happened (the materialized
   run stores the counter, the virtual one does not). *)

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let routine = "fxu_fwd"

(* [uniform] with its counter exposed: [threefry(threefry(self_id, seed), counter)], the counter
   pinned to the result's shape, the bits spread into [prec] lanes. Both runs create tensors in the
   same order after [unsafe_reinitialize], so the self ids (the keys) line up. *)
let run ~virtual_counter ~prec ~packed output_dims =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let grad_spec = Tensor.Prohibit_grad in
  let counter = NTDSL.range_of_shape ~label:[ "fxu_counter" ] ~output_dims () in
  let key =
    Operation.threefry4x32 (NTDSL.embed_self_id ()) (Tensor.get_random_seed ()) ~grad_spec ()
  in
  let bits =
    Operation.threefry4x32 ~spec:Operation.pin_counter_spec key counter ~grad_spec ~output_dims ()
  in
  let u =
    if packed then Operation.uint4x32_to_prec_uniform ~grad_spec bits ~label:[ "fxu" ] ()
    else Operation.uint4x32_to_prec_uniform1 ~grad_spec bits ~label:[ "fxu" ] ()
  in
  Ir.Tnode.update_prec u.value prec;
  if virtual_counter then Train.set_virtual counter.value else Train.set_materialized counter.value;
  Generated.arm routine;
  let ctx = Train.forward_once ctx u in
  let values = Context.get_values ctx u.value in
  (values, Generated.read routine, counter.value)

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* The kernel's statements only, an outlined kernel's parts included (gh-ocannl-1678): the prepended
   builtins define every conversion. *)
let body s = Generated.main_logic s

let check ~name ~prec ~packed output_dims =
  Verdict.case name @@ fun () ->
  let mat_vals, mat_src, counter = run ~virtual_counter:false ~prec ~packed output_dims in
  let vir_vals, vir_src, _ = run ~virtual_counter:true ~prec ~packed output_dims in
  let stores_counter src =
    String.is_substring (body src) ~substring:(Ir.Tnode.debug_name counter ^ "[")
  in
  Stdio.printf "%s: %d values drawn\n" name (Array.length mat_vals);
  p (name ^ ": the materialized run stores the counter") (stores_counter mat_src);
  p (name ^ ": the virtual run inlines the counter") (not (stores_counter vir_src));
  p_all2
    (name ^ ": inlined and materialized counters draw bitwise-equal values")
    vir_vals mat_vals ~f:bits_equal

let () =
  check ~name:"uniform1 single n=1" ~prec:Ir.Ops.single ~packed:false [ 1 ];
  check ~name:"packed uniform single n=1" ~prec:Ir.Ops.single ~packed:true [ 1 ];
  check ~name:"packed uniform half n=1" ~prec:Ir.Ops.half ~packed:true [ 1 ]
