open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train

(* gh-ocannl-1338: a tensor node's label becomes an identifier of the kernel that touches it -- a
   parameter, a pool-pointer local, a kernel-scope array -- so a label that is also a name the
   kernel's toolchain declares breaks the kernel. Three classes reach every backend:

   - a predefined macro ([__LINE__], [__STDC__]; on a GPU dialect [__CUDA_ARCH__],
   [__HIP_DEVICE_COMPILE__], [__METAL_VERSION__]): a declaration does not shadow a macro, it is
   expanded, so [float *__LINE__] reads as [float *42] whatever the body does; - a vendor header's
   macro ([CUDART_INF_FP16], [HIPRT_INF_FP16], Metal's [M_PI_F], [MAXFLOAT]): the same, for names
   outside the reserved namespace; - a vendor TYPE the body spells ([uint4] on Metal, the type of
   every uint4x32 value, there in the body as the scope-local a virtual RNG draw is held in;
   [float4], [half2]): a parameter of that name shadows the type for the rest of the kernel, so the
   next declaration using it does not parse.

   Each case below puts such nodes in one kernel and checks the executed value, not just that
   something compiled: the inputs are distinct powers of two, so the sum is exact and names every
   input it lost. The names are per backend, since only the dialect's own toolchain declares them;
   the C-family names ([__LINE__], [__STDC__], [unix], [linux]) apply to every backend. Which code
   name each node ends up with is the mechanism's business, not this test's (the GPU dialects prefix
   every node identifier, cc disambiguates by rule), so it is not printed. *)

let backend_names = function
  | "metal" -> [ "float4"; "uint4"; "M_PI_F"; "MAXFLOAT"; "HALF_MAX"; "__METAL_VERSION__" ]
  | "cuda" -> [ "float4"; "half2"; "CUDART_INF_FP16"; "CUDART_ONE_BF16"; "__CUDA_ARCH__" ]
  | "hip" -> [ "float4"; "half2"; "HIPRT_INF_FP16"; "__HIP_DEVICE_COMPILE__"; "__HIP__" ]
  | _ -> []

(* Predefined by every C and C++ front end ([__LINE__], [__STDC__] -- nvrtc, hiprtc and MSL's clang
   included), and by GCC and clang in their GNU dialects on the systems they name ([unix], [linux]):
   harmless where a compiler leaves one undefined, a compile failure where it defines it. *)
let common_names = [ "__LINE__"; "__STDC__"; "unix"; "linux" ]

(* 16 elements, so a backend that moves data in vector packs ([float4] on Metal) spells its vector
   type in the body. Small integers: every sum below is exact. *)
let width = 16

let input ~scale label =
  Tensor.term_init
    (Array.init width ~f:(fun i -> Float.of_int (scale * (i + 1))))
    ~label:[ label ] ~grad_spec:Prohibit_grad ~output_dims:[ width ] ()

(* One kernel per name, so a failure names its label: the node labelled [name] and a plainly
   labelled one, summed. [Ok] carries the executed sum, [Error] what compiling or running raised. *)
let sum_beside_plain name =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let named = input ~scale:1 name and plain = input ~scale:100 "plain" in
  let total = NTDSL.add named plain () in
  List.iter [ named; plain; total ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  match Train.forward_once ctx total with
  | ctx -> Ok (Context.get_values ctx total.value)
  | exception e -> Error (Stdlib.Printexc.to_string e)

let () =
  let backend = Context.backend_name (Context.auto ()) in
  let names = common_names @ backend_names backend in
  Stdio.eprintf "vendor_node_names: backend %s, labels %s (not part of the golden)\n%!" backend
    (String.concat ~sep:" " names);
  p_all "a kernel with a node named after a toolchain macro or vendor type computes its exact sum"
    names ~f:(fun name ->
      match sum_beside_plain name with
      | Ok got
        when Array.equal Float.equal got
               (Array.init width ~f:(fun i -> Float.of_int (101 * (i + 1)))) ->
          true
      | Ok got ->
          Stdio.eprintf "vendor_node_names: %s: wrong sum, first element %g, want 101\n%!" name
            got.(0);
          false
      | Error text ->
          Stdio.eprintf "vendor_node_names: %s: %s\n%!" name (String.prefix text 600);
          false)

(* A vendor type breaks a kernel only where the kernel spells it after the node that shadows it.
   Metal stores a uint4x32 value as [uint4], so a materialized uint4x32 node in the same kernel
   declares [device uint4* ...] beside the float node labelled [uint4]. The draw enters the sum as
   [relu (neg u)], exactly zero for a uniform draw in [0, 1), so the sum stays exact. Elsewhere the
   uint4x32 storage type is the emitter's own [uint4x32_t], and the claim holds trivially. *)
let () =
  let got =
    try
      Tensor.unsafe_reinitialize ();
      let ctx = Context.auto () in
      let named = input ~scale:1 "uint4" in
      let rng = NTDSL.threefry4x32 (NTDSL.bits 7L) (NTDSL.bits 9L) () in
      let draw = NTDSL.uint4x32_to_prec_uniform1 rng () in
      let total = NTDSL.add named (NTDSL.relu (NTDSL.neg draw ()) ()) () in
      List.iter [ named; rng; total ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
      let ctx = Train.forward_once ctx total in
      Ok (Context.get_values ctx total.value)
    with e -> Error (Stdlib.Printexc.to_string e)
  in
  (match got with
  | Error text ->
      Stdio.eprintf "vendor_node_names: uint4 beside uint4x32: %s\n%!" (String.prefix text 600)
  | Ok _ -> ());
  p "a node named uint4 coexists with a uint4x32 node of its kernel"
    (match got with
    | Ok got -> Array.equal Float.equal got (Array.init width ~f:(fun i -> Float.of_int (i + 1)))
    | Error _ -> false)
