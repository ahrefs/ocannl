(* gh-ocannl-1457: under a compile that pins contraction off, a tensorized matmul's [Tile_mma]
   scalar fallback rounds exactly as the serial loop does -- twice per k step, in serial order -- on
   NON-DYADIC f32 operands, where one rounding per k step and two differ.

   [Schedule.Tensorize] used to write every floating fallback as an explicit [FMA] (gh-ocannl-998),
   while the serial loop's [acc + a * b] is the C compiler's to contract. Under
   [cc_backend_fp_contract=off], which [profile=reproducible] pins, nothing contracts the serial
   loop, so a packed seed and the untuned kernel differed by an ulp wherever the seed won the timing
   (gh-ocannl-1682's residue). The renderer now fuses the fallback only where the compile leaves
   contraction to the compiler ([C_syntax_config.compiler_contraction_off]).

   The register tile itself is held to the same equality by [schedule_pad]'s
   [runtest-schedule_pad-contract_off] rule; this test pins the fallback, which is what the
   reproducible profile renders ([cc_vector_bytes=0] makes the register tile decline). Both pins are
   set in-process before the first compile. Values go through the bits, and the kernel is held to a
   host emulation of the serial f32 chain, so the claim says which arithmetic ran, not only that two
   kernels agree; the data are shown to separate one rounding from two by the same emulation fused.
   cc only: elsewhere every claim is skipped. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let () = Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" "off"
let () = Unix.putenv "OCANNL_CC_VECTOR_BYTES" "0"
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cc = List.mem [ "cc"; "multidev_cc" ] backend_name ~equal:String.equal
let accum_prec = Ll_test.backend_accum_prec

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let m_ext, n_ext, k_ext = (33, 40, 37)

(* Steps of a tenth and of a third, cycled over prime moduli, as in [schedule_pad]: every product
   and partial sum rounds at f32. *)
let a_vals =
  Array.init (m_ext * k_ext)
    ~f:(Ll_test.cycle_flat ~dims:[| m_ext; k_ext |] ~modulus:97 ~offset:(-1.3) ~stride:0.1)

let b_vals =
  Array.init (k_ext * n_ext)
    ~f:(Ll_test.cycle_flat ~dims:[| k_ext; n_ext |] ~modulus:89 ~offset:(-0.7) ~stride:(1. /. 3.))

(* Host emulation of the f32 serial chain [c[i,j] = sum_k a[i,k] * b[k,j]], k ascending from +0. An
   f32 product is exact in a double and an f32 sum's double rounding is innocuous (53 >= 2 * 24 +
   2), so [unfused] is the correctly rounded two-rounding chain. [fused] rounds the double [fma]
   once more to f32: not guaranteed correctly rounded, which is fine for its one use -- showing the
   data move under a single rounding. *)
let r32 x = Int32.float_of_bits (Int32.bits_of_float x)
let a32 i k = r32 a_vals.((i * k_ext) + k)
let b32 k j = r32 b_vals.((k * n_ext) + j)

let chain ~step =
  Array.init (m_ext * n_ext) ~f:(fun c ->
      let i = c / n_ext and j = c % n_ext in
      let acc = ref 0. in
      for k = 0 to k_ext - 1 do
        acc := step !acc (a32 i k) (b32 k j)
      done;
      !acc)

let unfused = chain ~step:(fun acc a b -> r32 (acc +. r32 (a *. b)))
let fused = chain ~step:(fun acc a b -> r32 (Stdlib.Float.fma a b acc))
let bits x = Int64.bits_of_float x
let bitwise a b = Array.equal (fun x y -> Int64.equal (bits x) (bits y)) a b

let mk name =
  let ma =
    TDSL.ndarray a_vals ~label:[ name ^ "a" ] ~input_dims:[ k_ext ] ~output_dims:[ m_ext ] ()
  in
  let mb =
    TDSL.ndarray b_vals ~label:[ name ^ "b" ] ~input_dims:[ n_ext ] ~output_dims:[ k_ext ] ()
  in
  let%op nc = ma * mb in
  nc

let cpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    simd_vector = { Ir.Vector_file.bytes = 32; registers = 16 };
  }

(* The claims, open on cc and skipped by name elsewhere, each written once (gh-ocannl-1703). *)
let cc_leg = Verdict.leg ~when_:on_cc ~on:backend_name ()

let results =
  lazy
    (let seeds = ref [] in
     let serial =
       let nc = mk "tfu0" in
       let ctx, routine =
         Context.compile
           ~lowered_transform:(fun opt ->
             seeds :=
               List.filter
                 (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt)
                 ~f:(fun p ->
                   p.Autotune.sk_mma && (not p.Autotune.sk_gpu) && (not p.Autotune.sk_epilogue)
                   && Option.is_none p.Autotune.sk_tile);
             [ opt ])
           (Context.auto ())
           (named "tfu_serial" (Train.forward nc))
           Ir.Indexing.Empty
       in
       Context.get_values (Context.run ctx routine) nc.Tensor.value
     in
     let runs =
       List.mapi (List.take !seeds 3) ~f:(fun i (p : Autotune.sketch_params) ->
           let nc = mk (Printf.sprintf "tfu%d" (i + 1)) in
           let transform opt = Sched.apply (Autotune.sketch_schedule ~accum_prec ~p opt) opt in
           let ctx, routine =
             Context.compile
               ~lowered_transform:(fun o -> [ transform o ])
               (Context.auto ())
               (named (Printf.sprintf "tfu_seed_%d" i) (Train.forward nc))
               Ir.Indexing.Empty
           in
           let got = Context.get_values (Context.run ctx routine) nc.Tensor.value in
           let max_diff =
             Array.foldi got ~init:0. ~f:(fun c acc g ->
                 Float.max acc (Float.abs (g -. unfused.(c))))
           in
           Stdio.eprintf
             "packed seed %d hoist=%b grid=%b: max |fallback - unfused chain| = %h (not part of \
              the golden)\n\
              %!"
             i p.sk_hoist p.sk_grid max_diff;
           (routine.Context.mma.Ir.C_syntax.tensorization, got))
     in
     (serial, runs))

let () =
  p "the host emulation separates one rounding per k step from two" (not (bitwise unfused fused));
  in_leg ~leg:cc_leg "the serial loop computes the two-rounding f32 chain bitwise" (fun label ->
      p label (bitwise (fst (Lazy.force results)) unfused));
  in_leg ~leg:cc_leg "every packed seed renders its Tile_mma through the scalar fallback"
    (fun label ->
      p_all ~min:2 label
        (snd (Lazy.force results))
        ~f:(fun (t, _) -> Ir.C_syntax.equal_tensorization t Ir.C_syntax.Scalar_fallback));
  in_leg ~leg:cc_leg "every packed seed's fallback computes the two-rounding f32 chain bitwise"
    (fun label ->
      p_all ~min:2 label (snd (Lazy.force results)) ~f:(fun (_, got) -> bitwise got unfused))
