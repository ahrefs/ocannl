(* gh-ocannl-1314, gh-ocannl-1529: a backend with a kernel code cache (HIP's [hip_kernel_cache_mib],
   CUDA's [cuda_kernel_cache_mib]) compiles only the kernels no earlier compile in the process
   produced, and reuses the rest -- without changing what any kernel computes.

   The routine is a materialized contraction feeding a reduction, which a search's fissioned
   candidates split into several kernels. Every input is a small integer, so every product and sum
   is exact in single precision whatever order a schedule reduces in, and the outputs are compared
   bitwise against a host reference (and the inputs vary along every axis, so a kernel reading the
   wrong element shows).

   - The same computation compiled twice: the second compile hands the vendor compiler no kernel,
   and its outputs equal the first's. - A search: some candidate compiles part of its kernels and
   reuses the rest -- its changed kernels miss and its unchanged ones hit, so the key both separates
   and joins -- and the tuned routine's outputs equal the reference.

   The reuse claims are gated on the compile census's [reuse_capable] (a compile went through a
   kernel code cache), never on a backend name, and that flag stays set with the cache disabled: a
   run with [OCANNL_HIP_KERNEL_CACHE_MIB=0] on HIP, or [OCANNL_CUDA_KERNEL_CACHE_MIB=0] on CUDA,
   must fail them, the negative control. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Asgns = Ir.Assignments
module CC = Ir.Compile_cost

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let b = 2
let n = 4
let k = 8
let m = 6

let xv =
  Array.init
    (b * n * k)
    ~f:(Ll_test.cycle_flat ~dims:[| b; n; k |] ~modulus:7 ~offset:(-3.) ~stride:1.)

let wv =
  Array.init (k * m) ~f:(Ll_test.cycle_flat ~dims:[| k; m |] ~modulus:5 ~offset:(-2.) ~stride:1.)

(* r[b,i] = sum_j sum_k x[b,i,k] w[k,j]. *)
let expected =
  Array.init (b * n) ~f:(fun bi ->
      let bb = bi / n and i = bi % n in
      let acc = ref 0. in
      for j = 0 to m - 1 do
        for kk = 0 to k - 1 do
          acc := !acc +. (xv.((((bb * n) + i) * k) + kk) *. wv.((kk * m) + j))
        done
      done;
      !acc)

let graph prefix =
  let x = TDSL.ndarray xv ~label:[ prefix ^ "x" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w = TDSL.ndarray wv ~label:[ prefix ^ "w" ] ~output_dims:[ k; m ] () in
  let%op z = x +* "b|ik;kj=>b|ij" w in
  Train.set_materialized z.Tensor.value;
  let%op r = z ++ "b|ij => b|i" in
  r

let bitwise_equal a e =
  Array.equal (fun x y -> Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)) a e

(* Whether this backend compiles through a kernel code cache, read off the first section's own
   compile: an invariant of the backend, not of the search population the second section
   quantifies. *)
let reuse_capable = ref None

let () =
  let r = graph "hkc_" in
  let comp = named "hkc_twice" (Train.forward r) in
  let compile_and_run () =
    let ctx = Context.auto () in
    let (ctx, routine), cost =
      CC.with_census (fun () -> Context.compile ctx comp Ir.Indexing.Empty)
    in
    let ctx = Context.run ctx routine in
    (Context.backend_name ctx, Context.get_values ctx r.Tensor.value, cost)
  in
  let backend, first, c1 = compile_and_run () in
  let _, second, c2 = compile_and_run () in
  reuse_capable := Some (backend, c1.reuse_capable);
  let on = backend and when_ = c1.reuse_capable in
  p "first compile: outputs equal the host reference" (bitwise_equal first expected);
  p "second compile: outputs equal the first compile's" (bitwise_equal second first);
  gated ~when_ ~on "the routine compiled at least one kernel" (c1.kernels >= 1);
  gated ~when_ ~on "first compile: every kernel it rendered was compiled"
    (c1.kernels_compiled = c1.kernels);
  gated ~when_ ~on "second compile: the same kernels rendered, none compiled"
    (c2.kernels = c1.kernels && c2.kernels_compiled = 0);
  Stdio.eprintf
    "gpu_kernel_cache (not part of the golden): first compile %d of %d kernels compiled, second %d \
     of %d; device %s s then %s s\n\
     %!"
    c1.kernels_compiled c1.kernels c2.kernels_compiled c2.kernels
    (Option.value_map c1.device_s ~default:"none" ~f:(Printf.sprintf "%.3f"))
    (Option.value_map c2.device_s ~default:"none" ~f:(Printf.sprintf "%.3f"))

let () =
  let r = graph "hkt_" in
  let comp = named "hkc_tune" (Train.forward r) in
  let records = ref [] in
  Autotune.install_compile_observer (fun rc -> records := rc :: !records);
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let backend = Context.backend_name ctx in
  let tuned = Context.get_values ctx r.Tensor.value in
  let records = List.rev !records in
  let candidates_compiled =
    List.filter records ~f:(fun (rc : Autotune.compile_record) ->
        String.equal rc.compile_stage "candidate" && String.equal rc.compile_verdict "compiled")
  in
  p "the tuned routine's outputs equal the host reference" (bitwise_equal tuned expected);
  let when_ = Option.value_map !reuse_capable ~default:false ~f:snd in
  gated_exists ~when_ ~on:backend "some candidate compiled part of its kernels and reused the rest"
    candidates_compiled ~f:(fun rc ->
      let c = rc.compile_cost in
      c.kernels_compiled > 0 && c.kernels_compiled < c.kernels);
  List.iter candidates_compiled ~f:(fun rc ->
      Stdio.eprintf "gpu_kernel_cache (not part of the golden): %s: %d of %d kernels compiled\n%!"
        rc.compile_attempt rc.compile_cost.kernels_compiled rc.compile_cost.kernels)
