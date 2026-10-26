(* gh-ocannl-1314, gh-ocannl-1529, gh-ocannl-1605: a backend with a kernel code cache (HIP's
   [hip_kernel_cache_mib], CUDA's [cuda_kernel_cache_mib], cc's [cc_kernel_cache_mib], Metal's
   [metal_kernel_cache_mib]) compiles only the kernels no earlier compile in the process produced,
   and reuses the rest -- without changing what any kernel computes. Where a link creates a pipeline
   object per kernel (Metal's compute pipeline states), a reused kernel reuses its pipeline too.

   The routine is a materialized contraction feeding a reduction, which a search's fissioned
   candidates split into several kernels. Every input is a small integer, so every product and sum
   is exact in single precision whatever order a schedule reduces in, and the outputs are compared
   bitwise against a host reference (and the inputs vary along every axis, so a kernel reading the
   wrong element shows).

   - The same computation compiled twice: the second compile hands the vendor compiler no kernel,
   and its outputs equal the first's. - A search: some candidate compiles part of its kernels and
   reuses the rest -- its changed kernels miss and its unchanged ones hit, so the key both separates
   and joins -- and the tuned routine's outputs equal the reference.

   - gh-ocannl-1525: an unchanged kernel keys alike whatever names the rest of the batch or the
   process made it take. A fresh build of a graph whose labels repeat (so its nodes are emitted
   under disambiguated, freshly numbered names) compiles no kernel; the same computation under
   another routine name compiles afresh, the routine's name being one of the things that decide
   whether its text compiles at all (only a segment's positional index is dropped from the key,
   which [kernel_code_cache] pins). (Two isomorphic kernels of one batch compiling once is pinned by
   [kernel_code_cache], against a fake compiler: a plain compile here renders the routine as one
   kernel on every backend.)

   The reuse claims are gated on the compile census's [reuse_capable] (a compile went through a
   kernel code cache), never on a backend name, and that flag stays set with the cache disabled: a
   run with [OCANNL_HIP_KERNEL_CACHE_MIB=0] on HIP, [OCANNL_CUDA_KERNEL_CACHE_MIB=0] on CUDA or
   [OCANNL_METAL_KERNEL_CACHE_MIB=0] on Metal must fail them, the negative control. The pipeline
   claims are gated on the census having seen the first compile create pipelines
   ([pipelines_created]), so a cache that reused libraries but recreated every pipeline fails
   them. *)

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

(* Whether this backend's links create a pipeline object per kernel, read off the same compile. *)
let creates_pipelines = ref false

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
  creates_pipelines := c1.pipelines_created > 0;
  let on = backend and when_ = c1.reuse_capable in
  p "first compile: outputs equal the host reference" (bitwise_equal first expected);
  p "second compile: outputs equal the first compile's" (bitwise_equal second first);
  gated ~when_ ~on "the routine compiled at least one kernel" (c1.kernels >= 1);
  gated ~when_ ~on "first compile: every kernel it rendered was compiled"
    (c1.kernels_compiled = c1.kernels);
  gated ~when_ ~on "second compile: the same kernels rendered, none compiled"
    (c2.kernels = c1.kernels && c2.kernels_compiled = 0);
  let when_ = !creates_pipelines in
  gated ~when_ ~on "first compile: a pipeline created for every kernel"
    (c1.pipelines_created = c1.kernels);
  gated ~when_ ~on "second compile: no pipeline created" (c2.pipelines_created = 0);
  Stdio.eprintf
    "gpu_kernel_cache (not part of the golden): first compile %d of %d kernels compiled, second %d \
     of %d; pipelines created %d then %d; device %s s then %s s\n\
     %!"
    c1.kernels_compiled c1.kernels c2.kernels_compiled c2.kernels c1.pipelines_created
    c2.pipelines_created
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
  gated_exists ~when_:!creates_pipelines ~on:backend
    "some candidate created pipelines for part of its kernels and reused the rest"
    candidates_compiled ~f:(fun rc ->
      let c = rc.compile_cost in
      c.pipelines_created > 0 && c.pipelines_created < c.kernels);
  List.iter candidates_compiled ~f:(fun rc ->
      Stdio.eprintf
        "gpu_kernel_cache (not part of the golden): %s: %d of %d kernels compiled, %d pipelines \
         created\n\
         %!"
        rc.compile_attempt rc.compile_cost.kernels_compiled rc.compile_cost.kernels
        rc.compile_cost.pipelines_created)

(* gh-ocannl-1525. Other extents than the first sections', so no kernel of theirs answers for these.
   Two inputs share one label, so the batch-wide disambiguation emits both under their numbered
   names ([n<id>_hkf_x]), which a fresh build numbers afresh. *)
let fn = 3
let fk = 5
let fm = 7

let fxv =
  Array.init
    (b * fn * fk)
    ~f:(Ll_test.cycle_flat ~dims:[| b; fn; fk |] ~modulus:7 ~offset:(-3.) ~stride:1.)

let fwv =
  Array.init (fk * fm)
    ~f:(Ll_test.cycle_flat ~dims:[| fk; fm |] ~modulus:5 ~offset:(-2.) ~stride:1.)

(* r[b,i] = 2 * sum_j sum_k x[b,i,k] w[k,j]: both inputs hold the same values. *)
let doubled =
  Array.init (b * fn) ~f:(fun bi ->
      let bb = bi / fn and i = bi % fn in
      let acc = ref 0. in
      for j = 0 to fm - 1 do
        for kk = 0 to fk - 1 do
          acc := !acc +. (fxv.((((bb * fn) + i) * fk) + kk) *. fwv.((kk * fm) + j))
        done
      done;
      2. *. !acc)

let collide_graph () =
  let x1 = TDSL.ndarray fxv ~label:[ "hkf_x" ] ~batch_dims:[ b ] ~output_dims:[ fn; fk ] () in
  let x2 = TDSL.ndarray fxv ~label:[ "hkf_x" ] ~batch_dims:[ b ] ~output_dims:[ fn; fk ] () in
  let w = TDSL.ndarray fwv ~label:[ "hkf_w" ] ~output_dims:[ fk; fm ] () in
  let%op z1 = x1 +* "b|ik;kj=>b|ij" w in
  let%op z2 = x2 +* "b|ik;kj=>b|ij" w in
  Train.set_materialized z1.Tensor.value;
  Train.set_materialized z2.Tensor.value;
  let%op r = z1 + z2 ++ "b|ij => b|i" in
  r

let compile_run name r =
  let comp = named name (Train.forward r) in
  let ctx = Context.auto () in
  let (ctx, routine), cost =
    CC.with_census (fun () -> Context.compile ctx comp Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  (Context.backend_name ctx, Context.get_values ctx r.Tensor.value, cost)

let () =
  let backend, first, c1 = compile_run "hkf_fresh" (collide_graph ()) in
  let _, second, c2 = compile_run "hkf_fresh" (collide_graph ()) in
  let _, renamed, c3 = compile_run "hkf_renamed" (collide_graph ()) in
  let on = backend and when_ = c1.reuse_capable in
  p "colliding labels: outputs equal the host reference" (bitwise_equal first doubled);
  p "fresh build: outputs equal the host reference" (bitwise_equal second doubled);
  p "renamed routine: outputs equal the host reference" (bitwise_equal renamed doubled);
  gated ~when_ ~on "fresh build, freshly numbered nodes: the same kernels rendered, none compiled"
    (c2.kernels = c1.kernels && c2.kernels_compiled = 0);
  gated ~when_ ~on "another routine name: the same kernels rendered, all compiled afresh"
    (c3.kernels = c1.kernels && c3.kernels_compiled = c3.kernels);
  let when_ = !creates_pipelines in
  gated ~when_ ~on "fresh build, freshly numbered nodes: no pipeline created"
    (c2.pipelines_created = 0);
  gated ~when_ ~on "another routine name: a pipeline created for every kernel"
    (c3.pipelines_created = c3.kernels);
  Stdio.eprintf
    "gpu_kernel_cache (not part of the golden): colliding labels %d of %d kernels compiled, fresh \
     build %d of %d, renamed %d of %d\n\
     %!"
    c1.kernels_compiled c1.kernels c2.kernels_compiled c2.kernels c3.kernels_compiled c3.kernels
