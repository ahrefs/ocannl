(* Accumulator-width policy (gh-ocannl-639): a reduction accumulator over narrow-float storage
   resides at compute precision across the whole reduction nest and narrows once at the store — for
   EVERY rendering, the plain serial fallback included — so the effective accumulation width is set
   by the numerics policy ([Numerics.cpu_compute_prec]), never by which schedule happened to place
   the accumulator in a register. Before gh-ocannl-639 the unscheduled lowering round-tripped the
   accumulator through storage on every reduction step ([mc[..] = single_to_bfloat16(fmaf(...,
   bfloat16_to_single(mc[..])))]), so a bf16 result depended on whether a register-tiling schedule
   ran.

   Inputs are exact in bf16 while their PARTIAL SUMS are not: products are multiples of 15/128 and
   their mean is nonzero (~0.23), so the running sum drifts past the range where bf16's 8
   significand bits can hold such multiples (a zero-mean b operand random-walks below 4 and every
   partial sum stays bf16-exact — the first draft of this test proved that the hard way), and
   per-k-step narrowing visibly diverges from whole-k f32 residency — the policy-off leg is the
   negative control proving the inputs discriminate. The whole reduction stays exact in f32
   (multiples of 1/128, magnitude far below 2^16), so the f64 host-side reference reproduces the
   kernel's fmaf chain exactly, and narrowing it once through the library's own bf16 conversion
   gives the normative result the widened kernel must match bitwise.

   The value claims are policy claims and run wherever the backend's accumulator resolution widens
   bf16 (gh-ocannl-663): the CPU backends ([Numerics.cpu_compute_prec]) and CUDA, whose mma legs
   hold f32 per-lane registers across the whole k extent — the hardware has no bf16 accumulate — so
   its serial legs must match — and HIP, whose [Bf16_auto] resolves wide since gh-ocannl-1051. On
   Metal the tensor unit accumulates in bf16 fragments and the serial legs deliberately keep bf16
   storage residency (width-uniform with its mma legs), so the bf16 widened claims are false there
   BY DESIGN (under the default [Bf16_auto]; the universal legs at the end run them under
   [Bf16_wide], gh-ocannl-838) and skipped — while the fp8 claim, which holds universally, executes
   on every backend. The structural claims grep cc's generated C and the
   SIMD/Workgroup_reduce-serialization legs exercise CPU-only renderings; they stay cc-only and
   print their passing golden line as skipped elsewhere. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* Every label this process has passed or skipped, newest first: [Verdict.passed_labels] carries the
   passes, and [skipped] splices each skip in at the point it happened, so the log keeps the order
   the claims were reported in. *)
let reported = ref []
let synced_passes = ref 0

let sync_reported () =
  let passed = Verdict.passed_labels () in
  reported := List.rev_append (List.drop passed !synced_passes) !reported;
  synced_passes := List.length passed

let skipped c =
  sync_reported ();
  reported := c :: !reported;
  Verdict.skipped ~backend:backend_name c

let on_cpu = Sched.backend_is_cpu backend_name
let codegen_capabilities = Context.codegen_capabilities (Context.auto ())

type rival_values = { once_narrowed : float; per_step : float }
type rival_fixture = { initial : float; increment : float; terms : int; narrow : float -> float }

let values { once_narrowed; per_step } = [ once_narrowed; per_step ]

let render_rivals { initial; increment; terms; narrow } =
  let increments = List.init terms ~f:(fun _ -> increment) in
  let once_narrowed = narrow (List.fold increments ~init:initial ~f:( +. )) in
  let per_step = List.fold increments ~init:initial ~f:(fun acc x -> narrow (acc +. x)) in
  { once_narrowed; per_step }

(* [Bf16_auto], the default bf16 mode, which the gated bf16 block below states its claims for (the
   universal legs at the end name [Bf16_wide]). The stanza declares OCANNL_BF16_ARITHMETIC, so the
   gate and the block both pin it rather than inheriting the ambient mode: under an ambient
   [Bf16_wide] the gate opens on Metal and the narrow_compute_f32 negative controls stay wide
   (gh-ocannl-1078). On HIP [Bf16_auto] itself opens the gate since gh-ocannl-1051; the controls
   that execute there are the fp8 one, whose HIP residency [narrow_compute_f32] still governs, and
   the cc-only bf16 ones skip. *)
let bf16_auto pol = { pol with Numerics.bf16_arithmetic = Numerics.Bf16_auto }

(* Read the same per-backend policy code generation applies, rather than reconstructing it from the
   backend name (gh-ocannl-822). *)
let widens_bf16 =
  Test_utils.with_policy bf16_auto (fun () ->
      not
        (Ir.Ops.equal_prec
           (codegen_capabilities.Ir.Backend_intf.accum_prec Ir.Ops.bfloat16)
           Ir.Ops.bfloat16))

(* Runs the leg only on cc; elsewhere prints the golden line as skipped (the leg exercises a
   CPU-only rendering or greps cc's generated C). *)
let cc_only_claims claims leg = if on_cpu then leg () else List.iter claims ~f:skipped
let cc_only claim leg = cc_only_claims [ claim ] leg

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The single-child chain of loops from the top of each top-level nest ([Ll_test.nest_paths]): used
   to address the reduction axis for the unroll legs. *)
let nest_paths = Ll_test.nest_paths

(* [Sched.Unroll] over the k axis of the matmul's i/j/k nest, in either representation. *)
let unroll_k ~materialize (opt : LL.optimized) : Sched.schedule =
  let k =
    match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ _; _; k ] -> k
    | _ -> assert false
  in
  [ Sched.Unroll { axis = k; materialize } ]

(* The i/r/s nest of the two-axis reduction: pick which reduction axis to transform. *)
let two_axis_sched ~f (opt : LL.optimized) : Sched.schedule =
  match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
  | [ _; r; s ] -> f ~r ~s
  | _ -> assert false

let run ~name ?schedule (out : Tensor.t) =
  let transform opt =
    match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
  in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named name (Train.forward out))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx out.Tensor.value

let n = 64

(* The operands' cells are exact in bf16 while the k-sum's partials are not: see {!Ll_test.cycle}
   for why the cycles are shaped this way, and the header above for this pair's own arithmetic
   (products are multiples of 15/128, mean ~0.23). *)
let fa = Ll_test.cycle ~dims:[| n; n |] ~modulus:3 ~offset:1. ~stride:0.375
let fb = Ll_test.cycle ~dims:[| n; n |] ~modulus:5 ~offset:(-1.5) ~stride:0.625
let claim_parity = "bf16 naive matmul equals the once-narrowed wide-accumulation reference"
let claim_shape = "the emitted serial k-loop narrows the accumulator once per cell, not per step"

let claim_merge_shape =
  "a merge-buffer read is not the accumulator's own cell (the update shape is recognized)"

let claim_off_value =
  "narrow_compute_f32=false recovers per-operator rounding: the result differs from the widened \
   default"

let claim_off_shape = "narrow_compute_f32=false brings back the per-k-step narrowing in the k-loop"

let claim_unroll_annot =
  "Unroll-annotated bf16 reduction keeps the wide accumulator (equals the serial result)"

let claim_unroll_mat =
  "materialized-unroll bf16 reduction keeps the wide accumulator (equals the serial result)"

let claim_2ax_ref = "two-axis bf16 reduction equals the once-narrowed wide-accumulation reference"

let claim_2ax_inner =
  "materialized-unrolled INNER reduction axis equals the serial result (the scope hoists through \
   the outer reduction loop)"

let claim_2ax_priv_outer =
  "a Privatize over the outer reduction axis, around the inner axis's materialized scope, equals \
   the serial result"

let claim_2ax_outer = "materialized-unrolled OUTER reduction axis equals the serial result"
let claim_2ax_annot = "Unroll-annotated both reduction axes equals the serial result"

let claim_2ax_both_mat =
  "BOTH reduction axes materialized sequentially keep the whole-nest accumulator (equals the \
   serial result)"

let claim_2ax_pad_mat =
  "Pad-guarded materialized unroll equals the serial result (the guard peels into the scope)"

let claim_partition =
  "Partition segments share one accumulator scope (equals the unsplit serial result)"

let claim_partition_compose =
  "a partition segment stays addressable by later schedule ops (unrolling a segment equals the \
   serial result)"

let claim_mat_then_inner =
  "inner loops stay reachable after an outer materializing unroll (annotating them equals the \
   serial result)"

let claim_vec_nested_structure = "the nested vectorized reduction rendering fires"

let claim_vec_nested =
  "a vectorized inner reduction axis folds into the whole-nest wide accumulator (equals the serial \
   result)"

let claim_wgr_nested =
  "a serialized nested Workgroup_reduce keeps the whole-nest accumulator (equals the serial result)"

let claim_mixed_scope =
  "a mixed-operator scope is not a reduction and keeps its per-iteration narrowing (256 +1 *1 \
   stays 256 at bf16)"

let claim_simd_tail_structure = "the non-divisible SIMD reduction rendering fires"

let claim_simd_tail =
  "the SIMD reduction folds its non-divisible tail into the wide total (equals the serial result)"

let claim_scope_recurrence =
  "a non-reduction recurrence through a pre-existing scope keeps its per-iteration narrowing (256 \
   -0.5 -0.5 stays 256 at bf16)"

let claim_adjacent =
  "adjacent accumulations into one cell keep their per-assignment narrowing (256 +1 +1 stays 256 \
   at bf16)"

let claim_guarded =
  "index-guarded bf16 reduction accumulates wide across the guard (256 +1x5 narrows once to 260)"

let claim_where_guarded =
  "an index-guarded Where-form update (virtualization's guarded-read shape) keeps the wide \
   accumulator (256 +1x5 narrows once to 260)"

let claim_init_round =
  "a widened scope's opening init keeps its own assignment's rounding (128 + 0.5 rounds before the \
   x3 reduction)"

let claim_off_fp8 = "narrow_compute_f32=false recovers per-step fp8 narrowing (16 + 8x0.5 stays 16)"

let claim_wgreduce =
  "a Workgroup_reduce loop serialized on cc keeps the Serial accumulator width (equals the serial \
   result)"

let claim_fp8 =
  "fp8 e5m2 reduction accumulates wide and narrows once (16 + 8x0.5 reaches 20, not 16)"

let fp8_fixture =
  {
    initial = 16.0;
    increment = 0.5;
    terms = 8;
    narrow = (fun x -> Ir.Ops.fp8_to_single (Ir.Ops.single_to_fp8 x));
  }

let f16_fixture =
  {
    initial = 2048.0;
    increment = 1.0;
    terms = 8;
    narrow = (fun x -> Ir.Ops.half_to_single (Ir.Ops.single_to_half x));
  }

let fp8_values = render_rivals fp8_fixture
let f16_values = render_rivals f16_fixture

(* === fp8 accumulates wide on every backend (gh-ocannl-663) === *)
(* No backend has an fp8 accumulator format (its arithmetic bridges through float per operator
   everywhere, and Metal computes fp8 in f32 wholesale), so fp8 reductions take f32 residency
   universally — the one leg that EXECUTES on every backend, HIP and Metal included, rather than
   riding the bf16 gate (Codex P2 on PR #396). e5m2's 2-bit mantissa makes the discrimination
   cheap: at 16 the spacing is 4, so per-step narrowing absorbs every +0.5 and leaves 16, while
   the wide accumulator reaches 20, exactly representable. *)
let fp8_sum ~name ~first_id () =
  let fp8 = Ir.Ops.fp8 in
  let f8node = Ll_test.node_factory ~prec:fp8 ~first_id ~dims:[| fp8_fixture.terms |] () in
  let f8acc = f8node ~dims:[| 1 |] (name ^ "_acc") in
  let f8xs = f8node (name ^ "_xs") in
  Ll_test.materialize f8acc;
  Ll_test.materialize f8xs;
  let f8i = Ll_test.sym () in
  let f8upd =
    Ll_test.set f8acc
      [| Ll_test.fixed 0 |]
      (LL.Binop
         ( Ir.Ops.Add,
           (Ll_test.get f8acc [| Ll_test.fixed 0 |], fp8),
           (Ll_test.get f8xs [| Ll_test.iter f8i |], fp8) ))
  in
  let f8o =
    Ll_test.optimize ~materialized:[ f8acc; f8xs ] ~name
      (Ll_test.loop_n f8i fp8_fixture.terms f8upd)
  in
  let f8vals =
    Ll_test.execute ~name f8o
      ~seed:
        [
          (f8acc, [| fp8_fixture.initial |]);
          (f8xs, Array.create ~len:fp8_fixture.terms fp8_fixture.increment);
        ]
      ~read:[ f8acc ]
  in
  (List.hd_exn f8vals).(0)

let fp8_leg () =
  p claim_fp8 (Float.equal (fp8_sum ~name:"aw_fp8" ~first_id:9700 ()) fp8_values.once_narrowed)

(* === f16 residency is the fp16_arithmetic policy's question (gh-ocannl-680) === *)
(* Universal legs, executed on every backend. Under the default [Fp16_auto] each backend keeps its
   structural residency — the CPU backends compute f16 in f32 ([narrow_compute_f32]'s blanket), the
   GPU backends keep storage residency, mirroring their f16-accumulate tensor-unit triples — and
   under [Fp16_wide] every backend resolves f32, narrowing once per nest. The discrimination: f16's
   spacing at 2048 is 2, so per-step narrowing absorbs every +1 (2049 ties to even, back to 2048)
   and leaves 2048, while the wide accumulator reaches 2056, exactly representable. NOTE the
   default-policy expectations pin [Fp16_auto]'s CURRENT resolution: auto deliberately retains
   latitude to resolve wide on hardware where that costs nothing (see Numerics.fp16_mode), and a
   backend exercising it would update these legs, not violate the policy. *)
let f16_sum ~name ~first_id () =
  let half = Ir.Ops.half in
  let hnode = Ll_test.node_factory ~prec:half ~first_id ~dims:[| f16_fixture.terms |] () in
  let hacc = hnode ~dims:[| 1 |] (name ^ "_acc") in
  let hxs = hnode (name ^ "_xs") in
  Ll_test.materialize hacc;
  Ll_test.materialize hxs;
  let hi = Ll_test.sym () in
  let hupd =
    Ll_test.set hacc
      [| Ll_test.fixed 0 |]
      (LL.Binop
         ( Ir.Ops.Add,
           (Ll_test.get hacc [| Ll_test.fixed 0 |], half),
           (Ll_test.get hxs [| Ll_test.iter hi |], half) ))
  in
  let ho =
    Ll_test.optimize ~materialized:[ hacc; hxs ] ~name (Ll_test.loop_n hi f16_fixture.terms hupd)
  in
  let hvals =
    Ll_test.execute ~name ho
      ~seed:
        [
          (hacc, [| f16_fixture.initial |]);
          (hxs, Array.create ~len:f16_fixture.terms f16_fixture.increment);
        ]
      ~read:[ hacc ]
  in
  (List.hd_exn hvals).(0)

let claim_f16_default =
  "the default-policy f16 reduction takes the backend's declared residency (2048 + 1x8: wide 2056 \
   on CPU, per-step 2048 on GPU)"

let claim_f16_wide =
  "Fp16_wide widens the f16 reduction on this backend too (2048 + 1x8 gives 2056)"

let claim_f16_wide_ncf32_off =
  "Fp16_wide holds the f32 residency even under narrow_compute_f32=false (2048 + 1x8 gives 2056)"

let claim_f16_vec_decline =
  "a Vectorized f16 reduction under Fp16_wide + narrow_compute_f32=false declines the SIMD chains"

let claim_f16_vec_wide =
  "the declined Vectorized f16 reduction keeps f32 residency (equals the once-narrowed wide \
   reference)"

let claim_f16_serial_wide =
  "the serial f16 reduction under Fp16_wide + narrow_compute_f32=false equals the once-narrowed \
   wide reference"

let claim_f16_wide_matmul =
  "under Fp16_wide the f16 naive matmul equals the once-narrowed wide-accumulation reference"

let claim_f16_default_matmul =
  "the default-policy f16 matmul matches the wide reference exactly where the backend widens (CPU) \
   and diverges where it keeps storage residency (GPU)"

let n16 = 128
let fa16 = Ll_test.cycle ~dims:[| n16; n16 |] ~modulus:3 ~offset:1. ~stride:0.375
let fb16 = Ll_test.cycle ~dims:[| n16; n16 |] ~modulus:5 ~offset:(-1.5) ~stride:0.625

(* The f16 twin of the bf16 parity leg's arithmetic, one exactness bracket up: cells are exact in
   f16 (multiples of 1/8, magnitude below 4), products are multiples of 15/64 with nonzero mean
   (~0.23), so at n = 128 the running sums drift past 16 — where f16's 1/64-multiple partials stop
   being representable and per-step narrowing visibly diverges — while the whole reduction stays
   exact in f32 (multiples of 1/64, magnitude far below 2^20), so the f64 host chain reproduces the
   kernel's f32 fmaf chain exactly and one narrowing through the library's own conversion gives the
   normative wide result. *)
let f16_matmul ~name () =
  let ma = NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.half ~i:[ n16 ] ~o:[ n16 ] ~f:fa16 () in
  let mb = NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.half ~i:[ n16 ] ~o:[ n16 ] ~f:fb16 () in
  let%op mc = ma * mb in
  Tn.update_prec mc.Tensor.value Ir.Ops.half;
  run ~name mc

let () =
  p_pairwise_distinct "the fp8 scalar rival-rendering values are pairwise distinct"
    (values fp8_values) ~equal:Float.equal ~to_string:Float.to_string;
  p_pairwise_distinct "the f16 scalar rival-rendering values are pairwise distinct"
    (values f16_values) ~equal:Float.equal ~to_string:Float.to_string;
  let wide16 =
    Float.equal
      (Test_utils.with_fp16_auto (f16_sum ~name:"aw_f16_auto" ~first_id:9740))
      f16_values.once_narrowed
  in
  Stdio.eprintf "accum_width: default-policy f16 residency on %s is %s (not part of the golden)\n%!"
    backend_name
    (if wide16 then "wide" else "storage");
  p claim_f16_default (Bool.equal wide16 on_cpu);
  let saved_policy = Numerics.get () in
  Numerics.set_policy { saved_policy with fp16_arithmetic = Numerics.Fp16_wide };
  p claim_f16_wide
    (Float.equal (f16_sum ~name:"aw_f16_wide" ~first_id:9760 ()) f16_values.once_narrowed);
  let got_wide16 = f16_matmul ~name:"aw_f16_naive_wide" () in
  (* The wide contract is unconditional: [narrow_compute_f32 = false] leaves f16 COMPUTE at storage
     width (per-operator rounding), but the ACCUMULATOR still resides in f32 and narrows once —
     [Numerics.cpu_accum_prec] diverging from [cpu_compute_prec] here is also what makes the
     register-tile renderings decline rather than accumulate narrowly (Codex P1 round 1 on staging
     PR #477). *)
  Numerics.set_policy
    { saved_policy with fp16_arithmetic = Numerics.Fp16_wide; narrow_compute_f32 = false };
  p claim_f16_wide_ncf32_off
    (Float.equal (f16_sum ~name:"aw_f16_wide_nco" ~first_id:9780 ()) f16_values.once_narrowed);
  (* The [Vectorized] retype's direct-cell SIMD form holds its register chains at COMPUTE precision
     — half, in this policy corner on a native-fp16 target — so it must decline rather than round
     narrowly while the serial schedule localizes at f32 (Codex P1 round 2 on staging PR #477). The
     localizer then wraps the nest at [accum_prec]; the SIMD rendering is attempted again inside the
     scope but [vec_expr]'s compute-width gates decline the half-compute contribution there too, so
     the honest outcome this leg pins is: no vector chains, the localized serial form at f32. The
     value claim discriminates against BOTH failure shapes by comparing to a host-side once-narrowed
     wide reference: half chains (the reported bug) and per-step RMW narrowing each diverge from it
     — cells are f16-exact ({!Ll_test.drift}, bf16-exact hence f16-exact) while the running sums
     (~25) sit past 16, where f16 cannot represent the 1/64 increments — and the whole reduction is
     exact in f32, so the comparison is bitwise. On a promoted-fp16 target the vector-capability
     gate declines the same candidates and the leg still holds. cc-only: the SIMD reduction
     rendering is a CPU form. *)
  cc_only_claims [ claim_f16_vec_decline; claim_f16_vec_wide; claim_f16_serial_wide ] (fun () ->
      let rows, cols = (4, 67) in
      let fv = Ll_test.drift ~dims:[| rows; cols |] in
      let run_f16_sum ~name ?schedule () =
        let xw = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.half ~o:[ rows; cols ] ~f:fv () in
        let outw =
          NTDSL.init ~l:(name ^ "_out") ~prec:Ir.Ops.half ~o:[ rows ] ~f:(fun _ -> 0.0) ()
        in
        Train.set_materialized outw.Tensor.value;
        let comp = named name [%cd outw =+ id xw ~logic:"is => i"] in
        let transform opt =
          match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
        in
        let ctx = Context.auto () in
        let ctx, routine =
          Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
        in
        let ctx = Context.run ctx routine in
        Context.get_values ctx outw.Tensor.value
      in
      let retype opt =
        match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 2) with
        | [ _; s ] -> [ Sched.Retype { axis = s; ty = LL.Vectorized } ]
        | _ -> assert false
      in
      let got_s = run_f16_sum ~name:"aw_f16v_serial" () in
      let got_v = run_f16_sum ~name:"aw_f16v_vec" ~schedule:retype () in
      let wide_v =
        Array.init rows ~f:(fun i ->
            let acc = ref 0.0 in
            for s = 0 to cols - 1 do
              acc := !acc +. fv [| i; s |]
            done;
            !acc)
      in
      let vref =
        NTDSL.init ~l:"aw_f16v_ref" ~prec:Ir.Ops.half ~o:[ rows ]
          ~f:(fun idcs -> wide_v.(idcs.(0)))
          ()
      in
      let want_v = run ~name:"aw_f16v_refc" vref in
      let vec_fired =
        String.is_substring
          (Generated.read ~ext:".c" "aw_f16v_vec")
          ~substring:"Vectorized reduction rendering"
      in
      p claim_f16_vec_decline (not vec_fired);
      p_all2 claim_f16_vec_wide got_v want_v ~f:Float.equal;
      p_all2 claim_f16_serial_wide got_s want_v ~f:Float.equal);
  Numerics.set_policy saved_policy;
  let wide_sums16 =
    Array.init (n16 * n16) ~f:(fun t ->
        let i = t / n16 and j = t % n16 in
        let acc = ref 0.0 in
        for k = 0 to n16 - 1 do
          acc := !acc +. (fa16 [| i; k |] *. fb16 [| k; j |])
        done;
        !acc)
  in
  let mref16 =
    NTDSL.init ~l:"aw_f16_ref" ~prec:Ir.Ops.half ~i:[ n16 ] ~o:[ n16 ]
      ~f:(fun idcs -> wide_sums16.((idcs.(0) * n16) + idcs.(1)))
      ()
  in
  let want16 = run ~name:"aw_f16_refc" mref16 in
  p_all2 claim_f16_wide_matmul got_wide16 want16 ~f:Float.equal;
  let got_auto16 = Test_utils.with_fp16_auto (f16_matmul ~name:"aw_f16_naive_auto") in
  p claim_f16_default_matmul
    ((not (Array.is_empty got_auto16))
    && Bool.equal (Array.for_all2_exn got_auto16 want16 ~f:Float.equal) on_cpu)

(* In execution order — the GPU skip lines must match the cc run's golden line for line. *)
let all_claims =
  [
    claim_parity;
    claim_shape;
    claim_merge_shape;
    claim_unroll_annot;
    claim_unroll_mat;
    claim_partition;
    claim_partition_compose;
    claim_2ax_ref;
    claim_2ax_inner;
    claim_2ax_outer;
    claim_2ax_annot;
    claim_2ax_both_mat;
    claim_2ax_pad_mat;
    claim_mat_then_inner;
    claim_2ax_priv_outer;
    claim_vec_nested_structure;
    claim_vec_nested;
    claim_wgr_nested;
    claim_adjacent;
    claim_guarded;
    claim_where_guarded;
    claim_scope_recurrence;
    claim_mixed_scope;
    claim_init_round;
    claim_wgreduce;
    claim_simd_tail_structure;
    claim_simd_tail;
    claim_fp8;
    claim_off_value;
    claim_off_fp8;
    claim_off_shape;
  ]

let default_bf16_block () =
  if not widens_bf16 then
    (* The fp8 claim holds on HIP and Metal too, so it executes rather than printing a green-by-skip
       line; every bf16-widening leg is skipped. *)
    List.iter all_claims ~f:(fun c -> if String.equal c claim_fp8 then fp8_leg () else skipped c)
  else begin
    sync_reported ();
    let reported_before = List.length !reported in
    let ma = NTDSL.init ~l:"ma" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
    let mb = NTDSL.init ~l:"mb" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
    let%op mc = ma * mb in
    Tn.update_prec mc.Tensor.value Ir.Ops.bfloat16;
    let got = run ~name:"aw_bf16_naive" mc in
    (* The reference: the f64 dot products are exact reproductions of the kernel's f32 fmaf chain
       (every partial sum is a multiple of 1/64 well within f32's mantissa), narrowed exactly once
       per cell by minting a bf16 tensor from them and reading it back. *)
    let wide_sums =
      Array.init (n * n) ~f:(fun t ->
          let i = t / n and j = t % n in
          let acc = ref 0.0 in
          for k = 0 to n - 1 do
            acc := !acc +. (fa [| i; k |] *. fb [| k; j |])
          done;
          !acc)
    in
    let mref =
      NTDSL.init ~l:"mref" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
        ~f:(fun idcs -> wide_sums.((idcs.(0) * n) + idcs.(1)))
        ()
    in
    let want = run ~name:"aw_bf16_ref" mref in
    p_all2 claim_parity got want ~f:Float.equal;
    cc_only claim_shape (fun () ->
        let src = Generated.read ~ext:".c" "aw_bf16_naive" in
        let has s = String.is_substring src ~substring:s in
        p claim_shape ((not (has "single_to_bfloat16(fmaf(")) && has "fmaf("));
    (* Structural: a merge-buffer read is a separate read-only staging buffer, so [p =+ p.merge]
       stays a recognizable accumulation (no same-node merge REDUCTION is constructible — merge
       buffers are node-shaped — so the recognizer shape is the whole reachable surface). *)
    (let pmrg =
       Ll_test.node_factory ~prec:Ir.Ops.bfloat16 ~first_id:9600 ~dims:[| 4 |] () "aw_mrg"
     in
     let km = Ll_test.sym () in
     let mllsc =
       LL.Binop
         ( Ir.Ops.Add,
           (LL.Get (pmrg, [| Ll_test.fixed 0 |]), Ir.Ops.bfloat16),
           (LL.Get_merge_buffer (pmrg, [| Ll_test.iter km |]), Ir.Ops.bfloat16) )
     in
     p claim_merge_shape
       (Option.is_some (LL.accum_update_parts ~tn:pmrg ~idcs:[| Ll_test.fixed 0 |] mllsc)));
    (* Both unroll representations autotune proposes over small reduction loops (annotated: codegen
       repeats the body; materialized: the IR carries one copy per step and no loop) must keep the
       wide accumulator — the reduction is exact in f32 and narrows at the same single point, so
       both are bitwise equal to the serial leg; per-repetition narrowing would visibly diverge on
       these inputs (the same discrimination as the policy-off arm). *)
    let matmul_leg ~claim ~name ~sched =
      let ma_u = NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
      let mb_u = NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
      let%op mc_u = ma_u * mb_u in
      Tn.update_prec mc_u.Tensor.value Ir.Ops.bfloat16;
      let got_u = run ~name ~schedule:sched mc_u in
      p_all2 claim got_u got ~f:Float.equal
    in
    matmul_leg ~claim:claim_unroll_annot ~name:"aw_bf16_unroll_annot"
      ~sched:(unroll_k ~materialize:false);
    matmul_leg ~claim:claim_unroll_mat ~name:"aw_bf16_unroll_mat"
      ~sched:(unroll_k ~materialize:true);
    (* Partition is an index-set specialization of one reduction: its segment seams must not become
       narrowing points, so one accumulator scope spans all the segments (unlike the naive reading
       where each sibling segment loop would widen separately and narrow at every breakpoint —
       visibly on these drifting sums). *)
    matmul_leg ~claim:claim_partition ~name:"aw_bf16_partition" ~sched:(fun opt ->
        let k =
          match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
          | [ _; _; k ] -> k
          | _ -> assert false
        in
        let pt, _segment_syms = Sched.partition ~axis:k ~breakpoints:[ 16; 40 ] in
        [ pt ]);
    (* The segment symbols Schedule.partition returns must stay usable AFTER the accumulation mint
       wrapped the segment loops in the scope: rewrite_loop descends into Local_scope bodies since
       gh-ocannl-639. *)
    matmul_leg ~claim:claim_partition_compose ~name:"aw_bf16_partition_compose" ~sched:(fun opt ->
        let k =
          match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
          | [ _; _; k ] -> k
          | _ -> assert false
        in
        let pt, segment_syms = Sched.partition ~axis:k ~breakpoints:[ 16; 40 ] in
        [ pt; Sched.Unroll { axis = List.hd_exn segment_syms; materialize = false } ]);
    (* === two-axis reduction out[i] = sum_{r,s} x[i,r,s]: unrolling EITHER reduction axis keeps the
       whole-nest accumulator. The inner-axis leg is the partial-materialization shape: the unroll
       mints a scope-form Set inside the still-serial outer reduction loop, and the codegen peel
       hoists that scope through it — without the hoist the accumulator would store and narrow once
       per outer iteration. Cell values are {!Ll_test.drift}: bf16-exact, while the running sums
       (~14.6) are not, and the whole reduction is exact in f32. *)
    let ni, nr, ns = (4, 6, 6) in
    let fx = Ll_test.drift ~dims:[| ni; nr; ns |] in
    let run2 ~name ?schedule () =
      let x2 = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; nr; ns ] ~f:fx () in
      let%op out2 = x2 ++ "irs => i" in
      Tn.update_prec out2.Tensor.value Ir.Ops.bfloat16;
      run ~name ?schedule out2
    in
    let got2 = run2 ~name:"aw2_serial" () in
    let wide2 =
      Array.init ni ~f:(fun i ->
          let acc = ref 0.0 in
          for r = 0 to nr - 1 do
            for s = 0 to ns - 1 do
              acc := !acc +. fx [| i; r; s |]
            done
          done;
          !acc)
    in
    let mref2 =
      NTDSL.init ~l:"aw2_ref" ~prec:Ir.Ops.bfloat16 ~o:[ ni ] ~f:(fun idcs -> wide2.(idcs.(0))) ()
    in
    let want2 = run ~name:"aw2_refc" mref2 in
    p_all2 claim_2ax_ref got2 want2 ~f:Float.equal;
    let leg2 ~claim ~name ~sched =
      let got_l = run2 ~name ~schedule:(two_axis_sched ~f:sched) () in
      p_all2 claim got_l got2 ~f:Float.equal
    in
    leg2 ~claim:claim_2ax_inner ~name:"aw2_unroll_inner" ~sched:(fun ~r:_ ~s ->
        [ Sched.Unroll { axis = s; materialize = true } ]);
    leg2 ~claim:claim_2ax_outer ~name:"aw2_unroll_outer" ~sched:(fun ~r ~s:_ ->
        [ Sched.Unroll { axis = r; materialize = true } ]);
    leg2 ~claim:claim_2ax_annot ~name:"aw2_unroll_annot" ~sched:(fun ~r ~s ->
        [
          Sched.Unroll { axis = r; materialize = false };
          Sched.Unroll { axis = s; materialize = false };
        ]);
    (* Sequential materialization of both axes: the second unroll must recognize the scope form the
       first one minted and reuse its accumulator — the outer loop is gone afterwards, so codegen's
       scope-base hoist could not recover a per-outer-iteration narrowing. *)
    leg2 ~claim:claim_2ax_both_mat ~name:"aw2_unroll_both_mat" ~sched:(fun ~r ~s ->
        [
          Sched.Unroll { axis = s; materialize = true };
          Sched.Unroll { axis = r; materialize = true };
        ]);
    (* Pad puts an [If (s < 6)] index guard on the leaf, then the materializing unroll must peel it
       into the scope: without that, the unroll would emit guarded per-copy Sets with no loop left,
       and every copy would narrow through storage. *)
    leg2 ~claim:claim_2ax_pad_mat ~name:"aw2_unroll_pad_mat" ~sched:(fun ~r:_ ~s ->
        [
          Sched.Pad { axis = s; to_multiple_of = 4 }; Sched.Unroll { axis = s; materialize = true };
        ]);
    (* After an outer materializing unroll, the copied inner s loops live inside the scope and keep
       their symbol: a later op targeting s must reach all of them. *)
    leg2 ~claim:claim_mat_then_inner ~name:"aw2_mat_then_inner" ~sched:(fun ~r ~s ->
        [
          Sched.Unroll { axis = r; materialize = true };
          Sched.Unroll { axis = s; materialize = false };
        ]);
    (* A PRIVATIZED outer reduction axis over the scope form the inner unroll minted (gh-ocannl-1116
       review round 4): swapping the output loop inside the outer reduction loop makes the tile's
       cell vary under it, so no peel reaches the tile — a storage tile would narrow once per outer
       iteration, while code generation hoists the scope-form base through the serial nest. The tile
       must take the residency for a scope-form base too, and the scope must be re-keyed to the
       tile: under [target]'s scope id its opening init re-narrows through bf16 on every outer
       iteration wherever bf16 computes at storage width (CUDA read 14.5625 for 14.5 before the
       re-key). *)
    let accum_target (opt : LL.optimized) =
      let rec go (llc : LL.t) =
        match llc with
        | LL.Set { tn; llsc; _ } when LL.scalar_touches_tn tn llsc -> Some tn
        | LL.Set { llsc = LL.Local_scope _; tn; _ } -> Some tn
        | LL.For_loop { body; _ } | LL.If { body; _ } -> go body
        | LL.Seq (a, b) -> Option.first_some (go a) (go b)
        | _ -> None
      in
      Option.value_exn (go opt.LL.llc)
    in
    let got_priv2 =
      run2 ~name:"aw2_unroll_inner_priv_outer"
        ~schedule:(fun opt ->
          let target = accum_target opt in
          two_axis_sched opt ~f:(fun ~r ~s ->
              let i =
                match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
                | i :: _ -> i
                | [] -> assert false
              in
              [
                Sched.Unroll { axis = s; materialize = true };
                Sched.Swap { outer = i; inner = r };
                Sched.privatize ~accum_prec:codegen_capabilities.Ir.Backend_intf.accum_prec ~target
                  ~over:r;
              ]))
        ()
    in
    p_all2 claim_2ax_priv_outer got_priv2 got2 ~f:Float.equal;
    (* === a VECTORIZED inner reduction axis === *)
    (* The nest peel rides through the Vectorized level, and the SIMD reduction rendering folds
       its chains into the scope LOCAL (no storage round-trip): the whole nest keeps one wide
       accumulator. Inner extent 32 clears the SIMD profitability gate; the reduction is exact in
       f32, so the vector reassociation is harmless and the comparison is bitwise. The structural
       conjunct asserts the vectorized rendering actually fired inside the scope. *)
    cc_only_claims [ claim_vec_nested_structure; claim_vec_nested ] (fun () ->
        let nvr, nvs = (4, 32) in
        let fxv = Ll_test.drift ~dims:[| ni; nvr; nvs |] in
        let run2v ~name ?schedule () =
          let xv =
            NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; nvr; nvs ] ~f:fxv ()
          in
          let%op outv = xv ++ "irs => i" in
          Tn.update_prec outv.Tensor.value Ir.Ops.bfloat16;
          run ~name ?schedule outv
        in
        let got_vn = run2v ~name:"aw_vecnest_serial" () in
        let got_vnv =
          run2v ~name:"aw_vecnest_vec"
            ~schedule:
              (two_axis_sched ~f:(fun ~r:_ ~s -> [ Sched.Retype { axis = s; ty = LL.Vectorized } ]))
            ()
        in
        let vecn_fired =
          String.is_substring
            (Generated.read ~ext:".c" "aw_vecnest_vec")
            ~substring:"Vectorized reduction rendering"
        in
        p claim_vec_nested_structure vecn_fired;
        p_all2 claim_vec_nested got_vnv got_vn ~f:Float.equal);
    (* A hardware-annotated inner reduction axis the backend serializes (cc binds no workgroup
       dimension) is a serial level to the peel, so the whole nest keeps one accumulator instead of
       narrowing once per outer iteration. The [=+]-into-pre-zeroed form again: the einsum
       lowering's whole-node init nest fails [validate_parallel] under a hardware annotation. *)
    let run_sum2 ~name ?schedule () =
      let xw = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; nr; ns ] ~f:fx () in
      let outw =
        NTDSL.init ~l:(name ^ "_out") ~prec:Ir.Ops.bfloat16 ~o:[ ni ] ~f:(fun _ -> 0.0) ()
      in
      Train.set_materialized outw.Tensor.value;
      let comp = named name [%cd outw =+ id xw ~logic:"irs => i"] in
      let transform opt =
        match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
      in
      let ctx = Context.auto () in
      let ctx, routine =
        Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      Context.get_values ctx outw.Tensor.value
    in
    cc_only claim_wgr_nested (fun () ->
        let got_wn = run_sum2 ~name:"aw2_wgr_serial" () in
        let got_wnn =
          run_sum2 ~name:"aw2_wgr_hw"
            ~schedule:
              (two_axis_sched ~f:(fun ~r:_ ~s ->
                   [ Sched.Retype { axis = s; ty = LL.Workgroup_reduce } ]))
            ()
        in
        p_all2 claim_wgr_nested got_wnn got_wn ~f:Float.equal);
    (* === adjacent accumulations: two SOURCE assignments into one cell are two stores, and each
       store narrows — they must NOT share an accumulator residency (that is the provenance
       boundary: only unrolled copies of one assignment may). 256 + 1 rounds to 256 at bf16 twice
       over; a rewrite merging the pair would produce 258. *)
    let s_acc = NTDSL.init ~l:"aw_s" ~prec:Ir.Ops.bfloat16 ~o:[ 1 ] ~f:(fun _ -> 256.0) () in
    let ax1 = NTDSL.init ~l:"aw_ax1" ~prec:Ir.Ops.bfloat16 ~o:[ 1 ] ~f:(fun _ -> 1.0) () in
    let ax2 = NTDSL.init ~l:"aw_ax2" ~prec:Ir.Ops.bfloat16 ~o:[ 1 ] ~f:(fun _ -> 1.0) () in
    Train.set_materialized s_acc.Tensor.value;
    let adj_comp = Asgns.sequence [ [%cd s_acc =+ ax1]; [%cd s_acc =+ ax2] ] in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ctx (named "aw_adjacent" adj_comp) Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    let adj = Context.get_values ctx s_acc.Tensor.value in
    p claim_adjacent (Float.equal adj.(0) 256.0);
    (* === index-guarded reduction: the gh-490 symbolic-extent guard shape [If (i < bound)] is
       transparent to the widening, so a runtime-bounded reduction keeps the compute-precision
       accumulator. Hand-built IR (the Assignments pipeline lowers clamped windows through
       interval-provable guards): 256 seeded, eight 1.0 contributions guarded to five — the wide
       accumulator reaches 261 and narrows once to 260 (round-to-even); per-step narrowing would
       absorb every +1 and leave 256. *)
    let bf16 = Ir.Ops.bfloat16 in
    let node = Ll_test.node_factory ~prec:bf16 ~first_id:9500 ~dims:[| 8 |] () in
    let gacc = node ~dims:[| 1 |] "aw_gacc" in
    let gxs = node "aw_gxs" in
    Ll_test.materialize gacc;
    Ll_test.materialize gxs;
    let gi = Ll_test.sym () in
    let iprec = Ir.Ops.index_prec () in
    let guard = LL.Binop (Ir.Ops.Cmplt, (Ll_test.embed gi, iprec), (LL.Constant 5.0, iprec)) in
    let upd =
      Ll_test.set gacc
        [| Ll_test.fixed 0 |]
        (LL.Binop
           ( Ir.Ops.Add,
             (Ll_test.get gacc [| Ll_test.fixed 0 |], bf16),
             (Ll_test.get gxs [| Ll_test.iter gi |], bf16) ))
    in
    let gllc = Ll_test.loop_n gi 8 (LL.If { cond = (guard, iprec); body = upd }) in
    let go = Ll_test.optimize ~materialized:[ gacc; gxs ] ~name:"aw_guarded" gllc in
    let gvals =
      Ll_test.execute ~name:"aw_guarded" go
        ~seed:[ (gacc, [| 256.0 |]); (gxs, Array.create ~len:8 1.0) ]
        ~read:[ gacc ]
    in
    p claim_guarded (Float.equal (List.hd_exn gvals).(0) 260.0);
    (* Scope-form legs below: a [Local_scope] over a MATERIALIZED node is a POST-optimize shape —
       the schedule mints and the codegen rewrite create it after [LL.optimize] has run, and the
       optimizer itself NORMALIZES such a scope to a plain [Get] of the node (the inlined
       computation of a non-virtual node is a read). So a hand-built scope must be injected past
       the optimizer: optimize a raw twin with the same nodes, reads and writes (for a valid
       traced store), then carry the scope form through the prelowered seam. The gh-639
       recurrence and mixed-operator legs used to optimize the scope form directly and were
       collapsing to identity copies whose value coincided with the expected one — green for the
       wrong reason. {!Ll_test.optimize_scoped} is that seam, and since gh-ocannl-681 it is the
       only route: [LL.optimize] now REJECTS a scope over a materialized node instead of
       normalizing it away. *)
    (* === virtualization's guarded-read update form === *)
    (* [inline_computation] guards a specialized reduction's update as [Set_local (id,
       Where (index cond, update, Get_local id))] — an expression-spelled guarded update whose
       else-arm carries the accumulator through. The census must classify it as a reduction
       (via [LL.accum_local_update_op]): treating the guarded self-read as a recurrence would
       leave a virtualized reduction narrow while its materialized serial twin widens —
       placement-dependent width. Same arithmetic as the [If]-guarded leg above: 256 seeded,
       eight guarded to five 1.0 contributions, wide 261 narrowing once to 260 (round-to-even);
       per-step narrowing would absorb every +1 and leave 256. *)
    let wacc = node ~dims:[| 1 |] "aw_wacc" in
    let wxs = node "aw_wxs" in
    Ll_test.materialize wacc;
    Ll_test.materialize wxs;
    let wi = Ll_test.sym () in
    let wid = LL.get_scope wacc in
    let w_guard = LL.Binop (Ir.Ops.Cmplt, (Ll_test.embed wi, iprec), (LL.Constant 5.0, iprec)) in
    let w_upd =
      LL.Binop (Ir.Ops.Add, (LL.Get_local wid, bf16), (Ll_test.get wxs [| Ll_test.iter wi |], bf16))
    in
    let w_body =
      LL.Seq
        ( LL.Set_local (wid, LL.Get (wacc, [| Ll_test.fixed 0 |])),
          Ll_test.loop_n wi 8
            (LL.Set_local
               ( wid,
                 LL.Ternop (Ir.Ops.Where, (w_guard, iprec), (w_upd, bf16), (LL.Get_local wid, bf16))
               )) )
    in
    let w_llc =
      LL.Set
        {
          tn = wacc;
          idcs = [| Ll_test.fixed 0 |];
          llsc =
            LL.Local_scope
              {
                id = wid;
                body = w_body;
                orig_indices = [| Ll_test.fixed 0 |];
                mint = LL.Schedule_minted;
              };
          debug = "";
        }
    in
    (* Raw twin: the same guarded accumulation spelled per-step, for the traced store. *)
    let w_raw =
      Ll_test.loop_n wi 8
        (Ll_test.set wacc
           [| Ll_test.fixed 0 |]
           (LL.Ternop
              ( Ir.Ops.Where,
                (w_guard, iprec),
                ( LL.Binop
                    ( Ir.Ops.Add,
                      (Ll_test.get wacc [| Ll_test.fixed 0 |], bf16),
                      (Ll_test.get wxs [| Ll_test.iter wi |], bf16) ),
                  bf16 ),
                (Ll_test.get wacc [| Ll_test.fixed 0 |], bf16) )))
    in
    let wo =
      Ll_test.optimize_scoped ~materialized:[ wacc; wxs ] ~name:"aw_whereg" ~raw:w_raw w_llc
    in
    let wvals =
      Ll_test.execute ~name:"aw_whereg" wo
        ~seed:[ (wacc, [| 256.0 |]); (wxs, Array.create ~len:8 1.0) ]
        ~read:[ wacc ]
    in
    p claim_where_guarded (Float.equal (List.hd_exn wvals).(0) 260.0);
    (* === a pre-existing scope carrying a NON-reduction recurrence must not be hoisted === *)
    (* The scope-base arm of the peel is licensed by the reduction reading alone: [local :=
       local - 0.5] narrows per enclosing iteration by the source's own semantics (256 - 0.5
       rounds back to 256 at bf16, twice over), and hoisting the enclosing loop into the scope
       would hold 255 instead. [valid_scope_updates] rejects the base. *)
    let gacc2 = node ~dims:[| 1 |] "aw_rec" in
    Ll_test.materialize gacc2;
    let ri = Ll_test.sym () in
    let rid = LL.get_scope gacc2 in
    let rec_scope_body =
      LL.Seq
        ( LL.Set_local (rid, LL.Get (gacc2, [| Ll_test.fixed 0 |])),
          LL.Set_local
            (rid, LL.Binop (Ir.Ops.Sub, (LL.Get_local rid, bf16), (LL.Constant 0.5, bf16))) )
    in
    let rec_llc =
      Ll_test.loop_n ri 2
        (LL.Set
           {
             tn = gacc2;
             idcs = [| Ll_test.fixed 0 |];
             llsc =
               LL.Local_scope
                 {
                   id = rid;
                   body = rec_scope_body;
                   orig_indices = [| Ll_test.fixed 0 |];
                   mint = LL.Schedule_minted;
                 };
             debug = "";
           })
    in
    let rec_raw =
      Ll_test.loop_n ri 2
        (Ll_test.set gacc2
           [| Ll_test.fixed 0 |]
           (LL.Binop
              (Ir.Ops.Sub, (Ll_test.get gacc2 [| Ll_test.fixed 0 |], bf16), (LL.Constant 0.5, bf16))))
    in
    let ro =
      Ll_test.optimize_scoped ~materialized:[ gacc2 ] ~name:"aw_recurrence" ~raw:rec_raw rec_llc
    in
    let rvals =
      Ll_test.execute ~name:"aw_recurrence" ro ~seed:[ (gacc2, [| 256.0 |]) ] ~read:[ gacc2 ]
    in
    p claim_scope_recurrence (Float.equal (List.hd_exn rvals).(0) 256.0);
    (* === individually reduce-shaped updates under MIXED operators are not a reduction === *)
    (* [local += 1; local *= 1] per iteration: 256 + 1 = 257 in f32, times 1, narrows back to
       256 — twice over. A hoisted scope would hold 257 then 258. [scope_updates_reduce_op]
       requires one uniform operator, so the base declines. *)
    let macc = node ~dims:[| 1 |] "aw_mix" in
    Ll_test.materialize macc;
    let mi = Ll_test.sym () in
    let mid = LL.get_scope macc in
    let mix_scope_body =
      LL.Seq
        ( LL.Set_local (mid, LL.Get (macc, [| Ll_test.fixed 0 |])),
          LL.Seq
            ( LL.Set_local
                (mid, LL.Binop (Ir.Ops.Add, (LL.Get_local mid, bf16), (LL.Constant 1.0, bf16))),
              LL.Set_local
                (mid, LL.Binop (Ir.Ops.Mul, (LL.Get_local mid, bf16), (LL.Constant 1.0, bf16))) ) )
    in
    let mix_llc =
      Ll_test.loop_n mi 2
        (LL.Set
           {
             tn = macc;
             idcs = [| Ll_test.fixed 0 |];
             llsc =
               LL.Local_scope
                 {
                   id = mid;
                   body = mix_scope_body;
                   orig_indices = [| Ll_test.fixed 0 |];
                   mint = LL.Schedule_minted;
                 };
             debug = "";
           })
    in
    let mix_raw =
      Ll_test.loop_n mi 2
        (LL.Seq
           ( Ll_test.set macc
               [| Ll_test.fixed 0 |]
               (LL.Binop
                  ( Ir.Ops.Add,
                    (Ll_test.get macc [| Ll_test.fixed 0 |], bf16),
                    (LL.Constant 1.0, bf16) )),
             Ll_test.set macc
               [| Ll_test.fixed 0 |]
               (LL.Binop
                  ( Ir.Ops.Mul,
                    (Ll_test.get macc [| Ll_test.fixed 0 |], bf16),
                    (LL.Constant 1.0, bf16) )) ))
    in
    let mo = Ll_test.optimize_scoped ~materialized:[ macc ] ~name:"aw_mixed" ~raw:mix_raw mix_llc in
    let mvals = Ll_test.execute ~name:"aw_mixed" mo ~seed:[ (macc, [| 256.0 |]) ] ~read:[ macc ] in
    p claim_mixed_scope (Float.equal (List.hd_exn mvals).(0) 256.0);
    (* === a widened scope's opening init keeps its own assignment's rounding === *)
    (* A virtual node initialized by one source assignment and reduced by another: the init
       [Set_local] is the inlined image of a SEPARATE assignment, so it renders at compute
       precision (its own store rounding) and only its result enters the residency — the
       provenance boundary of the adjacent-accumulations rule, seen from inside one scope. Here
       128 + 0.5 = 128.5, whose bf16 rounding is 128 (tie to even), then one multiplicative
       update by 3. Where the backend's compute precision is native bf16 (CUDA), the init rounds:
       128 * 3 = 384. On cc the compute precision IS f32 ([narrow_compute_f32]'s blanket over
       every intermediate — the pre-existing CPU semantics), so the init stays 128.5 and
       128.5 * 3 = 385.5 stores as 386; the invariant pinned is that the init renders at COMPUTE
       precision, never at the accumulator's residency. *)
    let iacc = node ~dims:[| 1 |] "aw_iacc" in
    let ia = node ~dims:[| 1 |] "aw_ia" in
    let ib = node ~dims:[| 1 |] "aw_ib" in
    let im = node ~dims:[| 1 |] "aw_im" in
    List.iter [ iacc; ia; ib; im ] ~f:Ll_test.materialize;
    let iid = LL.get_scope iacc in
    let i_body =
      LL.Seq
        ( LL.Set_local
            ( iid,
              LL.Binop
                ( Ir.Ops.Add,
                  (Ll_test.get ia [| Ll_test.fixed 0 |], bf16),
                  (Ll_test.get ib [| Ll_test.fixed 0 |], bf16) ) ),
          LL.Set_local
            ( iid,
              LL.Binop
                (Ir.Ops.Mul, (LL.Get_local iid, bf16), (Ll_test.get im [| Ll_test.fixed 0 |], bf16))
            ) )
    in
    let i_llc =
      LL.Set
        {
          tn = iacc;
          idcs = [| Ll_test.fixed 0 |];
          llsc =
            LL.Local_scope
              {
                id = iid;
                body = i_body;
                orig_indices = [| Ll_test.fixed 0 |];
                mint = LL.Schedule_minted;
              };
          debug = "";
        }
    in
    (* Raw twin: the two source assignments materialized, for the traced store. *)
    let i_raw =
      LL.Seq
        ( Ll_test.set iacc
            [| Ll_test.fixed 0 |]
            (LL.Binop
               ( Ir.Ops.Add,
                 (Ll_test.get ia [| Ll_test.fixed 0 |], bf16),
                 (Ll_test.get ib [| Ll_test.fixed 0 |], bf16) )),
          Ll_test.set iacc
            [| Ll_test.fixed 0 |]
            (LL.Binop
               ( Ir.Ops.Mul,
                 (Ll_test.get iacc [| Ll_test.fixed 0 |], bf16),
                 (Ll_test.get im [| Ll_test.fixed 0 |], bf16) )) )
    in
    let io =
      Ll_test.optimize_scoped ~materialized:[ iacc; ia; ib; im ] ~name:"aw_init_round" ~raw:i_raw
        i_llc
    in
    let ivals =
      Ll_test.execute ~name:"aw_init_round" io
        ~seed:[ (ia, [| 128.0 |]); (ib, [| 0.5 |]); (im, [| 3.0 |]) ]
        ~read:[ iacc ]
    in
    p claim_init_round (Float.equal (List.hd_exn ivals).(0) (if on_cpu then 386.0 else 384.0));
    (* === Workgroup_reduce serialized on cc: retyping the reduction axis to a hardware kind the
       backend cannot bind must not change the accumulator width relative to the Serial spelling.
       16 terms push the running sums past 4, where bf16 can no longer represent the 1/64
       increments, so a per-step regression in the serialized fallback would diverge. *)
    (* An explicit [=+] into a pre-zeroed materialized accumulator: an einsum-lowered sum's
       whole-node init nest would fail [validate_parallel] once the reduction axis is retyped to a
       hardware kind (whole-node zeroing is not distributed), and the init is not what these legs
       are about. *)
    let run_sum ~cols ~name ?schedule () =
      let fv = Ll_test.drift ~dims:[| ni; cols |] in
      let xw = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; cols ] ~f:fv () in
      let outw =
        NTDSL.init ~l:(name ^ "_out") ~prec:Ir.Ops.bfloat16 ~o:[ ni ] ~f:(fun _ -> 0.0) ()
      in
      Train.set_materialized outw.Tensor.value;
      let comp = named name [%cd outw =+ id xw ~logic:"is => i"] in
      let transform opt =
        match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
      in
      let ctx = Context.auto () in
      let ctx, routine =
        Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      Context.get_values ctx outw.Tensor.value
    in
    let retype_reduction ty opt =
      match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 2) with
      | [ _; s ] -> [ Sched.Retype { axis = s; ty } ]
      | _ -> assert false
    in
    cc_only claim_wgreduce (fun () ->
        let got_w = run_sum ~cols:16 ~name:"aw_wgr_serial" () in
        let got_wg =
          run_sum ~cols:16 ~name:"aw_wgr_hw" ~schedule:(retype_reduction LL.Workgroup_reduce) ()
        in
        p_all2 claim_wgreduce got_wg got_w ~f:Float.equal);
    (* === the SIMD reduction's scalar remainder === *)
    (* 67 is no multiple of any chains*lanes step, so the vector partial must fold into the wide
       total together with the tail contributions and narrow once — the pre-fix rendering stored
       the partial to bf16 mid-way (at running sums ~25, where 1/64 increments are lost) and then
       narrowed per tail step. The structural conjunct keeps the leg honest: if the vectorized
       rendering declines, serial-equals-serial would be a false green. *)
    cc_only_claims [ claim_simd_tail_structure; claim_simd_tail ] (fun () ->
        let got_v = run_sum ~cols:67 ~name:"aw_vec_serial" () in
        let got_vt =
          run_sum ~cols:67 ~name:"aw_vec_tail" ~schedule:(retype_reduction LL.Vectorized) ()
        in
        let vec_fired =
          String.is_substring
            (Generated.read ~ext:".c" "aw_vec_tail")
            ~substring:"Vectorized reduction rendering"
        in
        p claim_simd_tail_structure vec_fired;
        p_all2 claim_simd_tail got_vt got_v ~f:Float.equal);
    fp8_leg ();
    (* Negative controls: turning the policy off recovers per-step narrowing wherever that is
       schedule-uniform — which on these inputs must visibly differ from the widened default,
       proving the inputs discriminate the accumulator width. The bf16 control is cc-only: CUDA's
       bf16 residency is structural (its mma legs accumulate f32 in hardware, so a policy-narrowed
       serial leg would resurrect the schedule-dependent width — the serial legs stay wide there
       under either setting). The fp8 control runs on cc AND CUDA, where nothing tensorizes fp8
       destinations and the policy genuinely restores per-step semantics; on Metal it would stay
       wide (fp8 computes in f32 structurally), which the bf16 gate already skips. *)
    let saved_policy = Numerics.get () in
    Numerics.set_policy { saved_policy with narrow_compute_f32 = false };
    cc_only claim_off_value (fun () ->
        let ma2 = NTDSL.init ~l:"ma2" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
        let mb2 = NTDSL.init ~l:"mb2" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
        let%op mc2 = ma2 * mb2 in
        Tn.update_prec mc2.Tensor.value Ir.Ops.bfloat16;
        let got_off = run ~name:"aw_bf16_naive_off" mc2 in
        p claim_off_value (not (Array.for_all2_exn got_off got ~f:Float.equal)));
    p claim_off_fp8 (Float.equal (fp8_sum ~name:"aw_fp8_off" ~first_id:9720 ()) fp8_values.per_step);
    Numerics.set_policy saved_policy;
    cc_only claim_off_shape (fun () ->
        let src = Generated.read ~ext:".c" "aw_bf16_naive_off" in
        let has s = String.is_substring src ~substring:s in
        p claim_off_shape (has "single_to_bfloat16(fmaf("));
    (* The skip branch prints [all_claims] in its declared order, so that order must be exactly the
       sequence this branch reports, passes and skips alike: a claim missing from the list, a stale
       entry, or one out of place would each move a golden line on the skip branch only
       (gh-ocannl-1130). Silent on stdout: the goldens stay backend-uniform. *)
    sync_reported ();
    let this_branch =
      List.drop (List.rev !reported) reported_before
      |> List.remove_consecutive_duplicates ~equal:String.equal
    in
    let same = List.equal String.equal this_branch all_claims in
    if not same then
      Stdio.eprintf "reported by this branch:\n  %s\nall_claims:\n  %s\n"
        (String.concat ~sep:"\n  " this_branch)
        (String.concat ~sep:"\n  " all_claims);
    Verdict.claim "all_claims is exactly the sequence of claims this branch reports" same
  end

let () = Test_utils.with_policy bf16_auto default_bf16_block

(* === bf16 residency under the bf16_arithmetic policy (gh-ocannl-838) === *)
(* Universal legs, executed on every backend: the widened bf16 claims the gate above skips on HIP
   and Metal BY DESIGN hold there too once [Bf16_wide] asks for f32 residency, and the contract is
   unconditional — [narrow_compute_f32 = false] leaves CPU bf16 COMPUTE at storage width, but the
   accumulator still resides in f32 ([Numerics.cpu_accum_prec]). The default-policy leg is the
   two-sided complement: on the same discriminating inputs it matches the wide reference exactly
   where the backend already widens bf16, and diverges where it keeps storage residency — pinning
   [Bf16_auto]'s CURRENT resolution, which (like [Fp16_auto]) retains latitude per backend. *)
let claim_bf16_wide_matmul =
  "under Bf16_wide the bf16 naive matmul equals the once-narrowed wide-accumulation reference"

let claim_bf16_wide_ncf32_off =
  "under Bf16_wide + narrow_compute_f32=false the bf16 naive matmul still equals the wide reference"

let claim_bf16_default_matmul =
  "the default-policy bf16 matmul matches the wide reference exactly where the backend widens bf16 \
   and diverges where it keeps storage residency"

let () =
  let bf16_matmul ~name =
    let ma = NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
    let mb = NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
    let%op mc = ma * mb in
    Tn.update_prec mc.Tensor.value Ir.Ops.bfloat16;
    run ~name mc
  in
  let wide_sums =
    Array.init (n * n) ~f:(fun t ->
        let i = t / n and j = t % n in
        let acc = ref 0.0 in
        for k = 0 to n - 1 do
          acc := !acc +. (fa [| i; k |] *. fb [| k; j |])
        done;
        !acc)
  in
  let want =
    run ~name:"aw_bf16w_ref"
      (NTDSL.init ~l:"aw_bf16w_ref" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
         ~f:(fun idcs -> wide_sums.((idcs.(0) * n) + idcs.(1)))
         ())
  in
  (* Pinned rather than inherited: the stanza declares OCANNL_BF16_ARITHMETIC, and the claim is
     about [Bf16_auto]'s resolution, so both the run and the backend's widening are read under
     it. *)
  let saved_policy = Numerics.get () in
  Numerics.set_policy { saved_policy with bf16_arithmetic = Numerics.Bf16_auto };
  let got_auto = bf16_matmul ~name:"aw_bf16_naive_auto" in
  let auto_widens_bf16 =
    not
      (Ir.Ops.equal_prec
         (codegen_capabilities.Ir.Backend_intf.accum_prec Ir.Ops.bfloat16)
         Ir.Ops.bfloat16)
  in
  Numerics.set_policy { saved_policy with bf16_arithmetic = Numerics.Bf16_wide };
  let got_wide = bf16_matmul ~name:"aw_bf16_naive_wide" in
  Numerics.set_policy
    { saved_policy with bf16_arithmetic = Numerics.Bf16_wide; narrow_compute_f32 = false };
  let got_wide_nco = bf16_matmul ~name:"aw_bf16_naive_wide_nco" in
  Numerics.set_policy saved_policy;
  p_all2 claim_bf16_wide_matmul got_wide want ~f:Float.equal;
  p_all2 claim_bf16_wide_ncf32_off got_wide_nco want ~f:Float.equal;
  p claim_bf16_default_matmul
    ((not (Array.is_empty got_auto))
    && Bool.equal (Array.for_all2_exn got_auto want ~f:Float.equal) auto_widens_bf16)

(* === Privatize tiles reside at the backend's accumulator precision (gh-ocannl-1116) === *)
(* The autotuner seeds [Privatize] in its scalar matmul sketches (the CPU packing pipeline, the GPU
   register blocktiling), and before gh-ocannl-1116 the privatized tile was minted at the
   destination's STORAGE precision: every step of the reduction through it narrowed, while the
   serial rendering above widens — so a tuned schedule silently changed the reduction's width. The
   tile's cell varies with loops nested inside the reduction (the packing pipeline's [k_o, k_i, i_i,
   j_i] order, the blocktiling's materialized-unrolled tile), so codegen cannot localize the tile's
   accumulation into a wide scope of its own: the tile's precision IS the width.

   Universal legs, executed on every backend: the first unfused scalar seed the tuner enumerates for
   the site, built with the backend's accumulator resolution, must equal the serial rendering of
   the same policy bitwise — under [Bf16_auto] (whatever it resolves to here) and under the wide
   policies, where it must also equal the once-narrowed wide reference. The negative control
   rebuilds the same seed with a storage-precision tile — the pre-fix minting, still admissible as
   an explicit schedule — and must DIVERGE from the wide reference: it is what proves these inputs
   and this pipeline discriminate per-step narrowing, rather than parity holding because nothing
   could tell the widths apart. *)
let claim_priv_seeded =
  "the seeded scalar matmul sketch mints its Privatize tile at the backend's accumulator residency"

let claim_priv_auto =
  "the seeded privatized bf16 sketch equals the serial rendering under Bf16_auto"

let claim_priv_wide =
  "under Bf16_wide the seeded privatized bf16 sketch equals the once-narrowed wide-accumulation \
   reference"

let claim_priv_narrow =
  "a storage-precision Privatize tile narrows per step: under Bf16_wide the same seed diverges \
   from the wide reference"

let claim_priv_f16 =
  "under Fp16_wide the seeded privatized f16 sketch equals the once-narrowed wide-accumulation \
   reference"

let claim_priv_f16_narrow =
  "a storage-precision Privatize tile narrows per step: under Fp16_wide the same f16 seed diverges \
   from the wide reference"

let hardware_limits = lazy (Context.hardware_limits (Context.auto ()))

(* The first unfused scalar (non-tensorized) seed of the site — the pipelines that carry a
   [Privatize] — as the tuner would instantiate it. [~storage_tile] swaps its [Privatize] for the
   pre-gh-ocannl-1116 minting. [seeded] collects the precisions the built schedule's [Privatize]s
   carry, read before the swap. *)
let privatized_seed ~storage_tile ~seeded (opt : LL.optimized) : Sched.schedule =
  let p =
    List.find_exn
      (Autotune.sketch_seed_params
         ~is_gpu:(Sched.backend_is_gpu backend_name)
         ~is_cpu:on_cpu ~limits:(Lazy.force hardware_limits) opt)
      ~f:(fun p -> (not p.Autotune.sk_mma) && (not p.sk_conv) && not p.sk_epilogue)
  in
  Stdio.eprintf
    "accum_width: privatized seed on %s: %s pipeline, bm=%d bn=%d bk=%d (not part of the golden)\n\
     %!"
    backend_name
    (if p.sk_gpu then "GPU blocktiling" else "CPU packing")
    p.sk_bm p.sk_bn p.sk_bk;
  let sched =
    Autotune.sketch_schedule ~accum_prec:codegen_capabilities.Ir.Backend_intf.accum_prec ~p opt
  in
  List.map sched ~f:(function
    | Sched.Privatize { target; over; acc_prec } ->
        seeded := (Lazy.force target.Tn.storage_prec, acc_prec) :: !seeded;
        Sched.Privatize
          {
            target;
            over;
            acc_prec = (if storage_tile then Lazy.force target.Tn.storage_prec else acc_prec);
          }
    | op -> op)

(* One matmul instance per run (the routine name keys its build artifacts), over the given operand
   cycles; [schedule] as in {!run}. *)
let precision_matmul ~prec ~size ~fa ~fb ~name ?schedule () =
  let ma = NTDSL.init ~l:(name ^ "_a") ~prec ~i:[ size ] ~o:[ size ] ~f:fa () in
  let mb = NTDSL.init ~l:(name ^ "_b") ~prec ~i:[ size ] ~o:[ size ] ~f:fb () in
  let%op mc = ma * mb in
  Tn.update_prec mc.Tensor.value prec;
  run ~name ?schedule mc

(* The whole-k f64 dot products, exact reproductions of the kernels' f32 chains for these cycles,
   narrowed once per cell by minting a [prec] tensor from them. *)
let wide_reference ~prec ~size ~fa ~fb ~name =
  let sums =
    Array.init (size * size) ~f:(fun t ->
        let i = t / size and j = t % size in
        let acc = ref 0.0 in
        for k = 0 to size - 1 do
          acc := !acc +. (fa [| i; k |] *. fb [| k; j |])
        done;
        !acc)
  in
  run ~name
    (NTDSL.init ~l:name ~prec ~i:[ size ] ~o:[ size ]
       ~f:(fun idcs -> sums.((idcs.(0) * size) + idcs.(1)))
       ())

let () =
  let seeded = ref [] in
  let privatized ~prec ~size ~fa ~fb ~name ~storage_tile =
    precision_matmul ~prec ~size ~fa ~fb ~name ~schedule:(privatized_seed ~storage_tile ~seeded) ()
  in
  let bf16 = Ir.Ops.bfloat16 and f16 = Ir.Ops.half in
  let saved_policy = Numerics.get () in
  Numerics.set_policy { saved_policy with bf16_arithmetic = Numerics.Bf16_auto };
  let serial_auto = precision_matmul ~prec:bf16 ~size:n ~fa ~fb ~name:"aw_priv_serial_auto" () in
  let priv_auto = privatized ~prec:bf16 ~size:n ~fa ~fb ~name:"aw_priv_auto" ~storage_tile:false in
  let auto_resolution = codegen_capabilities.Ir.Backend_intf.accum_prec bf16 in
  Numerics.set_policy { saved_policy with bf16_arithmetic = Numerics.Bf16_wide };
  let want = wide_reference ~prec:bf16 ~size:n ~fa ~fb ~name:"aw_priv_ref" in
  let priv_wide = privatized ~prec:bf16 ~size:n ~fa ~fb ~name:"aw_priv_wide" ~storage_tile:false in
  let wide_resolution = codegen_capabilities.Ir.Backend_intf.accum_prec bf16 in
  let priv_narrow =
    privatized ~prec:bf16 ~size:n ~fa ~fb ~name:"aw_priv_narrow" ~storage_tile:true
  in
  Numerics.set_policy { saved_policy with fp16_arithmetic = Numerics.Fp16_wide };
  let want16 = wide_reference ~prec:f16 ~size:n16 ~fa:fa16 ~fb:fb16 ~name:"aw_priv16_ref" in
  let priv16 =
    privatized ~prec:f16 ~size:n16 ~fa:fa16 ~fb:fb16 ~name:"aw_priv16_wide" ~storage_tile:false
  in
  let f16_resolution = codegen_capabilities.Ir.Backend_intf.accum_prec f16 in
  let priv16_narrow =
    privatized ~prec:f16 ~size:n16 ~fa:fa16 ~fb:fb16 ~name:"aw_priv16_narrow" ~storage_tile:true
  in
  Numerics.set_policy saved_policy;
  (* Every build of the five privatized runs carried a Privatize, each at the resolution of the
     policy it was built under (read before the storage-tile swap): in build order, auto, wide,
     wide, f16-wide, f16-wide. *)
  let expected =
    [
      (bf16, auto_resolution);
      (bf16, wide_resolution);
      (bf16, wide_resolution);
      (f16, f16_resolution);
      (f16, f16_resolution);
    ]
  in
  p claim_priv_seeded
    (List.equal
       (fun (s, a) (s', a') -> Ir.Ops.equal_prec s s' && Ir.Ops.equal_prec a a')
       (List.rev !seeded) expected);
  p_all2 claim_priv_auto priv_auto serial_auto ~f:Float.equal;
  p_all2 claim_priv_wide priv_wide want ~f:Float.equal;
  p claim_priv_narrow
    ((not (Array.is_empty priv_narrow)) && not (Array.for_all2_exn priv_narrow want ~f:Float.equal));
  p_all2 claim_priv_f16 priv16 want16 ~f:Float.equal;
  p claim_priv_f16_narrow
    ((not (Array.is_empty priv16_narrow))
    && not (Array.for_all2_exn priv16_narrow want16 ~f:Float.equal))

(* === A Privatize tile widens only what code generation widens (gh-ocannl-1116 review) === *)
(* [acc_prec] is the residency of a RECOGNIZED widenable accumulation, as the backend's own is. A
   body code generation does not widen narrows at its own points in the serial rendering, so a
   Privatize tile taking it over must stay at storage precision, or a tuned schedule would change
   its rounding (Codex P1s, rounds 1-2 on staging PR #880): a non-reduction recurrence (256 - 0.5
   - 0.5 is 255 in f32 and stays 256 at bf16), a mixed-operator update (max(256 + 1, 0) twice is
   258 in f32 and stays 256 — not [* 1], which simplification folds into a plain reduction), and
   sibling accumulation statements into one cell, whose separate stores are their semantics: each
   update alone would widen, but the peel refuses a level carrying two (256 +1 +1 twice is 260
   wide and stays 256), and a reduction beside a statement writing another node, which the peel
   refuses the same way (256 + 1 + 1 stays 256; round 3) — or beside a [Declare_local] /
   [Set_local] computation, the shape online softmax's hoist emits (round 4), or beside a
   [Staged_compilation] statement, which is source and not schedule scratch (round 5). The positive
   control is one reduction statement the tile must widen with
   the serial rendering (256 + 1 + 1 reaches 258), so the declined legs' 256 is the gate and not a
   tile that never widened. All under [Bf16_wide], where the residency itself widens bf16 on every
   backend; the per-update gate is the shared [Low_level.accum_update_widens]. *)
let claim_priv_gate =
  "the shared widening predicate accepts the reduction and declines the recurrence and the \
   mixed-operator update"

let claim_priv_gate_add =
  "a privatized bf16 reduction widens with the serial rendering (256 + 1 + 1 reaches 258)"

let claim_priv_gate_sub =
  "a privatized non-reduction recurrence keeps per-step narrowing with the serial rendering (256 - \
   0.5 - 0.5 stays 256)"

let claim_priv_gate_sib =
  "privatized sibling accumulation statements keep their separate narrowings with the serial \
   rendering (256 +1 +1, twice, stays 256)"

let claim_priv_gate_other =
  "a privatized reduction beside another node's statement keeps per-step narrowing with the serial \
   rendering (256 + 1 + 1 stays 256)"

let claim_priv_gate_local =
  "a privatized reduction beside a scope-local computation keeps per-step narrowing with the \
   serial rendering (256 + 1 + 1 stays 256)"

let claim_priv_gate_staged =
  "a privatized reduction beside a staged-compilation statement keeps per-step narrowing with the \
   serial rendering (256 + 1 + 1 stays 256)"

let claim_priv_gate_hw =
  "a Privatize whose loop encloses a hardware-typed reduction level is refused"

let claim_priv_gate_two =
  "two accumulators privatized one after the other keep their per-step narrowing with the serial \
   rendering (256 + 1 + 1 stays 256 for each)"

let claim_priv_gate_scatter =
  "a privatized reduction beside a scatter into a schedule-minted partials node keeps per-step \
   narrowing with the serial rendering (256 + 1 + 1 stays 256)"

let claim_priv_gate_mixed =
  "a privatized mixed-operator update keeps per-step narrowing with the serial rendering (max(256 \
   + 1, 0) twice stays 256)"

let () =
  let bf16 = Ir.Ops.bfloat16 in
  let node = Ll_test.node_factory ~prec:bf16 ~first_id:9900 ~dims:[| 1 |] () in
  let cell = [| Ll_test.fixed 0 |] in
  let leg ?(siblings = 1) ?(other = false) ?(local = false) ?(staged = false) ~label ~update () =
    let acc = node label in
    Ll_test.materialize acc;
    let other_node = node (label ^ "_other") in
    Ll_test.materialize other_node;
    let k = Ll_test.sym () in
    let loc_node = node (label ^ "_loc") in
    Ll_test.virtualize loc_node;
    let loc = LL.get_scope loc_node in
    (* The local-computation leg's update is a recognized reduction whose contribution reads the
       scope local, so the per-update gate alone would widen it: only the sibling statements
       decline. *)
    let llsc =
      if local then
        LL.Binop
          ( Ir.Ops.Add,
            (Ll_test.get acc cell, bf16),
            (LL.Binop (Ir.Ops.Add, (LL.Get_local loc, bf16), (LL.Constant 1.0, bf16)), bf16) )
      else update (Ll_test.get acc cell)
    in
    let raw () =
      Ll_test.loop_n k 2
        (LL.unflat_lines
           ((if local then
               [
                 LL.Declare_local { id = loc; needs_init = false };
                 LL.Set_local (loc, LL.Constant 0.0);
               ]
             else [])
           @ List.init siblings ~f:(fun _ -> Ll_test.set acc cell llsc)
           @ (if other then [ Ll_test.set other_node cell (LL.Constant 7.0) ] else [])
           @
           if staged then [ LL.Staged_compilation (fun () -> PPrint.string "/* staged */") ] else []
           ))
    in
    let exec ~name o =
      (List.hd_exn (Ll_test.execute ~name o ~seed:[ (acc, [| 256.0 |]) ] ~read:[ acc ])).(0)
    in
    let serial =
      exec ~name:(label ^ "_serial")
        (Ll_test.optimize ~materialized:[ acc; other_node ] ~name:(label ^ "_serial") (raw ()))
    in
    let o = Ll_test.optimize ~materialized:[ acc; other_node ] ~name:(label ^ "_priv") (raw ()) in
    let priv =
      exec ~name:(label ^ "_priv")
        (Sched.apply
           [
             Sched.privatize ~accum_prec:codegen_capabilities.Ir.Backend_intf.accum_prec ~target:acc
               ~over:k;
           ]
           o)
    in
    (LL.accum_update_widens ~tn:acc ~idcs:cell llsc, serial, priv)
  in
  let bin op a b = LL.Binop (op, (a, bf16), (b, bf16)) in
  Test_utils.with_policy
    (fun pol -> { pol with Numerics.bf16_arithmetic = Numerics.Bf16_wide })
    (fun () ->
      let w_add, s_add, p_add =
        leg ~label:"aw_pg_add" ~update:(fun a -> bin Ir.Ops.Add a (LL.Constant 1.0)) ()
      in
      let w_sib, s_sib, p_sib =
        leg ~siblings:2 ~label:"aw_pg_sib" ~update:(fun a -> bin Ir.Ops.Add a (LL.Constant 1.0)) ()
      in
      let _, s_oth, p_oth =
        leg ~other:true ~label:"aw_pg_oth" ~update:(fun a -> bin Ir.Ops.Add a (LL.Constant 1.0)) ()
      in
      let _, s_loc, p_loc =
        leg ~local:true ~label:"aw_pg_loc" ~update:(fun a -> bin Ir.Ops.Add a (LL.Constant 1.0)) ()
      in
      let _, s_stg, p_stg =
        leg ~staged:true ~label:"aw_pg_stg" ~update:(fun a -> bin Ir.Ops.Add a (LL.Constant 1.0)) ()
      in
      let w_sub, s_sub, p_sub =
        leg ~label:"aw_pg_sub" ~update:(fun a -> bin Ir.Ops.Sub a (LL.Constant 0.5)) ()
      in
      let w_mix, s_mix, p_mix =
        leg ~label:"aw_pg_mix"
          ~update:(fun a -> bin Ir.Ops.Max (bin Ir.Ops.Add a (LL.Constant 1.0)) (LL.Constant 0.0))
          ()
      in
      Stdio.eprintf
        "accum_width: privatize gate legs serial/privatized: add %g/%g, siblings %g/%g, \
         beside-other %g/%g, beside-local %g/%g, beside-staged %g/%g, sub %g/%g, mixed %g/%g (not \
         part of the golden)\n\
         %!"
        s_add p_add s_sib p_sib s_oth p_oth s_loc p_loc s_stg p_stg s_sub p_sub s_mix p_mix;
      (* A hardware reduction level inside the privatized loop is refused outright (round 5): a
         per-thread tile would fold only its own lanes, and whether a backend serializes the level
         is not the transform's to know. *)
      let hw_refused =
        let acc = node "aw_pg_hw" in
        Ll_test.materialize acc;
        let xs = Ll_test.node_factory ~prec:bf16 ~first_id:9990 ~dims:[| 4 |] () "aw_pg_hw_x" in
        Ll_test.materialize xs;
        let k = Ll_test.sym () and w = Ll_test.sym () in
        let raw =
          Ll_test.loop_n k 2
            (Ll_test.loop_n ~axis:LL.Workgroup_reduce w 4
               (Ll_test.set acc cell
                  (bin Ir.Ops.Add (Ll_test.get acc cell) (Ll_test.get xs [| Ll_test.iter w |]))))
        in
        let o = Ll_test.optimize ~materialized:[ acc; xs ] ~name:"aw_pg_hw" raw in
        Result.is_error
          (Result.try_with (fun () ->
               Sched.apply
                 [
                   Sched.privatize ~accum_prec:codegen_capabilities.Ir.Backend_intf.accum_prec
                     ~target:acc ~over:k;
                 ]
                 o))
      in
      p claim_priv_gate_hw hw_refused;
      (* Two accumulators updated side by side, privatized one after the other (round 6): the first
         Privatize turns its update into a tile read-modify-write, which is still a sibling
         statement of the second's update — not staging scratch — so neither tile widens, as the
         serial rendering keeps both per-step (256 + 1 + 1 stays 256 for each). *)
      let two_targets =
        let a = node "aw_pg_two_a" and b = node "aw_pg_two_b" in
        Ll_test.materialize a;
        Ll_test.materialize b;
        let k = Ll_test.sym () in
        let raw () =
          Ll_test.loop_n k 2
            (LL.unflat_lines
               [
                 Ll_test.set a cell (bin Ir.Ops.Add (Ll_test.get a cell) (LL.Constant 1.0));
                 Ll_test.set b cell (bin Ir.Ops.Add (Ll_test.get b cell) (LL.Constant 1.0));
               ])
        in
        let exec ~name o =
          List.map
            (Ll_test.execute ~name o ~seed:[ (a, [| 256.0 |]); (b, [| 256.0 |]) ] ~read:[ a; b ])
            ~f:(fun v -> v.(0))
        in
        let serial =
          exec ~name:"aw_pg_two_serial"
            (Ll_test.optimize ~materialized:[ a; b ] ~name:"aw_pg_two_serial" (raw ()))
        in
        let priv =
          let o = Ll_test.optimize ~materialized:[ a; b ] ~name:"aw_pg_two_priv" (raw ()) in
          let pz target =
            Sched.privatize ~accum_prec:codegen_capabilities.Ir.Backend_intf.accum_prec ~target
              ~over:k
          in
          exec ~name:"aw_pg_two_priv" (Sched.apply [ pz a; pz b ] o)
        in
        Stdio.eprintf
          "accum_width: two privatized accumulators serial %s privatized %s (not part of the golden)\n\
           %!"
          (String.concat ~sep:"," (List.map serial ~f:Float.to_string))
          (String.concat ~sep:"," (List.map priv ~f:Float.to_string));
        List.equal Float.equal serial [ 256.0; 256.0 ] && List.equal Float.equal priv serial
      in
      p claim_priv_gate_two two_targets;
      (* A schedule-minted partials node written by a scatter beside the accumulation — the shape a
         [Split_reduce] of a one-hot scatter leaves (round 8) — is reduction work, not staging
         scratch, even though its node lives in the tile namespace: the tile must stay at storage
         precision with the serial rendering (256 + 1 + 1 stays 256). *)
      let beside_scatter =
        let acc = node "aw_pg_scat" in
        Ll_test.materialize acc;
        let partials =
          Tn.create ~namespace:"tile" (Tn.Specified bf16) ~id:9995 ~label:[ "aw_pg_scat_part" ]
            ~unpadded_dims:(lazy [| 1 |])
            ~padding:(lazy None)
            ()
        in
        Ll_test.materialize partials;
        let k = Ll_test.sym () in
        (* [Set_dynamic] is post-optimize IR (only the one-hot rewrite produces it), so the scatter
           twin reaches the backend through [optimize_scoped]: [raw] spells the same nodes with a
           plain write for the optimizer's traced store. *)
        let body ~scatter =
          Ll_test.loop_n k 2
            (LL.unflat_lines
               [
                 Ll_test.set acc cell (bin Ir.Ops.Add (Ll_test.get acc cell) (LL.Constant 1.0));
                 (if scatter then
                    LL.Set_dynamic
                      {
                        tn = partials;
                        idcs = cell;
                        dyn_axis = 0;
                        dyn_value = (LL.Constant 0.0, Ir.Ops.index_prec ());
                        llsc = LL.Constant 3.0;
                        debug = "";
                      }
                  else Ll_test.set partials cell (LL.Constant 3.0));
               ])
        in
        let scoped name =
          Ll_test.optimize_scoped ~materialized:[ acc; partials ] ~name ~raw:(body ~scatter:false)
            (body ~scatter:true)
        in
        let exec ~name o =
          (List.hd_exn
             (Ll_test.execute ~name o
                ~seed:[ (acc, [| 256.0 |]); (partials, [| 0.0 |]) ]
                ~read:[ acc ])).(0)
        in
        let serial = exec ~name:"aw_pg_scat_serial" (scoped "aw_pg_scat_serial") in
        let priv =
          let o = scoped "aw_pg_scat_priv" in
          exec ~name:"aw_pg_scat_priv"
            (Sched.apply
               [
                 Sched.privatize ~accum_prec:codegen_capabilities.Ir.Backend_intf.accum_prec
                   ~target:acc ~over:k;
               ]
               o)
        in
        Stdio.eprintf
          "accum_width: privatized beside a tile scatter serial %g privatized %g (not part of the \
           golden)\n\
           %!"
          serial priv;
        Float.equal serial 256.0 && Float.equal priv 256.0
      in
      p claim_priv_gate_scatter beside_scatter;
      p claim_priv_gate (w_add && w_sib && (not w_sub) && not w_mix);
      p claim_priv_gate_sib (Float.equal s_sib 256.0 && Float.equal p_sib 256.0);
      p claim_priv_gate_other (Float.equal s_oth 256.0 && Float.equal p_oth 256.0);
      p claim_priv_gate_local (Float.equal s_loc 256.0 && Float.equal p_loc 256.0);
      p claim_priv_gate_staged (Float.equal s_stg 256.0 && Float.equal p_stg 256.0);
      p claim_priv_gate_add (Float.equal s_add 258.0 && Float.equal p_add 258.0);
      p claim_priv_gate_sub (Float.equal s_sub 256.0 && Float.equal p_sub 256.0);
      p claim_priv_gate_mixed (Float.equal s_mix 256.0 && Float.equal p_mix 256.0))
