(* gh-ocannl-728: does splitting a matmul site's axis into a pre-split pair cost throughput, at a
   fixed tile geometry and fixed total work?

   The gpt2_mini q/k/v and out projections run at ~37% of gfx1151's sgemm peak while the FFN
   up-projection runs at ~85%. Two explanations were on the table: (H1) the batch loop is a
   pre-split ROW loop the tile cannot span, so every block sees M = 128 where the FFN sites were
   believed to see M = 1024; (H2) the projection kernels are simply small (134 MFLOP each against
   the FFN's 537) and launch/occupancy-bound at any tiling.

   This bench builds the shapes by hand at the SAME total FLOPs, on the SAME device, under the SAME
   sketch geometries, and times them warm. Each site is [d[b.., m, n..] += w[n.., k..] * x[b.., m,
   k..]]; the groups are:

   - A: 134.2 MFLOP, N = K = 256, 1024 rows split as B x M for B in {1,2,4,8} -- equal work, equal
   block count at every geometry, only the row split differs. This is the leg that isolates
   M-per-block from everything else. - B: the FFN up-projection's own shape (N = 1024, 537 MFLOP),
   merged and batched. - C: 134 / 268 / 537 / 1074 MFLOP at a fixed shape, batched and merged -- the
   size leg. - D: the out projection, whose weight carries two input axes (a multi-axis contraction,
   gh-ocannl-683), merged and batched. - E: the 2x2 factorial over the two pre-splits at constant
   FLOPs -- rows merged (1024) or split (8 x 128) against the column axis merged (256) or split into
   8 heads of 32, which is the column structure the gpt2_mini lowering actually gives q/k/v. - P: a
   1024^3 square GEMM -- the family's own ceiling, where no shape question arises.

   Two sites appear twice: A_rows1024 = E_qkv_rows1_heads1 and A_b8x128 = E_qkv_rows8_heads1. That
   is deliberate -- each is measured twice within one run, so a run reports its own session drift
   and a shape effect can be judged against it.

   Group [qkv] is gh-ocannl-728's Move 0 gate on its own: the real q/k/v site (rows 8 x 128, columns
   8 heads x 32) and its heads-merged twin (columns 256), each twice ([Q_heads8_a/b],
   [Q_heads1_a/b]). It closes with a gate table -- per candidate class, the merged pair against the
   split pair beside the duplicate pairs' own disagreement -- read off the finalists round below.

   Candidates per site: the untuned shipped default, then every geometry of the GPU blocktile sketch
   family applied as the pure IR transform it is -- plus, under [--with-mma], every tensorized seed
   the backend and numerics policy offer (f32 on CUDA needs [tf32_matmuls], e.g. the [approximate]
   profile; each site's phase-1 line counts what the menu offered); mode [tune] instead runs a full
   [Autotune.tune] search per site with [~search:true] and the disk cache DISABLED, so neither shape
   can replay the other's cached winner and a configuration that disabled searching cannot return
   the untuned default under a tuned label.

   Timing is ROUND-INTERLEAVED, not site-by-site: one round per base tile geometry ([bgrid] twins
   included, since a merged site's plain arm and a batched site's [bgrid] arm are the pair being
   compared), inside which every arm is timed batch by batch. A last FINALISTS round recompiles each
   site's best scalar and best tensorized seed (under each of the three statistics, from their own
   rounds) and its default, and times them again, interleaved with every other site's finalists and
   with the searches' crowned routines: the cross-geometry question "which candidate is best" is
   then answered inside one round rather than across rounds' drift, by numbers that are not the
   selection's own optimistic draw. The visiting order uses ONE rotation per adjacent PAIR of
   batches, mirrored on the odd member, so each such pair exchanges the positions of every pair of
   arms exactly -- the run-by-run A/B alternation, for all pairs at once and with no RNG. (A
   rotation that advances every batch does not do this: rotate-then-reverse leaves each arm at the
   same position in both halves.) An even batch count balances the pairs exactly, which is why the
   default is even. Timing one site to completion before the next puts the drift straight into the
   difference under test, and reversing the site order only moves that bias. Three statistics per
   arm: [repeats] dispatches queued back-to-back with one sync -- the sync only, never a
   device-to-host readback, which would put a transfer and a host allocation inside the timed region
   (what a kernel sustains inside a step, and the summary's statistic, taken at the MEDIAN batch),
   and [Autotune.time_routine] in each of its two modes ([Isolated], one dispatch and one sync,
   which reads up to 2.6x higher; and [Queued], its gh-ocannl-755 companion), minimized over the
   same interleaved passes. The last two are the TUNER'S instrument rather than a re-derivation of
   it, which is what makes the closing gh-ocannl-755 table a comparison of the ranking a search
   would produce against the ranking steady-state throughput produces, and not of two lookalikes.

   Which of the numbers on a line to believe depends on what the noise is. The median is the right
   summary when the noise is symmetric, which is why it is the summary. On a CONTENDED box it is
   not: interference only ever makes a batch slower, so the median wanders (20% between the two arms
   of a duplicated site on one measured run) while the two minima stay put (5% and 0.5% across every
   arm of the same geometry in that same run). The duplicated sites are the arbiter — when they
   disagree by more than the effect under test, the effect is not resolvable in that run, whichever
   statistic is read.

   Every candidate's whole output is compared cell by cell against a host-computed oracle -- built
   straight from the input formulas, so it is independent of the compiler under test -- and the
   inputs are chosen so that f32 and f64 accumulation agree exactly whatever order either uses (and
   survive a tensorized arm's operand rounding: a few significant bits each, well inside tf32's 11).
   Each line carries the launch dimensions of every kernel the compile shipped, read off the
   routine's own segments, so "same geometry" is read off the kernel -- for the untuned default and
   a crowned search too -- and a tensorized arm carries the routine's tensorization census; one that
   rendered only the scalar fallback is refused rather than timed under an [mma-] label. Any cell
   that fails parity, fails to compile or run, or (in [tune] mode) did not actually search, is
   counted and the process exits nonzero: a blank in the column the caller asked for is a failed
   experiment, not a missing number.

   Usage (bin/ cwd trap: pin the backend, and run from a directory holding an ocannl_config):
   OCANNL_BACKEND=hip <path to>/projection_shape_bench.exe \ [repeats] [batches] [group] [order]
   [mode] [--with-mma] Defaults 50 repeats, 6 timing batches, group "all"
   (a/b/c/d/e/p/abd/abde/all/qkv), order fwd (or rev, which reverses every round's complete arm
   order), mode seeds (or tune, or both). The batch count must be EVEN -- the visiting order mirrors
   in adjacent pairs -- so the measurement this bench was written for reads, in full:

   cd benchmarks # the nearest ocannl_config, and bin/ has the cwd trap OCANNL_BACKEND=hip
   ../_build/default/bin/projection_shape_bench.exe 200 8 abde fwd seeds

   and gh-ocannl-728's Move 0 is benchmarks/gh728_cells.sh, which drives group [qkv] beside the real
   gpt2_mini kernels.

   Group [smoke] is deliberately outside that measurement vocabulary: its one 2x2 matmul exists only
   so [@bin-smoke] can start and complete every phase cheaply on the cc backend. Its output is not
   evidence about the projection-shape question this benchmark measures (gh-ocannl-858).

   Two things the output does NOT claim. The [bestseed] column ranks arms measured in DIFFERENT
   rounds, so unlike every per-geometry comparison it is exposed to drift between rounds -- read it
   as indicative and the round lines as the measurement. And on the C backends the hoisted seeds
   mint packed-constant nodes per candidate, which land in the device-wide constant cache that
   [Context.release] cannot reclaim, so a long CPU run grows by one packed pool per hoisted
   candidate; the bench says so when it times them rather than pretending its cleanup covers that
   class. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Outcome = Ir.Schedule_outcome

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec =
  let caps = lazy (Context.codegen_capabilities (Context.auto ())) in
  fun p -> (Lazy.force caps).Ir.Backend_intf.accum_prec p

(* Flushed per line ([Bench_out]): a long remote run should be readable while it is still going. *)
let p fmt = Bench_out.p fmt

(* Process-level conditions are not candidate failures: containing them turns an OOM or an interrupt
   into one "failed cell" and keeps the run going, which prolongs thrashing and can make a
   minutes-long benchmark unstoppable. Same set the autotuner's own containment re-raises. *)
let is_fatal = function
  | Out_of_memory | Stack_overflow | Stdlib.Sys.Break | Assert_failure _ -> true
  | _ -> false

(* Cleanup on a path that is already failing must not itself abort the run -- a backend still
   reporting the original asynchronous error can raise from the release's device await -- but a
   fatal condition still propagates. *)
let release_quietly ctx = try Context.release ctx with e when not (is_fatal e) -> ()

(* A classified rejection is containable only when the backend says the device was not written: an
   illegal address, a device assertion or a launch timeout reports [Writes_may_have_occurred],
   leaves partial writes and a sticky context behind, and every later arm on that device is then
   suspect. The autotuner escalates that class for the same reason. *)
let escalate_if_wrote ?fatal ~candidate (c : Outcome.classified_cause) =
  match c.Outcome.execution_effect with
  | Outcome.No_device_writes -> ()
  | Outcome.Writes_may_have_occurred ->
      (* [fatal] is raised BEFORE the escalation, not after: the exception this renders is an
         ordinary one, so a containment guarded on that flag would otherwise catch it and go on
         using a sticky, possibly partially written device. *)
      Option.iter fatal ~f:(fun f -> f := true);
      Outcome.raise_failure (Outcome.Fatal (Outcome.fatal_of_classified ~candidate c))

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* [offset + stride * (flat index mod modulus)] over row-major [dims]: varies along every axis whose
   extent is not a multiple of [modulus]. Products are multiples of 1/8 and partial sums stay far
   below 2^24, so f32 addition is exact in any order and parity is bitwise. *)
let cycle ~dims ~modulus ~offset ~stride idcs =
  let flat = Array.foldi dims ~init:0 ~f:(fun i acc d -> (acc * d) + (idcs.(i) % d)) in
  offset +. (stride *. Float.of_int (flat % modulus))

(* The lowering alone, through the ANALYZE-ONLY entry point: [Context.compile] would need the
   backend to accept the untuned form, so one site whose default compile a backend rejects would
   abort the whole run before any explicitly scheduled seed -- which might compile perfectly -- got
   measured. It also mints no context to leak. *)
let capture fwd = Context.lowered_for_decisions (Context.auto ()) fwd Ir.Indexing.Empty

(* Every field two seeds of one site can differ by, so no two seeds share a label: the tables and
   the finalists round's recompile key on it. The register tile, swizzle, pack precision and conv
   flag print only when set, so the historical scalar labels are unchanged; a pair still colliding
   on a field this omits is a counted failure at phase 1, not a merged row. *)
let geom_label (q : Autotune.sketch_params) =
  Printf.sprintf "%s%s%s %dx%dx%d/%dx%d%s%s%s%s%s%s%s%s"
    (if q.sk_conv then "conv-" else "")
    (if q.sk_mma then "mma-" else "")
    (if q.sk_gpu then "gpu" else "cpu")
    q.sk_bm q.sk_bn q.sk_bk q.sk_tm q.sk_tn
    (if q.sk_depth > 1 then Printf.sprintf " pd%d" q.sk_depth else "")
    (match q.sk_tile with Some t -> " tile " ^ Ir.Register_tile.to_string t | None -> "")
    (match q.sk_swizzle with
    | Some Ir.Low_level.Swizzle_elem -> " swz-elem"
    | Some Ir.Low_level.Swizzle_b128 -> " swz-b128"
    | None -> "")
    (match q.sk_pack_prec with Some pr -> " pack" ^ Ir.Ops.prec_string pr | None -> "")
    (if q.sk_hoist then " hoist" else "")
    (if q.sk_grid then " grid" else "")
    (if q.sk_pack_rest then " packrest" else "")
    (if not q.sk_batch_grid then "" else if q.sk_batch_inner then " bgrid-in" else " bgrid")

(* A MONOTONIC counter, not a wall-clock timestamp: an NTP step or a VM clock correction during a
   long run would otherwise jump (or invert) an interval, and the corrupted batch feeds the median
   and the winner directly. Same clock [Autotune.time_routine] measures with. *)
let elapsed c = Mtime.Span.to_float_ns (Mtime_clock.count c) /. 1e9

(* The [autotune_repeats] default, pinned rather than read: the gh-ocannl-755 columns are meant to
   be the measurement a DEFAULT search takes of each arm, so an ambient OCANNL_AUTOTUNE_REPEATS
   would silently change what they are a comparison of. *)
let tuner_repeats = 3
let fst4 (a, _, _, _) = a

(* One site: a batched matmul [d[bs.., m, j] += a[bs.., m, kk..] * w[j, kk..]]. [ks] is the weight's
   input-axis list -- a singleton for the q/k/v shape, a pair for the out projection's multi-axis
   contraction. *)
type site = { tag : string; bs : int list; m : int; ns : int list; ks : int list }

let prod = List.fold ~init:1 ~f:( * )

let flops s =
  2.0
  *. Float.of_int (prod s.bs)
  *. Float.of_int s.m
  *. Float.of_int (prod s.ns)
  *. Float.of_int (prod s.ks)

let build s =
  let wdims = Array.of_list (s.ns @ s.ks) in
  let adims = Array.of_list (s.bs @ [ s.m ] @ s.ks) in
  let w =
    NTDSL.init ~l:("w_" ^ s.tag) ~prec:Ir.Ops.single ~o:s.ns ~i:s.ks
      ~f:(cycle ~dims:wdims ~modulus:11 ~offset:(-5.5) ~stride:0.5)
      ()
  in
  let a =
    NTDSL.init ~l:("a_" ^ s.tag) ~prec:Ir.Ops.single ~b:(s.bs @ [ s.m ]) ~o:s.ks
      ~f:(cycle ~dims:adims ~modulus:13 ~offset:0.25 ~stride:0.25)
      ()
  in
  let%op d = w * a in
  d

(* The independent oracle: the site's whole output computed on the host straight from the [cycle]
   formulas, in the same row-major layout the device writes. Every candidate is compared against it
   cell by cell, not through a scalar digest -- a checksum with a repeating coefficient cannot see a
   permutation of cells one period apart, and a reference taken from the default pipeline cannot see
   a defect the default pipeline shares. Products are multiples of 1/8 and every partial sum stays
   below 2^24, so the f64 accumulation here and the device's f32 accumulation agree exactly whatever
   order either uses, and the comparison is [Float.equal]. *)
let oracle s =
  let nn = prod s.ns and kk = prod s.ks and rows = prod s.bs * s.m in
  let wv = Array.init (nn * kk) ~f:(fun i -> -5.5 +. (0.5 *. Float.of_int (i % 11)))
  and av = Array.init (rows * kk) ~f:(fun i -> 0.25 +. (0.25 *. Float.of_int (i % 13))) in
  let d = Array.create ~len:(rows * nn) 0.0 in
  for r = 0 to rows - 1 do
    for j = 0 to nn - 1 do
      let acc = ref 0.0 in
      for k = 0 to kk - 1 do
        acc := !acc +. (av.((r * kk) + k) *. wv.((j * kk) + k))
      done;
      d.((r * nn) + j) <- !acc
    done
  done;
  d

(* One measured arm, in GFLOP/s: the median timing batch (the summary), the minimum batch, and the
   tuner's own [Queued] reading -- the objective a default search ranks by. *)
type stats = { st_med : float; st_min : float; st_queued : float }

(* The label prefix of an arm re-timed in the finalists round, which keeps it apart from the seed
   round's arm of the same geometry in every table. *)
let final_prefix = "FINAL "

(* The label of a search's crowned winner. *)
let tuned_label = "TUNED (full search)"

(* Everything one timed row needs, kept alive only for the round that times it. *)
type live = {
  lv_tag : string;
  lv_flops : float;
  lv_label : string;
  lv_ctx : Context.t ref;
  lv_routine : Context.routine;
  lv_launch : string;
  lv_mma : string;
      (** The compiled routine's own tensorization census ([Context.routine.mma]), so a line
          labelled as a tensorized seed says whether it measured tensor-core code or its scalar
          fallback. *)
  lv_parity : bool;
  lv_times : float list ref;
  lv_iso : float ref;
  lv_queued : float ref;
  lv_failed : bool ref;
      (** Set when a timed dispatch raises: the arm stops being visited, is not reported, and its
          cell is counted -- a recoverable per-candidate failure must not abort a run that still has
          matched arms to measure and release. *)
}

let () =
  let args = Bench_args.create "projection_shape_bench" in
  let repeats = Bench_args.int args 0 ~name:"repeats" ~default:50 in
  (* Even by default: the per-batch reversal below alternates every pair's order, so an even number
     of batches balances every pair exactly. *)
  let nbatches = Bench_args.int args 1 ~name:"batches" ~default:6 in
  (* Odd counts are refused rather than rounded: the visiting order mirrors in adjacent PAIRS of
     batches, so an unpaired final batch gives every arm one extra measurement in one visit order
     only -- exactly the asymmetry the mirroring exists to remove. *)
  if nbatches % 2 = 1 then
    invalid_arg
      (Printf.sprintf "batches must be even (got %d): each batch is mirrored by its partner"
         nbatches);
  (* [repeats] is the dispatch count inside a steady-state batch AND the sample count of the
     one-dispatch statistic, whose sweep is mirrored the same way; an odd value leaves that
     statistic's final sample unpaired. *)
  if repeats % 2 = 1 then
    invalid_arg
      (Printf.sprintf
         "repeats must be even (got %d): it is also the mirrored sample count of the one-dispatch \
          statistic"
         repeats);
  let group = String.lowercase (Bench_args.string args 2 ~default:"all") in
  (* [rev] reverses every round's complete arm order, so a residual position bias (a cold first arm,
     a site's twins always visited in one order) can be shown not to carry the conclusion. *)
  let order = String.lowercase (Bench_args.string args 3 ~default:"fwd") in
  (* [seeds] times the sketch seeds, one interleaved round per geometry; [tune] runs the full search
     per site (which cannot be interleaved -- a search is minutes of its own dispatches) with the
     disk cache disabled, so neither shape can replay the other's winner. *)
  let mode = String.lowercase (Bench_args.string args 4 ~default:"seeds") in
  if not (List.mem [ "seeds"; "tune"; "both" ] mode ~equal:String.equal) then
    invalid_arg ("unknown mode " ^ mode ^ " (seeds | tune | both)");
  let do_seeds = not (String.equal mode "tune") in
  let do_tune = List.mem [ "tune"; "both" ] mode ~equal:String.equal in
  (* [--with-mma] keeps the tensorized seeds in the menu (gh-ocannl-728 Move 0). Off by default so
     the historical groups keep meaning the scalar blocktile family they were measured over. Which
     tensorized seeds exist is the backend's and the numerics policy's call -- f32 operands on CUDA
     need [tf32_matmuls] (the [approximate] profile), HIP has no f32 tile shape -- so the phase-1
     line per site prints how many the menu offered, and a flag that yields none says so there. *)
  let with_mma = Bench_args.flag args ~name:"with-mma" in
  (* The backend comes from the CONTEXT, not from the [backend] setting: unpinned, that setting
     reads "cc" while [Context.auto] picks the first available GPU, and the harness would then seed
     the CPU families against a GPU context's limits and print the wrong name over the numbers. *)
  let probe_ctx = Context.auto () in
  let backend = String.lowercase (Context.backend_name probe_ctx) in
  let on_gpu =
    List.exists [ "metal"; "cuda"; "hip" ] ~f:(fun s -> String.is_substring backend ~substring:s)
  in
  let limits = Context.hardware_limits probe_ctx in
  Context.release probe_ctx;
  p "backend %s, repeats %d, batches %d, group %s, order %s, mode %s%s\n" backend repeats nbatches
    group order mode
    (if with_mma then ", tensorized seeds included" else "");
  (* The numerics gate the tensorized f32 seeds hinge on, off the effective policy the seeding
     consults -- so a log says which regime its mma arms (or their absence) belong to. *)
  p "numerics policy: tf32_matmuls=%b\n" (Ir.Numerics.get ()).Ir.Numerics.tf32_matmuls;
  p "%s\n" (if on_gpu then "GPU blocktile family" else "CPU families");
  let sites =
    let a =
      [
        { tag = "A_rows1024"; bs = []; m = 1024; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "A_b2x512"; bs = [ 2 ]; m = 512; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "A_b4x256"; bs = [ 4 ]; m = 256; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "A_b8x128"; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 256 ] };
      ]
    and b =
      [
        { tag = "B_ffn_rows1024"; bs = []; m = 1024; ns = [ 1024 ]; ks = [ 256 ] };
        { tag = "B_ffn_b8x128"; bs = [ 8 ]; m = 128; ns = [ 1024 ]; ks = [ 256 ] };
      ]
    and c =
      List.concat_map [ 8; 16; 32; 64 ] ~f:(fun bb ->
          [
            {
              tag = Printf.sprintf "C_b%dx128" bb;
              bs = [ bb ];
              m = 128;
              ns = [ 256 ];
              ks = [ 256 ];
            };
            {
              tag = Printf.sprintf "C_rows%d" (bb * 128);
              bs = [];
              m = bb * 128;
              ns = [ 256 ];
              ks = [ 256 ];
            };
          ])
    and d =
      [
        { tag = "D_outproj_rows1024"; bs = []; m = 1024; ns = [ 256 ]; ks = [ 8; 32 ] };
        { tag = "D_outproj_b8x128"; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 8; 32 ] };
      ]
    and e =
      [
        { tag = "E_qkv_rows8_heads8"; bs = [ 8 ]; m = 128; ns = [ 8; 32 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows1_heads8"; bs = []; m = 1024; ns = [ 8; 32 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows8_heads1"; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows1_heads1"; bs = []; m = 1024; ns = [ 256 ]; ks = [ 256 ] };
      ]
    and pk = [ { tag = "P_square1024"; bs = []; m = 1024; ns = [ 1024 ]; ks = [ 1024 ] } ]
    (* gh-ocannl-728 Move 0: the real q/k/v site (8 heads of 32 columns) against its heads-merged
       twin (256 columns), EACH TWICE, in alternation -- the two copies of a shape are the in-run
       noise floor the gate judges the merge against, and they share every round with the pair under
       test. *)
    and qkv =
      List.concat_map [ "a"; "b" ] ~f:(fun copy ->
          [
            { tag = "Q_heads8_" ^ copy; bs = [ 8 ]; m = 128; ns = [ 8; 32 ]; ks = [ 256 ] };
            { tag = "Q_heads1_" ^ copy; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 256 ] };
          ])
    and smoke = [ { tag = "smoke_2x2"; bs = []; m = 2; ns = [ 2 ]; ks = [ 2 ] } ] in
    match group with
    | "a" -> a
    | "b" -> b
    | "c" -> c
    | "d" -> d
    | "e" -> e
    | "p" -> pk
    | "abd" -> a @ b @ d
    | "abde" -> a @ b @ d @ e
    | "all" -> a @ b @ c @ d @ e @ pk
    | "qkv" -> qkv
    | "smoke" -> smoke
    | g -> invalid_arg ("unknown group " ^ g)
  in
  (* [rev] reverses each round's COMPLETE arm list (see [run_round]), not the site list: reversing
     sites alone would leave a site's own arms -- plain/bgrid twins, a finalist's scalar/tensorized
     pair -- and the finalists round's appended groups in their forward order. *)
  let reversed =
    match order with "fwd" -> false | "rev" -> true | o -> invalid_arg ("unknown order " ^ o)
  in
  (* Any cell of the experiment that could not be measured is a failure of the run, not a blank in a
     table: a harness that exits 0 with "n/a" in the column the caller asked for lets automation
     accept an invalid experiment.

     One cause of an unmeasured cell is not the run's fault, and is counted apart from the rest: a
     timing window refused because host contention dominated its samples (gh-ocannl-855) is a fact
     about the machine, and on a shared one every window of a tiny cell can go. That is what the
     [bin-smoke] canary hit on a GitHub runner (PR #608), where the cell is 2x2 and the question
     being asked is whether the pipeline runs end to end, not what it measures. So the canary passes
     [--allow-unmeasured] and a refusal leaves it green, while an ordinary run — which was asked for
     numbers and got none — still exits 1. Refusals print their [!!] line either way; a silent "n/a"
     is what neither caller wants. *)
  let failures = ref 0 in
  let unmeasured = ref 0 in
  let allow_unmeasured = Bench_args.flag args ~name:"allow-unmeasured" in
  let fail fmt =
    Printf.ksprintf
      (fun m ->
        Int.incr failures;
        p "   !! %s\n" m)
      fmt
  in
  let refused fmt =
    Printf.ksprintf
      (fun m ->
        Int.incr unmeasured;
        p "   !! %s\n" m)
      fmt
  in
  (* Phase 1: per site, the lowering, the detected shape, the seed list and the host oracle. *)
  let prepared =
    List.map sites ~f:(fun s ->
        let fl = flops s in
        p "\n== %s : rows %s | cols %s | contract %s  (%.1f MFLOP)\n" s.tag
          (String.concat ~sep:"x" (List.map (s.bs @ [ s.m ]) ~f:Int.to_string))
          (String.concat ~sep:"x" (List.map s.ns ~f:Int.to_string))
          (String.concat ~sep:"x" (List.map s.ks ~f:Int.to_string))
          (fl /. 1e6);
        (* One [build] per SITE, and the lowering probe uses THAT graph: a second [build] would mint
           a second pair of host-initialized operands, and those enter the per-device constant
           buffer cache which [Context.release] cannot reach, so a discarded probe graph would
           retain its operands for the rest of the run. *)
        let d = build s in
        let fwd = named (s.tag ^ "_t") (Train.forward d) in
        let opt = capture fwd in
        (match Autotune.detect_matmul opt.LL.llc with
        | None -> fail "%s: NOT DETECTED as a matmul site" s.tag
        | Some site ->
            p "   site: m=%d n=%d k=%d  outer-k [%s]  batch-outer [%s]  batch-inner [%s]\n"
              site.Autotune.m_ni site.Autotune.m_nj site.Autotune.m_nk
              (String.concat ~sep:","
                 (List.map site.Autotune.m_ko ~f:(fun (_, e) -> Int.to_string e)))
              (String.concat ~sep:","
                 (List.map site.Autotune.m_bo ~f:(fun (_, e) -> Int.to_string e)))
              (String.concat ~sep:","
                 (List.map site.Autotune.m_bi ~f:(fun (_, e) -> Int.to_string e))));
        let offered =
          Autotune.sketch_seed_params ~is_gpu:on_gpu ~is_cpu:(not on_gpu) ~limits opt
          |> List.filter ~f:(fun (q : Autotune.sketch_params) -> not q.sk_epilogue)
        in
        let seeds =
          List.filter offered ~f:(fun (q : Autotune.sketch_params) -> with_mma || not q.sk_mma)
        in
        p "   menu: %d scalar seed(s), %d tensorized seed(s) offered%s\n"
          (List.count offered ~f:(fun q -> not q.Autotune.sk_mma))
          (List.count offered ~f:(fun q -> q.Autotune.sk_mma))
          (if with_mma then "" else " (tensorized not timed: no --with-mma)");
        (* The seed's label is how every later table -- and the finalists round, which recompiles a
           site's best seed from it -- names the seed, so two seeds one label cannot tell apart
           would silently merge into one row. *)
        (match List.find_a_dup (List.map seeds ~f:geom_label) ~compare:String.compare with
        | Some l -> fail "%s: two seeds share the label %S -- their rows would merge" s.tag l
        | None -> ());
        (* The same comp is compiled for every candidate, which is what the sketch suites do too. *)
        (s, fl, seeds, lazy (oracle s), d, fwd))
  in
  (* Compile one candidate and check its parity. The routine name carries a run-wide counter so that
     under [output_debug_files_in_build_directory] each candidate's .cd/.ll/backend source survives
     its neighbours instead of being overwritten by them.

     A candidate that fails parity, or that raises anywhere between the compile and the readback, is
     RELEASED here and never returned: its context must not outlive the attempt (the pool tables
     strongly retain device slabs, docs/agent-notes/backend-memory.md), and an incorrect schedule
     must not be timed, ranked, or able to win a summary column that carries no parity marker. The
     context is held in a ref the cleanup can reach, so the release covers the exception path
     too. *)
  let counter = ref 0 in
  (* The two direct compiles go through [Context.compile_outcome] so that a FATAL compile, link or
     driver failure is not contained as one cell by the generic handler below: only a classified
     candidate rejection is recoverable. [Autotune.tune] is a raising API and does its own
     classification internally, so the tuned arm keeps the raising form. *)
  let compiled ~fatal_seen ?lowered_transform ~name ctx fwd =
    match
      Context.compile_outcome ?lowered_transform ~name ~provenance:Outcome.Candidate ~candidate:name
        ctx fwd Ir.Indexing.Empty
    with
    | Ok v -> v
    | Error (Outcome.Classified c) ->
        escalate_if_wrote ~fatal:fatal_seen ~candidate:name c;
        Outcome.raise_failure (Outcome.Classified c)
    | Error (Outcome.Fatal _ as f) ->
        fatal_seen := true;
        Outcome.raise_failure f
  in
  let arm ?(tensorized = false) (s, fl, _, orc, d, fwd) ~label ~compile =
    Int.incr counter;
    let name = Printf.sprintf "%s_c%d" s.tag !counter in
    let held = ref None in
    let drop () = Option.iter !held ~f:(fun c -> release_quietly !c) in
    (* Set when the classifier called a validation failure fatal: [raise_failure] re-raises the
       original exception with its backtrace, which the containment below would otherwise catch like
       any other. A classified decline is left to that containment on purpose. *)
    let fatal_seen = ref false in
    (* A tensorized seed whose every [Tile_mma] declined to the scalar fallback (or that emitted
       none) would be timed as scalar code under an [mma-] label -- the false perf number the
       routine's census exists to stop. Refused, and counted, rather than timed. *)
    let not_tensorized = ref None in
    match
      let ctx, routine = compile ~name ~fatal_seen fwd in
      let ctx = ref ctx in
      held := Some ctx;
      (* The warm-up launches and the synchronizing readback go through the same classifier the
         timed batches use: an unclassified driver failure here is fatal, not a candidate's own
         decline, and continuing to compile and time arms on an affected device is what the contract
         exists to stop. *)
      let protect phase f =
        match
          Outcome.protect ~classify_backend:(Context.failure_classifier !ctx)
            ~provenance:Outcome.Candidate ~phase ~candidate:label f
        with
        | Ok v -> v
        | Error (Outcome.Classified c) ->
            escalate_if_wrote ~fatal:fatal_seen ~candidate:label c;
            Outcome.raise_failure (Outcome.Classified c)
        | Error (Outcome.Fatal _ as f) ->
            fatal_seen := true;
            Outcome.raise_failure f
      in
      protect Outcome.Launch (fun () ->
          for _ = 1 to 3 do
            ctx := Context.run !ctx routine
          done);
      let got = protect Outcome.Sync (fun () -> Context.get_values !ctx d.Tensor.value) in
      let want = Lazy.force orc in
      let parity =
        Array.length got = Array.length want && Array.for_all2_exn got want ~f:Float.equal
      in
      (* The launch geometry of every kernel the compile SHIPPED, read off the routine's own
         segments -- so the untuned default and a crowned search, which lower themselves, report a
         verified geometry too, and a multi-kernel routine shows each launch. *)
      let launch =
        let pr a = String.concat ~sep:"x" (Array.to_list (Array.map a ~f:Int.to_string)) in
        String.concat ~sep:" + "
          (List.map routine.Context.segments ~f:(fun seg ->
               let dm = LL.launch_dims seg.LL.llc in
               Printf.sprintf "grid %s block %s" (pr dm.LL.grid) (pr dm.LL.block)))
      in
      let mma = routine.Context.mma in
      if
        tensorized
        && not
             (Ir.C_syntax.equal_tensorization mma.Ir.C_syntax.tensorization Ir.C_syntax.Tensorized)
      then not_tensorized := Some (Ir.C_syntax.mma_summary_string mma);
      {
        lv_tag = s.tag;
        lv_flops = fl;
        lv_label = label;
        lv_ctx = ctx;
        lv_routine = routine;
        lv_launch = launch;
        lv_mma = Ir.C_syntax.mma_summary_string mma;
        lv_parity = parity;
        lv_times = ref [];
        lv_iso = ref Float.infinity;
        lv_queued = ref Float.infinity;
        lv_failed = ref false;
      }
    with
    | _ when Option.is_some !not_tensorized ->
        fail "%s / %s: a tensorized seed did not tensorize (%s) -- not timed" s.tag label
          (Option.value_exn !not_tensorized);
        drop ();
        None
    | lv when lv.lv_parity -> Some lv
    | _ ->
        fail "%s / %s: PARITY FAILED against the host oracle -- not timed" s.tag label;
        drop ();
        None
    | exception exn when (not !fatal_seen) && not (is_fatal exn) ->
        fail "%s / %s: FAILED: %s" s.tag label
          (List.hd_exn (String.split_lines (Exn.to_string exn)));
        drop ();
        None
    | exception exn ->
        drop ();
        raise exn
  in
  (* One interleaved round: every arm of [lives] is timed batch by batch in rotation, so a monotone
     session drift lands on all of them alike instead of on whichever site came later in the run.
     A whole-site loop cannot do that -- reversing the site order only moves the bias, it does not
     cancel it (docs/agent-notes/training-and-performance.md's A/B protocol: alternate the arms RUN
     BY RUN). Each arm's own batch is [repeats] dispatches queued back-to-back with one sync. *)
  (* The order arms are visited in for batch [b]: ONE rotation per ADJACENT PAIR of batches
     ([b / 2]), mirrored on the odd member. Advancing the rotation between the forward and the
     reversed batch would leave every arm at the same position in both -- rotate-by-b then reverse
     maps arm 0 to position 0 at every even b and back to position 0 at every odd b -- so the
     mirror has to be of the SAME order the forward batch used. Then each pair of batches
     exchanges the positions of every pair of arms exactly, which is the run-by-run A/B
     alternation, for all pairs simultaneously and with no RNG; an even [nbatches] balances them
     exactly, which is why the default is even. *)
  let visit_order n b =
    let idx = Array.init n ~f:(fun i -> (i + (b / 2)) % n) in
    if b % 2 = 1 then Array.rev_inplace idx;
    idx
  in
  (* gh-ocannl-755's deliverable: per site, each arm's batched median beside the tuner's own two
     metrics, in seconds, so the closing table can rank the same population three ways. Filled by
     [report], which is the one place that has already decided an arm was measured. *)
  let instrument : (string, (string * float * float * float) list) Hashtbl.t =
    Hashtbl.create (module String)
  in
  let time_round lives =
    let arr = Array.of_list lives in
    let n = Array.length arr in
    (* Each timed batch is contained the way the autotuner's timing loop is, not by an exception
       taxonomy of this bench's own: [Schedule_outcome.protect] with the BACKEND's classifier
       decides whether a launch or sync failure is a candidate's own decline (contain it, mark that
       one arm, keep the round's other matched arms measurable) or fatal (re-raise with its
       backtrace). Deciding by "not one of four OCaml exceptions" would contain an unclassified
       driver failure -- device loss, say -- and then report the remaining arms as measurements,
       which is precisely what the classifier contract exists to stop. The narrow [tag]s tell a
       report whether the arm died at launch or at sync. *)
    let attempt lv f =
      if not !(lv.lv_failed) then
        match
          Outcome.protect
            ~classify_backend:(Context.failure_classifier !(lv.lv_ctx))
            ~provenance:Outcome.Candidate ~phase:Outcome.Launch ~candidate:lv.lv_label f
        with
        | Ok () -> ()
        | Error (Outcome.Classified c) ->
            escalate_if_wrote ~candidate:lv.lv_label c;
            lv.lv_failed := true;
            fail "%s / %s: a timed dispatch declined at %s: %s" lv.lv_tag lv.lv_label
              (Sexp.to_string (Outcome.sexp_of_phase c.Outcome.phase))
              (List.hd_exn (String.split_lines (Outcome.detail_of_cause c.Outcome.cause)))
        | Error (Outcome.Fatal fl) -> Outcome.raise_failure (Outcome.Fatal fl)
    in
    if n > 0 then begin
      for b = 0 to nbatches - 1 do
        let ord = visit_order n b in
        for i = 0 to n - 1 do
          let lv = arr.(ord.(i)) in
          attempt lv (fun () ->
              let c = Mtime_clock.counter () in
              for _ = 1 to repeats do
                Outcome.tag Outcome.Launch (fun () ->
                    lv.lv_ctx := Context.run !(lv.lv_ctx) lv.lv_routine)
              done;
              Outcome.tag Outcome.Sync (fun () -> Context.sync !(lv.lv_ctx));
              lv.lv_times := (elapsed c /. Float.of_int repeats) :: !(lv.lv_times))
        done
      done;
      (* gh-ocannl-755: the TUNER'S OWN instrument, in both of its modes, on the same arms in the
         same round -- not a re-derivation of it. This used to be a hand-rolled min-over-single-
         launches loop that resembled [Autotune.time_routine] without being it (a different sample
         count, no top-up), which is exactly the shape a ranking comparison must not have: the
         question is whether the tuner's metric crowns a different candidate than the batched one,
         so the isolated column has to be the metric a search actually ranks by. [tuner_repeats] is
         the [autotune_repeats] default, so each call is the measurement a default search would
         take; the min over [nbatches] such calls is this bench's usual defence against a batch that
         drew contention, and it is interleaved for the same reason the batches are.

         The returned context is dropped as the tuner drops it: [time_routine] threads its own
         [Context.run] chain internally and never hands the last one back, and the arm's [lv_ctx] is
         still the lineage every later dispatch continues from. *)
      for b = 0 to nbatches - 1 do
        let ord = visit_order n b in
        for i = 0 to n - 1 do
          let lv = arr.(ord.(i)) in
          let sample slot timing =
            attempt lv (fun () ->
                let timing_result =
                  Autotune.time_routine ~tag_failures:true ~timing ~repeats:tuner_repeats
                    !(lv.lv_ctx) lv.lv_routine
                in
                (* Through the shared gate, never the [contended] flag alone: contention and a clock
                   that resolved nothing are separate refusals (gh-ocannl-888), and a half-proof
                   here would store a zero as the best tuner time and report it as infinite
                   throughput. *)
                Option.iter (Autotune.admitted_timing_ms timing_result) ~f:(fun ms ->
                    let dt = ms /. 1000. in
                    if Float.(dt < !slot) then slot := dt))
          in
          sample lv.lv_iso Autotune.Isolated;
          sample lv.lv_queued Autotune.Queued
        done
      done
    end
  in
  (* Every measured arm's three throughput statistics, by site: what the finalists round and the
     gh-ocannl-728 gate read back. *)
  let measured : (string, (string * stats) list) Hashtbl.t = Hashtbl.create (module String) in
  (* The median is the summary and the winner-selection statistic: with a handful of batches on a
     loaded device the minimum is systematically optimistic and crowns whichever geometry drew the
     luckiest batch. The minimum is kept beside it as a diagnostic. *)
  let report lv =
    let sorted = Array.of_list !(lv.lv_times) in
    Array.sort sorted ~compare:Float.compare;
    let n = Array.length sorted in
    let best = sorted.(0) and worst = sorted.(n - 1) in
    let median =
      if n % 2 = 1 then sorted.(n / 2) else (sorted.((n / 2) - 1) +. sorted.(n / 2)) /. 2.
    in
    let g t = lv.lv_flops /. t /. 1e9 in
    p
      "   %-22s %-26s %8.1f GFLOP/s med (min %7.1f)  tuner iso %8.1f q %8.1f  spread %4.1f%%  %s%s\n"
      lv.lv_tag lv.lv_label (g median) (g best) (g !(lv.lv_iso)) (g !(lv.lv_queued))
      ((worst -. best) /. best *. 100.)
      lv.lv_launch
      (if String.is_prefix lv.lv_mma ~prefix:"not-requested" then "" else "  mma " ^ lv.lv_mma);
    (* The finalists round re-times arms under a [FINAL] label; they stay out of the gh-755 ranking,
       whose population is one tuning round's seeds. *)
    if not (String.is_prefix lv.lv_label ~prefix:final_prefix) then
      Hashtbl.update instrument
        ~f:(function
          | None -> [ (lv.lv_label, median, !(lv.lv_iso), !(lv.lv_queued)) ]
          | Some l -> (lv.lv_label, median, !(lv.lv_iso), !(lv.lv_queued)) :: l)
        lv.lv_tag;
    Hashtbl.add_multi measured ~key:lv.lv_tag
      ~data:(lv.lv_label, { st_med = g median; st_min = g best; st_queued = g !(lv.lv_queued) });
    g median
  in
  (* On the success path a release failure is a failed cell, not something to swallow: the arm's
     slabs stay in the pool tables and every later round is then timed under that growth.
     [release_quietly] stays for paths that are already reporting a failure. *)
  let release lv =
    try Context.release !(lv.lv_ctx)
    with e when not (is_fatal e) ->
      fail "%s / %s: releasing the measured arm failed: %s" lv.lv_tag lv.lv_label
        (List.hd_exn (String.split_lines (Exn.to_string e)))
  in
  let results : (string, (string * float) list) Hashtbl.t = Hashtbl.create (module String) in
  let record_result lv g =
    Hashtbl.update results lv.lv_tag ~f:(function
      | None -> [ (lv.lv_label, g) ]
      | Some l -> (lv.lv_label, g) :: l)
  in
  let run_round ~label lives =
    let lives = if reversed then List.rev lives else lives in
    if not (List.is_empty lives) then begin
      p "\n-- round: %s (%d arms interleaved)\n" label (List.length lives);
      time_round lives;
      List.iter lives ~f:(fun lv ->
          if (not !(lv.lv_failed)) && not (List.is_empty !(lv.lv_times)) then
            if Float.is_finite !(lv.lv_iso) && Float.is_finite !(lv.lv_queued) then
              record_result lv (report lv)
            else (
              lv.lv_failed := true;
              refused "%s / %s: every %s tuner timing was refused for host contention" lv.lv_tag
                lv.lv_label
                (if Float.is_finite !(lv.lv_iso) then "queued"
                 else if Float.is_finite !(lv.lv_queued) then "isolated"
                 else "isolated and queued")));
      List.iter lives ~f:release
    end
  in
  (* Round 0: the untuned shipped default, one arm per site. *)
  (* The default a caller of the runners actually gets: under [model_default_schedule] (the
     [performance]/[approximate] profiles) the benchmark runners' untuned compile is the cost
     model's pick, [Autotune.model_default], not the plain pipeline -- so that is the arm, labelled
     as such. It is a raising API, so a failure there is contained by [arm]'s handler as one cell. *)
  let model_default = Lazy.force Autotune.model_default_enabled in
  let default_label = if model_default then "default (model-picked)" else "default (untuned)" in
  let default_arm ?(prefix = "") pr =
    arm pr ~label:(prefix ^ default_label) ~compile:(fun ~name ~fatal_seen fwd ->
        if model_default then Autotune.model_default ~name (Context.auto ()) fwd Ir.Indexing.Empty
        else compiled ~fatal_seen ~name (Context.auto ()) fwd)
  in
  run_round ~label:default_label (List.filter_map prepared ~f:default_arm);
  (* A seed arm: the sketch geometry [q] applied to the site's lowering as the pure IR transform it
     is. Shared by the per-geometry rounds and the finalists round, so a finalist is the SAME
     schedule its geometry round measured, recompiled. *)
  let seed_arm ?(prefix = "") pr (q : Autotune.sketch_params) =
    arm ~tensorized:q.sk_mma pr
      ~label:(prefix ^ geom_label q)
      ~compile:(fun ~name ~fatal_seen fwd ->
        compiled ~fatal_seen
          ~lowered_transform:(fun o ->
            [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q o) o ])
          ~name (Context.auto ()) fwd)
  in
  (* One round per geometry, over the sites whose seed list offers it: that is the comparison the
     experiment makes, so that is the set that has to be interleaved. Menu order is preserved. *)
  if do_seeds then begin
    (* A round is a BASE geometry, [sk_batch_grid] twins included, because a merged site's
       [32x32x8/4x4] and a batched site's [32x32x8/4x4 bgrid] are the two arms the experiment
       actually compares -- putting the twins in separate rounds would leave exactly that pair
       un-interleaved. A batched site therefore contributes all of its batch arms to the round (with
       interior batch loops, the [bgrid-in] order of gh-ocannl-728 is the third). *)
    let base_geom g =
      String.chop_suffix_if_exists
        (String.chop_suffix_if_exists g ~suffix:" bgrid-in")
        ~suffix:" bgrid"
    in
    let geometries =
      List.concat_map prepared ~f:(fun (_, _, seeds, _, _, _) ->
          List.map seeds ~f:(fun q -> base_geom (geom_label q)))
      |> List.dedup_and_sort ~compare:String.compare
    in
    (* The hoisted CPU seeds mint packed-constant nodes per candidate; those land in the device-wide
       constant cache that [Context.release] cannot reclaim, so this branch grows by one packed pool
       per such candidate. Said out loud rather than papered over -- the alternative is to drop the
       family, and a measurement is worth more than a tidy footprint here. *)
    if (not on_gpu) && List.exists geometries ~f:(fun g -> String.is_substring g ~substring:"hoist")
    then
      p
        "   note: the hoisted CPU seeds each mint a packed constant pool that Context.release \
         cannot reclaim; this run's device footprint grows with them\n";
    List.iter geometries ~f:(fun g ->
        let lives =
          List.concat_map prepared ~f:(fun ((_, _, seeds, _, _, _) as pr) ->
              List.filter seeds ~f:(fun q -> String.equal (base_geom (geom_label q)) g)
              |> List.filter_map ~f:(seed_arm pr))
        in
        run_round ~label:g lives)
  end;
  (* The searches run one site at a time -- a search is minutes of its own dispatches, so there is
     nothing to interleave THEM with -- but the winners they crown are then timed together, in one
     interleaved round, exactly like a seed geometry. Timing each winner as its own singleton round
     would leave the tuned comparison confounded by the drift accumulated during the other sites'
     searches, which is the same defect the seed rounds were restructured to remove.

     Every parameter that shapes the search is passed explicitly and printed: [~search:true] alone
     would leave timing repeats and the model pre-filter to whatever OCANNL_* or the nearest
     ocannl_config says, so a pruning fraction below 1 would print "full search" over a search that
     was not; beam width and rounds follow the effective configuration (below) and are printed as
     used. And a completed search is not automatically a measurement: when every
     candidate declines, the outcome is still [Searched] while [best_ms] is infinite and the
     returned routine is the untuned default, so a finite time and a non-empty winner are required
     before the arm is admitted. *)
  (* Beam width and rounds are the EFFECTIVE configuration's -- the reference defaults 2/2, or a
     profile's (the [approximate] payload's 4/4) -- read with [Autotune.tune]'s own defaults and
     passed explicitly, so the search a leg runs is the one a caller under the same configuration
     gets, and the header prints what was used rather than restating a constant. *)
  let setting_int name v =
    match Int.of_string_opt (String.strip v) with
    | Some n when n >= 0 -> n
    | _ -> invalid_arg (Printf.sprintf "%s=%S is not a nonnegative integer" name v)
  in
  let tn_beam =
    max 1
      (setting_int "autotune_beam_width"
         (Utils.get_global_arg ~arg_name:"autotune_beam_width" ~default:"2"))
  and tn_rounds =
    setting_int "autotune_rounds" (Utils.get_global_arg ~arg_name:"autotune_rounds" ~default:"2")
  and tn_repeats = 3
  and tn_keep = 1.0
  and tn_split = 8 in
  let tuned = Hashtbl.create (module String) in
  let winners =
    if not do_tune then []
    else begin
      (* The four pinned arguments are not the whole treatment: [Autotune.tune] also consults gates
         that have no parameter -- the bound-pruning gate and the two roofline constants it prices
         candidates against -- so two identical invocations can publish differently pruned searches
         under one label. They cannot be pinned from here, so they are REPORTED, and a tune run is
         reproducible only against the line below. *)
      let shown = function "" -> "(unset)" | v -> v in
      p
        "\n\
         -- searches: beam_width %d, rounds %d, repeats %d, keep_fraction %.2f, \
         split_reduce_max_sites %d, cache disabled\n\
         -- ambient search gates: autotune_bound_pruning=%s autotune_timing=%s model_peak_flops=%s \
         model_peak_memory_bandwidth=%s\n"
        tn_beam tn_rounds tn_repeats tn_keep tn_split
        (Utils.get_global_arg ~arg_name:"autotune_bound_pruning" ~default:"false")
        (Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued")
        (shown (Utils.get_global_arg ~arg_name:"model_peak_flops" ~default:""))
        (shown (Utils.get_global_arg ~arg_name:"model_peak_memory_bandwidth" ~default:""));
      List.filter_map prepared ~f:(fun ((s, _, _, _, _, _) as pr) ->
          let lbl = ref "" and ms = ref Float.nan and outcome = ref None and contended = ref 0 in
          let lv =
            arm pr ~label:tuned_label ~compile:(fun ~name ~fatal_seen fwd ->
                Autotune.tune ~name ~search:true ~cache_dir:"" ~beam_width:tn_beam ~rounds:tn_rounds
                  ~repeats:tn_repeats ~keep_fraction:tn_keep ~max_split_reduce_sites:tn_split
                  ~report:(fun (r : Autotune.report) ->
                    lbl := r.Autotune.best_label;
                    contended := r.Autotune.timings_contended;
                    ms := r.Autotune.best_ms;
                    outcome := Some r.Autotune.outcome;
                    (* [tune] reports and then re-raises the original exception when a search dies
                       on a fatal failure; the report is the only place this callback learns that,
                       so the guard is set here rather than left to arm's exception taxonomy. *)
                    match r.Autotune.outcome with
                    | Autotune.Search_died _ | Autotune.Pre_search_failure _ -> fatal_seen := true
                    | _ -> ())
                  (Context.auto ()) fwd Ir.Indexing.Empty)
          in
          let admissible =
            match !outcome with
            | None ->
                if Option.is_some lv then fail "%s: the tune cell reported nothing" s.tag;
                false
            | Some Autotune.Searched when !contended > 0 ->
                (* A refused timing window leaves the search's measurement set incomplete: its crown
                   was chosen from a truncated comparison, so it is not the tuner's selection. *)
                fail
                  "%s: the search refused %d timing window(s) for contention -- its crown %S is \
                   from an incomplete comparison"
                  s.tag !contended !lbl;
                false
            | Some Autotune.Searched when Float.is_finite !ms && not (String.is_empty !lbl) ->
                p "   %-22s crowned: %s  (search best_ms %.4f)\n" s.tag !lbl !ms;
                Hashtbl.set tuned ~key:s.tag ~data:!lbl;
                true
            | Some Autotune.Searched ->
                fail
                  "%s: the search timed no candidate (best_ms %g, winner %S) -- the routine it \
                   returned is the untuned default, not a tuned measurement"
                  s.tag !ms !lbl;
                false
            | Some o ->
                fail
                  "%s: the tune cell did not search (%s) -- its number is not a tuned measurement"
                  s.tag (Autotune.outcome_name o);
                false
          in
          match lv with
          | Some lv when admissible -> Some lv
          | Some lv ->
              release lv;
              None
          | None -> None)
    end
  in
  (* The finalists round (gh-ocannl-728 Move 0). Each geometry round compares arms AT one geometry;
     "the best candidate a site has" is a comparison ACROSS rounds, which the bestseed column reads
     through whatever drift separated them. So each site's best scalar seed and best tensorized seed
     (by their own rounds' readings, under each statistic the gate uses) are recompiled and timed
     again here, interleaved with each other, with every other site's finalists, and with the
     searches' crowned winners -- one round holding every contender of the question "does the merged
     site beat the best the split site has", duplicated sites included. The finalists are chosen by
     one statistic and then re-measured, so the round's numbers are not the selection's own
     optimistic draw. The untuned default joins the round too: it is a candidate a site has, and
     nothing guarantees a seed or the queued search's crown beats it under every statistic the gate
     reads. *)
  let finalists =
    if not do_seeds then []
    else
      List.concat_map prepared ~f:(fun ((s, _, seeds, _, _, _) as pr) ->
          let seed_rows =
            Option.value (Hashtbl.find measured s.tag) ~default:[]
            |> List.filter_map ~f:(fun (l, st) ->
                List.find seeds ~f:(fun q -> String.equal (geom_label q) l)
                |> Option.map ~f:(fun q -> (q, st)))
          in
          (* The best seed of each class under EACH statistic the gate reads, deduplicated: when the
             statistics rank differently (the gh-755 question), the best under the min-batch or
             queued reading is in the round too, not only the median's pick. *)
          List.concat_map [ false; true ] ~f:(fun mma ->
              let rows =
                List.filter seed_rows ~f:(fun ((q : Autotune.sketch_params), _) ->
                    Bool.equal q.sk_mma mma)
              in
              List.filter_map
                [ (fun st -> st.st_med); (fun st -> st.st_min); (fun st -> st.st_queued) ]
                ~f:(fun stat ->
                  List.max_elt rows ~compare:(fun (_, a) (_, b) -> Float.compare (stat a) (stat b))
                  |> Option.map ~f:fst))
          |> List.dedup_and_sort ~compare:(fun a b -> String.compare (geom_label a) (geom_label b))
          |> List.filter_map ~f:(seed_arm ~prefix:final_prefix pr))
  in
  let contenders = finalists @ winners in
  run_round
    ~label:
      (match (finalists, winners) with
      | _ :: _, _ :: _ ->
          "finalists (each site's best seeds, the searches' crowned winners and the defaults)"
      | _ :: _, [] -> "finalists (each site's best scalar and tensorized seeds and the defaults)"
      | [], _ -> "TUNED winners and the defaults")
    (if List.is_empty contenders then []
     else contenders @ List.filter_map prepared ~f:(default_arm ~prefix:final_prefix));
  let f_opt = function None -> "     n/a" | Some g -> Printf.sprintf "%8.1f" g in
  let of_site tag lbl =
    Option.bind (Hashtbl.find results tag) ~f:(fun l ->
        List.find l ~f:(fun (l', _) -> String.equal l' lbl) |> Option.map ~f:snd)
  in
  (* The two named columns are the two geometries gh-ocannl-728 quotes as crowned on gfx1151 -- but
     only on a GPU backend: [geom_label] prefixes the CPU family's seeds with "cpu" and its
     blocktile menu is 16 and 8, so asking for the GPU spellings there would print n/a over
     measurements that were taken. A site's best arm at a geometry may be a twin (the [bgrid] one on
     a batched GPU site, the [hoist] one on CPU), so the twins are tried first. *)
  let col_a, col_b =
    if on_gpu then ("gpu 32x32x8/4x4", "gpu 16x16x8/2x2") else ("cpu 16x16x16/0x0", "cpu 8x8x8/0x0")
  in
  let crowned tag want =
    List.find_map [ want ^ " bgrid"; want ^ " hoist"; want ] ~f:(of_site tag)
  in
  (* The blocking, without the backend prefix or the register tile, so the header column stays 8
     wide whichever family named it. *)
  let short_col c =
    String.chop_prefix_if_exists ~prefix:"cpu " (String.chop_prefix_if_exists ~prefix:"gpu " c)
    |> String.split ~on:'/' |> List.hd_exn
  in
  (* gh-ocannl-755: does the tuner's objective rank the candidates the way steady-state throughput
     does? One table per site over the SEED arms only -- the untuned default and a crowned search
     compile their own lowering, so they are not members of the population a tuning round ranks.
     Ranks are printed rather than left to the reader because the whole question is whether two
     orderings differ, and a crown that moves is a rank-1 disagreement specifically. *)
  p "\n\n== gh-ocannl-755: candidate ranking under the two timing instruments ==\n";
  p
    "   batched = this bench's median batch (%d dispatches queued, one sync); iso/queued = \
     Autotune.time_routine ~repeats:%d in each mode, min over %d interleaved calls\n"
    repeats tuner_repeats nbatches;
  List.iter sites ~f:(fun s ->
      let arms =
        Option.value (Hashtbl.find instrument s.tag) ~default:[]
        |> List.filter ~f:(fun (l, _, _, _) ->
            (not (String.is_prefix l ~prefix:"default")) && not (String.is_prefix l ~prefix:"TUNED"))
      in
      if List.is_empty arms then p "\n   %s: no seed arm was measured\n" s.tag
      else begin
        (* Rank by a projection: position of each arm's label in that projection's ascending order
           of time. Derived from the same list all three columns come from, so a rank can never
           describe a different population than the number beside it. *)
        let rank_of key =
          let ordered =
            List.sort arms ~compare:(fun a b -> Float.compare (key a) (key b))
            |> List.mapi ~f:(fun i (l, _, _, _) -> (l, i + 1))
          in
          fun label -> List.Assoc.find_exn ordered label ~equal:String.equal
        in
        let batched (_, m, _, _) = m and iso (_, _, i, _) = i and queued (_, _, _, q) = q in
        let r_b = rank_of batched and r_i = rank_of iso and r_q = rank_of queued in
        let crown key =
          fst4 (List.hd_exn (List.sort arms ~compare:(fun a b -> Float.compare (key a) (key b))))
        in
        p "\n   %s (%.1f MFLOP)\n" s.tag (flops s /. 1e6);
        (* The last two columns subtract the two TUNER columns from each other, never the batched
           one: iso and queued are the same statistic (a min over the same interleaved passes) of
           the same instrument, so their difference is the per-launch round trip and nothing else,
           whereas batched is a median of noisy batches and differencing against it would mix the
           instrument offset with the choice of summary. *)
        p "   %-26s %10s %10s %10s   %5s %5s %5s   %9s %7s\n" "candidate" "batched" "iso" "queued"
          "rank" "rank" "rank" "iso-queued" "iso/";
        p "   %-26s %10s %10s %10s   %5s %5s %5s   %9s %7s\n" "" "ms" "ms" "ms" "bat" "iso" "que"
          "us" "queued";
        List.iter
          (List.sort arms ~compare:(fun a b -> Float.compare (batched a) (batched b)))
          ~f:(fun ((l, m, i, q) as a) ->
            p "   %-26s %10.4f %10.4f %10.4f   %5d %5d %5d   %9.1f %7.2f\n" l (m *. 1e3) (i *. 1e3)
              (q *. 1e3) (r_b l) (r_i l) (r_q l)
              ((iso a -. queued a) *. 1e6)
              (iso a /. queued a));
        let cb = crown batched and ci = crown iso and cq = crown queued in
        p "   crown: batched %S | iso %S | queued %S -- %s\n" cb ci cq
          (if String.equal cb ci then "the isolated instrument crowns the batched winner"
           else "THE CROWN MOVES between the two instruments")
      end);
  p "\n\n== summary (GFLOP/s, median timing batch) ==\n";
  p
    "   the named-geometry columns compare arms measured in ONE interleaved round; bestseed ranks \
     across rounds and is indicative\n";
  p "%-22s %9s  %8s  %8s  %8s  %8s  %8s  %s\n" "site" "MFLOP" "untuned" (short_col col_a)
    (short_col col_b) "bestseed" "tuned" "winner";
  List.iter sites ~f:(fun s ->
      let all = Option.value (Hashtbl.find results s.tag) ~default:[] in
      let seeds_only =
        List.filter all ~f:(fun (l, _) ->
            not
              (List.exists [ "default"; "TUNED"; final_prefix ] ~f:(fun prefix ->
                   String.is_prefix l ~prefix)))
      in
      let best_of =
        List.fold seeds_only ~init:None ~f:(fun acc (l, g) ->
            match acc with Some (_, bg) when Float.(bg >= g) -> acc | _ -> Some (l, g))
      in
      let tn = of_site s.tag tuned_label in
      p "%-22s %9.1f  %s  %s  %s  %s  %s  %s\n" s.tag
        (flops s /. 1e6)
        (f_opt (of_site s.tag default_label))
        (f_opt (crowned s.tag col_a))
        (f_opt (crowned s.tag col_b))
        (f_opt (Option.map best_of ~f:snd))
        (f_opt tn)
        (match Hashtbl.find tuned s.tag with
        | Some l when Option.is_some tn -> l
        | _ -> Option.value_map best_of ~default:"-" ~f:fst));
  (* gh-ocannl-728 Move 0's gate: does the heads-merged twin beat the BEST candidate the real
     heads-split site has by more than the in-run duplicate noise? Read off the finalists round
     only, where every contender -- each site's best scalar seed, best tensorized seed and the
     search's crown, for both copies of both shapes -- was timed interleaved. Per class and per
     statistic, a site's figure is its best contender of that class under that statistic. The noise
     is the larger disagreement of the two duplicated pairs; the effect is the merged pair's mean
     over the split pair's mean, and the worst case is the slower merged copy against the faster
     split one. *)
  if String.equal group "qkv" then begin
    let split = [ "Q_heads8_a"; "Q_heads8_b" ] and merged = [ "Q_heads1_a"; "Q_heads1_b" ] in
    let is_final l = String.is_prefix l ~prefix:final_prefix in
    let is_mma l = String.is_prefix l ~prefix:(final_prefix ^ "mma-") in
    let is_default = String.equal (final_prefix ^ default_label) in
    let classes =
      [
        ("scalar seeds", fun l -> is_final l && (not (is_mma l)) && not (is_default l));
        ("tensorized seeds", is_mma);
        ("untuned default", is_default);
        ("tuner's crown", String.equal tuned_label);
        ("any contender", fun l -> is_final l || String.equal l tuned_label);
      ]
    and statistics =
      [
        ("median batch", fun st -> st.st_med);
        ("min batch", fun st -> st.st_min);
        ("tuner queued", fun st -> st.st_queued);
      ]
    in
    let best tag in_class stat =
      Option.value (Hashtbl.find measured tag) ~default:[]
      |> List.filter ~f:(fun (l, _) -> in_class l)
      |> List.max_elt ~compare:(fun (_, a) (_, b) -> Float.compare (stat a) (stat b))
    in
    let pct x = Printf.sprintf "%+.1f%%" (x *. 100.) in
    p "\n\n== gh-ocannl-728 Move 0: the heads-merged twin against the best of the split site ==\n";
    p
      "   finalists round only (GFLOP/s); noise = larger |a-b|/min over the duplicated pairs; gain \
       = mean(heads1)/mean(heads8) - 1; worst = min(heads1)/max(heads8) - 1\n";
    p "   %-17s %-13s %9s %9s %9s %9s  %8s %8s %8s  %s\n" "class" "statistic" "heads8_a" "heads8_b"
      "heads1_a" "heads1_b" "noise" "gain" "worst" "verdict";
    List.iter classes ~f:(fun (cname, in_class) ->
        List.iter statistics ~f:(fun (sname, stat) ->
            let vals tags =
              List.map tags ~f:(fun t ->
                  Option.map (best t in_class stat) ~f:(fun (_, st) -> stat st))
            in
            match (Option.all (vals split), Option.all (vals merged)) with
            | Some ([ s1; s2 ] as sv), Some ([ m1; m2 ] as mv) ->
                let rel x y = Float.abs (x -. y) /. Float.min x y in
                let noise = Float.max (rel s1 s2) (rel m1 m2) in
                let gain = ((m1 +. m2) /. (s1 +. s2)) -. 1. in
                let worst = (Float.min m1 m2 /. Float.max s1 s2) -. 1. in
                p "   %-17s %-13s %9.1f %9.1f %9.1f %9.1f  %8s %8s %8s  %s\n" cname sname
                  (List.nth_exn sv 0) (List.nth_exn sv 1) (List.nth_exn mv 0) (List.nth_exn mv 1)
                  (Printf.sprintf "%.1f%%" (noise *. 100.))
                  (pct gain) (pct worst)
                  (if Float.(gain > noise) then "merged beats split beyond the noise"
                   else if Float.(gain < -.noise) then "merged LOSES beyond the noise"
                   else "within the noise")
            | _ -> p "   %-17s %-13s   (no contender of this class on every site)\n" cname sname);
        (* Which arm each site's median-best contender of the class is: the geometry the gate's
           headline figure belongs to. *)
        let names =
          List.map (split @ merged) ~f:(fun t ->
              match best t in_class (fun st -> st.st_med) with
              | Some (l, _) ->
                  Printf.sprintf "%s=%s" t (String.chop_prefix_if_exists l ~prefix:final_prefix)
              | None -> t ^ "=-")
        in
        p "   %-17s by median: %s\n" "" (String.concat ~sep:" | " names))
  end;
  if !failures > 0 || !unmeasured > 0 then
    p
      "\n\
       %d cell(s) of this experiment could not be measured or failed parity, and %d had every \
       timing refused for host contention -- see the !! lines.\n"
      !failures !unmeasured;
  if !failures > 0 || (!unmeasured > 0 && not allow_unmeasured) then Stdlib.exit 1
