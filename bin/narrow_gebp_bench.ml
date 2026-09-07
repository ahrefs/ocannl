(* Narrow-operand GEBP benchmark (gh-ocannl-575): the CPU register-tiled [Tile_mma] over 16-bit
   storage, with the widening folded into the packing [Stage] ([tile_prec]). Times, on the C
   backends, the serial naive kernel against the all-Serial packed GEBP and the pool-parallel
   grid-outermost per-chunk-packing flavor (schedule_bench's packmma / pm_bpk shapes), at the
   requested storage precision — the packed panels are minted at the compute precision the emission
   resolves ([Numerics.cpu_compute_prec] over [Context.hardware_limits]), so an f32 run measures the
   plain GEBP, a bf16/f16 run measures f32-GEBP-over-narrow-storage, and an f16 run with
   --ocannl_fp16_arithmetic=true on a native-arithmetic target (NEON, AVX512-FP16) measures the
   pure-f16 GEBP — the honest first comparison the issue asks for. On a target that merely promotes
   ([cc_fp16_arithmetic] probes it), the policy is ignored and the run stays f32-compute.

   Usage: OCANNL_BACKEND=cc dune exec bin/narrow_gebp_bench.exe -- [f32|bf16|f16] [n] [repeats] [bm]
   [bk] (defaults f32, 512, 20, 64, 256; [bm]/[bk] also as [--bm=]/[--bk=] flags). The packed
   variants schedule [Sched.split]s of factor [bm] over the i axis and [bk] over the k axis, so they
   need [n mod bm = 0] and [n mod bk = 0] (and an n big enough to leave a loop nest at all); an
   unschedulable n runs the unblocked naive variant alone, so an arbitrary extent still has
   something to compare against. Each line carries a position-weighted checksum of the whole output,
   which is what makes a mishandled edge region visible — the register-tiled micro-kernel peels row
   and column edges whenever its block shape does not cover the extent, and a single interior cell
   sees none of that. Since gh-ocannl-639 every leg (the naive serial fallback included) accumulates
   at compute precision, so with this bench's exact-by-design partial sums the checksums are
   comparable across the board even in a narrow-storage run. Each packed line carries its
   [C_syntax.mma_census] rendering, because a [Tile_mma] whose preconditions fail (a column extent
   below the compute vector width is one of several ways in) renders the scalar fallback while still
   reporting as "packmma" — read the bracket, not the variant name, when deciding what a timing
   measured. Readbacks stay outside the timed region (the [Context.get_values] trap,
   docs/agent-notes/backend-precision-and-simd.md). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

(* Flushed per line ([Bench_out]): see gh-ocannl-829 -- a buffered table is invisible until the
   process exits, which on a slow leg reads as a hang. *)
let p fmt = Bench_out.p fmt

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

(* The aperiodic mix used to mint the operand values below, and the whole-output checksum further
   down, both come from [Bench_checksum] (gh-ocannl-711) — shared with [schedule_bench], which had
   the pre-fix flat-offset forms of both. Why aperiodic and why keyed on the (row, column) pair is
   documented there; the short version is that a residue of a flattened offset loses its row
   dependence exactly when the modulus divides the row stride, and that a per-axis residue leaves a
   shift period that a user-chosen [bk] can hit. *)

let () =
  (* Positional geometry beside the [--ocannl_*] config flags, split by [Bench_args]
     (gh-ocannl-634): a bare [-64] is a positional, not an option, so it reaches the range check
     that names it instead of silently shifting every later positional into the wrong slot. Each
     integer is a positive extent or count, checked where it is read. *)
  let args = Bench_args.create "narrow_gebp_bench" in
  let prec_name = Bench_args.string args 0 ~default:"f32" in
  let prec =
    match prec_name with
    | "f32" -> Ir.Ops.single
    | "bf16" -> Ir.Ops.bfloat16
    | "f16" -> Ir.Ops.half
    | s -> invalid_arg ("narrow_gebp_bench: precision f32|bf16|f16 expected, got " ^ s)
  in
  let n = Bench_args.int args 1 ~name:"n" ~default:512 in
  let repeats = Bench_args.int args 2 ~name:"repeats" ~default:20 in
  let bm = Bench_args.int args 3 ~flag:"bm" ~name:"bm" ~default:64 in
  let bk = Bench_args.int args 4 ~flag:"bk" ~name:"bk" ~default:256 in
  (* What the packed variants require of n, in one place: an i/j/k nest to address (extent-1 loops
     are simplified away before the transform runs, leaving nothing to schedule), and extents
     divisible by the [Sched.split] factors. The naive variant has no blocking at all, so it runs
     for any n. *)
  let unschedulable =
    (if n < 2 then [ Printf.sprintf "n = %d leaves no i/j/k loop nest to schedule" n ] else [])
    @ List.filter_map
        [ ("bm", bm); ("bk", bk) ]
        ~f:(fun (name, f) ->
          if n % f = 0 then None
          else
            Some (Printf.sprintf "%s = %d does not divide n = %d (remainder %d)" name f n (n % f)))
  in
  let flops = 2.0 *. Float.of_int n *. Float.of_int n *. Float.of_int n in
  (* Operand values that vary with EVERY index at every n, drawn through [Bench_checksum.mix] so
     that no shift of any index is a symmetry, and exactly representable at every storage precision
     (the parity-test recipes). Two traps sit behind this, both of which cost a wrong version to
     find. The value must not be a modulus of the FLATTENED offset [(i * n + j) % p] — that loses
     its row dependence exactly when p divides n, which made an earlier ma constant along i at 3 | n
     and an earlier mb constant along k at 5 | n, with n = 960 collapsing both; a transform
     substituting the wrong row of a collapsed operand then computes the correct output, which no
     whole-output check can see. And reducing each index separately fixes that but leaves the
     period: with both operands drawn mod 5 in k, every packed K panel repeats under [k -> k + 5],
     so a run with [bk = 5] hides a panel-substitution bug just as thoroughly. The mix has no shift
     symmetry at any lag, so neither survives — measured over n = 2..2000 for the collapse, over
     lags 1..256 for the shift, and as zero duplicate full B~ panels at every bk from 1 to 64.

     The value SETS are the original ones — ma in {0.25, 0.5, 0.75}, mb in {-1, -0.5, 0, 0.5, 1} —
     because the resulting 1/8 product granularity is load-bearing in a narrow-storage run (see the
     checksum's note below; a finer-grained attempt overflowed the bf16 mantissa and split the legs
     at n = 256). ma is strictly positive, so a dropped term cannot cancel into something plausible;
     mb is signed and near-mean-zero, which makes partial sums random-walk rather than grow with n —
     measured max |partial sum| 18 at n = 256 and 31 at n = 512, against the ~0.075n a periodic-in-k
     pairing produced. *)
  let ma =
    NTDSL.init ~l:"ma" ~prec ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs ->
        Float.of_int (1 + (Bench_checksum.mix ~salt:0x5A17 idcs.(0) idcs.(1) % 3)) *. 0.25)
      ()
  in
  let mb =
    NTDSL.init ~l:"mb" ~prec ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs ->
        Float.of_int ((Bench_checksum.mix ~salt:0x3C6E idcs.(0) idcs.(1) % 5) - 2) *. 0.5)
      ()
  in
  let packed_schedule ~grid ~tile_prec ~mc (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      (* The i/j/k nest is what the packed schedule addresses. Extent-1 loops are simplified away
         before the transform runs, so a degenerate n leaves no 3-deep nest to schedule — report
         that rather than raising [Not_found_s] out of a [find_exn]. *)
      match List.find paths ~f:(fun p -> List.length p = 3) with
      | Some [ i; j; k ] -> (i, j, k)
      | _ ->
          failwith
            (Printf.sprintf
               "narrow_gebp_bench: no 3-deep i/j/k loop nest to schedule at n = %d (deepest nest \
                found: %d loops) — the packed variants need a non-degenerate extent"
               n
               (List.fold paths ~init:0 ~f:(fun acc p -> Int.max acc (List.length p))))
    in
    let outer_i = if grid then LL.Grid else LL.Serial in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = false;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec;
        }
    in
    let zops =
      if not grid then []
      else
        let ez, zsyms = Sched.expand_zero ~tn:mc in
        let zi = List.hd_exn zsyms in
        let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
        [ ez; sp_zi ]
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    zops @ [ sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ]
    @ (if grid then [] else sink i_o [ k_o ])
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ] ]
    @ [ tz ]
  in
  (* Every variant is compared against another one CELL BY CELL, and each call site says which and
     whether the equality is required (gh-ocannl-711 review). The checksum cannot carry this: it is
     a linear functional of the output, so a row permutation survives it whenever the value
     difference is orthogonal to the weight difference, and an elementwise comparison has nothing to
     cancel.

     Which comparison is REQUIRED is what the comparability note below is about, and it is not
     uniform across the legs. naive narrows once per cell while the packed variants narrow once per
     k block, so past the extent at which the block partials stay storage-exact a naive-vs-packed
     difference is legitimate. The two packed variants share their reduction structure exactly, so
     their equality is required at EVERY extent — which is why packmma_par is compared against
     packmma rather than against naive. Folding it into the naive comparison would mix a
     packmma_par-only defect in with expected block-boundary rounding and leave the one equality
     that still holds unchecked. *)
  let disagreements = ref 0 in
  let expected_differences = ref 0 in
  let bench ~variant ~schedule ~against () =
    let%op mc = ma * mb in
    Ir.Tnode.update_prec mc.Tensor.value prec;
    let comp = named ("ngb_" ^ variant) (Train.forward mc) in
    let transform opt =
      match schedule with None -> opt | Some s -> Sched.apply (s ~mc:mc.Tensor.value opt) opt
    in
    let ctx = Context.auto () in
    (* The census records how codegen actually rendered each [Tile_mma] (gh-ocannl-479). A
       [Tile_mma] whose preconditions fail renders the scalar fallback and still reports as
       "packmma", so an unchecked run can present fallback timings as register-tiled ones — the
       column extent below the compute vector width is one way in, but so are a narrow
       [vector_bytes], a mixed operand precision, and [debug_log_from_routines]. Collecting the
       census only appends to a list, so it does not perturb what is compiled or timed. Since
       gh-ocannl-626 it travels on the compiled routine, and the "did this tensorize" predicate is
       shared with [schedule_bench] rather than re-derived here. *)
    let ctx, routine =
      Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
    in
    let mma = routine.Context.mma in
    let ctx = Context.run ctx routine in
    let _ = Context.get_values ctx mc.Tensor.value in
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx (Stdlib.Array.make repeats ())
    in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    (* Readback OUTSIDE the timed region (the [Context.get_values] trap,
       docs/agent-notes/backend-precision-and-simd.md): the cc scheduler is synchronous, so the run
       loop needs no readback to complete. *)
    let values = Context.get_values ctx mc.Tensor.value in
    (* Element [1][1] of the n*n result — an interior cell, away from the corners — except at n = 1,
       where the whole output is one element. One interior cell cannot see a remainder region, so
       the whole output is checksummed too: every correct variant prints the identical value, a
       tail-mishandling one does not.

       Which remainder, precisely — the obvious answer is the wrong one. A [Sched.split] whose
       factor does not divide its extent is NOT the case covered here: [unschedulable] rejects
       exactly those factors, so at a non-dividing [bm]/[bk] the packed variants never run and the
       naive line has nothing to be compared against. What IS covered is the register-tiled
       micro-kernel's own edge peel, which needs no split remainder: the [Tile_mma] covers full
       blocks only, and n is free of any width constraint because j is never split. At n = 77 with
       bm = 7, bk = 11 the emitted kernel carries two scalar peel loops — a column peel over j in
       [76, 77) and a row peel over i in [4, 7) — that the same kernel at n = 76 does not have, and
       all three variants agree on the checksum across it. So the tail this check guards is the
       micro-kernel peel and the packing [Stage] edges, not the block remainder. Position-weighted
       for a different reason than schedule_bench's: with the per-axis residues above there is no
       divisibility class on which a plain sum vanishes (unlike the flattened form, whose collapse
       also zeroed it). What an unweighted sum cannot see is a PERMUTATION — a tail written with the
       right values at the wrong offsets leaves the multiset intact, and the multiset is all a plain
       sum reads — and a misplaced edge is exactly what the peel risks.

       [Bench_checksum.whole_output] runs the weight through the same mix on the (row, column) pair
       for the same reason the operands do, and the flat-offset form is the trap it avoids: a weight
       of [1 + (t mod 251)] over the flat offset t = i*n + j collapses to [1 + j] when 251 divides
       n, giving every row identical weights, so at n = 251 (or 502, 753, ...) a row permutation was
       invisible to the checksum AND to the spot cell at once. Same degeneracy as the operands', one
       line away, and it came in with the port. Weights stay capped at 251 so that products of these
       exact-in-binary operands stay exact in the double accumulator, and the printed [chk a/b] is
       one sum per weight stream: at a narrow n a single capped stream runs out of distinct row
       weight vectors and two rows collide, whose swap no weighting of that stream can see.

       Cross-variant equality is EXACT in an f32 run, at every extent: the products are multiples of
       1/8 bounded by 0.75n, so the whole reduction is exact in the f32 accumulator and independent
       of the order the variants sum in. Since gh-ocannl-639 it holds in narrow storage runs too, at
       every extent where the per-k-block partial sums stay exact at storage precision: the naive
       serial fallback now holds its accumulator at compute precision across the whole k extent and
       narrows once at the store (before gh-ocannl-639 it narrowed at EVERY k step, which split it
       from the packed legs at bf16 n = 320 and forced the run to announce the non-comparability
       outright), while the packed variants narrow once per k block (the C-tile stores back at [bk]
       boundaries — the narrowing points remain a property of the schedule's reduction structure;
       only the accumulator's WIDTH is policy). The near-mean-zero mb keeps partial sums
       random-walking (max 31 at n = 512, multiples of 1/8 — bf16-exact), so every rounding any
       variant performs is exact and the checksums agree bitwise; well beyond that extent an inexact
       block-boundary partial could split naive from packed again, far more rarely than the per-step
       narrowing did. Both checks are outside the timed region. *)
    let checksum = Bench_checksum.whole_output ~row_stride:n values in
    let agreement =
      match against with
      | None -> "reference"
      | Some (name, r, required) ->
          let d = Bench_checksum.first_difference ~reference:r values in
          if Option.is_some d then
            Int.incr (if required then disagreements else expected_differences);
          Bench_checksum.render_agreement ~name d
    in
    let spot = Int.min (n + 1) (Array.length values - 1) in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    (* Printed on EVERY timing line, untensorized variants included: an absent suffix is one a table
       reader does not notice (gh-ocannl-626). *)
    p "%-12s %8.3f ms  %8.2f GFLOP/s  (spot check [%d] %.2f, chk %s, %s)  [%s]\n" variant
      (secs *. 1e3)
      (flops /. secs /. 1e9)
      spot values.(spot) (Bench_checksum.render checksum) agreement
      (Ir.C_syntax.mma_summary_string mma);
    (secs, mma, values)
  in
  let ctx0 = Context.auto () in
  let limits = Context.hardware_limits ctx0 in
  let cprec =
    Numerics.cpu_compute_prec ~native_fp16_arithmetic:limits.Ir.Backend_intf.native_fp16_arithmetic
      prec
  in
  let tile_prec = if Ir.Ops.equal_prec cprec prec then None else Some cprec in
  p "GEBP n=%d, %d repeats, blocking bm=%d bk=%d, storage %s, compute %s, packed panels %s\n" n
    repeats bm bk (Ir.Ops.prec_string prec) (Ir.Ops.prec_string cprec)
    (Option.value_map tile_prec ~default:"(storage)" ~f:Ir.Ops.prec_string);
  (* Say it at runtime rather than only in a comment: since gh-ocannl-639 every variant accumulates
     at compute precision (naive narrows once per cell, packed once per k block). Whether the
     checksums are comparable depends on the extent: every rounding any variant performs is exact as
     long as the per-k-block partial sums stay storage-exact, which for these operands is measured
     through n = 512 at bf16 (max |partial| 31, multiples of 1/8); beyond that an inexact block
     partial can legitimately split naive from packed. *)
  (* One predicate, used by the note AND by the exit status below, so what the run SAYS about
     comparability and what it ENFORCES cannot drift apart. It governs the naive-vs-packed
     comparison only: the two packed variants share a reduction structure, so their equality is
     required at every extent. *)
  let naive_comparable = Option.is_none tile_prec || n <= 512 in
  if Option.is_some tile_prec then
    if naive_comparable then
      p
        "note: all variants accumulate in %s (gh-ocannl-639): naive narrows to %s once per cell,\n\
        \      packed variants once per k block — at this extent every such rounding is exact for\n\
        \      this bench's operands, so the checksums are comparable across the board.\n"
        (Ir.Ops.prec_string cprec) (Ir.Ops.prec_string prec)
    else
      p
        "note: all variants accumulate in %s (gh-ocannl-639), but naive narrows to %s once per\n\
        \      cell while packed variants narrow once per k block — at n = %d an inexact block\n\
        \      partial can legitimately split naive from packed; the packed variants remain\n\
        \      comparable with each other, and an f32 run is the cross-variant oracle.\n"
        (Ir.Ops.prec_string cprec) (Ir.Ops.prec_string prec) n;
  let t_naive, _, v_naive = bench ~variant:"naive" ~schedule:None ~against:None () in
  match unschedulable with
  | _ :: _ ->
      p "skipping the packed variants — this n is not schedulable with this blocking:\n";
      List.iter unschedulable ~f:(fun reason -> p "  %s\n" reason);
      if n >= 2 then p "pass a compatible blocking, e.g. --bm=<f> --bk=<f> with f dividing %d.\n" n
  | [] ->
      let t_pack, r_pack, v_pack =
        bench ~variant:"packmma"
          ~schedule:(Some (packed_schedule ~grid:false ~tile_prec))
          ~against:(Some ("naive", v_naive, naive_comparable))
          ()
      in
      (* Against packmma, not naive, and REQUIRED at every extent: the two packed variants narrow at
         the same k-block boundaries, so nothing the comparability note excuses can separate them —
         and comparing par against naive instead would bury a par-only defect in the same expected
         DIFFERS output as block-boundary rounding. *)
      let t_par, r_par, _ =
        bench ~variant:"packmma_par"
          ~schedule:(Some (packed_schedule ~grid:true ~tile_prec))
          ~against:(Some ("packmma", v_pack, true))
          ()
      in
      p "speedups vs naive: packmma %.1fx, packmma_par %.1fx\n" (t_naive /. t_pack)
        (t_naive /. t_par);
      (* Count the fallback rather than "anything that is not [Mma_register_tiled]": on the C
         backends this bench targets the two are the same set, but the latter also indicts
         [Mma_intrinsics], so it would false-warn the moment an arm runs on a GPU. That predicate
         now lives once, in [C_syntax] (gh-ocannl-626), so the two benches cannot disagree. *)
      let all = Ir.C_syntax.merge_mma_summaries [ r_pack; r_par ] in
      let declined = all.Ir.C_syntax.scalar_fallbacks in
      if declined > 0 then
        p
          "WARNING: %d of %d Tile_mma statements rendered the scalar fallback (see the census\n\
           above) — these are NOT register-tiled timings. Re-run with\n\
           --ocannl_schedule_log_declines=true for the per-rule reason.\n"
          declined all.Ir.C_syntax.statements;
      (* A variant that computed something ELSE, which the checksum can miss and this cannot. Where
         the comparison was REQUIRED — every extent for the two packed variants against each other,
         and the naive comparison wherever the note above calls the legs comparable — every leg's
         reduction is exact whatever order it sums in, so a difference is a wrong result and not
         rounding. It therefore EXITS NONZERO, after every variant has been reported: a guard that
         only prints leaves an automated run free to keep the speedup of a kernel already known to
         be wrong. A difference the note calls legitimate is counted separately and stays
         non-fatal. *)
      if !expected_differences > 0 then
        p
          "note: %d naive-vs-packed difference(s), which this run's comparability note says are\n\
           legitimate at this extent and storage precision. The packed variants are still\n\
           required to agree with each other, and were checked.\n"
          !expected_differences;
      if !disagreements > 0 then (
        p
          "WRONG RESULT: %d required comparison(s) failed — the DIFFERS lines above name the\n\
           first cell and both values. At these operands the compared legs round identically, so\n\
           this is not rounding.\n"
          !disagreements;
        Stdlib.exit 1)
