(* Matmul schedule benchmark (docs/proposals/schedule-ir-optops.md, phases S1-S3; gh-ocannl-412
   acceptance): times the naive 1x1-launch kernel against three hand-written schedules on the
   configured backend --

   - parallel: one thread per output element (Split i / Split j into Grid x Workgroup; the S1 shape,
   Boehm kernel 1 equivalent); - smem: + Split k, operands staged through workgroup-shared tiles,
   the output accumulator privatized to a per-thread scalar (S2 + Privatize, Boehm kernel 3); -
   regtile: + second-level splits with materialized-unroll TM x TN register tiles accumulating into
   a privatized per-thread tile (S3 + Privatize, Boehm kernel 4/5 shape).

   On the C backends the shared schedules are rejected and the CPU variants run instead: cpupack (S4
   cache tiling + operand packing, all-Serial), tensorize (whole-triple register-tiled Tile_mma,
   gh-ocannl-469), packmma (Tile_mma composed with cache tiling + packing, all-Serial GEBP),
   packmma_par (the same composition with pool-parallel Grid row blocks and per-chunk privatized A~
   tiles; the B~ pack at k_o re-enters the parallel construct once per k-block), and the three
   grid-outermost flavors — one dispatch spanning the whole GEBP triple (gh-ocannl-473 /
   gh-ocannl-475): pm_hoist (hoisted constant-pool B~ panel, A read in place; autotune's sk_grid &&
   sk_hoist seed), pm_mixed (hoisted B~ plus an in-kernel per-chunk A~ pack; the sk_pack_rest seed),
   and pm_bpk (both operands packed in-kernel, each chunk re-packing its own B~ panel — needs the
   panel under the per-chunk privatization cap, config cc_grid_private_bytes_cap, or it silently
   declines to serial: check with --ocannl_schedule_log_declines=true).

   Usage: dune exec bin/schedule_bench.exe -- [n] [repeats] [m] [k] [naive_repeats] (defaults 256,
   20, n, n, repeats; the output is m x n, the reduction depth k — deep-K gradient-GEMM geometries
   are m,n << k). [naive_repeats] defaults to [repeats], which makes the naive leg the whole wall
   clock as soon as the size grows: at m = n = k = 512 on an M4 Max one 1x1-launch run is ~10 s
   against milliseconds for every scheduled variant, and queueing a second one behind it costs far
   more than that ratio predicts (gh-ocannl-828), so the default 20 repeats put the run past ten
   minutes with nothing to show for it. That is what the fifth argument is for — pass a small count,
   or 0 to skip the naive TIMING (speedups then print as nan). It does not skip the computation
   outright: the oracle every variant is compared against is then materialized by one untimed run,
   compiled without a [lowered_transform] so that the backend's default schedule applies, which is
   why 0 costs milliseconds on the GPU backends and the naive leg costs seconds. Output is flushed
   per line and every leg slow enough to be worth waiting for — that reference run included —
   announces itself BEFORE it runs and reports what it cost, so a run in progress is distinguishable
   from a hang.

   Each scheduled variant needs its own [Sched.split] factors to divide the extents they split (i
   over m, j over n, k over k), and every one of them needs a non-degenerate nest to address; a
   variant whose requirement fails is skipped by name with the reason. Two are exempt from the
   divisibility half — [parallel] on the GPU backends and [tensorize] on the C ones, which put
   nothing downstream of the split that a remainder guard would break — so an arbitrary
   (non-degenerate) extent still gets a scheduled measurement on either branch, and the naive kernel
   runs at any size whatsoever. Each line carries a position-weighted checksum of the whole output,
   which is what makes a remainder-region error visible, and every variant carrying a [Tile_mma]
   carries its [C_syntax.mma_census] rendering: a [Tile_mma] whose preconditions fail (a column
   extent below the compute vector width is one way in, and arbitrary extents reach it) renders the
   scalar fallback while still reporting under its schedule's name, so read the bracket, not the
   variant name, when deciding what a timing measured. Run with OCANNL_BACKEND=metal (or cuda); the
   C backends reject the shared schedules. Timing includes kernel executions and one device-to-host
   transfer per variant (runs queue on the stream; get_values synchronizes). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

(* CUDA's uniform-f32 MMA arm is gated on tf32 (the Numerics policy): without it the [Tile_mma]s of
   the mma_pd* variants render the lane-0 scalar fallback and the labels would time a kernel that
   never tensorizes ("timed is not tensorized", docs/agent-notes/scheduling-and-autotune.md).
   Metal's f32 [simdgroup_matrix] path has no such gate and is unaffected. Cross-check a new
   backend's emission with --ocannl_schedule_log_declines=true before trusting the labels.

   The tf32 rounding is not free of the bench's guard, and never was: the comparison oracle is the
   UNSCHEDULED kernel, which carries no [Tile_mma] and so rounds nothing, while an mma variant
   rounds both its operands to tf32 — so [= reference] on an mma line is a claim that the operands
   survive that rounding exactly. They do, by construction rather than by luck: ma is a multiple of
   1/16 below 3 (six significant bits) and mb an integer in -8..8 (four), against tf32's 11-bit
   significand. Confirmed on an RTX 5070 Ti under CUDA 13.3 at the gh-ocannl-738 granularity bump —
   mma_pd1 and mma_pd2 both report [= reference] at 256^3, 512^3, 128x128x1024 and 320x192x64, with
   the census confirming they tensorized. The check is not vacuous: reminting ma at 1/4096 (twelve
   significant bits) turns exactly those two legs red — [DIFFERS from reference at [0]] — and leaves
   parallel, smem and regtile green, since only the tensorized legs round. That is the negative
   control to re-run before raising the granularity again, or when porting the claim to a backend
   whose mma input format is narrower than tf32. *)
let () = Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = true }

(* Flushed per line ([Bench_out]): an unflushed table reaches the reader only when the process
   exits, which on a leg that is seconds to minutes per run reads as a hang (gh-ocannl-829). *)
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

let () =
  (* Positional integer args only — the --ocannl_* config flags share the argv, and [Bench_args]
     holds the split (gh-ocannl-634). Getting it wrong is worse here than in a one-argument bench:
     with five positionals, an argument silently dropped loses its own value AND shifts every later
     one into the wrong slot (`schedule_bench 256 20 -64 512` would run with m defaulting to n and k
     = 512 read as m), so the bad value never reaches the range check that names it. Every argument
     is a positive extent or count, except [naive_repeats]. *)
  let args = Bench_args.create "schedule_bench" in
  let n = Bench_args.int args 0 ~name:"n" ~default:256 in
  let repeats = Bench_args.int args 1 ~name:"repeats" ~default:20 in
  let m = Bench_args.int args 2 ~name:"m" ~default:n in
  let k = Bench_args.int args 3 ~name:"k" ~default:n in
  (* The naive 1x1-launch kernel is minutes per run at large sizes (a single GPU thread); cap its
     repeats separately so the scheduled variants can be timed at scale (arg 5, default = repeats; 0
     skips the naive leg entirely, including its warmup run — speedups then print as nan). *)
  let naive_repeats = Bench_args.int args 4 ~name:"naive_repeats" ~least:0 ~default:repeats in
  (* Blocking factors, named once and shared between each schedule below and the divisibility gate
     at the bottom, so the requirement and what is actually scheduled cannot drift apart. *)
  let par_b = 16 in
  let smem_bm, smem_bn, smem_bk = (16, 16, 8) in
  let reg_bm, reg_bn, reg_bk, reg_tm, reg_tn = (64, 64, 8, 8, 8) in
  let mma_bm, mma_bn, mma_bk, mma_w = (32, 32, 32, 32) in
  let cpu_bm, cpu_bn, cpu_bk = (64, 64, 16) in
  let gebp_bm, gebp_bk = (64, 64) in
  (* Every scheduled variant addresses the i/j/k nest, so an extent of 1 makes ALL of them
     unschedulable — [tensorize] and the unblocked ones included: the extent-1 loop is simplified
     away before the transform runs, leaving [accum_syms] no 3-deep nest to find. That is a
     precondition of the gate, not a requirement of any one variant's blocking, so it is folded into
     [unmet] rather than listed per variant. *)
  let degenerate =
    List.filter_map
      [ ("m", m); ("n", n); ("k", k) ]
      ~f:(fun (name, e) ->
        if e >= 2 then None
        else
          Some
            (Printf.sprintf
               "%s = %d leaves no i/j/k loop nest to schedule (an extent-1 loop is simplified away)"
               name e))
  in
  (* A [Sched.split] of factor [f] over an axis of extent [e] needs [e mod f = 0]: the remainder
     guard it constructs otherwise survives into the staged/tensorized micro-kernels these schedules
     build on top of it. Report which factor does not divide which extent, per variant, instead of
     tripping one bare assert for all of them. *)
  let unmet reqs =
    degenerate
    @ List.filter_map reqs ~f:(fun (fname, f, ename, e) ->
        if e % f = 0 then None
        else
          Some
            (Printf.sprintf "%s = %d does not divide %s = %d (remainder %d)" fname f ename e (e % f)))
    |> List.dedup_and_sort ~compare:String.compare
  in
  (* Operand values that vary with EVERY index at every extent, drawn through [Bench_checksum]'s
     (row, column) mix rather than a residue of the FLATTENED offset (gh-ocannl-711). The flat form
     is degenerate exactly where this bench runs: [(t mod 17) - 8] over the k x n mb gives every row
     of mb the identical values whenever 17 divides n (and [(t mod 13)] the same for ma whenever 13
     divides k), and a schedule that substitutes the wrong row of a collapsed operand then computes
     the correct output, which no whole-output check can see. Keying on the (row, column) pair
     removes the class: the row index enters the value in its own right, so no divisibility relation
     between a modulus and a stride can erase it. The value RANGE is unchanged — ma over (0, 3], mb
     the integers -8..8 — so the products stay exact in binary and the checksum's exactness argument
     is the one it was.

     ma's GRANULARITY is 1/16, 48 levels rather than the 12 of 1/4 it was minted at through
     gh-ocannl-711 (gh-ocannl-738). Keying on the (row, column) pair does not answer how many rows a
     generator can keep distinct over a k-wide row: that is bounded by [levels ^ k] whatever it
     encodes, so at the narrowest reduction this bench accepts, k = 2, 12 levels left only 144 rows
     to draw from and ma's rows first repeated at row 11. 48 levels take the bound to 2304 and the
     measured first repeat to 33; every k above 2 moves out by a comparable factor
     ([test/operations/bench_checksum_discrimination] prints the table). The finer levels stay exact
     in binary — a multiple of 1/16 below 3 needs six significant bits, which f32, tf32 (11) and f16
     (11) all hold — so no leg's reduction rounds, and the CUDA mma variants were re-run to confirm
     it rather than argued into it: see the [tf32_matmuls] note at the top of this file.

     ma drops the zero from its set ([Bench_checksum.positive_level] mints [1..levels] rather than
     [0..levels]), which the flat form did not need and the mixed one does: ma's row spans the
     reduction, so an all-zero ma row zeroes the whole output row, which against a zero-initialized
     destination is indistinguishable from a schedule that dropped that row. Under the flat form no
     ma row could be all-zero for k >= 2 (its values marched with the column); under the mix each
     entry is independent, so a row of k of them drawn from a zero-admitting residue is all-zero
     with probability (levels + 1)^-k — at 13 levels and k = 2 that is one row in 169, and m = 465
     has one. More levels only make that rarer, never impossible, which is why the fix is strict
     positivity and not granularity. mb keeps its zero: with ma strictly positive no output row is
     systematically zero, and mb's near-mean-zero set is what keeps the partial sums random-walking
     rather than growing with k. *)
  let mav =
    Array.init (m * k)
      ~f:(Bench_checksum.positive_level ~salt:0x5A17 ~row_stride:k ~levels:48 ~scale:0.0625)
  in
  let mbv =
    Array.init (k * n) ~f:(fun t ->
        Float.of_int (Bench_checksum.residue ~salt:0x3C6E ~row_stride:n ~modulus:17 t) -. 8.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ k ] ~output_dims:[ m ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ k ] () in
  let flops = 2.0 *. Float.of_int m *. Float.of_int n *. Float.of_int k in

  (* The i/j/k nest (i over m, j over n, k over k) is what every schedule below addresses. Extent-1
     loops are simplified away before the transform runs, so a degenerate size leaves no 3-deep nest
     — report that rather than raising [Not_found_s] out of a [find_exn]. *)
  let accum_syms opt =
    let paths = nest_paths opt.LL.llc in
    match List.find paths ~f:(fun p -> List.length p = 3) with
    | Some [ i; j; k ] -> (i, j, k)
    | _ ->
        failwith
          (Printf.sprintf
             "schedule_bench: no 3-deep i/j/k loop nest to schedule at m=%d n=%d k=%d (deepest \
              nest found: %d loops) — the scheduled variants need non-degenerate extents"
             m n k
             (List.fold paths ~init:0 ~f:(fun acc p -> Int.max acc (List.length p))))
  in

  (* One thread per output element. *)
  let parallel_schedule ~mc opt =
    let i, j, k = accum_syms opt in
    ignore k;
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:par_b ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:par_b ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, _ = Sched.split ~axis:i ~factor:par_b ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, _ = Sched.split ~axis:j ~factor:par_b ~outer:LL.Grid ~inner:LL.Workgroup in
    [ ez; sp_zi; sp_zj; sp_i; sp_j ]
  in

  (* + shared-memory operand tiles (32x32x8, 32x32 threads is over most limits: use 16x16x8). *)
  let smem_schedule ~mc opt =
    let bm, bn, bk = (smem_bm, smem_bn, smem_bk) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    [
      ez;
      sp_zi;
      sp_zj;
      sp_i;
      sp_j;
      sp_k;
      Sched.Stage
        {
          source = ma.Tensor.value;
          tile_loops = [ i_i; k_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.Stage
        {
          source = mb.Tensor.value;
          tile_loops = [ k_i; j_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.Privatize { target = mc; over = k_o };
    ]
  in

  (* CPU cache tiling + operand packing (all-Serial; the S4 shape, Boehm's packed CPU kernel). *)
  let cpupack_schedule ~mc opt =
    let bm, bn, bk = (cpu_bm, cpu_bn, cpu_bk) in
    let i, j, k = accum_syms opt in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    [ sp_i; sp_j; sp_k ]
    @ sink i_i [ j_o; j_i; k_o; k_i ]
    @ sink j_i [ k_o; k_i; i_i ]
    @ [
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Stage
          {
            source = mb.Tensor.value;
            tile_loops = [ k_i; j_i ];
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Privatize { target = mc; over = k_o };
      ]
  in

  (* + TM x TN register tiles via materialized unroll (64x64 block, 8x8 per thread). *)
  let regtile_schedule ~mc opt =
    let bm, bn, bk, tm, tn = (reg_bm, reg_bn, reg_bk, reg_tm, reg_tn) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    [ ez; sp_zi; sp_zi2; sp_zj; sp_zj2; sp_i; sp_i2; sp_j; sp_j2; sp_k ]
    @ sink i_t [ j_o; j_w; j_t; k_o; k_i ]
    @ sink j_t [ k_o; k_i ]
    @ [
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_w; i_t; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Stage
          {
            source = mb.Tensor.value;
            tile_loops = [ k_i; j_w; j_t ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Privatize { target = mc; over = k_o };
        Sched.Unroll { axis = i_t; materialize = true };
        Sched.Unroll { axis = j_t; materialize = true };
      ]
  in

  (* Staged + tensorized simdgroup mma (the autotune staged-seed shape, gh-ocannl-487): both operand
     tiles cooperatively staged at the serial [k_o] anchor and the micro-kernel tensorized, with the
     software-pipelining depth as the knob — [pipeline_depth = 1] is the strictly phased form, [2]
     the double-buffered one, bitwise identical by the transform's invariant, so the paired timing
     difference is the prefetch overlap's alone. *)
  let mma_staged_schedule ~pipeline_depth ~mc opt =
    let bm, bn, bk, w = (mma_bm, mma_bn, mma_bk, mma_w) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:w ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:k_i ~simd_width:w () in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = Some w;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth;
          tile_prec = None;
        }
    in
    [ ez; sp_zi; sp_zj; sp_i; sp_j; sp_k ]
    @ sink i_i [ j_o ] @ sink j_i [ k_o ] @ sink i_i [ k_o ]
    @ [ stage ma.Tensor.value [ i_i; k_i ]; stage mb.Tensor.value [ k_i; j_i ]; tz ]
  in

  (* Register-tiled Tile_mma micro-kernel (gh-ocannl-469): the whole i x j x k triple becomes one
     Tile_mma statement, which the C backends render tinyBLAS-style — the C-tile held in an RM x RN
     grid of vector registers across the entire k-loop, edges peeled. The zeroing must cover the
     lane slot (validate_parallel's coverage rule), so its column loop becomes the Workgroup axis
     and the lane width matches its extent (the lane loop renders serially on the C backends,
     executing the guarded statement once). *)
  let tensorize_schedule ~mc opt =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:n () in
    [ ez; rz; tz ]
  in

  (* Tile_mma composed with cache tiling + operand packing (the GEBP shape; the closing piece of
     gh-ocannl-469, autotune's [cpu_mma_pack_sketch_schedule]): pack the B panel [bk x n] at k_o
     (reused across all row blocks) and the A tile [bm x bk] at i_o, then tensorize the inner triple
     — the register-tiled micro-kernel streams the contiguous packed tiles. All-Serial with a unit
     lane, so the whole-node zeroing stays legal. *)
  let packmma_schedule ~mc:_ opt =
    let bm, bk = (gebp_bm, gebp_bk) in
    let i, j, k = accum_syms opt in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
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
          tile_prec = None;
        }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    [ sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ]; tz ]
  in

  (* The fully parallel packed GEMM (gh-ocannl-469 follow-up): the row-block loop is Grid-typed and
     pool-parallelizes — the per-row-block A~ tile is privatized to per-chunk block-scope storage by
     the renderer, the B~ panel packed at k_o is read-only inside the Grid body (behind a pointer
     alias under the blocks extension). The whole-node zeroing is no longer legal beside a
     hardware-annotated loop, so it expands with the same Grid row geometry. *)
  let packmma_par_schedule ~mc opt =
    let bm, bk = (gebp_bm, gebp_bk) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi = match zsyms with [ zi; _ ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
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
          tile_prec = None;
        }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ]; tz ]
  in

  (* Grid-outermost flavors (one dispatch spanning the whole GEBP triple; gh-ocannl-473 /
     gh-ocannl-475): the row-block Grid loop stays outermost — no [sink i_o [k_o]] — so anything
     packed at [k_o] lands inside the Grid body. [pack_b = `Hoist] packs the (constant) B~ panel at
     link time into the constant pool; [`Chunk] packs it in-kernel, each chunk re-packing its own
     panel (redundant work, but no dispatch-per-k-block; the panel must fit the per-chunk
     privatization cap). [pack_a] toggles the in-kernel per-chunk A~ pack vs. reading A in place. *)
  let packmma_outer_schedule ~pack_a ~pack_b ~mc opt =
    let bm, bk = (gebp_bm, gebp_bk) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi = match zsyms with [ zi; _ ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let stage ~hoisted source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = false;
          cooperative = None;
          hoisted;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    let stage_b =
      match pack_b with
      | `Hoist -> [ stage ~hoisted:true mb.Tensor.value [ k_i; j ] ]
      | `Chunk -> [ stage ~hoisted:false mb.Tensor.value [ k_i; j ] ]
    in
    let stage_a = if pack_a then [ stage ~hoisted:false ma.Tensor.value [ i_i; k_i ] ] else [] in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ stage_b @ stage_a @ [ tz ]
  in

  (* What every variant is compared against, cell by cell (gh-ocannl-711 review). That comparison,
     not the checksum, is what decides whether a variant computed the right thing: a checksum is a
     linear functional of the output, so a row permutation survives it whenever the value difference
     is orthogonal to the weight difference — by the weights colliding, or by plain cancellation,
     both of which are reachable at the narrow extents this bench accepts. An elementwise comparison
     has nothing to cancel. The checksum is still printed: one number per line fingerprints a run
     and travels into a report.

     The reference is the UNSCHEDULED computation, not "whichever variant completed first". Those
     coincide while the naive leg runs, and diverge exactly where it matters: under [naive_repeats =
     0] the naive leg is skipped, and taking the first scheduled variant instead would label an
     unvalidated output "reference" — with a single schedulable variant, comparing it against
     nothing at all, and with several, hiding any defect they share. So skipping the expensive naive
     TIMING costs the timing only: the oracle is materialized by one untimed run of the same
     unscheduled kernel, on demand and once. When the naive leg does run, its own output is that
     oracle and no extra run happens. *)
  let reference = ref None in
  let disagreements = ref 0 in
  let unscheduled_output () =
    (* Announced like any other work that can make the reader wait: this runs only when the naive
       leg was skipped, and a run that has just been told the leg is skipped should not then sit
       through an unexplained kernel. It is compiled WITHOUT a [lowered_transform], so unlike the
       naive leg it is not the 1x1 launch — the backend's own default schedule applies, which on the
       GPU backends is the difference between milliseconds and the naive leg's seconds — but that is
       a fact about this backend, not a promise, so the elapsed time is reported whenever it crosses
       the same floor a slow warmup does. *)
    p "%-10s materializing the comparison oracle — one untimed run of the unscheduled computation\n"
      "reference";
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let%op mc = ma * mb in
    let comp = named "mm_reference" (Train.forward mc) in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    let values = Context.get_values ctx mc.Tensor.value in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 in
    if Float.(secs >= 0.5) then p "%-10s %.1f s\n" "reference" secs;
    values
  in
  let reference_output () =
    match !reference with
    | Some r -> r
    | None ->
        let r = unscheduled_output () in
        reference := Some r;
        r
  in
  let bench ?(repeats = repeats) ~variant ~schedule () =
    let%op mc = ma * mb in
    let comp = named ("mm_" ^ variant) (Train.forward mc) in
    let transform opt =
      match schedule with None -> opt | Some s -> Sched.apply (s ~mc:mc.Tensor.value opt) opt
    in
    let ctx = Context.auto () in
    (* What codegen actually did with each [Tile_mma] (gh-ocannl-479). A [Tile_mma] whose
       preconditions fail renders the scalar fallback and still reports under its schedule's name —
       "timed is not tensorized", docs/agent-notes/scheduling-and-autotune.md — so a variant name is
       not evidence of tensorization and every variant carrying one has to be read from the census
       instead. The decline rules include a column extent below the compute vector width (which
       arbitrary extents now reach), a narrow [vector_bytes], mixed operand precisions, an
       accumulation not in FMA form, and [debug_log_from_routines]. Collecting the census only
       appends to a list, so it perturbs neither what is compiled nor what is timed. Since
       gh-ocannl-626 it travels on the compiled routine, so this bench cannot forget to ask and
       cannot disagree with [narrow_gebp_bench] about what "tensorized" means. *)
    let ctx, routine =
      Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
    in
    let mma = routine.Context.mma in
    (* The unscheduled leg is the 1x1 launch: one GPU thread walking the whole m x n x k nest, which
       is seconds per run by n = 512 and minutes beyond it. Say so BEFORE the first kernel rather
       than after the last, and say how to cap it — a reader who does not know that the fifth
       positional exists cannot tell a long naive leg from a hang. Keyed on [schedule = None], which
       is what makes the leg unscheduled, so it cannot drift from which leg is the slow one. *)
    if Option.is_none schedule then
      p
        "%-10s 1x1 launch over m*n*k = %d — one thread, 1 warmup + %d timed run(s); cap with the \
         5th argument (0 skips the timing)\n"
        variant
        (m * n * k)
        repeats;
    (* Warmup (includes any lazy initialization and host transfers). Timed, so that a leg whose cost
       only shows up at run time can still announce it before the timed loop multiplies it. *)
    let warm_start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx = Context.run ctx routine in
    let _ = Context.get_values ctx mc.Tensor.value in
    let warm_stop = Time_now.nanoseconds_since_unix_epoch () in
    (* A second read with nothing queued behind it has no kernel to wait for, so it times the
       device-to-host transfer alone. The timed loop below queues [repeats] runs and reads back
       ONCE, so multiplying the whole warmup would multiply a cost that is paid once — on a fast
       kernel with a large output that is the whole estimate, and the announcement would be wrong by
       a factor of [repeats]. Outside the timed region, and one extra transfer of an output the
       backend has already computed. *)
    let _ = Context.get_values ctx mc.Tensor.value in
    let read_stop = Time_now.nanoseconds_since_unix_epoch () in
    let warmup = Float.of_int63 Int63.(warm_stop - warm_start) /. 1e9 in
    let readback = Float.of_int63 Int63.(read_stop - warm_stop) /. 1e9 in
    (* Half a second of warmup already means [repeats] half-seconds of silence ahead, which is long
       enough to want an estimate for; the naive leg's warmup at m = n = k = 256 lands just under a
       second, so a one-second floor would have stayed quiet on the very shape this reports. A fast
       variant stays a single line, so the table is unchanged wherever there was nothing to wait
       for. The per-run figure still carries whatever lazy initialization the first run paid, which
       the timed loop does not repeat either, so the estimate is an upper bound — [Float.max 0.]
       because a readback that jitters above its own warmup would otherwise flip its sign. *)
    if Float.(warmup >= 0.5) then
      p "%-10s warmup %.1f s (readback %.1f s) — timing %d run(s), about %.0f s\n" variant warmup
        readback repeats
        ((Float.of_int repeats *. Float.max 0. (warmup -. readback)) +. readback);
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx (Stdlib.Array.make repeats ())
    in
    let values = Context.get_values ctx mc.Tensor.value in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    (* Element [1][1] of the m x n result — an interior cell, away from the corners — except where
       the output is too small to have one; print which cell was checked. One interior cell cannot
       see the remainder region an arbitrary extent creates, and a [Sched.split] whose factor does
       not divide its extent puts the last partial block exactly there, so the whole output is
       checksummed too: every correct variant prints the identical value, and one that drops or
       repeats tail work does not.

       Position-weighted through [Bench_checksum.whole_output], for two reasons. A plain sum of THIS
       data is 0 whenever 17 divides n (each mb row spans a full cycle of its 17 values, and the
       total factors as sum_k (sum_i a) (sum_j b)) — exactly the arbitrary-extent regime the
       checksum is for. And a plain sum reads only the multiset, so it cannot see a PERMUTATION,
       which is what a misplaced row-edge peel produces. The weight is keyed on the (row, column)
       pair rather than on the flat offset t = i*n + j, because [1 + (t mod 251)] collapses to [1 +
       j] whenever 251 divides n — every row then carries the identical weight vector, a row
       permutation is invisible, and the spot cell at [1][1] is blind to other rows at the same
       time, so both halves of the check fail together at n = 251, 502, 753, … (gh-ocannl-711).
       Weights stay capped at 251 so that products of these exact-in-binary operands stay exact in
       the double accumulator, and the printed [chk a/b] is one sum per weight stream: at a narrow n
       a single capped stream runs out of distinct row weight vectors and two rows collide, whose
       swap no weighting of that stream can see. Both checks are outside the timed region. *)
    let checksum = Bench_checksum.whole_output ~row_stride:n values in
    (* The unscheduled leg, when it runs, IS the oracle — same kernel, so re-running it would only
       burn time. Any other variant is compared against it. *)
    if Option.is_none schedule && Option.is_none !reference then reference := Some values;
    let agreement =
      let r = reference_output () in
      if phys_equal r values then "reference"
      else begin
        let d = Bench_checksum.first_difference ~reference:r values in
        if Option.is_some d then Int.incr disagreements;
        Bench_checksum.render_agreement ~name:"reference" d
      end
    in
    let spot = Int.min (n + 1) (Array.length values - 1) in
    (* The label is printed on EVERY timing line, including the untensorized variants: a suffix that
       appears only when there is something to say is a suffix a table reader does not miss when it
       is absent (gh-ocannl-626). *)
    p "%-10s %8.3f ms  %8.2f GFLOP/s  (spot [%d] %.1f, chk %s, %s)  [%s]\n" variant (secs *. 1e3)
      (flops /. secs /. 1e9)
      spot values.(spot) (Bench_checksum.render checksum) agreement
      (Ir.C_syntax.mma_summary_string mma);
    (secs, mma)
  in
  (* A static gate cannot enumerate every way a schedule can be rejected at a given size —
     [validate_parallel]'s coverage rules, per-kernel hardware limits and backend declines all
     surface as exceptions out of [Context.compile], and small operands bring in whole new
     interactions (below m,n,k = 5 the constants are initialized in-kernel, which the [tensorize]
     variant's Workgroup retype of the zeroing loop then makes illegal). Charge such a failure to
     the variant, not to the run: name it, keep measuring the rest, and exit nonzero at the end so a
     scripted run still sees that something failed. *)
  let failures = ref 0 in
  (* Every rendering any variant produced, for the closing verdict on whether the tensorized labels
     measured tensorized kernels. *)
  let census = ref [] in
  let attempt ?repeats ~variant ~schedule () =
    try
      let secs, mma = bench ?repeats ~variant ~schedule () in
      census := mma :: !census;
      secs
    with e ->
      Int.incr failures;
      p "%-10s FAILED at this size: %s\n" variant
        (List.hd_exn (String.split_lines (Exn.to_string e)));
      Float.nan
  in
  (* A variant runs when the gate is satisfied, and is skipped by name with every reason it found
     when it is not, so an unschedulable size still measures everything that is schedulable at it. A
     skipped variant times as nan, like the naive leg under [naive_repeats = 0]. The naive leg needs
     no gate — it addresses no loop nest at all, so it runs at any size, degenerate included. *)
  let bench_v ?repeats ~variant ~reqs ~schedule () =
    match unmet reqs with
    | [] -> attempt ?repeats ~variant ~schedule ()
    | reasons ->
        p "%-10s skipped — this size is not schedulable with this blocking:\n" variant;
        List.iter reasons ~f:(fun reason -> p "             %s\n" reason);
        Float.nan
  in
  p "matmul m=%d n=%d k=%d, %d repeats (naive: %d), backend from config/OCANNL_BACKEND\n" m n k
    repeats naive_repeats;
  let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  (* HIP belongs here too: the backend renders workgroup-shared placement exactly as CUDA and Metal
     do, so the shared/staged/tensorized variants — including the gh-ocannl-487 [mma_pd1]/[mma_pd2]
     pipelining pair — are expressible on it. Omitting it silently routed every HIP run into the CPU
     branch below, which is why the pipelining pair had never been timed on an AMD GPU. *)
  let has_shared =
    String.is_substring backend ~substring:"metal"
    || String.is_substring backend ~substring:"cuda"
    || String.is_substring backend ~substring:"hip"
  in
  let t_naive =
    if naive_repeats > 0 then attempt ~repeats:naive_repeats ~variant:"naive" ~schedule:None ()
    else Float.nan
  in
  (if has_shared then
     let mma_reqs =
       [
         ("bm", mma_bm, "m", m); ("bn", mma_bn, "n", n); ("w", mma_w, "n", n); ("bk", mma_bk, "k", k);
       ]
     in
     (* [parallel] asks for nothing beyond a nest to address: it only splits the zeroing and
        accumulation loops, with no swap, staging, privatization or tensorization downstream of the
        split, so [Sched.split]'s remainder guard is all the partial block needs. Measured at 17^3,
        33^3, 100^3 and 100x17x33 on metal — the checksum matches the naive leg's at each, tail
        included. That does NOT generalize to the others: with their reqs disabled, smem, regtile
        and both mma_pd variants raise [Invalid_argument] at 100^3, as do all six C-backend blocked
        variants, so those gates report a real requirement rather than an assumed one. *)
     let t_par = bench_v ~variant:"parallel" ~reqs:[] ~schedule:(Some parallel_schedule) () in
     let t_smem =
       bench_v ~variant:"smem"
         ~reqs:[ ("bm", smem_bm, "m", m); ("bn", smem_bn, "n", n); ("bk", smem_bk, "k", k) ]
         ~schedule:(Some smem_schedule) ()
     in
     let t_reg =
       bench_v ~variant:"regtile"
         ~reqs:[ ("bm", reg_bm, "m", m); ("bn", reg_bn, "n", n); ("bk", reg_bk, "k", k) ]
         ~schedule:(Some regtile_schedule) ()
     in
     let t_mma1 =
       bench_v ~variant:"mma_pd1" ~reqs:mma_reqs
         ~schedule:(Some (mma_staged_schedule ~pipeline_depth:1))
         ()
     in
     let t_mma2 =
       bench_v ~variant:"mma_pd2" ~reqs:mma_reqs
         ~schedule:(Some (mma_staged_schedule ~pipeline_depth:2))
         ()
     in
     p
       "speedups vs naive: parallel %.1fx, smem %.1fx, regtile %.1fx, mma_pd1 %.1fx, mma_pd2 %.1fx\n"
       (t_naive /. t_par) (t_naive /. t_smem) (t_naive /. t_reg) (t_naive /. t_mma1)
       (t_naive /. t_mma2)
   else
     (* The GEBP family blocks i and k only — j is left whole — so it asks nothing of n. The
        [tensorize] variant has no blocking at all: the whole triple becomes one [Tile_mma], so any
        non-degenerate extent still gets a scheduled measurement. *)
     let gebp_reqs = [ ("bm", gebp_bm, "m", m); ("bk", gebp_bk, "k", k) ] in
     let t_pack =
       bench_v ~variant:"cpupack"
         ~reqs:[ ("bm", cpu_bm, "m", m); ("bn", cpu_bn, "n", n); ("bk", cpu_bk, "k", k) ]
         ~schedule:(Some cpupack_schedule) ()
     in
     let t_tmma = bench_v ~variant:"tensorize" ~reqs:[] ~schedule:(Some tensorize_schedule) () in
     let t_pmma = bench_v ~variant:"packmma" ~reqs:gebp_reqs ~schedule:(Some packmma_schedule) () in
     let t_pmmap =
       bench_v ~variant:"packmma_par" ~reqs:gebp_reqs ~schedule:(Some packmma_par_schedule) ()
     in
     let t_hoist =
       bench_v ~variant:"pm_hoist" ~reqs:gebp_reqs
         ~schedule:(Some (packmma_outer_schedule ~pack_a:false ~pack_b:`Hoist))
         ()
     in
     let t_mixed =
       bench_v ~variant:"pm_mixed" ~reqs:gebp_reqs
         ~schedule:(Some (packmma_outer_schedule ~pack_a:true ~pack_b:`Hoist))
         ()
     in
     let t_bpk =
       bench_v ~variant:"pm_bpk" ~reqs:gebp_reqs
         ~schedule:(Some (packmma_outer_schedule ~pack_a:true ~pack_b:`Chunk))
         ()
     in
     p
       "speedups vs naive: cpupack %.1fx, tensorize %.1fx, packmma %.1fx, packmma_par %.1fx, \
        pm_hoist %.1fx, pm_mixed %.1fx, pm_bpk %.1fx\n"
       (t_naive /. t_pack) (t_naive /. t_tmma) (t_naive /. t_pmma) (t_naive /. t_pmmap)
       (t_naive /. t_hoist) (t_naive /. t_mixed) (t_naive /. t_bpk));
  (* The closing verdict the variant names cannot give: a [Tile_mma] that declined rendered the
     lane-0 scalar loop, so the line above timed a scalar kernel under a tensorized label. Reported
     rather than rejected — the census is honest for every decline rule at once, whereas a minimum
     extent would re-introduce exactly the kind of arbitrary size constraint this change removes. *)
  let all = Ir.C_syntax.merge_mma_summaries !census in
  let declined = all.Ir.C_syntax.scalar_fallbacks in
  if declined > 0 then
    p
      "WARNING: %d of %d Tile_mma statements rendered the scalar fallback — the tensorized lines \
       above are NOT tensorized timings. Re-run with --ocannl_schedule_log_declines=true for the \
       per-rule reason (at these extents, most likely n below the compute vector width).\n"
      declined all.Ir.C_syntax.statements;
  (* A variant that computed something ELSE is the failure this bench's guard exists for, and it is
     worse than one that failed to compile: a compile failure is loud, whereas a wrong result under
     a fast timing is exactly what a report carries forward. So it EXITS NONZERO, after every
     variant has been reported — a guard that only prints leaves an automated run free to keep the
     speedup of a kernel already known to be wrong, which is the same hazard `Verdict` exists for on
     the test side. There is no rounding to excuse it: ma and mb are exact in binary with at most
     six and four significant bits respectively, so every product is exact (in tf32 and f16 as well
     as f32 — both carry an 11-bit significand) and every leg's reduction is exact whatever order it
     sums in, at any extent this bench runs. *)
  if !disagreements > 0 then
    p
      "WRONG RESULT: %d variant(s) did not reproduce the reference output cell for cell — the \
       DIFFERS lines above name the first cell and both values. At these operands every variant's \
       reduction is exact whatever order it sums in, so this is not rounding.\n"
      !disagreements;
  if !failures > 0 then
    p "%d variant(s) failed at m=%d n=%d k=%d — see the FAILED lines above.\n" !failures m n k;
  if !failures > 0 || !disagreements > 0 then Stdlib.exit 1
