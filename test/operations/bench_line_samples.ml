(* gh-ocannl-1767: lines the benchmark harness's own emitters print, for [bench_line_readers.py] to
   run benchmarks/bench_output.py's readers over. Nothing here formats a line: each mode drives the
   code that prints it, so a change to an emitter reaches the readers' check the day it lands.

   - [harness]: the dominant-kernel instrument ([Bench_harness.dominant_kernel]) over four small
   compiled routines, with its per-kernel table on ([BENCH_KERNEL_TABLE=1], from the rule) on stderr
   -- one kernel through the instrument's own [time_kernel] seam as declined, one as refused, so the
   table holds every kind of row -- and the launch-geometry trace of their compiles
   ([schedule_log_launches], from the command line) beside it. A CPU backend ships a routine as one
   kernel, hence four routines. Its [dominant_kernel] record goes to stdout. The run states
   [autotune_timing=isolated] itself, as a kprof cell pins it, and restates it [queued] once the
   last kernel's timing calls return: the record must name the mode those calls were made under, not
   the configuration it is written under (gh-ocannl-1860). - [trace]: the timing trace
   ([BENCH_TIMING_TRACE=1], from the rule) of a seed-only search of a small matmul, then the
   end-of-search marker; the summary prints at exit. - [trace-raise]: the same search with one
   failure injected (gh-ocannl-2246): the first candidate after the baseline to start its timing
   call's validation raises [Device_memory_exhausted] from its first batch
   ([Autotune.on_batch_wall], inside the tagged call and past its validation) -- the candidate's
   classified decline, which the search survives (an expired wait would abandon the device,
   gh-ocannl-2133). The run prints the injected candidate's label and exception on stdout,
   [%S]-quoted as the trace quotes them, for the reader to find in the trace's [raised] line. -
   [trace-cut]: the same search on a scripted clock ([Autotune.on_batch_wall], as
   [autotune_hopeless_window] drives it, gh-ocannl-2330): the baseline and the floors read 12 ms,
   and every other attempt 100 ms -- over four times the floor -- until the first window is cut as
   hopeless, every attempt after it the floor again, so exactly one window is cut. The run prints
   the cut candidate's label on stdout, [%S]-quoted, read off the exception the raised-call observer
   is handed. - [trace-stop]: the same search under a budget whose virtual clock
   ([Autotune.budget_elapsed]) passes it during the first non-floor candidate's timing, so exactly
   that window is stopped ([Autotune.Timing_stopped]); the run prints its label as [stopped: %S]. -
   [seeded]: the recipes' untuned compile ([Autotune.seeded_default], what
   [Autotune.untuned_compile] is with the model pick off) of a 64x64 matmul on the run's own
   backend, under [autotune_log] (from the rule): where the backend seeds its untuned default
   ([Autotune.seeded_default_backend]: CUDA and HIP) the compile goes through
   [Autotune.seeded_apply], which prints the [seeded_default:] lines on stderr; its one leg claim
   checks that the compile shipped a seeded segment without falling back, and that the routine's
   product is the host's, bitwise. Elsewhere the leg is skipped, and [bench_line_readers.py] reads
   its [SKIPPED] line as the format's not being produced there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module H = Bench_harness

let harness () =
  Utils.state_config ~key:"autotune_timing" ~data:"isolated";
  let ctx = ref (Context.auto ()) in
  let routines =
    List.init 4 ~f:(fun k ->
        let a =
          NTDSL.range_of_shape ~label:[ Printf.sprintf "bls_a%d" k ] ~output_dims:[ 8; 8 ] ()
        in
        let%op c = a +* "ij; jk => ik" a in
        Train.set_materialized c.Tensor.value;
        let ctx', routine =
          Context.compile ~name:(Printf.sprintf "bls_k%d" k) !ctx (Train.forward c)
            Ir.Indexing.Empty
        in
        Train.run ctx' routine;
        ctx := ctx';
        routine)
  in
  let time_kernel ~i timing =
    match i with
    | 0 -> Error (H.Declined "injected by bench_line_samples:\na detail over two lines")
    | 1 ->
        Error
          (H.Untimed
             {
               Bench_retime.ms = Float.nan;
               lo_ms = Float.nan;
               hi_ms = Float.nan;
               admitted = 0;
               refused = 4;
             })
    | i ->
        let timed = timing () in
        if i = List.length routines - 1 then
          Utils.state_config ~key:"autotune_timing" ~data:"queued";
        timed
  in
  (* The record is the result line's [dominant_kernel] object, printed on stdout for the reader of
     its timing mode (gh-ocannl-1860); the rows are read off stderr. *)
  Stdio.print_endline
    (H.dominant_kernel ~calls:4 ~time_kernel ~ctx:!ctx ~bindings:Train.IDX.empty routines)

(* Installed after the trace, so the trace's own seams run first and count what this injects. *)
let inject_one_raise () =
  let current = ref "" and armed = ref false and injected = ref false in
  let prev_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  Autotune.Seam.set Autotune.on_candidate_attempt (fun label ->
      prev_attempt label;
      current := label);
  let prev_preflight = Autotune.Seam.get Autotune.on_candidate_preflight in
  Autotune.Seam.set Autotune.on_candidate_preflight (fun name ->
      prev_preflight name;
      armed := (not !injected) && not (String.equal !current "baseline"));
  let prev_wall = Autotune.Seam.get Autotune.on_batch_wall in
  Autotune.Seam.set Autotune.on_batch_wall (fun ~depth ~ordinal wall ->
      let wall = prev_wall ~depth ~ordinal wall in
      if not !armed then wall
      else begin
        armed := false;
        injected := true;
        let exn =
          Ir.Schedule_outcome.Device_memory_exhausted
            {
              backend = "cc";
              requested = 4096;
              free = None;
              detail = Printf.sprintf "injected by bench_line_samples at depth %d" depth;
            }
        in
        Stdio.printf "injected: %S %S\n%!" !current (Stdlib.Printexc.to_string exn);
        raise exn
      end)

(* Installed after the trace, like [inject_one_raise]. Which attempts are floors is
   [autotune_hopeless_window]'s rule: the baseline, the backend's default seed and the seeded
   singles, whose readings the others are ranked against. *)
let is_floor label =
  String.equal label "baseline"
  || Option.exists
       (Autotune.default_seed_label ~facts:Ir.Backend_intf.cc_facts)
       ~f:(String.equal label)
  || String.is_prefix label ~prefix:"F_seeded"

let cut_one_window () =
  let floor_ms = 12. and hopeless_ms = 100. in
  let current = ref "" and cut = ref false in
  let prev_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  Autotune.Seam.set Autotune.on_candidate_attempt (fun label ->
      prev_attempt label;
      current := label);
  Autotune.install_timing_call_raised_observer (fun (r : Autotune.timing_call_raised) ->
      match r.raised_exn with
      | Autotune.Timing_hopeless _ ->
          cut := true;
          Stdio.printf "cut: %S\n%!" r.raised_label
      | _ -> ());
  let prev_wall = Autotune.Seam.get Autotune.on_batch_wall in
  Autotune.Seam.set Autotune.on_batch_wall (fun ~depth ~ordinal wall ->
      ignore (prev_wall ~depth ~ordinal wall : float);
      let ms = if !cut || is_floor !current then floor_ms else hopeless_ms in
      ms *. Float.of_int depth)

(* Installed after the trace, like [cut_one_window]. The budget reads a virtual clock
   ([Autotune.budget_elapsed]): zero, until the first batch of a non-floor candidate's timing, and
   then far past the budget, so the check before its next batch stops the window; zero again once
   the window has stopped, which leaves the budget spent. *)
let stop_one_window () =
  let current = ref "" and armed = ref false and stopped = ref false in
  let prev_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  Autotune.Seam.set Autotune.on_candidate_attempt (fun label ->
      prev_attempt label;
      current := label);
  Autotune.install_timing_call_raised_observer (fun (r : Autotune.timing_call_raised) ->
      match r.raised_exn with
      | Autotune.Timing_stopped ->
          armed := false;
          stopped := true;
          Stdio.printf "stopped: %S\n%!" r.raised_label
      | _ -> ());
  let prev_wall = Autotune.Seam.get Autotune.on_batch_wall in
  Autotune.Seam.set Autotune.on_batch_wall (fun ~depth ~ordinal wall ->
      if not (!stopped || is_floor !current) then armed := true;
      prev_wall ~depth ~ordinal wall);
  Autotune.Seam.set Autotune.budget_elapsed (fun _real -> if !armed then 1e9 else 0.)

let trace ?(cut = false) ?(stop = false) ~inject () =
  H.install_timing_trace ();
  if inject then inject_one_raise ();
  if cut then cut_one_window ();
  if stop then stop_one_window ();
  let a = NTDSL.range_of_shape ~label:[ "bls_a" ] ~output_dims:[ 16; 16 ] () in
  let b = NTDSL.range_of_shape ~label:[ "bls_b" ] ~output_dims:[ 16; 16 ] () in
  let%op c = a +* "ik; kj => ij" b in
  Train.set_materialized c.Tensor.value;
  let ctx = Context.auto () in
  let t0 = Unix.gettimeofday () in
  let _ctx, _routine =
    Autotune.tune ~search:true ~rounds:0 ~cache_dir:"" ~name:"bls_matmul" ctx (Train.forward c)
      Ir.Indexing.Empty
  in
  H.trace_search_done ~compile_s:(Unix.gettimeofday () -. t0);
  if (not inject) && (not cut) && not stop then (
    (* Pin Autotune's settle and admission vocabularies for bench_line_readers.py so
       gh1253_gate_compare.py's closed allowlist stays tied to the producer (gh-ocannl-2376). *)
    let settle_name (s : Autotune.batch_settle) =
      match s with
      | Isolated_objective | Single_estimate | Singles_reach_target | Singles_far_over_target
      | Prior_depth | Measured_batch | Affine_projection | Wall_scaled | Cap_measured
      | Cap_unresolved | Budget_exhausted ->
          Autotune.batch_settle_name s
    in
    let admission_name (a : Autotune.batch_admission) =
      match a with
      | Admitted | Refused_unbatched | Refused_contended | Refused_unreadable ->
          Autotune.batch_admission_name a
    in
    let settles =
      List.map
        [
          Autotune.Isolated_objective;
          Single_estimate;
          Singles_reach_target;
          Singles_far_over_target;
          Prior_depth;
          Measured_batch;
          Affine_projection;
          Wall_scaled;
          Cap_measured;
          Cap_unresolved;
          Budget_exhausted;
        ]
        ~f:settle_name
    in
    let admissions =
      List.map
        [ Autotune.Admitted; Refused_unbatched; Refused_contended; Refused_unreadable ]
        ~f:admission_name
    in
    Stdio.printf "timing-vocab: settles=%s\n" (String.concat ~sep:"," settles);
    Stdio.printf "timing-vocab: admissions=%s\n%!" (String.concat ~sep:"," admissions))

let seeded () =
  let ctx = Context.auto () in
  let leg =
    Verdict.leg
      ~when_:(Autotune.seeded_default_backend (Context.backend_facts ctx))
      ~on:(Context.backend_name ctx) ()
  in
  Verdict.in_leg ~leg
    "seeded: the untuned matmul ran seeded, with no fallback, and computes the host's product"
    (fun label ->
      let n = 64 in
      (* Small integers, so every product and sum is exact in f32 (and tf32): the comparison is
         bitwise, and no two adjacent rows or columns of the product coincide. *)
      let cyc ~l ~modulus ~offset =
        NTDSL.init ~l ~prec:Ir.Ops.single ~o:[ n; n ]
          ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus ~offset ~stride:1.)
          ()
      in
      let a = cyc ~l:"bls_sa" ~modulus:5 ~offset:(-2.) in
      let b = cyc ~l:"bls_sb" ~modulus:7 ~offset:(-3.) in
      let%op c = a +* "ik; kj => ij" b in
      List.iter [ a; b; c ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
      (* [Autotune.untuned_compile] is [seeded_default] under [model_default_schedule=false], which
         the rule states; its report says what the compile shipped. *)
      let choice = ref None in
      let ctx, routine =
        Autotune.seeded_default ~name:"bls_seeded"
          ~report:(fun c -> choice := Some c)
          ctx (Train.forward c) Ir.Indexing.Empty
      in
      Train.run ctx routine;
      let av = Context.get_values ctx a.Tensor.value
      and bv = Context.get_values ctx b.Tensor.value
      and cv = Context.get_values ctx c.Tensor.value in
      let want =
        Array.init (n * n) ~f:(fun ij ->
            List.sum
              (module Float)
              (List.range 0 n)
              ~f:(fun k -> av.((ij / n * n) + k) *. bv.((k * n) + (ij % n))))
      in
      let shipped_seeded =
        match !choice with
        | Some { Autotune.sd_seeded = _ :: _; sd_fell_back = false; _ } -> true
        | Some _ | None ->
            Stdio.eprintf "seeded: the compile shipped no seeded segment, or fell back\n%!";
            false
      in
      Verdict.Claims.p label (shipped_seeded && Verdict.within label cv want ~tol:(fun _ -> 0.)))

let () =
  (* The mode comes first; the [--ocannl_*] settings after it are the configuration's. *)
  match Array.to_list (Sys.get_argv ()) with
  | _ :: "harness" :: _ -> harness ()
  | _ :: "trace" :: _ -> trace ~inject:false ()
  | _ :: "trace-raise" :: _ -> trace ~inject:true ()
  | _ :: "trace-cut" :: _ -> trace ~cut:true ~inject:false ()
  | _ :: "trace-stop" :: _ -> trace ~stop:true ~inject:false ()
  | _ :: "seeded" :: _ -> seeded ()
  | _ ->
      failwith
        "usage: bench_line_samples.exe harness|trace|trace-raise|trace-cut|trace-stop|seeded \
         [--ocannl_<key>=<value> ...]"
