(* gh-ocannl-1702: an autotune report carries the renderer's Grid record per candidate.

   Every compiled routine records, for each outermost [Grid] loop, whether it rendered on the CPU
   pool or which rule kept it serial ([Context.routine.grid], gh-ocannl-1668). The serial loop
   computes the same values, so a Grid-seeded candidate the renderer kept serial is timed and ranked
   under a Grid label -- the failure [best_tensorization] closed for tensorization. The report now
   carries the crowned candidate's record ([best_grid]) and every ranked Grid-asking candidate that
   rendered a Grid loop serially ([grid_serial_timed]), and the [autotune_log] flags each one.

   The cap is forced below a Grid seed's footprint: [cc_grid_private_bytes_cap] is set to 64 bytes
   before anything resolves it (it is read once per process, so it is stated before the first
   compile, and the resolved value is pinned below). The hoisted-only Grid shape of the packed
   matmul sketch has no tile the seeding pre-filter weighs against the cap -- both operands are
   host-init constants, packed at link time -- yet its row-block loop contracts a [bm x bn]
   accumulator fragment per chunk, which the renderer privatizes and weighs. So the seed is
   proposed, compiled and timed, and the renderer keeps its row-block loop serial for the cap: the
   report must say so, with the bytes and the cap. The cc vector file is pinned
   ([Fissioned_chain.small_menu_envelope]), so the menu does not follow the host's width.

   Ranked is the timing policy's verdict, not the seed's, and on a loaded host the policy drops
   these seeds: a window under a co-tenant is refused as host contention, and a whole-routine seed
   whose first samples read four times the beam head's is cut as hopeless -- on a 4 us routine, a
   scheduler hiccup. The 2026-11-01 sweep, at the full suite's width on minix/multidev_cc, ranked
   one Grid seed of the eight and none over the cap. So the windows are opted out of host contention
   ([with_uncontended_test_windows], gh-ocannl-1156) and every admitted reading passes through
   [on_candidate_measured] unchanged (gh-ocannl-1027): the readings, and so the ranking and the
   crown, stay the clock's, but a scripted reading turns the hopeless cut off (gh-ocannl-1527) --
   only what the policy keeps out of the ranking changes, and every seed the menu proposes is
   ranked.

   The record is relational, not restated: each listed candidate's own census is serial on a pool
   backend, the crowned candidate's record is the one its routine carries, and a crowned Grid-asking
   schedule rendered serially is among the listed. The tuned routine also computes the reference
   product bitwise, on a premise the last item derives, so the serial rendering is the same values.

   On a backend with no CPU pool a [Grid] loop is a hardware grid and nothing is recorded, so the
   whole leg is skipped there by name. *)

(* cc-fp-contract: exact-operands -- Ll_test.exact_partials derives the operands' sums exact at
   f32 *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Cs = Ir.C_syntax
open Verdict.Claims

let forced_cap = 64

(* Before anything resolves the cap: [Cs.per_chunk_private_bytes_cap] is a process-lifetime lazy. *)
let () =
  Hashtbl.set Utils.config_file_args ~key:"cc_grid_private_bytes_cap"
    ~data:(Int.to_string forced_cap)

(* The cc vector file the sketch menu follows, pinned for the process (the width auto-probes from
   the host, and the register tile's geometry follows it): the claims are about a hoisted-grid
   seed's record, which needs the menu that proposes one. Also before the first search. *)
let () = Fissioned_chain.small_menu_envelope ()
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)

(* The census exists only where the pool does: gated on the backend's own predicate, never on the
   report under test. *)
let leg =
  Verdict.leg ~when_:(on_cpu && Context.Cc_backend.pool_parallel_grid ()) ~on:backend_name ()

let n = 64

(* The operands' init functions: they vary with every index pair, and their products are multiples
   of 1/4 small enough for the derivation below. *)
let fa = Ll_test.cycle ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25
let fb = Ll_test.cycle ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.

let () =
  p "the renderer resolved the forced per-chunk privatization cap"
    (Lazy.force Cs.per_chunk_private_bytes_cap = forced_cap)

let tuned =
  lazy
    (let av = Array.init (n * n) ~f:(fun t -> fa [| t / n; t % n |]) in
     let bv = Array.init (n * n) ~f:(fun t -> fb [| t / n; t % n |]) in
     let expected =
       Array.init (n * n) ~f:(fun idx ->
           let i = idx / n and j = idx % n in
           let acc = ref 0. in
           for k = 0 to n - 1 do
             acc := !acc +. (av.((i * n) + k) *. bv.((k * n) + j))
           done;
           !acc)
     in
     let a = TDSL.ndarray av ~label:[ "agr_a" ] ~output_dims:[ n; n ] () in
     let b = TDSL.ndarray bv ~label:[ "agr_b" ] ~output_dims:[ n; n ] () in
     let%op c = a +* "ik;kj=>ij" b in
     let comp = Train.forward c in
     let reports = ref [] in
     let ctx, routine =
       (* The clock's own reading, through the seam: see the header. *)
       Autotune.with_uncontended_test_windows (fun () ->
           Autotune.Seam.with_value Autotune.on_candidate_measured
             (fun ~label:_ ~digest:_ ms -> ms)
             (fun () ->
               Autotune.tune ~name:"agr_matmul" ~search:true ~beam_width:1 ~rounds:0 ~repeats:1
                 ~cache_dir:"" ~timing_ctx:(Context.auto ())
                 ~report:(fun r -> reports := r :: !reports)
                 (Context.auto ()) comp Ir.Indexing.Empty))
     in
     let ctx = Context.run ctx routine in
     let got = Context.get_values ctx c.Tensor.value in
     let r =
       match !reports with [ r ] -> r | _ -> failwith "expected exactly one tuning report"
     in
     (* For a reader of a failed run. *)
     List.iter r.Autotune.grid_serial_timed ~f:(fun g ->
         Stdio.eprintf "agr: %s: %s (not part of the golden)\n%!" g.Autotune.grid_label
           (Cs.grid_summary_string g.Autotune.grid_record));
     Stdio.eprintf "agr: crowned %s: %s (not part of the golden)\n%!" r.Autotune.best_label
       (Option.value_map r.Autotune.best_grid ~default:"no record" ~f:Cs.grid_summary_string);
     (r, routine, got, expected))

let over_cap (g : Cs.grid_summary) =
  List.exists g.loops ~f:(fun e ->
      match e.Cs.site with
      | Cs.Grid_serial (Grid_private_over_cap { private_bytes; cap }) ->
          cap = forced_cap && private_bytes > cap
      | _ -> false)

let () =
  in_leg ~leg "agr: the timing policy kept no timed seed out of the ranking" (fun label ->
      let r, _, _, _ = Lazy.force tuned in
      p label (r.Autotune.timings_contended = 0 && r.Autotune.timings_hopeless = 0));
  in_leg ~leg "agr: a ranked Grid seed's record shows its row-block loop serial for the cap"
    (fun label ->
      let r, _, _, _ = Lazy.force tuned in
      p_exists label r.Autotune.grid_serial_timed ~f:(fun g -> over_cap g.Autotune.grid_record));
  in_leg ~leg
    "agr: every listed candidate asked for a Grid loop the pool kept serial, per its own census"
    (fun label ->
      let r, _, _, _ = Lazy.force tuned in
      (* Non-empty by the claim above; [p_all] says so if it were not. *)
      p_all label r.Autotune.grid_serial_timed ~f:(fun g ->
          Autotune.grid_rendered_serially g.Autotune.grid_record
          && Float.is_finite g.Autotune.grid_ms));
  in_leg ~leg "agr: the crowned candidate's record is the one its shipped routine carries"
    (fun label ->
      let r, routine, _, _ = Lazy.force tuned in
      p label
        (match r.Autotune.best_grid with
        | Some g ->
            Int.equal g.Cs.serial routine.Context.grid.Cs.serial
            && Int.equal g.Cs.pool_rendered routine.Context.grid.Cs.pool_rendered
            && Bool.equal g.Cs.pool routine.Context.grid.Cs.pool
        | None -> false));
  in_leg ~leg "agr: a crowned Grid-asking schedule rendered serially is among the listed"
    (fun label ->
      let r, _, _, _ = Lazy.force tuned in
      p label
        ((not
            (Autotune.schedule_asks_grid r.Autotune.best_schedule
            && Option.exists r.Autotune.best_grid ~f:Autotune.grid_rendered_serially))
        || List.exists r.Autotune.grid_serial_timed ~f:(fun g ->
            String.equal g.Autotune.grid_label r.Autotune.best_label)));
  in_leg ~leg "agr: the tuned matmul matches the reference bitwise" (fun label ->
      let _, _, got, expected = Lazy.force tuned in
      p_all2 label got expected ~f:Float.equal)

(* The bitwise claim's premise: every partial sum of every cell is exact at an f32 accumulator, in
   any association, so every correct rendering returns the reference's bits. Derived from the
   operands' own init functions; [~any_sign] because the leg runs on the C backends, no tensor
   unit. *)
let () =
  p_all "agr: every reference cell's partial sums are exact at an f32 accumulator"
    (Ll_test.matmul_products ~m:n ~n ~k:n fa fb)
    ~f:(Ll_test.exact_partials ~any_sign:true ~sig_bits:24)
