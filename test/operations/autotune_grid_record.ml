(* gh-ocannl-1702: an autotune report carries the renderer's Grid record per candidate;
   gh-ocannl-1793: the seeding pre-filter weighs what that record would refuse for the per-chunk
   privatization cap.

   Every compiled routine records, for each outermost [Grid] loop, whether it rendered on the CPU
   pool or which rule kept it serial ([Context.routine.grid], gh-ocannl-1668). The serial loop
   computes the same values, so a Grid-seeded candidate the renderer kept serial is timed and ranked
   under a Grid label -- the failure [best_tensorization] closed for tensorization. The report
   carries the crowned candidate's record ([best_grid]) and every ranked Grid-asking candidate that
   rendered a Grid loop serially ([grid_serial_timed]), and the [autotune_log] flags each one.

   The cap is forced to [forced_cap] before anything resolves it (it is read once per process, so it
   is stated before the first compile, and the resolved value is pinned below). The packed matmul
   sketch's grid-outermost shapes keep the row-block loop outside the reduction blocks, so their
   contraction mints a [bm x bn] accumulator fragment inside that loop's body, which the renderer
   privatizes per chunk and weighs against the cap. Until gh-ocannl-1793 the pre-filter weighed only
   the in-kernel packed tiles, and the hoisted-only Grid shape has none (both operands are host-init
   constants, packed at link time): a seed whose fragment exceeds the cap was proposed, compiled and
   timed with its row-block loop serial.

   The seeding leg judges the pre-filter against the renderer, on two sites and one cap: every
   proposed grid-outermost hoisted seed of a 64^3 and a 128^3 matmul, and every [Hoisted] seed
   re-asked as its hoisted-grid twin (the pre-filter's judgment of that geometry under the Grid
   shape), is compiled and its record read. A proposed seed must render no Grid loop serially for
   the cap, and a refused one the renderer would privatize must be over it: the pre-filter refuses
   what the renderer refuses, and no more. Both sides occur: the 128^3 site's fragments are over the
   cap (red before gh-ocannl-1793, which proposed them), and the 64^3 site's [16 x 64] f32 fragment
   is exactly [forced_cap], which the renderer admits (its decline is strictly over the cap), so it
   is proposed and renders on the pool.

   The tuning leg searches the 64^3 site: no ranked candidate rendered serially for the cap, and the
   report still lists the Grid seeds the launch floor ([cc_grid_min_updates]) kept serial -- each
   grid-split shape zeroes the output in its own Grid nest, whose 64^2 updates are under the floor.

   Ranked is the timing policy's verdict, not the seed's, and on a loaded host the policy drops
   seeds: a window under a co-tenant is refused as host contention, and a whole-routine seed whose
   first samples read four times the beam head's is cut as hopeless -- on a 4 us routine, a
   scheduler hiccup. So the windows are opted out of host contention
   ([with_uncontended_test_windows], gh-ocannl-1156) and every admitted reading passes through
   [on_candidate_measured] unchanged (gh-ocannl-1027): the readings, and so the ranking and the
   crown, stay the clock's, but a scripted reading turns the hopeless cut off (gh-ocannl-1527) --
   only what the policy keeps out of the ranking changes, and every seed the menu proposes is
   ranked.

   The record is relational, not restated: each listed candidate's own census is serial on a pool
   backend, the crowned candidate's record is the one its routine carries, and a crowned Grid-asking
   schedule rendered serially is among the listed. The tuned routine also computes the reference
   product bitwise, on a premise the last item derives, so a serial rendering is the same values.

   On a backend with no CPU pool a [Grid] loop is a hardware grid and nothing is recorded, so both
   legs are skipped there by name. *)

(* cc-fp-contract: exact-operands -- Ll_test.exact_partials derives the operands' sums exact at
   f32 *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Cs = Ir.C_syntax
open Verdict.Claims

let forced_cap = 4096

(* Before anything resolves the cap: [Cs.per_chunk_private_bytes_cap] is a process-lifetime lazy. *)
let () =
  Hashtbl.set Utils.config_file_args ~key:"cc_grid_private_bytes_cap"
    ~data:(Int.to_string forced_cap)

(* The cc vector file the sketch menu follows, pinned for the process (the width auto-probes from
   the host, and the register tile's geometry follows it): the claims are about hoisted-grid seeds'
   records, which need the menu that proposes them. Also before the first search. *)
let () = Fissioned_chain.small_menu_envelope ()
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)

(* The census exists only where the pool does: gated on the backend's own predicate, never on the
   report under test. *)
let leg =
  Verdict.leg ~when_:(on_cpu && Context.Cc_backend.pool_parallel_grid ()) ~on:backend_name ()

let () =
  p "the renderer resolved the forced per-chunk privatization cap"
    (Lazy.force Cs.per_chunk_private_bytes_cap = forced_cap)

(* The operands' init functions: they vary with every index pair, and their products are multiples
   of 1/4 small enough for the derivation below. *)
let operand_inits n =
  ( Ll_test.cycle ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25,
    Ll_test.cycle ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1. )

(* An [n x n] product of two host-init constants -- both operands hoistable -- with the operands'
   values. *)
let matmul ~tag n =
  let fa, fb = operand_inits n in
  let av = Array.init (n * n) ~f:(fun t -> fa [| t / n; t % n |]) in
  let bv = Array.init (n * n) ~f:(fun t -> fb [| t / n; t % n |]) in
  let a = TDSL.ndarray av ~label:[ tag ^ "_a" ] ~output_dims:[ n; n ] () in
  let b = TDSL.ndarray bv ~label:[ tag ^ "_b" ] ~output_dims:[ n; n ] () in
  let%op c = a +* "ik;kj=>ij" b in
  (c, av, bv)

let over_cap (g : Cs.grid_summary) =
  List.exists g.loops ~f:(fun e ->
      match e.Cs.site with
      | Cs.Grid_serial (Grid_private_over_cap { private_bytes; cap }) ->
          cap = forced_cap && private_bytes > cap
      | _ -> false)

let below_floor (g : Cs.grid_summary) =
  List.exists g.loops ~f:(fun e ->
      match e.Cs.site with Cs.Grid_serial (Grid_below_min_updates _) -> true | _ -> false)

(* A loop the renderer put on the pool privatizing a per-chunk local, within the cap. *)
let pool_privatizing (g : Cs.grid_summary) =
  List.exists g.loops ~f:(fun e ->
      match e.Cs.site with
      | Cs.Grid_pool { private_bytes } -> 0 < private_bytes && private_bytes <= forced_cap
      | _ -> false)

(* --- The seeding leg: the pre-filter against the renderer. *)

type judged = { j_proposed : bool; j_grid : Cs.grid_summary option }

let equal_params a b =
  Sexp.equal (Autotune.sexp_of_sketch_params a) (Autotune.sexp_of_sketch_params b)

(* The hoisted-grid shape's geometries on the [n]-site: its proposed leaves, and each [Hoisted] leaf
   re-asked with the Grid flag -- the hoisted-grid twin of that geometry, proposed or not. Each is
   compiled (its schedule applied as the search would) and its Grid record read. *)
let judge_site n =
  let tag = Printf.sprintf "agr_s%d" n in
  let c, _, _ = matmul ~tag n in
  let lowering = Ll_test.captured_lowering ~name:(tag ^ "_probe") (Train.forward c) in
  let limits = Context.hardware_limits (Context.auto ()) in
  let _family, seeds = Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits lowering in
  let packed_hoisted (sp : Autotune.sketch_params) =
    sp.sk_mma && sp.sk_bk > 0 && sp.sk_hoist && not sp.sk_pack_rest
  in
  let candidates =
    List.filter_map seeds ~f:(fun sp ->
        if packed_hoisted sp then Some { sp with sk_grid = true } else None)
    |> List.dedup_and_sort ~compare:(fun a b ->
        Sexp.compare (Autotune.sexp_of_sketch_params a) (Autotune.sexp_of_sketch_params b))
  in
  List.mapi candidates ~f:(fun i sp ->
      let label = Autotune.sketch_params_label sp in
      let proposed = List.mem seeds sp ~equal:equal_params in
      let name = Printf.sprintf "%s_c%d" tag i in
      let c, _, _ = matmul ~tag:name n in
      let grid =
        match
          Context.compile ~name
            ~lowered_transform:(fun opt ->
              [
                Ir.Schedule.apply
                  (Autotune.sketch_schedule ~accum_prec:Ll_test.backend_accum_prec ~p:sp opt)
                  opt;
              ])
            (Context.auto ()) (Train.forward c) Ir.Indexing.Empty
        with
        | ctx, routine ->
            Context.release ctx;
            Some routine.Context.grid
        | exception exn ->
            Stdio.eprintf "agr: %s: did not compile (not part of the golden): %s\n%!" label
              (Exn.to_string exn);
            None
      in
      Stdio.eprintf "agr: n=%d %s %s: %s (not part of the golden)\n%!" n
        (if proposed then "proposed" else "refused")
        label
        (Option.value_map grid ~default:"no record" ~f:Cs.grid_summary_string);
      { j_proposed = proposed; j_grid = grid })

let judged = lazy (List.concat_map [ 64; 128 ] ~f:judge_site)
let grid_of j ~f = Option.exists j.j_grid ~f

let () =
  in_leg ~leg "agr-seed: every judged candidate compiled" (fun label ->
      p_all label (Lazy.force judged) ~f:(fun j -> Option.is_some j.j_grid));
  in_leg ~leg "agr-seed: no proposed hoisted-grid seed renders a Grid loop serially for the cap"
    (fun label ->
      p_none label
        (List.filter (Lazy.force judged) ~f:(fun j -> j.j_proposed))
        ~f:(grid_of ~f:over_cap));
  in_leg ~leg "agr-seed: every refused geometry the renderer would privatize is over the cap"
    (fun label ->
      p_all label
        (List.filter (Lazy.force judged) ~f:(fun j ->
             (not j.j_proposed) && grid_of j ~f:(fun g -> over_cap g || pool_privatizing g)))
        ~f:(grid_of ~f:over_cap));
  in_leg ~leg "agr-seed: the pre-filter refuses a geometry whose fragment is over the cap"
    (fun label ->
      p_exists label (Lazy.force judged) ~f:(fun j -> (not j.j_proposed) && grid_of j ~f:over_cap));
  in_leg ~leg "agr-seed: the pre-filter keeps a geometry that privatizes on the pool within the cap"
    (fun label ->
      p_exists label (Lazy.force judged) ~f:(fun j -> j.j_proposed && grid_of j ~f:pool_privatizing))

(* --- The tuning leg. *)

let n = 64

let tuned =
  lazy
    (let c, av, bv = matmul ~tag:"agr" n in
     let expected =
       Array.init (n * n) ~f:(fun idx ->
           let i = idx / n and j = idx % n in
           let acc = ref 0. in
           for k = 0 to n - 1 do
             acc := !acc +. (av.((i * n) + k) *. bv.((k * n) + j))
           done;
           !acc)
     in
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

let () =
  in_leg ~leg "agr: the timing policy kept no timed seed out of the ranking" (fun label ->
      let r, _, _, _ = Lazy.force tuned in
      p label (r.Autotune.timings_contended = 0 && r.Autotune.timings_hopeless = 0));
  in_leg ~leg "agr: no ranked candidate rendered a Grid loop serially for the cap" (fun label ->
      let r, _, _, _ = Lazy.force tuned in
      p_none label r.Autotune.grid_serial_timed ~f:(fun g -> over_cap g.Autotune.grid_record));
  in_leg ~leg "agr: a ranked Grid seed's record shows a loop serial for the launch floor"
    (fun label ->
      let r, _, _, _ = Lazy.force tuned in
      p_exists label r.Autotune.grid_serial_timed ~f:(fun g -> below_floor g.Autotune.grid_record));
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
  let fa, fb = operand_inits n in
  p_all "agr: every reference cell's partial sums are exact at an f32 accumulator"
    (Ll_test.matmul_products ~m:n ~n ~k:n fa fb)
    ~f:(Ll_test.exact_partials ~any_sign:true ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single))
