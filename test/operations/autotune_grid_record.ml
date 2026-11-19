(* gh-ocannl-1702: an autotune report carries the renderer's Grid record per candidate;
   gh-ocannl-1793: the seeding pre-filter weighs what that record would refuse for the per-chunk
   privatization cap; gh-ocannl-1857: it also weighs the site launch floor.

   Every compiled routine records, for each outermost [Grid] loop, whether it rendered on the CPU
   pool or which rule kept it serial ([Context.routine.grid], gh-ocannl-1668). The serial loop
   computes the same values, so the timing line includes the compiled routine's Grid record. The
   report carries the crowned candidate's record ([best_grid]) and every ranked Grid-asking
   candidate that rendered a Grid loop serially ([grid_serial_timed]).

   The cap is forced to [forced_cap] before anything resolves it (it is read once per process, so it
   is stated before the first compile, and the resolved value is pinned below). The packed matmul
   sketch's grid-outermost shapes keep the row-block loop outside the reduction blocks, so their
   contraction mints a [bm x bn] accumulator fragment inside that loop's body, which the renderer
   privatizes per chunk and weighs against the cap. Until gh-ocannl-1793 the pre-filter weighed only
   the in-kernel packed tiles, and the hoisted-only Grid shape has none (both operands are host-init
   constants, packed at link time): a seed whose fragment exceeds the cap was proposed, compiled and
   timed with its row-block loop serial.

   The seeding leg judges the pre-filter against the renderer, on two sites and one cap: every
   proposed grid-outermost hoisted seed of a 96^3 and a 128^3 matmul, and every [Hoisted] seed
   re-asked as its hoisted-grid twin (the pre-filter's judgment of that geometry under the Grid
   shape), is compiled and its record read. A proposed seed must render no Grid loop serially for
   the cap, and a refused one the renderer would privatize must be over it: the pre-filter refuses
   what the renderer refuses, and no more. Both sides occur: the 128^3 site's fragments are over the
   cap (red before gh-ocannl-1793, which proposed them), and the 96^3 site's [16 x 96] f32 fragment
   is exactly [forced_cap], which the renderer admits (its decline is strictly over the cap), so it
   is proposed and renders on the pool.

   The floor leg re-asks every Serial packed geometry as its Grid twin at 64^3, and the unsplit
   whole-triple geometry with a 16-row Grid block at 32x16x16. The scheduled contraction loop's
   symbol identifies the site independently of the output zeroing. No proposed site may render
   serially for the floor or lack a matching census entry. Both witnesses are refused at the default
   floor, and every below-floor geometry is proposed and renders on the pool when the floor is
   lowered to 4096. The lowered-floor candidates also exercise the proposed-site claims. The same
   parameters connect the two runs; an unrelated pool-rendered geometry cannot satisfy the control.

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

let forced_cap = 6144

(* Before anything resolves the cap: [Cs.per_chunk_private_bytes_cap] is a process-lifetime lazy. *)
let () = Utils.state_config ~key:"cc_grid_private_bytes_cap" ~data:(Int.to_string forced_cap)

(* The cc vector file the sketch menu follows, pinned for the process (the width auto-probes from
   the host, and the register tile's geometry follows it): the claims are about hoisted-grid seeds'
   records, which need the menu that proposes them. Also before the first search. *)
let () = Fissioned_chain.small_menu_envelope ()
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)

(* The census exists only where the pool does: gated on the backend's own predicate, never on the
   report under test. *)
let leg =
  let backend_leg = Verdict.leg ~when_:on_cpu ~on:backend_name () in
  Verdict.sub_leg backend_leg ~aggregation:`Environment
    ~when_:(lazy (Context.Cc_backend.pool_parallel_grid ()))
    ~on:"cc without a native pool"

let () =
  p "the renderer resolved the forced per-chunk privatization cap"
    (Lazy.force Cs.per_chunk_private_bytes_cap = forced_cap)

(* The operands' init functions: they vary with every index pair, and their products are multiples
   of 1/4 small enough for the derivation below. *)
let operand_inits ?rows ?depth n =
  let m = Option.value rows ~default:n and k = Option.value depth ~default:n in
  ( Ll_test.cycle ~dims:[| m; k |] ~modulus:13 ~offset:0. ~stride:0.25,
    Ll_test.cycle ~dims:[| k; n |] ~modulus:17 ~offset:(-8.) ~stride:1. )

(* A product of two host-init constants -- both operands hoistable -- with their values. *)
let matmul ?rows ?depth ~tag n =
  let m = Option.value rows ~default:n and k = Option.value depth ~default:n in
  let fa, fb = operand_inits ?rows ?depth n in
  let av = Array.init (m * k) ~f:(fun t -> fa [| t / k; t % k |]) in
  let bv = Array.init (k * n) ~f:(fun t -> fb [| t / n; t % n |]) in
  let a = TDSL.ndarray av ~label:[ tag ^ "_a" ] ~output_dims:[ m; k ] () in
  let b = TDSL.ndarray bv ~label:[ tag ^ "_b" ] ~output_dims:[ k; n ] () in
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

let matmul_grid_loop llc =
  let loops =
    List.filter_map (Ll_test.loop_sites llc) ~f:(fun loop ->
        match loop.ls_stmt with
        | Ir.Low_level.For_loop { axis = Grid; index; body; _ } ->
            let tiles = Ll_test.count_stmt body ~f:(function Tile_mma _ -> true | _ -> false) in
            Option.some_if (tiles > 0) index
        | _ -> None)
  in
  match loops with [ index ] -> Some (Ir.Indexing.symbol_ident index) | _ -> None

let grid_site loop entries =
  Option.bind loop ~f:(fun loop ->
      match List.filter entries ~f:(fun e -> String.equal e.Cs.loop loop) with
      | [ entry ] -> Some entry.Cs.site
      | _ -> None)

(* --- The seeding leg: the pre-filter against the renderer. *)

type judged = {
  j_params : Autotune.sketch_params;
  j_dims : int * int * int;
  j_bound : int;
  j_padded : bool;
  j_proposed : bool;
  j_grid : Cs.grid_summary option;
  j_site_loop : string option;
}

let equal_params a b =
  Sexp.equal (Autotune.sexp_of_sketch_params a) (Autotune.sexp_of_sketch_params b)

(* The selected shape's geometries on the site: its proposed leaves, and each [Hoisted] leaf
   re-asked with the Grid flag -- the hoisted-grid twin of that geometry, proposed or not. Each is
   compiled (its schedule applied as the search would) and its Grid record read. *)
let judge_site ?rows ?depth ?(shape = `Hoisted) n =
  let tag =
    Printf.sprintf "agr_s%dx%dx%d_%s_f%d" (Option.value rows ~default:n) n
      (Option.value depth ~default:n)
      (match shape with `Hoisted -> "h" | `Grid -> "g" | `Whole -> "w")
      (Cs.grid_min_updates ())
  in
  let c, _, _ = matmul ?rows ?depth ~tag n in
  let lowering = Ll_test.captured_lowering ~name:(tag ^ "_probe") (Train.forward c) in
  let limits = Context.hardware_limits (Context.auto ()) in
  let _family, seeds = Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits lowering in
  let site = Option.value_exn (Autotune.detect_matmul lowering.Ir.Low_level.llc) in
  let packed_hoisted (sp : Autotune.sketch_params) =
    sp.sk_mma && sp.sk_bk > 0 && sp.sk_hoist && not sp.sk_pack_rest
  in
  let candidates =
    List.filter_map seeds ~f:(fun sp ->
        match shape with
        | `Whole when sp.sk_mma && sp.sk_bk = 0 && sp.sk_bm = 0 -> Some { sp with sk_bm = 16 }
        | `Grid when sp.sk_mma && sp.sk_bk > 0 && (not sp.sk_hoist) && not sp.sk_pack_rest ->
            Some { sp with sk_grid = true }
        | `Hoisted when packed_hoisted sp -> Some { sp with sk_grid = true }
        | _ -> None)
    |> List.dedup_and_sort ~compare:(fun a b ->
        Sexp.compare (Autotune.sexp_of_sketch_params a) (Autotune.sexp_of_sketch_params b))
  in
  List.mapi candidates ~f:(fun i sp ->
      let label = Autotune.sketch_params_label sp in
      let proposed = List.mem seeds sp ~equal:equal_params in
      let name = Printf.sprintf "%s_c%d" tag i in
      let c, _, _ = matmul ?rows ?depth ~tag:name n in
      let site_loop = ref None in
      let grid =
        match
          Context.compile ~name
            ~lowered_transform:(fun opt ->
              let scheduled =
                Ir.Schedule.apply
                  (Autotune.sketch_schedule ~accum_prec:Ll_test.backend_accum_prec ~p:sp opt)
                  opt
              in
              (match shape with
              | `Grid | `Whole -> site_loop := matmul_grid_loop scheduled.llc
              | `Hoisted -> ());
              [ scheduled ])
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
      let bound = Autotune.cpu_mma_grid_updates_bound site sp ~cap:Int.max_value in
      Stdio.eprintf "agr: %s %s %s (bound %d): %s (not part of the golden)\n%!" tag
        (if proposed then "proposed" else "refused")
        label bound
        (Option.value_map grid ~default:"no record" ~f:Cs.grid_summary_string);
      let divides b e = b = 0 || e % b = 0 in
      {
        j_params = sp;
        j_dims = (site.m_ni, site.m_nj, site.m_nk);
        j_bound = bound;
        j_padded =
          not
            (divides sp.sk_bm site.m_ni && divides sp.sk_bn site.m_nj && divides sp.sk_bk site.m_nk);
        j_proposed = proposed;
        j_grid = grid;
        j_site_loop = !site_loop;
      })

let judged = lazy (List.concat_map [ 96; 128 ] ~f:(fun n -> judge_site n))

let site_of ?(census_order = Fn.id) j =
  Option.bind j.j_grid ~f:(fun g -> grid_site j.j_site_loop (census_order g.Cs.loops))

let site_below_floor j =
  match site_of j with Some (Cs.Grid_serial (Grid_below_min_updates _)) -> true | _ -> false

let site_has_no_floor_decline j =
  match site_of j with
  | None | Some (Cs.Grid_serial (Grid_below_min_updates _)) -> false
  | Some _ -> true

let floor_sites () = judge_site ~shape:`Grid 64 @ judge_site ~rows:72 ~depth:64 ~shape:`Grid 32
let floor_judged = lazy (floor_sites ())
let whole_floor_site () = judge_site ~rows:32 ~depth:16 ~shape:`Whole 16
let whole_floor_judged = lazy (whole_floor_site ())

let floor_control =
  lazy
    (Scripted_search.with_settings
       [ ("cc_grid_min_updates", "4096") ]
       (fun () ->
         let packed = floor_sites () in
         (packed, whole_floor_site ())))

let floor_control_candidates () =
  let packed, whole = Lazy.force floor_control in
  packed @ whole

let same_geometry a b = Poly.equal a.j_dims b.j_dims && equal_params a.j_params b.j_params
let grid_of j ~f = Option.exists j.j_grid ~f

let () =
  in_leg ~leg "agr-seed: every judged candidate compiled" (fun label ->
      p_all label
        (Lazy.force judged @ Lazy.force floor_judged @ Lazy.force whole_floor_judged
       @ floor_control_candidates ())
        ~f:(fun j -> Option.is_some j.j_grid));
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

(* --- The plain-Grid seeding leg (gh-ocannl-1856): the pre-filter's weighed bytes against the
   renderer's record, where the pad masks force the fragment.

   The plain [grid] packing shape sinks its row-block [Grid] loop under the reduction blocks, so no
   loop contracts the accumulator; but a row or column block that does not divide its extent forces
   [Tensorize]'s contraction at the statement site, inside the row-block body, and the renderer
   privatizes that [bm x bn] tile per chunk beside the A~ tile. On a site no menu block divides,
   every plain-Grid geometry the shared menu proposes for the [Serial] shape is re-asked under the
   Grid shape (proposed or not) and compiled. The pre-filter's sum
   ([Autotune.cpu_pack_grid_chunk_bytes], what it weighs) is the renderer's recorded private bytes
   on every candidate whose row-block loop privatizes; a proposed one renders no loop serially for
   the cap and a refused one is over it; a padded geometry whose A~ tile alone fits under the cap is
   refused because its fragment does not, and a padded one inside the cap stays proposed and renders
   on the pool. *)

type grid_judged = {
  g_proposed : bool;
  g_masked : bool;
  g_a_tile : int;
  g_weighed : int;
  g_grid : Cs.grid_summary option;
}

(* An [m x k] by [k x n] product of two host-init constants. *)
let rect_matmul ~tag ~m ~k ~n =
  let fa = Ll_test.cycle ~dims:[| m; k |] ~modulus:13 ~offset:0. ~stride:0.25
  and fb = Ll_test.cycle ~dims:[| k; n |] ~modulus:17 ~offset:(-8.) ~stride:1. in
  let a =
    TDSL.ndarray
      (Array.init (m * k) ~f:(fun t -> fa [| t / k; t % k |]))
      ~label:[ tag ^ "_a" ]
      ~output_dims:[ m; k ] ()
  in
  let b =
    TDSL.ndarray
      (Array.init (k * n) ~f:(fun t -> fb [| t / n; t % n |]))
      ~label:[ tag ^ "_b" ]
      ~output_dims:[ k; n ] ()
  in
  let%op c = a +* "ik;kj=>ij" b in
  c

let recorded_private (g : Cs.grid_summary) =
  List.filter_map g.loops ~f:(fun e ->
      match e.Cs.site with
      | Cs.Grid_pool { private_bytes } when private_bytes > 0 -> Some private_bytes
      | Cs.Grid_serial (Grid_private_over_cap { private_bytes; _ }) -> Some private_bytes
      | _ -> None)

let judge_plain_grid ?bk (m, k, n) =
  let tag = Printf.sprintf "agr_g%dx%dx%d" m k n in
  let c = rect_matmul ~tag ~m ~k ~n in
  let lowering = Ll_test.captured_lowering ~name:(tag ^ "_probe") (Train.forward c) in
  let site = Option.value_exn (Autotune.detect_matmul lowering.Ir.Low_level.llc) in
  let limits = Context.hardware_limits (Context.auto ()) in
  let _family, seeds = Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits lowering in
  let serial_packed (sp : Autotune.sketch_params) =
    sp.sk_mma && sp.sk_bk > 0 && (not sp.sk_hoist) && (not sp.sk_grid) && (not sp.sk_pack_rest)
    && Option.value_map bk ~default:true ~f:(Int.equal sp.sk_bk)
  in
  let candidates =
    List.filter_map seeds ~f:(fun sp ->
        if serial_packed sp then Some { sp with sk_grid = true } else None)
    |> List.dedup_and_sort ~compare:(fun a b ->
        Sexp.compare (Autotune.sexp_of_sketch_params a) (Autotune.sexp_of_sketch_params b))
  in
  List.mapi candidates ~f:(fun i sp ->
      let label = Autotune.sketch_params_label sp in
      let proposed = List.mem seeds sp ~equal:equal_params in
      let prec_bytes =
        Ir.Ops.prec_in_bytes
          (Option.value sp.sk_pack_prec ~default:(Lazy.force site.m_a.Ir.Tnode.storage_prec))
      in
      let divides b e = b <= e && e % b = 0 in
      let masked = (not (divides sp.sk_bm m)) || (sp.sk_bn > 0 && not (divides sp.sk_bn n)) in
      let weighed = Autotune.cpu_pack_grid_chunk_bytes site ~shape:`Grid ~prec_bytes sp in
      let name = Printf.sprintf "%s_c%d" tag i in
      let c = rect_matmul ~tag:name ~m ~k ~n in
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
      Stdio.eprintf "agr: %dx%dx%d %s %s%s, weighed %d bytes: %s (not part of the golden)\n%!" m k n
        (if proposed then "proposed" else "refused")
        label
        (if masked then " (padded)" else "")
        weighed
        (Option.value_map grid ~default:"no record" ~f:Cs.grid_summary_string);
      {
        g_proposed = proposed;
        g_masked = masked;
        g_a_tile = sp.sk_bm * sp.sk_bk * prec_bytes;
        g_weighed = weighed;
        g_grid = grid;
      })

let grid_judged =
  (* No menu row block divides 40, 72 or 200 rows. At 40x64x96 the [16 x 96] f32 fragment alone is
     the cap, so with the [16 x 16] A~ tile the geometry is over it; at 200x64x32 the [16 x 32]
     fragment and the A~ tile fit, and the row-block launch is past [cc_grid_min_updates] (at 72
     rows it is not, so the loop stays serial for the floor). *)
  lazy
    (List.concat_map
       [ (40, 64, 96); (72, 64, 32); (200, 64, 32) ]
       ~f:(fun dims -> judge_plain_grid dims))

let g_grid_of j ~f = Option.exists j.g_grid ~f

let () =
  in_leg ~leg "agr-grid: every judged plain-Grid candidate compiled" (fun label ->
      p_all label (Lazy.force grid_judged) ~f:(fun j -> Option.is_some j.g_grid));
  in_leg ~leg "agr-grid: the pre-filter weighs what the renderer privatizes, byte for byte"
    (fun label ->
      p_all label
        (List.filter (Lazy.force grid_judged) ~f:(fun j ->
             g_grid_of j ~f:(fun g -> not (List.is_empty (recorded_private g)))))
        ~f:(fun j ->
          g_grid_of j ~f:(fun g -> List.for_all (recorded_private g) ~f:(fun b -> b = j.g_weighed))));
  in_leg ~leg "agr-grid: no proposed plain-Grid seed renders a Grid loop serially for the cap"
    (fun label ->
      p_none label
        (List.filter (Lazy.force grid_judged) ~f:(fun j -> j.g_proposed))
        ~f:(g_grid_of ~f:over_cap));
  in_leg ~leg "agr-grid: every refused geometry the renderer would privatize is over the cap"
    (fun label ->
      p_all label
        (List.filter (Lazy.force grid_judged) ~f:(fun j ->
             (not j.g_proposed) && g_grid_of j ~f:(fun g -> over_cap g || pool_privatizing g)))
        ~f:(g_grid_of ~f:over_cap));
  in_leg ~leg
    "agr-grid: a padded geometry whose A~ tile fits is refused for its mask-forced fragment"
    (fun label ->
      p_exists label (Lazy.force grid_judged) ~f:(fun j ->
          j.g_masked && j.g_a_tile <= forced_cap && (not j.g_proposed) && g_grid_of j ~f:over_cap));
  in_leg ~leg "agr-grid: a padded geometry inside the cap is proposed and privatizes on the pool"
    (fun label ->
      p_exists label (Lazy.force grid_judged) ~f:(fun j ->
          j.g_masked && j.g_proposed && g_grid_of j ~f:pool_privatizing))

let without_launch_floor f = Scripted_search.with_settings [ ("cc_grid_min_updates", "0") ] f

(* Pool eligibility at 40^3 and 72^3 (gh-ocannl-2275), with the launch floor out of the way. A k
   partition duplicates the row-block loop but leaves its A~ tile shared across segments. *)
let () =
  List.iter [ 40; 72 ] ~f:(fun n ->
      let judged = lazy (without_launch_floor (fun () -> judge_plain_grid ~bk:16 (n, n, n))) in
      let claim = Printf.sprintf "agr-grid: no %d^3 bk=16 plain-Grid seed is proposed" n in
      in_leg ~leg claim (fun label -> p_none label (Lazy.force judged) ~f:(fun j -> j.g_proposed));
      let witness =
        Printf.sprintf "agr-grid: at %d^3 both k segments share scratch within the cap" n
      in
      in_leg ~leg witness (fun label ->
          p_exists label (Lazy.force judged) ~f:(fun j ->
              j.g_weighed <= forced_cap
              && g_grid_of j ~f:(fun g ->
                  match
                    List.filter_map g.Cs.loops ~f:(fun e ->
                        match e.Cs.site with
                        | Cs.Grid_serial (Grid_local_accessed_outside tile) -> Some tile
                        | _ -> None)
                  with
                  | [ full; edge ] -> String.equal full edge
                  | _ -> false))));
  in_leg ~leg "agr-grid: a single partial k block is proposed and renders on the pool" (fun label ->
      let judged = without_launch_floor (fun () -> judge_plain_grid ~bk:16 (40, 12, 40)) in
      p_exists label judged ~f:(fun j -> j.g_proposed && g_grid_of j ~f:pool_privatizing))

(* The 41-row output pads at both tested block sizes. Fission exposes an actual panel Grid. At 32
   rows the staged tiles and masked fragment reach the cap with 8 input channels and exceed it with
   12. *)
let judge_conv_panel ~ic bm =
  let name = Printf.sprintf "agr_conv_ic%d_bm%d" ic bm in
  let x =
    NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.single ~o:[ 5; 43; ic ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let w =
    NTDSL.init ~l:(name ^ "_w") ~prec:Ir.Ops.single ~i:[ 3; 3; ic ] ~o:[ 32 ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op y = x +* "|oh<+kh, ow<+kw, ic; |kh, kw, ic -> oc => |oh, ow, oc" w in
  let comp = Train.forward y in
  let judgment = ref None in
  let schedule _ (seg : Ir.Low_level.optimized) =
    Option.map (Autotune.detect_conv seg.llc) ~f:(fun site ->
        let limits = Context.hardware_limits (Context.auto ()) in
        let _, seeds = Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits seg in
        let serial =
          List.find_exn seeds ~f:(fun sp ->
              sp.Autotune.sk_conv && sp.sk_bm = 0 && (not sp.sk_grid) && (not sp.sk_epilogue)
              && Option.is_none sp.sk_tile)
        in
        let panel = { serial with sk_bm = bm } in
        if site.c_zeroed || not (Autotune.conv_acc_masked site panel) then
          failwith "expected a pad-masked conv panel in a fission segment";
        judgment :=
          Some
            ( List.mem seeds panel ~equal:equal_params,
              Autotune.cpu_conv_panel_chunk_bytes site panel );
        Autotune.sketch_schedule ~accum_prec:Ll_test.backend_accum_prec ~p:panel seg)
  in
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(Ll_test.tuner_transform ~schedule)
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  Context.release ctx;
  let grid = routine.Context.grid in
  let proposed, weighed = Option.value_exn !judgment in
  Stdio.eprintf "agr: conv ic=%d bm=%d %s: %s (not part of the golden)\n%!" ic bm
    (if proposed then "proposed" else "refused")
    (Cs.grid_summary_string grid);
  (proposed, grid, weighed)

let () =
  let panel ~ic bm = lazy (without_launch_floor (fun () -> judge_conv_panel ~ic bm)) in
  let within = panel ~ic:12 16 and at_cap = panel ~ic:8 32 and over = panel ~ic:12 32 in
  in_leg ~leg "agr-conv: a padded row panel within the cap is proposed and renders on the pool"
    (fun label ->
      let proposed, grid, _ = Lazy.force within in
      p label (proposed && List.length grid.Cs.loops = 1 && pool_privatizing grid));
  in_leg ~leg "agr-conv: a padded row panel at the cap is proposed and renders on the pool"
    (fun label ->
      let proposed, grid, _ = Lazy.force at_cap in
      p label
        (proposed
        &&
        match grid.Cs.loops with
        | [ { Cs.site = Grid_pool { private_bytes }; _ } ] -> private_bytes = forced_cap
        | _ -> false));
  in_leg ~leg "agr-conv: a padded row panel over the cap is refused" (fun label ->
      let proposed, grid, _ = Lazy.force over in
      p label ((not proposed) && List.length grid.Cs.loops = 1 && over_cap grid));
  in_leg ~leg "agr-conv: the pre-filter weighs what the renderer privatizes, byte for byte"
    (fun label ->
      p_all label
        [ Lazy.force within; Lazy.force at_cap; Lazy.force over ]
        ~f:(fun (_, grid, weighed) -> List.equal Int.equal (recorded_private grid) [ weighed ]))

let () =
  in_leg ~leg "agr-seed: no proposed packed Grid seed renders its site serially for the floor"
    (fun label ->
      let packed, _ = Lazy.force floor_control in
      p_all label
        (Lazy.force floor_judged @ packed)
        ~f:(fun j -> (not j.j_proposed) || site_has_no_floor_decline j));
  in_leg ~leg "agr-seed: the floor refuses a packed Grid geometry whose site renders below it"
    (fun label ->
      p_exists label (Lazy.force floor_judged) ~f:(fun j ->
          (not j.j_proposed) && site_below_floor j));
  in_leg ~leg "agr-seed: lowering the floor keeps each below-floor geometry on the pool"
    (fun label ->
      let controls = floor_control_candidates () in
      p_all label
        (List.filter (Lazy.force floor_judged @ Lazy.force whole_floor_judged) ~f:site_below_floor)
        ~f:(fun refused ->
          List.exists controls ~f:(fun j ->
              (* The zeroing launch can stay serial while the contraction runs on the pool. *)
              same_geometry j refused && j.j_proposed
              &&
              match site_of ~census_order:List.rev j with
              | Some (Cs.Grid_pool _) -> true
              | _ -> false)));
  in_leg ~leg "agr-seed: every floor-only refusal renders its site below the floor" (fun label ->
      let controls = floor_control_candidates () in
      p_all label
        (List.filter
           (Lazy.force floor_judged @ Lazy.force whole_floor_judged)
           ~f:(fun j ->
             (not j.j_proposed)
             && List.exists controls ~f:(fun control ->
                 same_geometry j control && control.j_proposed)))
        ~f:site_below_floor);
  in_leg ~leg "agr-seed: the seeding bound never undercounts a below-floor site's renderer count"
    (fun label ->
      p_all label
        (List.filter (Lazy.force floor_judged @ Lazy.force whole_floor_judged) ~f:site_below_floor)
        ~f:(fun j ->
          match site_of j with
          | Some (Cs.Grid_serial (Grid_below_min_updates { updates; _ })) -> updates <= j.j_bound
          | _ -> false));
  in_leg ~leg "agr-seed: unpadded below-floor estimates equal the renderer count" (fun label ->
      p_all label
        (List.filter
           (Lazy.force floor_judged @ Lazy.force whole_floor_judged)
           ~f:(fun j -> (not j.j_padded) && site_below_floor j))
        ~f:(fun j ->
          match site_of j with
          | Some (Cs.Grid_serial (Grid_below_min_updates { updates; _ })) -> updates = j.j_bound
          | _ -> false));
  in_leg ~leg "agr-seed: no proposed whole-triple Grid seed renders its site serially for the floor"
    (fun label ->
      let _, whole = Lazy.force floor_control in
      p_all label
        (Lazy.force whole_floor_judged @ whole)
        ~f:(fun j -> (not j.j_proposed) || site_has_no_floor_decline j));
  in_leg ~leg "agr-seed: the floor refuses a whole-triple Grid geometry below it" (fun label ->
      p_exists label (Lazy.force whole_floor_judged) ~f:(fun j ->
          (not j.j_proposed) && site_below_floor j))

(* gh-ocannl-2285: the fused tail contributes a store per padded output cell and per Workgroup
   iteration. Re-ask refused seeds at their own renderer count and the admitted boundary seed one
   update above the floor. The contraction and output zeroing have independent Grid decisions. *)
let judge_fused_grid ~select ~m ~n ~k =
  let tag = Printf.sprintf "agr_fused_%dx%dx%d" m n k in
  let comp name =
    let prod = rect_matmul ~tag:name ~m ~n ~k in
    Train.set_materialized prod.Tensor.value;
    let bias =
      TDSL.ndarray
        (Array.init n ~f:(fun j -> Float.of_int (j % 3) -. 1.))
        ~label:[ name ^ "_bias" ]
        ~output_dims:[ n ] ()
    in
    let%op out = relu (prod + bias) in
    Train.forward out
  in
  let lowering = Ll_test.captured_lowering ~name:tag (comp tag) in
  let limits = Context.hardware_limits (Context.auto ()) in
  let seeds () = Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits lowering in
  let proposed = seeds () in
  let candidates =
    List.filter (without_launch_floor seeds) ~f:(fun p ->
        p.Autotune.sk_mma && p.sk_epilogue && select p)
  in
  List.mapi candidates ~f:(fun i p ->
      let compile phase ~census_order =
        let name = Printf.sprintf "%s_%s_%d" tag phase i in
        let site_loop = ref None in
        let ctx, routine =
          Context.compile ~name
            ~lowered_transform:(fun opt ->
              let scheduled =
                Ir.Schedule.apply
                  (Autotune.sketch_schedule ~accum_prec:Ll_test.backend_accum_prec ~p opt)
                  opt
              in
              site_loop := matmul_grid_loop scheduled.llc;
              [ scheduled ])
            (Context.auto ()) (comp name) Ir.Indexing.Empty
        in
        Context.release ctx;
        let grid = routine.Context.grid in
        Stdio.eprintf "agr: %s %s: %s (not part of the golden)\n%!" name
          (Autotune.sketch_params_label p) (Cs.grid_summary_string grid);
        grid_site !site_loop (census_order grid.Cs.loops)
      in
      let site = compile "default" ~census_order:Fn.id in
      let control_floor =
        match site with
        | Some (Cs.Grid_serial (Grid_below_min_updates { updates; _ })) -> updates
        | _ -> Cs.grid_min_updates () + 1
      in
      let control_proposed, control_site =
        Scripted_search.with_settings
          [ ("cc_grid_min_updates", Int.to_string control_floor) ]
          (* Swapping the contraction and zeroing entries must preserve the site's identity. *)
          (fun () ->
            (List.mem (seeds ()) p ~equal:equal_params, compile "control" ~census_order:List.rev))
      in
      (List.mem proposed p ~equal:equal_params, site, control_proposed, control_site))

let () =
  let packed = lazy (judge_fused_grid ~select:(fun p -> p.Autotune.sk_grid) ~m:33 ~n:65 ~k:16) in
  let whole k =
    judge_fused_grid ~select:(fun p -> p.Autotune.sk_bk = 0 && p.sk_bm > 0) ~m:32 ~n:16 ~k
  in
  let whole_below = lazy (whole 8) and whole_at_floor = lazy (whole 16) in
  let refused_below = function
    | false, Some (Cs.Grid_serial (Grid_below_min_updates _)), true, Some (Cs.Grid_pool _) -> true
    | _ -> false
  in
  in_leg ~leg "agr-fused: packed Grid seeds below the launch floor are refused" (fun label ->
      p_all label ~min:2 (Lazy.force packed) ~f:refused_below);
  in_leg ~leg "agr-fused: the whole-triple Grid seed below the launch floor is refused"
    (fun label -> p_all label (Lazy.force whole_below) ~f:refused_below);
  in_leg ~leg "agr-fused: epilogue stores keep the whole-triple Grid seed at the launch floor"
    (fun label ->
      let default_floor = Cs.grid_min_updates () in
      p_all label (Lazy.force whole_at_floor) ~f:(function
        | ( true,
            Some (Cs.Grid_pool _),
            false,
            Some (Cs.Grid_serial (Grid_below_min_updates { updates; floor })) ) ->
            updates = default_floor && floor = default_floor + 1
        | _ -> false))

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
