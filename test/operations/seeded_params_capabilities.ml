(* The scalar-blocktile seeding rule ([Autotune.seeded_params], the untuned CUDA/HIP default's
   per-segment sketch) prices workgroup memory with the capabilities it is handed, as the gate that
   judges the seeded kernel does ([Schedule.kernel_workgroup_bytes], gh-ocannl-1330): the seeding
   refutes a geometry exactly where [Schedule.check_hardware_limits_classified] would refuse its
   built schedule, so a seed the gate dooms never costs a compile. Every caller holding a compiling
   context passes that context's [Context.codegen_capabilities]; the default,
   [no_codegen_capabilities], charges the exact tile sizes.

   On every backend, on synthetic device limits: an f32 matmul is seeded with no workgroup-memory
   limit; its seeded geometry's schedule is built and read by the gate's own estimate under a
   stand-in capability whose allocation granule (12 bytes) rounds every seeded tile up — a tile is
   [bm * bk] f32 elements, a power of two, so no granule with an odd factor divides it. At a limit
   of that estimate the rule still seeds the geometry and the gate admits it; one byte below, the
   gate refuses it and the rule seeds something else, while a rule handed no capabilities (exact
   sizes, which fit) still seeds it — the compile a doomed seed would cost. The same pair of
   verdicts is claimed of the seeded plan ([Autotune.seeded_plan], derived for "cuda") and of its
   decision rule ([sp_start.decide], the staffed search's per-key start), which carry the
   capabilities down to the rule.

   Metal's own granule (4 bytes) changes no verdict here: the seeded geometries' tiles are multiples
   of 8 elements. The stand-in pins the threading, not a behaviour change on today's hardware. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let standin = { Ir.Backend_intf.no_codegen_capabilities with workgroup_allocation_alignment = 12 }
let limits_at cap = { Ir.Backend_intf.no_hardware_limits with max_workgroup_memory_bytes = cap }
let accum_prec = Fn.id

let () =
  let a =
    TDSL.ndarray
      (Array.create ~len:(128 * 64) 1.)
      ~label:[ "spc_a" ] ~input_dims:[ 64 ] ~output_dims:[ 128 ] ()
  in
  let b =
    TDSL.ndarray
      (Array.create ~len:(64 * 128) 1.)
      ~label:[ "spc_b" ] ~input_dims:[ 128 ] ~output_dims:[ 64 ] ()
  in
  let%op c = a * b in
  let lowered =
    Context.lowered_for_decisions ~name:"spc_matmul" (Context.auto ()) (Train.forward c)
      Ir.Indexing.Empty
  in
  let seeded ?capabilities cap =
    Autotune.seeded_params ?capabilities ~limits:(limits_at cap) ~accum_prec ~min_reduction:1
      (scratch lowered)
  in
  let unconstrained = seeded ~capabilities:standin None in
  p "the rule seeds the matmul with no workgroup-memory limit" (Option.is_some unconstrained);
  match unconstrained with
  | None -> ()
  | Some p0 ->
      Stdio.eprintf "seeded_params_capabilities: seeded %s (not part of the golden)\n%!"
        (Autotune.sketch_params_label p0);
      let built =
        let o = scratch lowered in
        Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:p0 o) o
      in
      let est = Sched.workgroup_memory_bytes ~capabilities:standin built in
      let exact =
        Sched.workgroup_memory_bytes ~capabilities:Ir.Backend_intf.no_codegen_capabilities built
      in
      Stdio.eprintf
        "seeded_params_capabilities: gate estimate %d rounded, %d exact (not part of the golden)\n\
         %!"
        est exact;
      p "the stand-in granule rounds the seeded tiles above their exact sizes" (est > exact);
      let is_p0 = Option.equal Poly.equal (Some p0) in
      (* [Some requested] where the gate refuses on workgroup memory, [None] where it admits. *)
      let gate_refusal cap =
        match
          Sched.check_hardware_limits_classified ~name:"spc_gate" ~limits:(limits_at (Some cap))
            ~capabilities:standin built
        with
        | () -> None
        | exception
            SO.Cause_at (_, SO.Resource_exceeded { resource = SO.Workgroup_memory; requested; _ })
          ->
            Some requested
      in
      p
        "the gate admits the seeded schedule at a limit of its estimate, and refuses it one byte \
         below, requesting the estimate"
        (Option.is_none (gate_refusal est)
        && Option.equal Int.equal (gate_refusal (est - 1)) (Some est));
      p "the rule handed the stand-in seeds the geometry at a limit of its gate estimate"
        (is_p0 (seeded ~capabilities:standin (Some est)));
      p
        "the rule handed the stand-in refutes it one byte below, where a rule handed no \
         capabilities still seeds it"
        ((not (is_p0 (seeded ~capabilities:standin (Some (est - 1)))))
        && is_p0 (seeded (Some (est - 1))));
      let plan ?capabilities cap =
        Autotune.seeded_plan ~min_reduction:1 ?capabilities ~backend:"cuda" ~limits:(limits_at cap)
          ~accum_prec ~static_indices:[] (scratch lowered)
      in
      (* [None] where no plan is derived, so a refutation claim cannot pass vacuously. *)
      let plan_seeds ?capabilities cap =
        match plan ?capabilities (Some cap) with
        | Some { Autotune.sp_segments = Some (_, segments); _ } ->
            Some
              (List.exists segments ~f:(function
                | Autotune.Seeded_sketch { params; _ } -> Poly.equal params p0
                | _ -> false))
        | _ -> None
      in
      let yes = Option.equal Bool.equal (Some true) and no = Option.equal Bool.equal (Some false) in
      p
        "the seeded plan handed the stand-in seeds the geometry at the gate estimate and not one \
         byte below, where a plan handed no capabilities does"
        (yes (plan_seeds ~capabilities:standin est)
        && no (plan_seeds ~capabilities:standin (est - 1))
        && yes (plan_seeds (est - 1)));
      let decides ?capabilities cap =
        match plan ?capabilities (Some cap) with
        | Some { Autotune.sp_start = Some start; _ } ->
            Some (is_p0 (start.decide (scratch lowered)))
        | _ -> None
      in
      p
        "the plan's decision rule handed the stand-in decides the geometry at the gate estimate \
         and not one byte below, where a rule handed no capabilities does"
        (yes (decides ~capabilities:standin est)
        && no (decides ~capabilities:standin (est - 1))
        && yes (decides (est - 1)))
