(* gh-ocannl-1205: the schedule-level shared-memory estimate counts the workgroup memory a tile-MMA
   emission scope declares beside the staged tiles. Metal's converted destination boundary (a float
   accumulator over half storage under [Fp16_wide], gh-ocannl-1075) initializes a coordinate table
   in [threadgroup] memory once per scope. Before this the estimate summed the staged tiles alone,
   so a candidate that fit them but not the table reached backend compilation, where
   [newComputePipelineStateWithFunction] refused it with an untyped [Failure] before the post-link
   allocation check could classify it: a fatal to a search, not a decline (measured on an M4 Max;
   the classification gap is gh-ocannl-1226).

   The legs work at the device's own limit [L], on a staged + tensorized uniform-half matmul whose
   accumulator is a resident fragment across two [k_o] blocks (one converted scope).

   R, the refusal: staged tiles of exactly [L] bytes. Under [Fp16_auto] the triple converts nothing,
   so the kernel compiles, links and runs at the limit, tensorized (the control: the tiles alone fit
   and the staged accounting is exact there). Under [Fp16_wide] the same schedule is refused at
   [Hardware_limits], before any backend compilation, requesting [L + scratch], where [scratch] is
   the backend's own [mma_scope_workgroup_bytes] for the triple (positive on Metal).

   F, the fit at the limit: under [Fp16_wide], tiles of [L - scratch] bytes (an A-tile stride pad
   makes the sum land exactly), so the estimate is exactly [L]. It compiles, links and runs, so the
   compiled kernel's own static allocation is not above the estimate (Metal's pipeline creation
   would refuse it otherwise). The emitted source declares exactly as many tables as the estimate
   counted scopes.

   The scratch size is never restated here: it is read off the capability the emitter derives it
   from. Only Metal converts at a threadgroup-resident boundary; elsewhere the legs are skipped. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_metal = String.is_substring backend_name ~substring:"metal"

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The lane width, the row block and the column extent. The zeroing nest's column loop is the
   [Workgroup] axis, and barrier-strength uniformity requires it to equal the lane loop's extent
   once a [Tile_mma] is present, so the column extent is the width. *)
let simd_width = 32
let bm = 32
let n = simd_width
let m = 2 * bm
let half_bytes = Ir.Ops.prec_in_bytes Ir.Ops.half

(* Staged bytes of the two tiles: A is [bm x bk] (minor dim padded to [a_ld]), B is [bk x n]. *)
let staged_bytes ~bk ~a_ld = half_bytes * ((bm * a_ld) + (bk * n))

(* The staged + tensorized composition of schedule_mma_matmul (lane-aware [Stage] of both operands
   at [k_o], the [i_i x j x k_i] micro-kernel tensorized), with the k block [bk] and an optional
   stride pad on the A tile. *)
let staged_schedule ~bk ~a_pad ~out ~src_a ~src_b (opt : LL.optimized) : Sched.schedule =
  let paths = Ll_test.nest_paths opt.LL.llc in
  let i, j, k =
    match List.find_exn paths ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
  let stage ?pad_stride source tile_loops =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = true;
        cooperative = Some simd_width;
        hoisted = false;
        swizzle = None;
        pad_stride;
        pipeline_depth = 1;
        tile_prec = None;
      }
  in
  [
    ez;
    sp_zi;
    rz;
    sp_i;
    sp_k;
    Sched.Swap { outer = j; inner = k_o };
    Sched.Swap { outer = i_i; inner = k_o };
    stage ?pad_stride:(Option.map a_pad ~f:(fun pad -> bk + pad)) src_a [ i_i; k_i ];
    stage src_b [ k_i; j ];
    tz;
  ]

(* [c = a * b] over [k = 2 * bk]: [a] selects row [i]'s own [k = i] (an identity block, [m <= k]),
   so [c[i, j] = b[i, j]] — small integers, exact in half under either policy and varying with both
   output symbols, so a cell computed from the wrong row or column is caught. *)
let b_value ~row ~col = Float.of_int (((row + (2 * col)) % 17) + 1)
let expected = Array.init (m * n) ~f:(fun flat -> b_value ~row:(flat / n) ~col:(flat % n))

type leg = {
  routine : string;
  result : (Context.t * Context.routine) SO.outcome;
  estimate : int;  (** [Sched.workgroup_memory_bytes] of the transformed code, under its policy. *)
  out : Tensor.t;
}

let compile_leg ~name ~fp16 ~bk ~a_pad =
  let k = 2 * bk in
  let a =
    NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.half ~i:[ k ] ~o:[ m ]
      ~f:(fun idcs -> if idcs.(0) = idcs.(1) then 1. else 0.)
      ()
  in
  let b =
    NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.half ~i:[ n ] ~o:[ k ]
      ~f:(fun idcs -> b_value ~row:idcs.(0) ~col:idcs.(1))
      ()
  in
  let saved = Numerics.get () in
  Numerics.set_policy { saved with fp16_arithmetic = fp16 };
  let%op c = a * b in
  Tn.update_prec c.Tensor.value Ir.Ops.half;
  let ctx = Context.auto () in
  let ctx_caps = Context.codegen_capabilities ctx in
  let estimate = ref (-1) in
  let transform opt =
    let o =
      Sched.apply
        (staged_schedule ~bk ~a_pad ~out:c.Tensor.value ~src_a:a.Tensor.value ~src_b:b.Tensor.value
           opt)
        opt
    in
    (* Read under the leg's policy: the triple converts only under [Fp16_wide]. *)
    estimate := Sched.workgroup_memory_bytes ~capabilities:ctx_caps o;
    o
  in
  let result =
    Exn.protect
      ~f:(fun () ->
        Context.compile_outcome
          ~lowered_transform:(fun o -> [ transform o ])
          ~provenance:SO.Candidate ~candidate:name ctx
          (named name (Train.forward c))
          Ir.Indexing.Empty)
      ~finally:(fun () -> Numerics.set_policy saved)
  in
  { routine = name; result; estimate = !estimate; out = c }

let run_values leg =
  match leg.result with
  | Ok (ctx, routine) -> Some (Context.get_values (Context.run ctx routine) leg.out.value)
  | _ -> None

let describe_failure leg =
  match leg.result with
  | Ok _ -> "compiled"
  | Error (SO.Classified c) -> Sexp.to_string_hum (SO.sexp_of_classified_cause c)
  | Error (SO.Fatal { exn; _ }) -> "fatal: " ^ Exn.to_string exn

let claim_r_control =
  "R control: tiles of exactly the limit, no converted boundary, compile and run"

let claim_r_control_source =
  "R control: the unconverted kernel is tensorized and declares no coordinate table"

let claim_scratch = "the converted half boundary charges a positive per-scope scratch"

let claim_r_refused =
  "R wide: the same tiles plus the scope scratch are refused at Hardware_limits, requesting \
   exactly tiles + scratch"

let claim_f_estimate = "F wide: tiles of limit - scratch make the estimate exactly the limit"
let claim_f_runs = "F wide: the at-limit estimate compiles, links and runs"
let claim_f_scopes = "F wide: the source declares one table per scope the estimate counted"

let all_claims =
  [
    claim_scratch;
    claim_r_control;
    claim_r_control_source;
    claim_r_refused;
    claim_f_estimate;
    claim_f_runs;
    claim_f_scopes;
  ]

let () =
  let limits = Context.hardware_limits (Context.auto ()) in
  let mma = Option.is_some limits.Ir.Backend_intf.mma in
  match (on_metal && mma, limits.max_workgroup_memory_bytes) with
  | true, Some limit ->
      let scratch =
        let saved = Numerics.get () in
        Numerics.set_policy { saved with fp16_arithmetic = Numerics.Fp16_wide };
        Exn.protect
          ~f:(fun () ->
            (Context.codegen_capabilities (Context.auto ())).mma_scope_workgroup_bytes
              ~d_prec:Ir.Ops.half ~a_prec:Ir.Ops.half ~b_prec:Ir.Ops.half)
          ~finally:(fun () -> Numerics.set_policy saved)
      in
      p claim_scratch (scratch > 0);
      (* R: [bk] so that the unpadded tiles total exactly [limit]. *)
      let tile_row = half_bytes * (bm + n) in
      let bk_r = limit / tile_row in
      (* F: one k block narrower, the A tile's stride padded until the tiles total [limit -
         scratch]. *)
      let bk_f = bk_r - 8 in
      let pad_bytes = limit - scratch - staged_bytes ~bk:bk_f ~a_ld:bk_f in
      let a_pad = pad_bytes / (half_bytes * bm) in
      let shapes_ok =
        limit % tile_row = 0
        && bk_r % 8 = 0
        && bk_f > 0 && pad_bytes > 0
        && pad_bytes % (half_bytes * bm) = 0
        && staged_bytes ~bk:bk_f ~a_ld:(bk_f + a_pad) = limit - scratch
        && m <= 2 * bk_f
      in
      if not shapes_ok then (
        (* The arithmetic above assumes the M-series limit (32 KiB) and a scratch that is a whole
           number of padded A-tile columns; a device where it does not land says so rather than
           testing a different boundary. *)
        Stdio.eprintf
          "schedule_mma_scope_scratch: no exact near-limit shapes for limit %d, scratch %d\n%!"
          limit scratch;
        List.iter all_claims ~f:(fun c -> if not (String.equal c claim_scratch) then skipped c))
      else
        let r_auto =
          compile_leg ~name:"scratch_r_auto" ~fp16:Numerics.Fp16_auto ~bk:bk_r ~a_pad:None
        in
        let r_values = run_values r_auto in
        if Option.is_none r_values then
          Stdio.eprintf "schedule_mma_scope_scratch: R control: %s\n%!" (describe_failure r_auto);
        p claim_r_control
          (r_auto.estimate = limit
          && Option.value_map r_values ~default:false ~f:(fun got ->
              Array.equal Float.equal got expected));
        (* Without the intrinsic census a scalar-fallback rendering would also lack the table. *)
        let tensorized =
          match r_auto.result with
          | Ok (_, routine) ->
              let census = List.map routine.Context.mma.Ir.C_syntax.renderings ~f:snd in
              (not (List.is_empty census))
              && List.for_all census ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_intrinsics)
          | Error _ -> false
        in
        p claim_r_control_source
          (Option.is_some r_values && tensorized
          && not (String.is_substring (Generated.read r_auto.routine) ~substring:"ocannl_mma_rc8"));
        let r_wide =
          compile_leg ~name:"scratch_r_wide" ~fp16:Numerics.Fp16_wide ~bk:bk_r ~a_pad:None
        in
        let refused =
          match r_wide.result with
          | Error
              (SO.Classified
                 {
                   phase = SO.Hardware_limits;
                   cause =
                     SO.Resource_exceeded
                       { resource = SO.Workgroup_memory; requested; limit = Some l; _ };
                   _;
                 }) ->
              requested = limit + scratch && l = limit && r_wide.estimate = requested
          | _ ->
              Stdio.eprintf "schedule_mma_scope_scratch: R wide: %s\n%!" (describe_failure r_wide);
              false
        in
        p claim_r_refused refused;
        let f_wide =
          compile_leg ~name:"scratch_f_wide" ~fp16:Numerics.Fp16_wide ~bk:bk_f ~a_pad:(Some a_pad)
        in
        p claim_f_estimate (f_wide.estimate = limit);
        let f_values = run_values f_wide in
        if Option.is_none f_values then
          Stdio.eprintf "schedule_mma_scope_scratch: F wide: %s\n%!" (describe_failure f_wide);
        p claim_f_runs
          (Option.value_map f_values ~default:false ~f:(fun got ->
               Array.equal Float.equal got expected));
        p claim_f_scopes
          (Option.is_some f_values
          &&
          let src = Generated.read f_wide.routine in
          let tables =
            List.length
              (String.substr_index_all src ~may_overlap:false
                 ~pattern:"threadgroup float ocannl_mma_rc8[")
          in
          tables >= 1
          && tables * scratch = f_wide.estimate - staged_bytes ~bk:bk_f ~a_ld:(bk_f + a_pad))
  | _ -> List.iter all_claims ~f:skipped
