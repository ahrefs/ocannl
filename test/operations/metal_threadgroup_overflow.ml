(* gh-ocannl-1226, gh-ocannl-1330: Metal's threadgroup-memory accounting, at the device limit [L],
   on both sides of the schedule-level gate.

   Metal rounds every [threadgroup] array up to 4 bytes (measured on an M4 Max). So two [half] tiles
   of odd length whose exact bytes total [L] need [L + 4] — what pipeline creation requests. Since
   gh-ocannl-1330 [Schedule.workgroup_memory_bytes] rounds each allocation by the backend's
   [workgroup_allocation_alignment] the same way, and the gate refuses them before compiling
   anything; before it the estimate summed exact sizes, let them through, and only pipeline creation
   refused them.

   The kernel is the outer sum [c[i, j] = u[i] + v[j]] with [i] split into [Grid x Workgroup] blocks
   of [bi]: [u] is staged over the workgroup axis (a [bi]-element tile), [v] over the serial [j]
   loop (a [J]-element tile), [bi + J = L / 2] in every leg, so the exact sum is [L] in every leg.

   C, the control: [bi] and [J] even. Each tile is a multiple of 4 bytes, the rounded estimate is
   exactly [L], and the kernel compiles, links and runs at the limit.

   G, the gate (gh-ocannl-1330): [bi] and [J] odd, one element moved between the tiles. The rounded
   estimate is over [L], and the schedule is refused at [Hardware_limits] requesting exactly it. Run
   against the backend before gh-ocannl-1330, leg G compiled through to the driver's refusal.

   O, the overflow behind the gate (gh-ocannl-1226): leg G's kernel with the gate judging by exact
   sizes ([Schedule.with_gate_capabilities] sets the alignment to 1 — the pre-gh-ocannl-1330
   estimate, the one route left past an estimate that is now exact). Pipeline creation refuses the
   kernel, and the refusal must arrive as a contained [Resource_exceeded Workgroup_memory] at
   [Backend_link], carrying the driver's limit (the device limit the backend reports) and its
   request, which is leg G's rounded estimate — the rounding model is the driver's — and, through
   the public [Context.compile], as the [Utils.User_error] a hand-written schedule gets for any
   resource refusal. Run against the backend before gh-ocannl-1226, leg O came back [Fatal].

   Metal only: the other GPU backends' shared-memory rounding is not this test's subject, and the C
   backends cannot express shared placement. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_metal = String.is_substring backend_name ~substring:"metal"
let half_bytes = Ir.Ops.prec_in_bytes Ir.Ops.half

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Small integers, exact in half, varying with both output symbols. *)
let u_value i = Float.of_int ((i % 7) + 1)
let v_value j = Float.of_int ((j % 11) - 5)

type 'r leg = {
  result : 'r;
  estimate : int;
      (** [Sched.workgroup_memory_bytes] of the transformed code under the backend's capabilities:
          each allocation rounded. *)
  exact : int;  (** The same with the alignment 1: the allocations' exact sum. *)
  c : Tensor.t;
  rows : int;
  cols : int;
}

(* The gate's view in leg O: exact sizes, as before gh-ocannl-1330. *)
let exact_sizes (caps : Ir.Backend_intf.codegen_capabilities) =
  { caps with Ir.Backend_intf.workgroup_allocation_alignment = 1 }

let schedule ~bi ~u ~v (opt : LL.optimized) : Sched.schedule =
  let i, j =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 2) with
    | [ i; j ] -> (i, j)
    | _ -> assert false
  in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bi ~outer:LL.Grid ~inner:LL.Workgroup in
  let stage source tile_loops =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = true;
        cooperative = None;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      }
  in
  [ sp_i; stage u [ i_i ]; stage v [ j ] ]

(* The leg's compile, handed to [compile] with the context, the named computation and the transform;
   the transform records the estimate of the code it hands on. *)
let leg ~name ~bi ~cols compile =
  let rows = 2 * bi in
  let u =
    NTDSL.init ~l:(name ^ "_u") ~prec:Ir.Ops.half ~o:[ rows ] ~f:(fun idcs -> u_value idcs.(0)) ()
  in
  let v =
    NTDSL.init ~l:(name ^ "_v") ~prec:Ir.Ops.half ~o:[ cols ] ~f:(fun idcs -> v_value idcs.(0)) ()
  in
  let%op c = u +++ "i; j => ij" v in
  let ctx = Context.auto () in
  let caps = Context.codegen_capabilities ctx in
  let estimate = ref (-1) and exact = ref (-1) in
  let lowered_transform opt =
    let o = Sched.apply (schedule ~bi ~u:u.Tensor.value ~v:v.Tensor.value opt) opt in
    estimate := Sched.workgroup_memory_bytes ~capabilities:caps o;
    exact := Sched.workgroup_memory_bytes ~capabilities:(exact_sizes caps) o;
    [ o ]
  in
  let result = compile ~lowered_transform ctx (named name (Train.forward c)) in
  { result; estimate = !estimate; exact = !exact; c; rows; cols }

(* [~exact_gate]: leg O's route past the gate. *)
let gated ~exact_gate f = if exact_gate then Sched.with_gate_capabilities exact_sizes ~f else f ()

let compile_leg ?(exact_gate = false) ~name ~bi ~cols () =
  gated ~exact_gate (fun () ->
      leg ~name ~bi ~cols (fun ~lowered_transform ctx comp ->
          Context.compile_outcome ~lowered_transform ~provenance:SO.Candidate ~candidate:name ctx
            comp Ir.Indexing.Empty))

(* The public boundary: a hand-written schedule, whose refusal is rendered into an exception. *)
let public_leg ~name ~bi ~cols =
  let { result; _ } =
    gated ~exact_gate:true (fun () ->
        leg ~name ~bi ~cols (fun ~lowered_transform ctx comp ->
            match Context.compile ~lowered_transform ctx comp Ir.Indexing.Empty with
            | (_ : Context.t * Context.routine) -> Ok ()
            | exception exn -> Error exn))
  in
  result

let describe (leg : (Context.t * Context.routine) SO.outcome leg) =
  match leg.result with
  | Ok _ -> "compiled"
  | Error (SO.Classified c) -> Sexp.to_string_hum (SO.sexp_of_classified_cause c)
  | Error (SO.Fatal { exn; phase; _ }) ->
      "fatal at " ^ Sexp.to_string (SO.sexp_of_phase phase) ^ ": " ^ Exn.to_string exn

let claim_control =
  "C: even-length half tiles estimated at exactly the limit compile, link and run correctly"

let claim_gate =
  "G: odd-length half tiles whose exact bytes total the limit are refused at Hardware_limits, \
   requesting their rounded estimate, over the limit"

let claim_estimate = "O: with exact sizes the gate estimates the odd tiles at exactly the limit"

let claim_refused =
  "O: pipeline creation's refusal is a contained Workgroup_memory decline at Backend_link, \
   requesting leg G's rounded estimate over the device limit"

let claim_public = "O: the public compile reports the refusal as a Utils.User_error"
let all_claims = [ claim_control; claim_gate; claim_estimate; claim_refused; claim_public ]

let () =
  let limits = Context.hardware_limits (Context.auto ()) in
  match (on_metal, limits.Ir.Backend_intf.max_workgroup_memory_bytes) with
  | true, Some limit when limit % (4 * half_bytes) = 0 ->
      let elems = limit / half_bytes in
      (* C: [bi] even, so [J = elems - bi] is even too ([elems] is a multiple of 4). *)
      let control = compile_leg ~name:"tgo_control" ~bi:32 ~cols:(elems - 32) () in
      let values =
        match control.result with
        | Ok (ctx, routine) ->
            Some (Context.get_values (Context.run ctx routine) control.c.Tensor.value)
        | Error _ ->
            Stdio.eprintf "metal_threadgroup_overflow: C: %s\n%!" (describe control);
            None
      in
      let expected =
        Array.init (control.rows * control.cols) ~f:(fun flat ->
            u_value (flat / control.cols) +. v_value (flat % control.cols))
      in
      p claim_control
        (control.estimate = limit && control.exact = limit
        && Option.value_map values ~default:false ~f:(fun got ->
            Array.equal Float.equal got expected));
      (* G: one element moved from [v]'s tile to [u]'s, both odd. *)
      let gate = compile_leg ~name:"tgo_gate" ~bi:33 ~cols:(elems - 33) () in
      let gate_refused =
        match gate.result with
        | Error
            (SO.Classified
               {
                 phase = SO.Hardware_limits;
                 cause =
                   SO.Resource_exceeded
                     { resource = SO.Workgroup_memory; requested; limit = Some l; _ };
                 _;
               }) ->
            gate.exact = limit && requested = gate.estimate && requested > limit && l = limit
        | _ ->
            Stdio.eprintf "metal_threadgroup_overflow: G: %s\n%!" (describe gate);
            false
      in
      p claim_gate gate_refused;
      (* O: leg G's kernel, judged by exact sizes. *)
      let overflow =
        compile_leg ~exact_gate:true ~name:"tgo_overflow" ~bi:33 ~cols:(elems - 33) ()
      in
      p claim_estimate (overflow.exact = limit);
      let refused =
        match overflow.result with
        | Error
            (SO.Classified
               {
                 phase = SO.Backend_link;
                 cause =
                   SO.Resource_exceeded
                     { resource = SO.Workgroup_memory; requested; limit = Some l; _ };
                 execution_effect = SO.No_device_writes;
               }) ->
            l = limit && requested > limit && requested = gate.estimate
        | _ -> false
      in
      (* The driver's counts, for the record (not part of the golden). *)
      Stdio.eprintf "metal_threadgroup_overflow: O: %s\n%!" (describe overflow);
      p claim_refused refused;
      let public = public_leg ~name:"tgo_public" ~bi:33 ~cols:(elems - 33) in
      p claim_public
        (match public with
        | Error (Utils.User_error msg) -> String.is_substring msg ~substring:"threadgroup memory"
        | Error exn ->
            Stdio.eprintf "metal_threadgroup_overflow: public: %s\n%!" (Exn.to_string exn);
            false
        | Ok () ->
            Stdio.eprintf "metal_threadgroup_overflow: public: compiled\n%!";
            false)
  | _ -> List.iter all_claims ~f:skipped
