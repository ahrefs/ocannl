(* gh-ocannl-1226: a kernel whose threadgroup memory is over the device limit, but which the
   schedule-level estimate lets through, comes back from Metal's pipeline creation as a typed
   [Backend_link] refusal — an ordinary decline to a search — instead of an untyped [Failure] that
   strict classification makes fatal.

   The route past the estimate is a real one, measured on an M4 Max: Metal rounds every
   [threadgroup] array up to 4 bytes, while [Schedule.workgroup_memory_bytes] sums the staged tiles'
   exact sizes. So two [half] tiles of odd length whose exact bytes total the limit [L] pass the
   [Hardware_limits] gate, and the pipeline needs [L + 4].

   The kernel is the outer sum [c[i, j] = u[i] + v[j]] with [i] split into [Grid x Workgroup] blocks
   of [bi]: [u] is staged over the workgroup axis (a [bi]-element tile), [v] over the serial [j]
   loop (a [J]-element tile), [bi + J = L / 2] in both legs, so the estimate is exactly [L] in both.

   C, the control: [bi] and [J] even. Each tile is a multiple of 4 bytes, the pipeline needs exactly
   [L], and the kernel compiles, links and runs at the limit.

   O, the overflow: [bi] and [J] odd, one element moved between the tiles. The estimate is still
   [L]; pipeline creation refuses the kernel, and the refusal must arrive as a contained
   [Resource_exceeded Workgroup_memory] at [Backend_link], carrying the driver's limit (the device
   limit the backend reports) and a request over it — and, through the public [Context.compile], as
   the [Utils.User_error] a hand-written schedule gets for any resource refusal. Run against the
   backend before gh-ocannl-1226, leg O came back [Fatal].

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

type leg = {
  result : (Context.t * Context.routine) SO.outcome;
  estimate : int;  (** [Sched.workgroup_memory_bytes] of the transformed code. *)
  c : Tensor.t;
  rows : int;
  cols : int;
}

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
  let estimate = ref (-1) in
  let lowered_transform opt =
    let o = Sched.apply (schedule ~bi ~u:u.Tensor.value ~v:v.Tensor.value opt) opt in
    estimate := Sched.workgroup_memory_bytes ~capabilities:caps o;
    [ o ]
  in
  let result = compile ~lowered_transform ctx (named name (Train.forward c)) in
  (result, !estimate, c, rows, cols)

let compile_leg ~name ~bi ~cols =
  let result, estimate, c, rows, cols =
    leg ~name ~bi ~cols (fun ~lowered_transform ctx comp ->
        Context.compile_outcome ~lowered_transform ~provenance:SO.Candidate ~candidate:name ctx comp
          Ir.Indexing.Empty)
  in
  { result; estimate; c; rows; cols }

(* The public boundary: a hand-written schedule, whose refusal is rendered into an exception. *)
let public_leg ~name ~bi ~cols =
  let result, _, _, _, _ =
    leg ~name ~bi ~cols (fun ~lowered_transform ctx comp ->
        match Context.compile ~lowered_transform ctx comp Ir.Indexing.Empty with
        | (_ : Context.t * Context.routine) -> Ok ()
        | exception exn -> Error exn)
  in
  result

let describe leg =
  match leg.result with
  | Ok _ -> "compiled"
  | Error (SO.Classified c) -> Sexp.to_string_hum (SO.sexp_of_classified_cause c)
  | Error (SO.Fatal { exn; phase; _ }) ->
      "fatal at " ^ Sexp.to_string (SO.sexp_of_phase phase) ^ ": " ^ Exn.to_string exn

let claim_control =
  "C: even-length half tiles estimated at exactly the limit compile, link and run correctly"

let claim_estimate = "O: odd-length half tiles are estimated at exactly the limit too"

let claim_refused =
  "O: pipeline creation's refusal is a contained Workgroup_memory decline at Backend_link, over \
   the device limit"

let claim_public = "O: the public compile reports the refusal as a Utils.User_error"
let all_claims = [ claim_control; claim_estimate; claim_refused; claim_public ]

let () =
  let limits = Context.hardware_limits (Context.auto ()) in
  match (on_metal, limits.Ir.Backend_intf.max_workgroup_memory_bytes) with
  | true, Some limit when limit % (4 * half_bytes) = 0 ->
      let elems = limit / half_bytes in
      (* C: [bi] even, so [J = elems - bi] is even too ([elems] is a multiple of 4). *)
      let control = compile_leg ~name:"tgo_control" ~bi:32 ~cols:(elems - 32) in
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
        (control.estimate = limit
        && Option.value_map values ~default:false ~f:(fun got ->
            Array.equal Float.equal got expected));
      (* O: one element moved from [v]'s tile to [u]'s, both odd. *)
      let overflow = compile_leg ~name:"tgo_overflow" ~bi:33 ~cols:(elems - 33) in
      p claim_estimate (overflow.estimate = limit);
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
            l = limit && requested > limit
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
