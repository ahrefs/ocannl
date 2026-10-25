(* gh-ocannl-1520: the dominant kernel's [bytes] counts global memory only.

   [Bench_json.dominant_kernel_object] scores the dominant kernel's byte count against a DRAM
   bandwidth ceiling, so a staged kernel charged its workgroup tiles and thread-private scratch read
   above its true attainment. [Bench_harness.dominant_kernel_record] -- the record the
   dominant-kernel instrument fills -- now takes its bytes from [Bench_harness.global_counts], the
   accounting the kernel table's [io:] field already used (gh-ocannl-1444).

   Hand-built IR: an untiled weight gradient scheduled by the GPU sketch family under a GPU-shaped
   [hardware_limits] record, as test/operations/cost_model_reuse does; nothing is compiled or
   executed. The claims: on a staged blocktile the record's bytes are below the whole-kernel
   [Ir.Cost_model.analyze] total by exactly the bytes of the nodes the ranking score's reuse
   extraction keeps off the DRAM leg ([ru_scratch], derived independently of the harness), and equal
   the [io:] field's read plus written; the op count stays the whole kernel's; and on the unstaged
   default plan nothing is left out. *)

open Base
module LL = Ir.Low_level
module Sched = Ir.Schedule
module CM = Ir.Cost_model
module Tn = Ir.Tnode
module H = Bench_harness
open Verdict.Claims

let nk = 256
and nm = 128
and nn = 128

let node = Ll_test.node_factory ~first_id:152000 ~dims:[| 1 |] ()
let dy = node ~dims:[| nk; nm |] "dkb_dy"
let x = node ~dims:[| nk; nn |] "dkb_x"
let dw = node ~dims:[| nm; nn |] "dkb_dw"
let () = List.iter [ dy; x; dw ] ~f:Ll_test.materialize
let it = Ll_test.iter

(* [dw[o,i] += dy[k,o] * x[k,i]], loops (o, i, k): the contraction innermost, as the matmul
   detection reads it. *)
let wgrad () =
  let o = Ll_test.sym () and i = Ll_test.sym () and k = Ll_test.sym () in
  Ll_test.loop_n o nm
    (Ll_test.loop_n i nn
       (Ll_test.loop_n k nk
          (Ll_test.set dw
             [| it o; it i |]
             (LL.Ternop
                ( Ir.Ops.FMA,
                  (Ll_test.get dy [| it k; it o |], Ll_test.single),
                  (Ll_test.get x [| it k; it i |], Ll_test.single),
                  (Ll_test.get dw [| it o; it i |], Ll_test.single) )))))

let limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    max_threads_per_workgroup = Some 1024;
    max_workgroup_memory_bytes = Some 49152;
    max_workgroup_dims = Some (1024, 1024, 64);
    max_grid_yz = Some 65535;
    simdgroup_width = Some 32;
  }

let base = Ll_test.optimize ~name:"dkb_wgrad" (wgrad ())

let hermetic (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let scheduled sched = Sched.apply sched (hermetic base)
let untiled = scheduled (Sched.default_gpu ~limits (hermetic base))

let applied q =
  match scheduled (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q (hermetic base)) with
  | o -> Some o
  | exception (Invalid_argument _ | Ir.Schedule_outcome.Cause_at _) -> None

let plans =
  List.filter_map (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits base) ~f:applied

let record (o : LL.optimized) =
  H.dominant_kernel_record ~segment:0 ~segments:1 ~declined:0 ~seg_ms:1. ~segments_ms:1.
    ~tensorization:"none" o

(* The nodes the ranking score keeps off the DRAM leg, and their whole-kernel bytes. *)
let scratch (o : LL.optimized) =
  (CM.reuse_traffic ~is_global:(CM.global_in o) o.LL.llc).CM.ru_scratch

let scratch_bytes (o : LL.optimized) (s : CM.summary) =
  List.sum
    (module Int)
    (scratch o)
    ~f:(fun tn ->
      List.Assoc.find s.CM.per_node tn ~equal:Tn.equal
      |> Option.value_map ~default:0 ~f:(fun fp -> fp.CM.fp_read_bytes + fp.CM.fp_write_bytes))

(* The kernel table's [io:<read>/<written>] field, read back as the bytes it names (its trailing
   [~]/[?] flags set aside). *)
let io_total (o : LL.optimized) =
  let flag c = Char.equal c '~' || Char.equal c '?' in
  match String.chop_prefix (H.segment_io o) ~prefix:"io:" with
  | None -> None
  | Some rest -> (
      match String.split (String.rstrip rest ~drop:flag) ~on:'/' with
      | [ r; w ] ->
          Option.map
            (Option.both (Int.of_string_opt r) (Int.of_string_opt w))
            ~f:(fun (r, w) -> r + w)
      | _ -> None)

let () =
  let staged = List.filter plans ~f:(fun o -> not (List.is_empty (scratch o))) in
  Stdio.eprintf "dominant_kernel_bytes: %d plans, %d staged (not part of the golden)\n"
    (List.length plans) (List.length staged);
  List.iter staged ~f:(fun o ->
      let k, s = record o in
      Stdio.eprintf
        "  staged: bytes=%d analyze total=%d scratch=%d io=%s (not part of the golden)\n"
        k.Bench_json.bytes (CM.total_bytes s) (scratch_bytes o s) (H.segment_io o));
  p_all ~min:1 "a staged blocktile's dominant-kernel bytes are below the whole-kernel total" staged
    ~f:(fun o ->
      let k, s = record o in
      k.Bench_json.bytes < CM.total_bytes s);
  p_all ~min:1
    "a staged blocktile's dominant-kernel bytes leave out exactly its on-chip nodes' bytes" staged
    ~f:(fun o ->
      let k, s = record o in
      k.Bench_json.bytes + scratch_bytes o s = CM.total_bytes s);
  p_all ~min:1
    "the dominant-kernel bytes are the kernel table's io: read plus written, on every plan"
    (untiled :: plans) ~f:(fun o ->
      let k, _ = record o in
      Option.equal Int.equal (Some k.Bench_json.bytes) (io_total o));
  p_all ~min:1 "the dominant kernel's op count stays the whole kernel's, on every plan"
    (untiled :: plans) ~f:(fun o ->
      let k, s = record o in
      k.Bench_json.flops = s.CM.flops);
  p_none "the unstaged default plan touches no on-chip node" (CM.analyze untiled.LL.llc).CM.per_node
    ~f:(fun (tn, _) -> not (CM.global_in untiled tn));
  p "on the unstaged default plan nothing is left out"
    (let k, s = record untiled in
     k.Bench_json.bytes = CM.total_bytes s)
