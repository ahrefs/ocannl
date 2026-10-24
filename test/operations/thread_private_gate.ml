(* gh-ocannl-1313: a kernel whose routine-scoped scratch exceeds the device compiler's frame limit
   is refused at [Hardware_limits], before codegen, as a typed [Thread_scratch] -- not left for the
   device compiler to refuse untyped after compiling all of it.

   What motivated it: on HIP gfx1102, [Train.tune_placements]' default-placements arm on the
   [gpt2_mini_train] step died at its base compile after 52 s, hiprtc reporting [stack frame size
   (360544) exceeds limit (262136)]. The serial base form declares 89 [Local] arrays, 360452 bytes,
   and under strict classification an untyped compiler error is fatal, so the arm ended in
   [pre-search-failure]. With the gate, the base and every whole-routine candidate are declined in
   about a second each and the arm searches what fits.

   The decision is pre-compile, so this test needs no GPU: it judges a hand-built kernel against
   MOCKED [hardware_limits] records, refusing and accepting cap side by side (a gate that refuses
   everything must not read as a pass). Two relationships are pinned rather than restated:

   - The estimate is the PURE peek of what codegen does: every scratch node's placement is still
   open when it is computed, and after a real link the arrays codegen declared add up to exactly the
   estimate, node for node in count. A peek that drifted from the forcing family -- counting an
   observable [Never_virtual] node, or one over the stack threshold, both of which codegen puts
   [On_device] -- fails here rather than as a false refusal on a device. - Asking settles nothing:
   the open placements are still open after the estimate, and no node's [size_in_bytes] is forced by
   it, nor by the sibling workgroup-memory estimate [Schedule.workgroup_memory_bytes]. HIP is the
   one backend running these checks, so a forced size is node state only HIP leaves behind.
   [Tn.header] once printed that state ([mem in bytes: <not-in-yet>] until forced), turning the
   backend-uniform [test/einsum/moons_demo_variant] golden red on HIP alone; it now prints the size
   derived from the node (gh-ocannl-1369), which the header claims below pin: the same text before
   and after the size is forced. *)

open Base
open Ll_test
open Verdict.Claims
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module BI = Ir.Backend_intf
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Generated.init ~backend_name
let mk = node_factory ~first_id:13130 ~dims:[| 64 |] ()
let never_virtual tn = Tn.update_memory_mode tn Tn.Never_virtual (Site "99:test-setup")

(* One materialized output and four scratch nodes, each filled forward and drained in REVERSE into
   the output (a forward drain lets a device compiler forward every store and drop the array):

   - [declared]: intent [Local] -- settled scratch; - [small]: [Never_virtual], 256 bytes -- settles
   [Local] under the stack threshold; - [big]: [Never_virtual], 32 KiB -- settles [On_device], over
   the default 16 KiB threshold; - [observed]: [Never_virtual] and observable -- settles [On_device]
   whatever its size.

   Every value varies with its loop symbol, so no array is a constant the optimizer can fold
   away. *)
let kernel () =
  let out = mk "tpg_out" in
  materialize out;
  let declared = mk "tpg_declared" in
  Tn.update_memory_mode declared Tn.Local (Site "99:test-setup");
  let small = mk "tpg_small" in
  never_virtual small;
  let big = mk ~dims:[| 8192 |] "tpg_big" in
  never_virtual big;
  let observed = mk "tpg_observed" in
  never_virtual observed;
  Tn.set_observable observed;
  let fill tn n =
    let s = sym () in
    loop_n s n (set tn [| iter s |] (tick s))
  in
  let y = sym () in
  let rev n = aff [ (-1, y) ] (n - 1) in
  let drain =
    loop_n y 64
      (set out
         [| iter y |]
         (add
            (add (get declared [| rev 64 |]) (get small [| rev 64 |]))
            (add (get big [| rev 8192 |]) (get observed [| rev 64 |]))))
  in
  let llc =
    List.fold_right
      [ fill declared 64; fill small 64; fill big 8192; fill observed 64 ]
      ~init:drain ~f:seq
  in
  (optimize ~materialized:[ out ] ~name:"tpg_kernel" llc, declared, small, big, observed)

let refusal ~limits opt =
  match
    Sched.check_hardware_limits_classified ~capabilities:BI.no_codegen_capabilities
      ~name:"tpg_kernel" ~limits opt
  with
  | () -> None
  | exception SO.Cause_at (phase, SO.Resource_exceeded { resource; requested; limit; _ }) ->
      Some (phase, resource, requested, limit)
  | exception _ -> None

let accepts ~limits opt =
  match
    Sched.check_hardware_limits_classified ~capabilities:BI.no_codegen_capabilities
      ~name:"tpg_kernel" ~limits opt
  with
  | () -> true
  | exception _ -> false

let private_cap n = { BI.no_hardware_limits with max_thread_private_bytes = Some n }

(* The arrays a generated kernel declares at function scope, as (count, bytes): the lines between
   codegen's [Local declarations] and [Main logic] markers, each [<type> <ident>[<n>]]. Every node
   here is single precision. *)
let declared_arrays src =
  let lines = String.split_lines src in
  let rec after_marker = function
    | [] -> []
    | l :: rest ->
        if String.is_substring l ~substring:"Local declarations and initialization" then rest
        else after_marker rest
  in
  let section =
    List.take_while (after_marker lines) ~f:(fun l ->
        not (String.is_substring l ~substring:"Main logic"))
  in
  List.fold section ~init:(0, 0) ~f:(fun (count, bytes) l ->
      match String.lsplit2 (String.strip l) ~on:'[' with
      | Some (decl, rest) when String.contains decl ' ' -> (
          match String.lsplit2 rest ~on:']' with
          | Some (n, _) when (not (String.is_empty n)) && String.for_all n ~f:Char.is_digit ->
              (count + 1, bytes + (4 * Int.of_string n))
          | _ -> (count, bytes))
      | _ -> (count, bytes))

let () =
  let opt, declared, small, big, observed = kernel () in
  let plc = opt.LL.optimize_ctx.LL.placements in
  let size tn = Lazy.force tn.Tn.size_in_bytes in
  let scratch = [ declared; small; big; observed ] in
  let estimate = Sched.thread_private_bytes opt in
  (* Every scratch node as a workgroup-shared tile: the capability-free workgroup estimate then
     counts exactly their sizes. *)
  let wg_estimate =
    Sched.workgroup_memory_bytes ~capabilities:BI.no_codegen_capabilities
      { opt with LL.workgroup_shared = Set.of_list (module Tn) scratch }
  in
  (* Read before anything below forces a size through [size]: each node with whether its size was
     forced by then, and its header at that point. *)
  let sizes_after_estimate = List.map scratch ~f:(fun tn -> Lazy.is_val tn.Tn.size_in_bytes) in
  let headers_unforced = List.map scratch ~f:Tn.header in

  (* The premises: which scratch node settles where. Without them the estimate below could match for
     the wrong reason, e.g. every node silently virtualized. *)
  p "premise: big is over the stack threshold, so a Never_virtual big settles On_device"
    (Tn.equal_memory_mode (Tn.most_local_materialized_mode big) Tn.On_device);
  p "premise: small is under it, so it settles Local"
    (Tn.equal_memory_mode (Tn.most_local_materialized_mode small) Tn.Local);
  p_all "premise: every scratch node is in the kernel's traced store"
    [ declared; small; big; observed ]
    ~f:(Hashtbl.mem opt.LL.traced_store);
  p_all "premise: the Never_virtual nodes are still open before codegen" [ small; big; observed ]
    ~f:(Tn.Placements.mode_is_unspecified plc);

  (* The estimate, from the pure peek. *)
  p "estimate: the Local and the under-threshold nodes, nothing else"
    (estimate = size declared + size small);
  p_all "estimate: asking settles nothing" [ small; big; observed ]
    ~f:(Tn.Placements.mode_is_unspecified plc);
  p_none
    "estimate: neither thread_private_bytes nor workgroup_memory_bytes forces a node's \
     size_in_bytes"
    sizes_after_estimate ~f:Fn.id;
  p "estimate: a workgroup-shared tile is declared per workgroup, not counted per work-item"
    (Sched.thread_private_bytes
       { opt with LL.workgroup_shared = Set.singleton (module Tn) declared }
    = size small);
  p "workgroup estimate: the shared tiles' sizes, nothing else"
    (wg_estimate = List.sum (module Int) scratch ~f:size);

  (* The header reports the node, not the lazy's state: every size is forced by now. *)
  p_all "premise: every scratch node's size_in_bytes is forced by now" scratch ~f:(fun tn ->
      Lazy.is_val tn.Tn.size_in_bytes);
  p_all "header: unchanged by forcing size_in_bytes, and printing the size it forced to"
    (List.zip_exn scratch headers_unforced) ~f:(fun (tn, unforced) ->
      String.equal unforced (Tn.header tn)
      && String.is_substring unforced ~substring:("mem in bytes: " ^ Int.to_string_hum (size tn)));

  (* The gate, refusing and accepting side by side. *)
  p "gate: no cap accepts" (accepts opt ~limits:BI.no_hardware_limits);
  p "gate: a cap at the estimate accepts" (accepts opt ~limits:(private_cap estimate));
  p "gate: a cap one below refuses at Hardware_limits as a typed Thread_scratch excess"
    (match refusal opt ~limits:(private_cap (estimate - 1)) with
    | Some (SO.Hardware_limits, SO.Thread_scratch, requested, Some limit) ->
        requested = estimate && limit = estimate - 1
    | _ -> false);

  (* The relationship: what codegen actually declares. A real link settles the placements by the
     forcing family, so the arrays in the emitted source are its verdict, read beside the peek's. *)
  let _ctx, _routine = link ~name:"tpg_kernel" opt in
  let count, bytes = declared_arrays (Generated.read "tpg_kernel") in
  p "codegen: the declared arrays add up to the estimate" (bytes = estimate);
  p "codegen: one array per counted node" (count = 2);
  p "codegen: the forcing family settled the peek's two as Local and the other two On_device"
    (known_local opt small
    && Tn.Placements.is_materialized_peek plc big
    && Tn.Placements.is_materialized_peek plc observed);
  Stdio.print_endline "thread_private_gate: done"
