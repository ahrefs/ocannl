(* gh-ocannl-1452: the workgroup-memory estimate ([Schedule.workgroup_memory_bytes]) charges exactly
   what the kernels declare.

   1. Under: the multi-warp form of a warp-shuffle reduction into a cell
   ([C_syntax.try_warp_reduce]) declares one workgroup-shared [wred_partials_] slot per warp, which
   the estimate did not count. Over sums of 32, 64 and 128 lanes (one, two and four warps of 32)
   retyped [Workgroup_reduce], the estimate's charge beyond the tiles is, on every backend, exactly
   the bytes of the [wred_partials_] arrays the emitted kernel declares, read off the source (none
   on the C backends, which render the serial loop). On the GPU backends the two- and four-warp
   kernels declare some, and the gate admits each at exactly its estimate and refuses it one byte
   below. Every kernel also runs and returns the sum. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module L = Ll_test
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let () =
  Stdio.eprintf "workgroup_memory_accounting: backend=%s (not part of the golden)\n%!" backend_name

let () = Generated.init ~backend_name
let on_gpu = Sched.backend_is_gpu backend_name

(* The workgroup-shared bytes of the [wred_partials_*] arrays a generated kernel declares. An
   element type other than float/double reads as an impossible count, so the claim fails. *)
let declared_partial_bytes src =
  List.sum
    (module Int)
    (String.split_lines src)
    ~f:(fun line ->
      let line = String.strip line in
      match String.substr_index line ~pattern:"wred_partials_" with
      | Some i when String.is_suffix line ~suffix:"];" && not (String.mem line '=') -> (
          let ty = List.last_exn (String.split (String.rstrip (String.prefix line i)) ~on:' ') in
          let lb = String.index_exn line '[' and rb = String.rindex_exn line ']' in
          let slots = Int.of_string (String.sub line ~pos:(lb + 1) ~len:(rb - lb - 1)) in
          match ty with "float" -> 4 * slots | "double" -> 8 * slots | _ -> -1_000_000)
      | _ -> 0)

(* === 1. The warp-reduce partials. === *)

type reduce_leg = {
  name : string;
  lanes : int;
  scheduled : LL.optimized;  (** The transformed code, as rendered. *)
  sum : float;
  expected : float;
}

let reduce_leg lanes =
  let name = Printf.sprintf "wmem_wred%d" lanes in
  (* Integer-valued terms: every partial sum is exact, whatever the association. *)
  let vv = Array.init lanes ~f:(fun k -> Float.of_int ((k * 7 % 13) - 4)) in
  let v = TDSL.ndarray vv ~label:[ name ^ "_v" ] ~output_dims:[ lanes ] () in
  let%op s = v ++ "i=>0" in
  let captured = ref None in
  (* The lowered serial reduction replaced by one [Workgroup_reduce] accumulation loop: the renderer
     owns the communication. The transform drops the lowered [Zero_out], so allocation zeroes the
     accumulator instead. *)
  let transform (opt : LL.optimized) =
    (LL.get_node opt.traced_store s.Tensor.value).LL.zero_initialized_by_code <- false;
    let i = L.sym () in
    let f0 = L.fixed 0 in
    let opt =
      {
        opt with
        llc =
          L.loop_n ~axis:LL.Workgroup_reduce i lanes
            (L.set_at s.Tensor.value f0
               (L.add (L.get s.Tensor.value [| f0 |]) (L.get v.Tensor.value [| L.iter i |])));
      }
    in
    captured := Some opt;
    opt
  in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun o -> [ transform o ])
      ctx (Train.forward s) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  {
    name;
    lanes;
    scheduled = Option.value_exn !captured;
    sum = (Context.get_values ctx s.Tensor.value).(0);
    expected = Array.fold vv ~init:0. ~f:( +. );
  }

let () =
  let capabilities = Context.codegen_capabilities (Context.auto ()) in
  let legs = List.map [ 32; 64; 128 ] ~f:reduce_leg in
  let charged leg =
    Sched.workgroup_memory_bytes ~capabilities leg.scheduled
    - Sched.workgroup_memory_bytes ~capabilities:Ir.Backend_intf.no_codegen_capabilities
        leg.scheduled
  in
  let declared leg = declared_partial_bytes (Generated.read leg.name) in
  p_all "every reduction runs and returns the sum" legs ~f:(fun leg ->
      Float.equal leg.sum leg.expected);
  p_all "the estimate charges exactly the warp-reduce partials the emitted kernel declares" legs
    ~f:(fun leg -> charged leg = declared leg);
  let multi = List.filter legs ~f:(fun leg -> leg.lanes > 32) in
  gated_all ~when_:on_gpu ~on:backend_name
    "past one warp the emitted kernel declares warp-reduce partials" multi ~f:(fun leg ->
      declared leg > 0);
  gated_all ~when_:on_gpu ~on:backend_name "within one warp the emitted kernel declares none"
    (List.filter legs ~f:(fun leg -> leg.lanes <= 32))
    ~f:(fun leg -> declared leg = 0);
  let admits leg bytes =
    match
      Sched.check_hardware_limits ~name:leg.name
        ~limits:{ Ir.Backend_intf.no_hardware_limits with max_workgroup_memory_bytes = Some bytes }
        ~capabilities leg.scheduled
    with
    | () -> true
    | exception Utils.User_error _ -> false
  in
  gated_all ~when_:on_gpu ~on:backend_name
    "past one warp the gate admits the kernel at exactly its declared partials and refuses it one \
     byte below"
    multi ~f:(fun leg ->
      let bytes = Sched.workgroup_memory_bytes ~capabilities leg.scheduled in
      bytes = declared leg && admits leg bytes && not (admits leg (bytes - 1)))
