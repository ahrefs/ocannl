(* gh-ocannl-1452: the workgroup-memory estimate ([Schedule.workgroup_memory_bytes]) charges exactly
   what the kernels declare, and the conv family's seeding prediction charges exactly the estimate.

   1. Under: the multi-warp form of a warp-shuffle reduction into a cell
   ([C_syntax.try_warp_reduce]) declares one workgroup-shared [wred_partials_] slot per warp, which
   the estimate did not count. Over sums of 32, 64 and 128 lanes (one, two and four warps of 32)
   retyped [Workgroup_reduce], the estimate's charge beyond the tiles is, on every backend, exactly
   the bytes of the [wred_partials_] arrays the emitted kernel declares, read off the source (none
   on the C backends, which render the serial loop). On the GPU backends the two- and four-warp
   kernels declare some, and the gate admits each at exactly its estimate and refuses it one byte
   below. Every kernel also runs and returns the sum.

   2. Over: the conv family's GPU leg priced both staged operand tiles at the conv OUTPUT's
   precision, so a mixed-precision conv (f16 operands, f32 output, on a stand-in f16 x f16 -> f32
   tile) was charged double and refused seeds that fit. And under: a masked flavor (a pad multiple,
   or a row block not dividing the rows) places its accumulator fragment in workgroup memory beside
   the tiles, which the leg did not charge (a reduction pad alone masks nothing, its guard being
   discharged). Over an unmasked conv (every extent a tile multiple), a reduction-padded one and a
   masked one, every GPU conv seed, built and read by the gate's own estimate, is kept at a limit of
   exactly that estimate and refused one byte below it: the prediction is the gate's charge, each
   tile priced at its own operand's precision. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode
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

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

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

(* === 2. The conv family's pricing. === *)

(* A mixed-precision conv: f16 input and kernel, f32 output; [hw] input rows and columns, [ic] input
   channels, 8 output channels, a 3x3 window. *)
let make_conv tag ~hw ~ic =
  let x =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.half ~b:[ 2 ] ~o:[ hw; hw; ic ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.half ~i:[ 3; 3; ic ] ~o:[ 8 ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op y =
    x +* "...| oh<+kh, ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  Tn.update_prec y.Tensor.value Ir.Ops.single;
  y

(* The routine's whole lowering, captured on its way through a compile. *)
let lowering name t =
  let captured = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile ~name
      ~lowered_transform:(fun opt ->
        captured := Some (scratch opt);
        [ opt ])
      ctx (Train.forward t) Ir.Indexing.Empty
  in
  Option.value_exn !captured

(* A device whose only tile takes f16 operands into an f32 accumulator. *)
let f16_limits cap =
  let f16 = Ir.Backend_intf.Mma_f16 and f32 = Ir.Backend_intf.Mma_f32 in
  {
    Ir.Backend_intf.no_hardware_limits with
    max_workgroup_memory_bytes = cap;
    mma =
      Some
        {
          Ir.Backend_intf.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f16, f16, f32), (8, 8, 8)) ];
        };
  }

(* A stand-in charging Metal-like 4-byte allocation rounding, so the prediction rounds as the gate
   does. *)
let standin = { Ir.Backend_intf.no_codegen_capabilities with workgroup_allocation_alignment = 4 }

(* Each GPU conv seed of the lowering beside the gate's estimate of its built schedule. *)
let conv_leg ~what opt =
  let seeds cap =
    List.filter
      (Autotune.sketch_seed_params ~capabilities:standin ~is_gpu:true ~is_cpu:false
         ~limits:(f16_limits cap) (scratch opt))
      ~f:(fun p -> p.Autotune.sk_gpu)
  in
  let accum_prec = (Context.codegen_capabilities (Context.auto ())).accum_prec in
  let built p =
    let o = scratch opt in
    Sched.apply (Autotune.sketch_schedule ~accum_prec ~p o) o
  in
  let mixed =
    match
      Autotune.seeding_census ~capabilities:standin ~is_gpu:true ~is_cpu:false
        ~limits:(f16_limits None) (scratch opt)
    with
    | { sc_family = Autotune.Seeding_conv { judgment = { cj_site = site; _ }; _ }; _ } ->
        let prec tn = Lazy.force tn.Tn.storage_prec in
        Ir.Ops.equal_prec (prec site.c_a) Ir.Ops.half
        && Ir.Ops.equal_prec (prec site.c_b) Ir.Ops.half
        && Ir.Ops.equal_prec (prec site.c_d) Ir.Ops.single
    | _ -> false
  in
  p (what ^ ": the conv site is mixed precision, f16 operands into an f32 output") mixed;
  let judged =
    List.map (seeds None) ~f:(fun p ->
        let o = built p in
        (p, Sched.workgroup_memory_bytes ~capabilities:standin o, o))
  in
  Stdio.eprintf
    "workgroup_memory_accounting: %s: %d GPU conv seeds, estimates %s (not part of the golden)\n%!"
    what (List.length judged)
    (String.concat ~sep:"," (List.map judged ~f:(fun (_, e, _) -> Int.to_string e)));
  let kept cap p = List.mem (seeds (Some cap)) p ~equal:Poly.equal in
  p_all (what ^ ": every GPU conv seed is kept at a limit of exactly its gate estimate") judged
    ~f:(fun (p, est, _) -> kept est p);
  p_all (what ^ ": every GPU conv seed is refused one byte below its gate estimate") judged
    ~f:(fun (p, est, _) -> not (kept (est - 1) p));
  judged

let () =
  (* 16 output rows and 8 input channels: every extent a multiple of the 8x8x8 tile, so no flavor is
     masked and the staged operand tiles are the whole charge. *)
  let unmasked =
    conv_leg ~what:"unmasked" (lowering "wmem_conv_u" (make_conv "wmu" ~hw:18 ~ic:8))
  in
  p_all "unmasked: no flavor places its accumulator fragment in workgroup memory" unmasked
    ~f:(fun (_, _, o) ->
      Set.for_all o.LL.workgroup_shared ~f:(fun tn ->
          match tn.Tn.label with "fragment" :: _ -> false | _ -> true));
  (* 16 output rows and 4 input channels: only the reduction pads, and [Tensorize] discharges that
     guard (the padded operand slots are zeros), so the accumulator stays unmasked and no fragment
     is workgroup-shared (Codex P2 on the first push charged one). *)
  let k_padded =
    conv_leg ~what:"reduction-padded" (lowering "wmem_conv_k" (make_conv "wmk" ~hw:18 ~ic:4))
  in
  p_all "reduction-padded: every flavor pads the reduction" k_padded ~f:(fun (p, _, _) ->
      p.Autotune.sk_bk > 0);
  p_all "reduction-padded: no flavor places its accumulator fragment in workgroup memory" k_padded
    ~f:(fun (_, _, o) ->
      Set.for_all o.LL.workgroup_shared ~f:(fun tn ->
          match tn.Tn.label with "fragment" :: _ -> false | _ -> true));
  (* 9 output rows and 4 input channels, padded to the tile: every flavor's accumulator is masked
     (the row pad or the non-dividing row block), and its accumulator fragment, at the f32 output
     precision, is workgroup-shared beside the tiles. *)
  let masked = conv_leg ~what:"masked" (lowering "wmem_conv_m" (make_conv "wmm" ~hw:11 ~ic:4)) in
  p_all "masked: every flavor places an f32 accumulator fragment in workgroup memory" masked
    ~f:(fun (_, _, o) ->
      Set.exists o.LL.workgroup_shared ~f:(fun tn ->
          (match tn.Tn.label with "fragment" :: _ -> true | _ -> false)
          && Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single))
