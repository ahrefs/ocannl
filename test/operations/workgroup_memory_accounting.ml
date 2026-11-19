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
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

let () =
  Stdio.eprintf "workgroup_memory_accounting: backend=%s (not part of the golden)\n%!" backend_name

let () = Generated.init ~backend_name
let on_gpu = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)

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
  p_all
    (name ^ ": warp-reduce terms have f32-exact partial sums")
    [ Array.to_list vv ]
    ~f:(Ll_test.exact_partials ~any_sign:true ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single));
  let v = TDSL.ndarray vv ~label:[ name ^ "_v" ] ~output_dims:[ lanes ] () in
  let%op s = v ++ "i=>0" in
  let captured = ref None in
  (* The lowered serial reduction replaced by one [Workgroup_reduce] accumulation loop: the renderer
     owns the communication. The transform drops the lowered [Zero_out], so [Context.run] below
     starts from a stated zero: allocation's is not one to rely on (gh-ocannl-1873). *)
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
  let ctx = Context.run (Context.set_values ctx s.Tensor.value [| 0. |]) routine in
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
      bytes = declared leg && admits leg bytes && not (admits leg (bytes - 1)));
  (* The structural half (gh-ocannl-1626): per-PR CI runs cc, whose dialect renders the serial loop
     and charges nothing, so the gated claims above are skipped there. The charge and the gate are
     host-side, so the same schedules are judged here under a GPU-like dialect's derived
     capabilities ([Ll_test.gpu_like_capabilities]): partials past one warp, none within it, and the
     gate's boundary at the estimate. That the charge is what the kernel declares stays the GPU
     claim's: it reads the emitted source. *)
  let gpu_like = Ll_test.gpu_like_capabilities () in
  let gpu_like_charge leg =
    Sched.workgroup_memory_bytes ~capabilities:gpu_like leg.scheduled
    - Sched.workgroup_memory_bytes ~capabilities:Ir.Backend_intf.no_codegen_capabilities
        leg.scheduled
  in
  p_all "structural: under a GPU-like dialect, past one warp the estimate charges partials" multi
    ~f:(fun leg -> gpu_like_charge leg > 0);
  p_all "structural: under a GPU-like dialect, within one warp it charges none"
    (List.filter legs ~f:(fun leg -> leg.lanes <= 32))
    ~f:(fun leg -> gpu_like_charge leg = 0);
  p_all
    "structural: under a GPU-like dialect, past one warp the gate admits the kernel at exactly its \
     estimate and refuses it one byte below"
    multi ~f:(fun leg ->
      let bytes = Sched.workgroup_memory_bytes ~capabilities:gpu_like leg.scheduled in
      let admits bytes =
        match
          Sched.check_hardware_limits ~name:leg.name
            ~limits:
              { Ir.Backend_intf.no_hardware_limits with max_workgroup_memory_bytes = Some bytes }
            ~capabilities:gpu_like leg.scheduled
        with
        | () -> true
        | exception Utils.User_error _ -> false
      in
      admits bytes && not (admits (bytes - 1)))

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
    | { sc_family = Autotune.Seeding_conv { site; _ }; _ } ->
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
  (* An accumulator tile is known by membership in the sets the contraction fills, never by its
     label (gh-ocannl-1721): the masked tile -- since gh-ocannl-1736 the padded scratch tile a
     register fragment is staged through -- in [masked_fragments], a fragment in
     [simdgroup_fragments]. *)
  let is_accumulator_tile (o : LL.optimized) tn =
    Set.mem o.LL.masked_fragments tn || Set.mem o.LL.simdgroup_fragments tn
  in
  let unmasked =
    conv_leg ~what:"unmasked" (lowering "wmem_conv_u" (make_conv "wmu" ~hw:18 ~ic:8))
  in
  p_all "unmasked: no flavor places its accumulator fragment in workgroup memory" unmasked
    ~f:(fun (_, _, o) ->
      Set.for_all o.LL.workgroup_shared ~f:(fun tn -> not (is_accumulator_tile o tn)));
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
      Set.for_all o.LL.workgroup_shared ~f:(fun tn -> not (is_accumulator_tile o tn)));
  (* 9 output rows and 4 input channels, padded to the tile: every flavor's accumulator is masked
     (the row pad or the non-dividing row block). Since gh-ocannl-1736 the masked tile is a padded
     scratch tile at the f32 output precision, workgroup-shared beside the operand tiles, and the
     accumulator itself is a register fragment staged through it, never in workgroup memory. *)
  let masked = conv_leg ~what:"masked" (lowering "wmem_conv_m" (make_conv "wmm" ~hw:11 ~ic:4)) in
  let scratch_tiles (o : LL.optimized) =
    Set.to_list (Set.inter o.LL.workgroup_shared o.LL.masked_fragments)
  in
  p_all
    "masked: every flavor stages a register fragment through one f32 scratch tile, the only \
     accumulator tile in workgroup memory"
    masked ~f:(fun (_, _, o) ->
      (match scratch_tiles o with
        | [ tn ] -> Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single
        | _ -> false)
      && (not (Set.is_empty o.LL.simdgroup_fragments))
      && Set.for_all o.LL.simdgroup_fragments ~f:(fun tn -> not (Set.mem o.LL.workgroup_shared tn)));
  (* The estimate charges the scratch tile and nothing for the fragment: taking the scratch tile out
     of workgroup memory lowers it by exactly the tile's bytes (at the stand-in's 4-byte granule,
     which an f32 tile already meets), so the masked estimates above exceed the unmasked ones by the
     scratch tile alone. *)
  p_all "masked: the gate's estimate charges exactly the scratch tile beyond the operand tiles"
    masked ~f:(fun (_, est, o) ->
      match scratch_tiles o with
      | [ tn ] ->
          let bytes =
            Array.fold (Lazy.force tn.Tn.dims) ~init:1 ~f:( * )
            * Ir.Ops.prec_in_bytes (Lazy.force tn.Tn.storage_prec)
          in
          let without =
            Sched.workgroup_memory_bytes ~capabilities:standin
              { o with LL.workgroup_shared = Set.remove o.LL.workgroup_shared tn }
          in
          est - without = bytes
      | _ -> false)

(* === 3. Every family's seeds (gh-ocannl-1530). ===

   The two-boundary check above, over the matmul family's seeds and the coalesced branch's beside
   the conv family's: each GPU seed, built on a scratch copy and read by the gate's own estimate, is
   kept at a limit of exactly that estimate (no over-charge drops it) and, where the estimate is
   positive, refused one byte below it (no under-charge proposes a kernel the gate refuses). Seeds
   are judged on synthetic GPU limits advertising an f32 tile with pipelining depth 2 and a swizzled
   staged layout, so the staged, tensorized, depth, swizzle and [Fuse_epilogue] twins all seed on
   every host, and under two stand-in capabilities: Metal-like rounding with no tile-MMA scratch,
   and the same with a positive per-kernel scratch for any tile-MMA kernel, so the scratch term is
   charged where the gate charges it. [compiled_workgroup_bytes] holds the estimate to what the
   device compiler allocates. *)

let synthetic_limits cap =
  let f32 = Ir.Backend_intf.Mma_f32 in
  {
    Ir.Backend_intf.no_hardware_limits with
    max_workgroup_memory_bytes = cap;
    mma =
      Some
        {
          Ir.Backend_intf.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
          mma_staged_layouts = [ ((f32, f32, f32), Ir.Backend_intf.Mma_swizzled_b128) ];
          mma_pipeline_depths = [ 2 ];
        };
  }

(* The stand-in with a per-kernel tile-MMA scratch: 256 bytes for any kernel holding a scope. *)
let scratch_standin =
  {
    standin with
    Ir.Backend_intf.mma_kernel_workgroup_bytes =
      (fun ~triples -> if List.is_empty triples then 0 else 256);
  }

type judged_seed = { seed : Autotune.sketch_params; estimate : int; built : LL.optimized }

let family_leg ~what ~capabilities opt =
  let accum_prec = (Context.codegen_capabilities (Context.auto ())).accum_prec in
  let seeding = Hashtbl.Poly.create () in
  let seeds cap =
    Hashtbl.find_or_add seeding cap ~default:(fun () ->
        let limits = synthetic_limits cap in
        List.filter
          (Autotune.sketch_seed_params ~capabilities ~is_gpu:true ~is_cpu:false ~limits
             (scratch opt)
          @ Autotune.coalesced_seed_params ~capabilities ~is_gpu:true ~is_cpu:false ~limits
              (scratch opt))
          ~f:(fun p -> p.Autotune.sk_gpu))
  in
  let judged =
    List.map (seeds None) ~f:(fun seed ->
        let o = scratch opt in
        let built = Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:seed o) o in
        { seed; estimate = Sched.workgroup_memory_bytes ~capabilities built; built })
  in
  let kept cap p = List.mem (seeds (Some cap)) p ~equal:Poly.equal in
  List.iter judged ~f:(fun j ->
      let at = kept j.estimate j.seed and below = j.estimate > 0 && kept (j.estimate - 1) j.seed in
      if (not at) || below then
        Stdio.eprintf
          "workgroup_memory_accounting: %s: seed %s, estimate %d: %s (not part of the golden)\n%!"
          what
          (Autotune.sketch_params_label j.seed)
          j.estimate
          (if not at then "REFUSED at its estimate" else "KEPT one byte below its estimate"));
  Stdio.eprintf "workgroup_memory_accounting: %s: %d GPU seeds (not part of the golden)\n%!" what
    (List.length judged);
  p_all ~min:1 (what ^ ": every GPU seed is kept at a limit of exactly its gate estimate") judged
    ~f:(fun j -> kept j.estimate j.seed);
  p_all ~min:1
    (what ^ ": every GPU seed with a positive estimate is refused one byte below it")
    (List.filter judged ~f:(fun j -> j.estimate > 0))
    ~f:(fun j -> not (kept (j.estimate - 1) j.seed));
  judged

(* A GEMM [z = x . w] with an elementwise tail [z + pos]; [materialize] keeps [z] on device, else
   the fused tail is its only consumer. *)
let gemm_lowering name ~materialize ~m ~k ~n =
  let x =
    NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.single ~o:[ m; k ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let w =
    NTDSL.init ~l:(name ^ "_w") ~prec:Ir.Ops.single ~o:[ k; n ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let pos =
    NTDSL.init ~l:(name ^ "_pos") ~prec:Ir.Ops.single ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 5) -. 2.)
      ()
  in
  let%op z = x +* "ik;kj=>ij" w in
  let%op y = z + pos in
  if materialize then Train.set_materialized z.Tensor.value;
  (lowering name y, z.Tensor.value)

(* The attention out projection [d[b,s,j] += w[j,h,e] * x[b,s,h,e]]: the coalesced branch's site. *)
let out_proj_lowering name =
  let w =
    NTDSL.init ~l:(name ^ "_w") ~prec:Ir.Ops.single ~o:[ 64 ] ~i:[ 4; 16 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2; 3 |] ~modulus:7 ~offset:(-3.) ~stride:1.)
      ()
  in
  let att =
    NTDSL.init ~l:(name ^ "_att") ~prec:Ir.Ops.single ~b:[ 2; 32 ] ~o:[ 4; 16 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2; 3; 1 |] ~modulus:5 ~offset:1. ~stride:1.)
      ()
  in
  let%op z = w * att in
  lowering name z

(* An f32 conv with a per-channel bias tail, the product left to the tail alone. *)
let conv_tail_lowering name =
  let x =
    NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 10; 10; 8 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(name ^ "_k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 8 ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let bias =
    NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.single ~o:[ 8 ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
      ()
  in
  let%op z =
    x +* "...| oh<+kh, ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  let%op y = z + bias in
  (lowering name y, z.Tensor.value)

let () =
  let is_mma j = j.seed.Autotune.sk_mma in
  let dense, _ = gemm_lowering "wmem_mm_d" ~materialize:true ~m:32 ~k:64 ~n:64 in
  let d = family_leg ~what:"matmul" ~capabilities:standin dense in
  p "matmul: a staged scalar seed, a staged tensorized one and an unstaged tensorized one"
    (List.exists d ~f:(fun j -> (not (is_mma j)) && j.seed.sk_bk > 0)
    && List.exists d ~f:(fun j -> is_mma j && j.seed.sk_bk > 0)
    && List.exists d ~f:(fun j -> is_mma j && j.seed.sk_bk = 0));
  p "matmul: a depth-2 twin, a swizzled twin and a fused twin"
    (List.exists d ~f:(fun j -> j.seed.sk_depth > 1)
    && List.exists d ~f:(fun j -> Option.is_some j.seed.sk_swizzle)
    && List.exists d ~f:(fun j -> j.seed.sk_epilogue));
  let _ : judged_seed list =
    family_leg ~what:"matmul, tile-MMA scratch" ~capabilities:scratch_standin dense
  in
  (* The product left to the fused tail alone: a tensorized fused twin places it in workgroup-shared
     memory, whole ([Schedule.apply_fuse_epilogue]'s shared placement). *)
  let local, z = gemm_lowering "wmem_mm_l" ~materialize:false ~m:32 ~k:32 ~n:32 in
  let l = family_leg ~what:"matmul, routine-local product" ~capabilities:standin local in
  p_exists
    "matmul, routine-local product: a fused tensorized twin places the product in shared memory" l
    ~f:(fun j -> is_mma j && j.seed.sk_epilogue && Set.mem j.built.LL.workgroup_shared z);
  let padded, _ = gemm_lowering "wmem_mm_p" ~materialize:true ~m:20 ~k:70 ~n:50 in
  let _ : judged_seed list = family_leg ~what:"matmul, padded" ~capabilities:standin padded in
  let c = family_leg ~what:"contraction" ~capabilities:standin (out_proj_lowering "wmem_cp") in
  p_exists "contraction: a contraction-coalesced staged tensorized seed" c ~f:(fun j ->
      is_mma j && j.seed.sk_bk > 0
      && Poly.equal j.seed.Autotune.sk_coalesce (Some Autotune.Coalesce_contraction));
  let conv, z = conv_tail_lowering "wmem_cv_l" in
  let cv = family_leg ~what:"conv, routine-local product" ~capabilities:standin conv in
  p_exists "conv, routine-local product: a fused twin places the product in shared memory" cv
    ~f:(fun j -> j.seed.Autotune.sk_epilogue && Set.mem j.built.LL.workgroup_shared z)
