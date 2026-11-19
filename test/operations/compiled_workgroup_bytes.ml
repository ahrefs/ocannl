(* gh-ocannl-1530: the device compiler's own static workgroup allocation, kernel by kernel, against
   the schedule layer's charge for it, over every candidate a search seeds.

   [Schedule.workgroup_memory_bytes] is the charge the hardware gate holds a kernel to before its
   compile, and the one the seeding floors reproduce through [Schedule.kernel_workgroup_bytes]
   ([workgroup_memory_accounting] holds the floors to it at both boundaries). Nothing else checks it
   against what a compiled kernel actually allocates, so an allocation a transform or a renderer
   adds without a term there surfaces only as a pipeline the driver refuses, or under the limit as
   an occupancy misprediction. Here the driver answers: [Ir.Compiled_kernels] records, per linked
   kernel, Metal's [staticThreadgroupMemoryLength], CUDA's and HIP's
   [FUNC_ATTRIBUTE_SHARED_SIZE_BYTES].

   The claim is AT MOST, not equality: a device compiler may allocate less than the kernel declares
   (NVVM, ptxas and the Metal compiler drop an array they prove dead; the lane all-reduce partials
   are charged at the wider of two precisions), and an over-charge only costs candidates, while an
   under-charge is the defect. Each kernel's slack is printed to stderr. So that the claim can fail,
   some kernel's figure must equal its charge exactly and be positive: a dropped term of the charge
   then shows on it (the negative control the PR reports).

   Candidates: every whole-routine seed [Autotune.search_seeding] proposes under the device's own
   limits and capabilities (sketch and coalesced; staged, tensorized and [Fuse_epilogue] twins
   included), each compiled through the seed's own schedule, over fixtures chosen for their
   allocations: a dividing GEMM with a tail (staged tiles, fused twins), a padded one (the masked
   accumulator tile), two f16-operand ones (half tiles; an f16 product, whose accumulator boundary
   converts on Metal: the per-kernel tile-MMA term), the attention out projection (the
   contraction-coalesced K menu's deep staged tiles) and a conv (the conv family's staged tiles). A
   candidate the gate or the backend refuses is outside the comparison (stderr says why); a crash
   fails. Kernel figures pair with the routine's shipped segments in launch order, which the
   census's count must match.

   The C backends declare no workgroup memory and record nothing: the leg is skipped there, its
   fixtures still built and seeded. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let accum_prec = Ll_test.backend_accum_prec
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let facts = Context.backend_facts_of_name backend_name
let is_gpu = Ir.Backend_intf.binds_hardware_grid facts
let is_cpu = Ir.Backend_intf.renders_on_host_pool facts

let () =
  Stdio.eprintf "compiled_workgroup_bytes: backend=%s (not part of the golden)\n%!" backend_name

let leg = Verdict.leg ~when_:is_gpu ~on:backend_name ()

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* One compiled kernel beside its segment's charge, and the share of that charge that is the
   backend's per-kernel tile-MMA scratch. *)
type kernel = { name : string; compiled : int; charged : int; mma_scratch : int }

type outcome =
  | Compiled of kernel list
  | Unpaired of string  (** The census and the shipped segments disagree in count. *)
  | Refused of string
      (** A classified refusal ([Context.compile_outcome]'s [Error (Classified _)]): the gate's, the
          backend's. *)
  | Crashed of string

let all_kernels : kernel list ref = ref []
let all_outcomes : (string * outcome) list ref = ref []

let fixture ?(materialize = true) ?(fp16_wide = false) ?scratch_triple tag build =
  let saved = Ir.Numerics.get () in
  if fp16_wide then Ir.Numerics.set_policy { saved with fp16_arithmetic = Ir.Numerics.Fp16_wide };
  Exn.protect
    ~finally:(fun () -> Ir.Numerics.set_policy saved)
    ~f:(fun () ->
      Tensor.unsafe_reinitialize ();
      let z, y = build () in
      if materialize then Train.set_materialized z.Tensor.value;
      let fwd = named ("cwb_" ^ tag) (Train.forward y) in
      let ctx = Context.auto () in
      let limits = Context.hardware_limits ctx in
      let capabilities = Context.codegen_capabilities ctx in
      let capture = ref None in
      let base_ctx, _routine =
        Context.compile
          ~lowered_transform:(fun opt ->
            capture := Some opt;
            [ opt ])
          ctx fwd Ir.Indexing.Empty
      in
      Context.release base_ctx;
      let opt = Option.value_exn ~here:[%here] !capture in
      let no_scratch =
        { capabilities with Ir.Backend_intf.mma_kernel_workgroup_bytes = (fun ~triples:_ -> 0) }
      in
      let seeding =
        Autotune.search_seeding ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices:[] opt
      in
      let seeds = seeding.Autotune.ss_whole in
      let one_line msg = String.prefix (String.tr ~target:'\n' ~replacement:' ' msg) 300 in
      let run q =
        let label = Autotune.sketch_params_label q in
        let outcome =
          match
            Ir.Compiled_kernels.with_census (fun () ->
                Context.compile_outcome ~provenance:Ir.Schedule_outcome.Candidate ~candidate:label
                  ~lowered_transform:(fun o ->
                    [ Sched.apply_classified (Autotune.sketch_schedule ~accum_prec ~p:q o) o ])
                  (Context.auto ()) fwd Ir.Indexing.Empty)
          with
          | Ok (cctx, routine), records ->
              let segments = routine.Context.segments in
              let outcome =
                if List.length records <> List.length segments then
                  Unpaired
                    (Printf.sprintf "%d compiled figures for %d segments" (List.length records)
                       (List.length segments))
                else
                  Compiled
                    (List.map2_exn records segments ~f:(fun r seg ->
                         let charged = Sched.workgroup_memory_bytes ~capabilities seg in
                         {
                           name = r.Ir.Compiled_kernels.kernel;
                           compiled = r.static_workgroup_bytes;
                           charged;
                           mma_scratch =
                             charged - Sched.workgroup_memory_bytes ~capabilities:no_scratch seg;
                         }))
              in
              (* Each candidate's working pools go back before the next links. *)
              Context.release cctx;
              outcome
          | Error (Ir.Schedule_outcome.Classified c), _ ->
              Refused (Ir.Schedule_outcome.detail_of_cause c.cause)
          | Error (Ir.Schedule_outcome.Fatal f), _ -> Crashed (Exn.to_string f.exn)
          | exception exn -> Crashed (Exn.to_string exn)
        in
        (match outcome with
        | Compiled ks ->
            List.iter ks ~f:(fun k ->
                Stdio.eprintf
                  "%s: %s kernel %s: compiled %d, charged %d%s (not part of the golden)\n%!" tag
                  label k.name k.compiled k.charged
                  (if k.compiled > k.charged then " UNDER-CHARGED" else ""))
        | Unpaired msg ->
            Stdio.eprintf "%s: %s UNPAIRED: %s (not part of the golden)\n%!" tag label msg
        | Refused msg ->
            Stdio.eprintf "%s: %s refused: %s (not part of the golden)\n%!" tag label (one_line msg)
        | Crashed msg ->
            Stdio.eprintf "%s: %s CRASHED: %s (not part of the golden)\n%!" tag label (one_line msg));
        (label, outcome)
      in
      let outcomes = if is_gpu then List.map seeds ~f:run else [] in
      Stdio.eprintf "%s: %d seeds, %d compiled (not part of the golden)\n%!" tag (List.length seeds)
        (List.count outcomes ~f:(function _, Compiled _ -> true | _ -> false));
      (* The conv family seeds its GPU leg only through a tensorized pipeline, so a conv site has no
         seed where the backend withholds a tile for its f32 operands (CUDA outside tf32, HIP). The
         gate is that capability, the seeder's own judgment, never the seed list under test. *)
      let capability =
        match fst (Autotune.family_seeding ~capabilities ~is_gpu ~is_cpu ~limits opt) with
        | Autotune.Seeding_conv _ ->
            Ll_test.tensorized_capability ~is_gpu ~is_cpu ~limits ~a_prec:Ir.Ops.single
              ~b_prec:Ir.Ops.single
              ~d_prec:(Lazy.force z.Tensor.value.Ir.Tnode.storage_prec)
        | Autotune.Seeding_matmul _ | Autotune.Seeding_none -> `Advertised
      in
      let when_, aggregation =
        match capability with
        | `Advertised -> (true, None)
        | `Withheld aggregation -> (false, Some aggregation)
      in
      gated ?aggregation ~when_ ~on:backend_name
        (tag ^ ": the search seeds at least one candidate here")
        (not (List.is_empty seeds));
      let kernels = List.concat_map outcomes ~f:(function _, Compiled ks -> ks | _ -> []) in
      (* The per-kernel tile-MMA term has a compiled witness where this fixture's triple is
         tensorized and the backend charges it a positive scratch: both read off the capabilities
         under the fixture's policy, never off which candidates compiled. *)
      Option.iter scratch_triple ~f:(fun ((d_prec, a_prec, b_prec) as triple) ->
          let scratch =
            capabilities.Ir.Backend_intf.mma_kernel_workgroup_bytes ~triples:[ triple ]
          in
          let when_, aggregation =
            match Ll_test.tensorized_capability ~is_gpu ~is_cpu ~limits ~a_prec ~b_prec ~d_prec with
            | `Advertised -> (is_gpu && scratch > 0, None)
            | `Withheld aggregation -> (false, Some aggregation)
          in
          gated_exists ?aggregation ~when_ ~on:backend_name
            (tag
           ^ ": a compiled kernel charged the per-kernel tile-MMA scratch allocates exactly its \
              charge") kernels ~f:(fun k -> k.mma_scratch > 0 && k.compiled = k.charged));
      all_outcomes := !all_outcomes @ outcomes;
      all_kernels := !all_kernels @ kernels)

(* A GEMM [z = x . w] with an elementwise tail [z + pos]; [prec] the operands', [out_prec] the
   product's. *)
let gemm ?(prec = Ir.Ops.single) ?(out_prec = Ir.Ops.single) ~m ~k ~n tag () =
  let x =
    NTDSL.init ~l:(tag ^ "_x") ~prec ~o:[ m; k ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let w =
    NTDSL.init ~l:(tag ^ "_w") ~prec ~o:[ k; n ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let pos =
    NTDSL.init ~l:(tag ^ "_pos") ~prec:Ir.Ops.single ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 5) -. 2.)
      ()
  in
  let%op z = x +* "ik;kj=>ij" w in
  Ir.Tnode.update_prec z.Tensor.value out_prec;
  let%op y = z + pos in
  (z, y)

(* The attention out projection [d[b,s,j] += w[j,h,e] * x[b,s,h,e]], with a tail. *)
let out_proj ~bb ~ss ~jj ~hh ~ee tag () =
  let w =
    NTDSL.init ~l:(tag ^ "_w") ~prec:Ir.Ops.single ~o:[ jj ] ~i:[ hh; ee ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2; 3 |] ~modulus:7 ~offset:(-3.) ~stride:1.)
      ()
  in
  let att =
    NTDSL.init ~l:(tag ^ "_att") ~prec:Ir.Ops.single ~b:[ bb; ss ] ~o:[ hh; ee ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2; 3; 1 |] ~modulus:5 ~offset:1. ~stride:1.)
      ()
  in
  let pos =
    NTDSL.init ~l:(tag ^ "_pos") ~prec:Ir.Ops.single ~o:[ jj ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
      ()
  in
  let%op z = w * att in
  let%op y = z + pos in
  (z, y)

(* A stride-1 convolution [b, h, w, ic] * [kh, kw, ic -> oc] with a per-channel bias tail. *)
let conv ~hw ~ic ~oc tag () =
  let x =
    NTDSL.init ~l:(tag ^ "_x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ hw; hw; ic ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "_k") ~prec:Ir.Ops.single ~i:[ 3; 3; ic ] ~o:[ oc ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let bias =
    NTDSL.init ~l:(tag ^ "_b") ~prec:Ir.Ops.single ~o:[ oc ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
      ()
  in
  let%op z =
    x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  let%op y = z + bias in
  (z, y)

let () =
  fixture "dense" (gemm ~m:32 ~k:64 ~n:64 "cwb_dense");
  (* The product left to the fused tail alone: a tensorized fused twin places it, routine-local, in
     workgroup-shared memory whole. *)
  fixture ~materialize:false "dense_local" (gemm ~m:32 ~k:32 ~n:32 "cwb_dense_local");
  fixture "padded" (gemm ~m:20 ~k:70 ~n:50 "cwb_padded");
  fixture "half" (gemm ~prec:Ir.Ops.half ~m:32 ~k:48 ~n:40 "cwb_half");
  (* An f16 product under the wide fp16 policy, an f32 accumulator over half storage: on Metal its
     fragment boundary converts, and the kernel declares the coordinate table that is the per-kernel
     tile-MMA term (gh-ocannl-1205). *)
  fixture ~fp16_wide:true
    ~scratch_triple:Ir.Ops.(half, half, half)
    "half_out"
    (gemm ~prec:Ir.Ops.half ~out_prec:Ir.Ops.half ~m:32 ~k:32 ~n:32 "cwb_half_out");
  fixture "contraction" (out_proj ~bb:2 ~ss:32 ~jj:64 ~hh:4 ~ee:16 "cwb_contraction");
  fixture "conv" (conv ~hw:10 ~ic:4 ~oc:8 "cwb_conv")

let () =
  let outcomes = !all_outcomes and kernels = !all_kernels in
  in_leg ~leg "every seeded candidate compiles, or is refused typed" (fun label ->
      p_all ~min:1 label outcomes ~f:(function _, Crashed _ -> false | _ -> true));
  in_leg ~leg "every compiled candidate reports one figure per shipped kernel" (fun label ->
      p_all ~min:1 label
        (List.filter outcomes ~f:(function _, Refused _ -> false | _ -> true))
        ~f:(function _, Compiled _ -> true | _ -> false));
  in_leg ~leg "no compiled kernel allocates more workgroup memory than its charge" (fun label ->
      p_all ~min:1 label kernels ~f:(fun k -> k.compiled <= k.charged));
  in_leg ~leg "some compiled kernel allocates exactly its positive charge" (fun label ->
      p_exists label kernels ~f:(fun k -> k.compiled > 0 && k.compiled = k.charged))
