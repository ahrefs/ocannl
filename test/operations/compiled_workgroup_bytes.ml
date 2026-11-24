(* gh-ocannl-1530, gh-ocannl-2426: the device compiler's own static workgroup allocation, kernel by
   kernel, against the schedule layer's charge for it, over every candidate a search seeds.

   [Schedule.workgroup_memory_bytes] is the charge the hardware gate holds a kernel to before its
   compile, and the one the seeding floors reproduce through [Schedule.kernel_workgroup_bytes]
   ([workgroup_memory_accounting] holds the floors to it at both boundaries). Nothing else checks it
   against what a compiled kernel actually allocates, so an allocation a transform or a renderer
   adds without a term there surfaces only as a pipeline the driver refuses, or under the limit as
   an occupancy misprediction. Here the driver answers: [Ir.Compiled_kernels] records, per linked
   kernel, Metal's [staticThreadgroupMemoryLength], CUDA's and HIP's
   [FUNC_ATTRIBUTE_SHARED_SIZE_BYTES].

   Each backend's driver rounds that reported figure up to its own reporting granule
   ([Backend_intf.codegen_capabilities.workgroup_reporting_granule], gh-ocannl-2426): Metal rounds
   the kernel's 4 B-rounded total up to 16 B, while CUDA and HIP report to 1 B. The upper bound is
   therefore [charged] rounded up to that granule — not [charged] bare, which flags unaligned Metal
   kernels as under-charged by up to 12 B.

   The claim is AT MOST, not equality: a device compiler may allocate less than the kernel declares
   (NVVM, ptxas and the Metal compiler drop an array they prove dead; the lane all-reduce partials
   are charged at the wider of two precisions), and an over-charge only costs candidates, while an
   under-charge is the defect. Each kernel's slack is printed to stderr. So that the claim can fail,
   some aligned kernel's figure must equal its positive charge exactly (a dropped term of the charge
   shows on it), a single-array 37 B [byte] staged probe must equal its granule-rounded bound (48 B
   on Metal, 37 B on CUDA and HIP — catching granule 1 vs 2 as well as coarser Metal granules, with
   no MMA requirement). On HIP, two-array probes pin inter-array padding separately from the
   trailing array. On CUDA, odd byte pairs pin the exact sum of array sizes. On CUDA and Metal,
   where f16 tensorization is advertised, some unaligned seeded kernel must equal its
   granule-rounded bound below a 32 B multiple.

   Candidates: every whole-routine seed [Autotune.search_seeding] proposes under the device's own
   limits and capabilities (sketch and coalesced; staged, tensorized and [Fuse_epilogue] twins
   included), each compiled through the seed's own schedule, over fixtures chosen for their
   allocations: a dividing GEMM with a tail (staged tiles, fused twins), a routine-local GEMM (whole
   aligned fused product), two odd f16 routine-local GEMMs (9x21 and 9x11: unaligned fused products
   of 378 and 198 B beside aligned tiles, pinning the reporting granule below and above), a padded
   GEMM (the masked accumulator tile), two f16-operand ones (half tiles; an f16 product, whose
   accumulator boundary converts on Metal: the per-kernel tile-MMA term), the attention out
   projection (the contraction-coalesced K menu's deep staged tiles) and a conv (the conv family's
   staged tiles). A candidate the gate or the backend refuses is outside the comparison (stderr says
   why); a crash fails. Kernel figures pair with the routine's shipped segments in launch order,
   which the census's count must match.

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

let round_up ~granule bytes =
  let g = max 1 granule in
  (bytes + g - 1) / g * g

(* One compiled kernel beside its segment's charge, that charge rounded up to the backend's
   reporting granule, and the share of that charge that is the backend's per-kernel tile-MMA
   scratch. *)
type kernel = {
  name : string;
  compiled : int;
  charged : int;
  reported_bound : int;
  mma_scratch : int;
}

type outcome =
  | Compiled of kernel list
  | Unpaired of string  (** The census and the shipped segments disagree in count. *)
  | Refused of string
      (** A classified refusal ([Context.compile_outcome]'s [Error (Classified _)]): the gate's, the
          backend's. *)
  | Crashed of string

type byte_pair = { kernel : kernel; metal_charge : int; hip_min : int; hip_charge : int }

let all_kernels : kernel list ref = ref []
let all_outcomes : (string * outcome) list ref = ref []
let probe_kernels : kernel list ref = ref []
let probe_outcomes : (string * outcome) list ref = ref []
let byte_pairs : byte_pair list ref = ref []
let two_tier_outcomes : (string * outcome) list ref = ref []
let one_line msg = String.prefix (String.tr ~target:'\n' ~replacement:' ' msg) 300

let kernels_of ~capabilities ~no_scratch records segments =
  if List.length records <> List.length segments then
    Unpaired
      (Printf.sprintf "%d compiled figures for %d segments" (List.length records)
         (List.length segments))
  else
    let granule = capabilities.Ir.Backend_intf.workgroup_reporting_granule in
    Compiled
      (List.map2_exn records segments ~f:(fun r seg ->
           let charged = Sched.workgroup_memory_bytes ~capabilities seg in
           {
             name = r.Ir.Compiled_kernels.kernel;
             compiled = r.static_workgroup_bytes;
             charged;
             reported_bound = round_up ~granule charged;
             mma_scratch = charged - Sched.workgroup_memory_bytes ~capabilities:no_scratch seg;
           }))

let compile_and_measure ~capabilities ~no_scratch ~candidate ~lowered_transform ctx fwd =
  match
    Ir.Compiled_kernels.with_census (fun () ->
        Context.compile_outcome ~provenance:Ir.Schedule_outcome.Candidate ~candidate
          ~lowered_transform ctx fwd Ir.Indexing.Empty)
  with
  | Ok (cctx, routine), records ->
      let outcome = kernels_of ~capabilities ~no_scratch records routine.Context.segments in
      (* Each candidate's working pools go back before the next links. *)
      Context.release cctx;
      outcome
  | Error (Ir.Schedule_outcome.Classified c), _ ->
      Refused (Ir.Schedule_outcome.detail_of_cause c.cause)
  | Error (Ir.Schedule_outcome.Fatal f), _ -> Crashed (Exn.to_string f.exn)
  | exception exn -> Crashed (Exn.to_string exn)

let report ~granule ~tag ~label outcome =
  match outcome with
  | Compiled ks ->
      List.iter ks ~f:(fun k ->
          Stdio.eprintf
            "%s: %s kernel %s: compiled %d, charged %d, bound %d (granule %d)%s (not part of the \
             golden)\n\
             %!"
            tag label k.name k.compiled k.charged k.reported_bound granule
            (if k.compiled > k.reported_bound then " UNDER-CHARGED" else ""))
  | Unpaired msg -> Stdio.eprintf "%s: %s UNPAIRED: %s (not part of the golden)\n%!" tag label msg
  | Refused msg ->
      Stdio.eprintf "%s: %s refused: %s (not part of the golden)\n%!" tag label (one_line msg)
  | Crashed msg ->
      Stdio.eprintf "%s: %s CRASHED: %s (not part of the golden)\n%!" tag label (one_line msg)

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
      let granule = capabilities.Ir.Backend_intf.workgroup_reporting_granule in
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
      let run q =
        let label = Autotune.sketch_params_label q in
        let outcome =
          compile_and_measure ~capabilities ~no_scratch ~candidate:label
            ~lowered_transform:(fun o ->
              [ Sched.apply_classified (Autotune.sketch_schedule ~accum_prec ~p:q o) o ])
            (Context.auto ()) fwd
        in
        report ~granule ~tag ~label outcome;
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

(* A staged [byte] outer-sum probe (no MMA needed): stages [u] of length [bi] in workgroup-shared
   memory and, when [stage_v] is true, [v] of length [cols] beside it. *)
let run_byte_probe ~stage_v ~tag ~bi ~cols =
  Tensor.unsafe_reinitialize ();
  let u =
    NTDSL.init ~l:(tag ^ "_u") ~prec:Ir.Ops.byte
      ~o:[ 2 * bi ]
      ~f:(fun idcs -> Float.of_int ((idcs.(0) % 7) + 1))
      ()
  in
  let v =
    NTDSL.init ~l:(tag ^ "_v") ~prec:Ir.Ops.byte ~o:[ cols ]
      ~f:(fun idcs -> Float.of_int ((idcs.(0) % 11) + 1))
      ()
  in
  let%op c = u +++ "i; j => ij" v in
  let fwd = named tag (Train.forward c) in
  let ctx = Context.auto () in
  let capabilities = Context.codegen_capabilities ctx in
  let granule = capabilities.Ir.Backend_intf.workgroup_reporting_granule in
  let lowered_transform (opt : Ir.Low_level.optimized) =
    let i, j =
      match List.find_exn (Ll_test.nest_paths opt.llc) ~f:(fun p -> List.length p = 2) with
      | [ i; j ] -> (i, j)
      | _ -> assert false
    in
    let sp_i, _, i_i =
      Sched.split ~axis:i ~factor:bi ~outer:Ir.Low_level.Grid ~inner:Ir.Low_level.Workgroup
    in
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
    let ops =
      sp_i :: stage u.Tensor.value [ i_i ]
      :: (if stage_v then [ stage v.Tensor.value [ j ] ] else [])
    in
    [ Sched.apply ops opt ]
  in
  let outcome =
    compile_and_measure ~capabilities ~no_scratch:capabilities ~candidate:tag ~lowered_transform ctx
      fwd
  in
  report ~granule ~tag ~label:tag outcome;
  ((tag, outcome), match outcome with Compiled ks -> ks | _ -> [])

let on_metal = String.equal backend_name "metal"
let on_hip = String.equal backend_name "hip"
let on_cuda = String.equal backend_name "cuda"
let two_tier_leg = Verdict.leg ~when_:on_metal ~on:backend_name ()
let hip_lds_leg = Verdict.leg ~when_:on_hip ~on:backend_name ()
let cuda_shared_leg = Verdict.leg ~when_:on_cuda ~on:backend_name ()

let () =
  fixture "dense" (gemm ~m:32 ~k:64 ~n:64 "cwb_dense");
  (* The product left to the fused tail alone: a tensorized fused twin places it, routine-local, in
     workgroup-shared memory whole. *)
  fixture ~materialize:false "dense_local" (gemm ~m:32 ~k:32 ~n:32 "cwb_dense_local");
  (* Odd f16 products left to the fused tail alone (378 B and 198 B). On Metal, 9x11's reported
     bound is 4304 B (16 mod 32), and on CUDA it is 4294 B. HIP can charge up to 15 B extra here
     because the compiler can leave the odd product at the end of the LDS layout. *)
  fixture ~materialize:false "half_local_odd"
    (gemm ~prec:Ir.Ops.half ~out_prec:Ir.Ops.half ~m:9 ~k:32 ~n:21 "cwb_half_local_odd");
  fixture ~materialize:false "half_local_9x11"
    (gemm ~prec:Ir.Ops.half ~out_prec:Ir.Ops.half ~m:9 ~k:32 ~n:11 "cwb_half_local_9x11");
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
  fixture "conv" (conv ~hw:10 ~ic:4 ~oc:8 "cwb_conv");
  (* Single-array 37 B [byte] probe: no MMA needed and no second array to inter-array-align; catches
     granule 1 vs 2 directly on CUDA and HIP (37 vs 38 B), while Metal charges 40 B and reports 48 B
     (16 mod 32). *)
  if is_gpu then (
    let out, ks = run_byte_probe ~stage_v:false ~tag:"cwb_byte_odd_37" ~bi:37 ~cols:16 in
    probe_outcomes := [ out ];
    probe_kernels := ks);
  (* Metal reports 48 B for both 4 B-rounded sums (36 B and 40 B). On HIP, the two possible orders
     of 17+15 B are bounded by 47 B, and either order of 17+17 B occupies 49 B. The equal sizes also
     expose any padding charged after the final array. *)
  if on_metal || on_hip then (
    let o1, k1 = run_byte_probe ~stage_v:true ~tag:"cwb_byte_pair_17_15" ~bi:17 ~cols:15 in
    let o2, k2 = run_byte_probe ~stage_v:true ~tag:"cwb_byte_pair_17_17" ~bi:17 ~cols:17 in
    two_tier_outcomes := [ o1; o2 ];
    byte_pairs :=
      List.map k1 ~f:(fun kernel -> { kernel; metal_charge = 36; hip_min = 33; hip_charge = 47 })
      @ List.map k2 ~f:(fun kernel -> { kernel; metal_charge = 40; hip_min = 49; hip_charge = 49 }))

let () =
  let outcomes = !all_outcomes @ !probe_outcomes @ !two_tier_outcomes
  and kernels = !all_kernels @ !probe_kernels @ List.map !byte_pairs ~f:(fun pair -> pair.kernel)
  and seeded_kernels = !all_kernels
  and probe = !probe_kernels
  and byte_pairs = !byte_pairs in
  let limits = Context.hardware_limits (Context.auto ()) in
  let seeded_f16_leg =
    let when_, aggregation =
      match
        Ll_test.tensorized_capability ~is_gpu ~is_cpu ~limits ~a_prec:Ir.Ops.half
          ~b_prec:Ir.Ops.half ~d_prec:Ir.Ops.half
      with
      | `Advertised -> (on_metal || on_cuda, None)
      | `Withheld aggregation -> (false, Some aggregation)
    in
    Verdict.leg ?aggregation ~when_ ~on:backend_name ()
  in
  in_leg ~leg "every seeded candidate compiles, or is refused typed" (fun label ->
      p_all ~min:1 label outcomes ~f:(function _, Crashed _ -> false | _ -> true));
  in_leg ~leg "every compiled candidate reports one figure per shipped kernel" (fun label ->
      p_all ~min:1 label
        (List.filter outcomes ~f:(function _, Refused _ -> false | _ -> true))
        ~f:(function _, Compiled _ -> true | _ -> false));
  in_leg ~leg
    "no compiled kernel allocates more workgroup memory than its charge rounded to the reporting \
     granule" (fun label -> p_all ~min:1 label kernels ~f:(fun k -> k.compiled <= k.reported_bound));
  in_leg ~leg "some aligned compiled kernel allocates exactly its positive charge" (fun label ->
      p_exists label seeded_kernels ~f:(fun k ->
          k.charged % 16 = 0 && k.compiled > 0 && k.compiled = k.charged));
  in_leg ~leg "the 37-byte unaligned staged kernel allocates exactly its granule-rounded bound"
    (fun label ->
      p_all ~min:1 label probe ~f:(fun k ->
          k.charged % 16 <> 0 && k.compiled = k.reported_bound && k.reported_bound % 32 <> 0));
  in_leg ~leg:two_tier_leg
    "two-array odd-byte probes distinguish 16 B kernel-total rounding over 4 B-aligned arrays from \
     exact-sum and per-array rounding" (fun label ->
      p_all ~min:2 label byte_pairs ~f:(fun { kernel = k; metal_charge; _ } ->
          k.charged = metal_charge && k.compiled = k.reported_bound && k.compiled = 48));
  in_leg ~leg:hip_lds_leg
    "HIP odd-byte pairs charge inter-array padding and preserve the trailing array size"
    (fun label ->
      p_all ~min:2 label byte_pairs ~f:(fun { kernel = k; hip_min; hip_charge; _ } ->
          k.charged = hip_charge && k.compiled >= hip_min));
  in_leg ~leg:cuda_shared_leg "CUDA odd-byte pairs charge and allocate exactly their array sizes"
    (fun label ->
      p_all ~min:2 label [ 17; 33 ] ~f:(fun size ->
          let tag = Printf.sprintf "cwb_byte_pair_%d_%d" size size in
          match run_byte_probe ~stage_v:true ~tag ~bi:size ~cols:size with
          | _, [ k ] -> k.charged = 2 * size && k.compiled = 2 * size
          | _ -> false));
  in_leg ~leg:seeded_f16_leg
    "some unaligned seeded kernel allocates exactly its granule-rounded bound, not a coarser \
     multiple" (fun label ->
      p_exists label seeded_kernels ~f:(fun k ->
          k.charged % 16 <> 0 && k.compiled = k.reported_bound && k.reported_bound % 32 <> 0))
