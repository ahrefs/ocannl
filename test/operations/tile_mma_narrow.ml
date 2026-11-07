(* Tile_mma register tiling for narrow (16-bit) operands (gh-ocannl-575, re-scoped from
   gh-ocannl-530, originally gh-ocannl-516/517).

   Three executed legs of the packed GEBP pipeline (the [schedule_pack_mma_matmul] composition: B~
   packed at [k_o], A~ at [i_o], [Tensorize] at unit lane width), each against its serial twin:

   - bf16 storage computing in f32 (the default [narrow_compute_f32] policy): the packing [Stage]s
   mint their tiles at f32 ([tile_prec = Some single]) so the widening conversion rides the packing
   copy — the micro-kernel reads uniform f32 panels and only the accumulator crosses the memory
   boundary through the narrow bridge ("narrow storage bridged: d:bfloat16"). - half storage with a
   HOISTED f32 B~ panel (gh-ocannl-470 composed with the widening pack): the host-side
   [pack_constant_tile] converts half host-init data into the f32 constant-pool panel. - pure fp16
   (gh-ocannl-516 task 3): [cc_fp16_arithmetic=native] (forced via the environment — on a merely
   promoted target the compiler still implements per-op half rounding, so values are unchanged and
   only the throughput claim would be off) plus the [fp16_arithmetic] policy keep the compute
   precision at half: the register tiling runs at doubled lanes with f16 accumulators and identity
   bridges. Skipped (with a stderr note) where the toolchain has no [_Float16] at all — the
   forced-native kernel then fails to compile.

   Inputs are exactly representable at their storage precision with all partial sums exact (the
   [schedule_mma_matmul] discipline), so every parity check is BITWISE; the structural pins are what
   assert the rendering. Since gh-ocannl-639 the bitwise parity no longer depends on that input
   discipline for the accumulator: the serial fallback holds its accumulator at compute precision
   across the whole k extent and narrows once, exactly like the register tiling — the final bf16
   section checks the parity on inputs whose partial sums are NOT bf16-exact, which the pre-639
   per-k-step narrowing would visibly split (test/operations/accum_width.ml pins that divergence as
   its policy-off negative control). On GPU backends the trailing [Tensorize] is dropped (a
   unit-lane [Tile_mma] must not reach hardware intrinsics): the widened packing is still exercised
   for value parity. *)

(* cc-fp-contract: exact-operands -- dyadic operands keep every product and sum exact, the native
   f16 accumulators included *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

(* Before any backend touch: the pure-f16 leg's probe override (read per call, env is live). *)
let () = Unix.putenv "OCANNL_CC_FP16_ARITHMETIC" "native"
let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let facts = Context.backend_facts_of_name backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })
let n = 64
let bm, bk = (16, 16)

(* Pure-f16 compute runs the register file at the machine's full vector width in 2-byte elements, so
   the typedef the kernel mints is per-machine (8 lanes on NEON, 16 on AVX2, 32 on AVX-512) and is
   computed rather than spelled — a hardcoded list of widths reads as a failure on the first machine
   wider than the ones it lists. GPU backends report 0 bytes and never reach this. *)
let half_vec_typ =
  let simd_vector_bytes =
    (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.simd_vector.bytes
  in
  Printf.sprintf "ocannl_vec%dh"
    (Option.value ~default:0
       (Ir.Backend_intf.simd_lanes_for ~vector_bytes:simd_vector_bytes
          ~elt_bytes:(Ir.Ops.prec_in_bytes Ir.Ops.half)
          ~extent:n))

(* The composed packed pipeline, parameterized by the packed tiles' precision override (and, for the
   gh-ocannl-639 section, by the k blocking — [bk = n] makes the single register tile cover the
   whole k extent, so the C-tile narrows once per cell like the serial fallback). *)
let composed_schedule ?(bk = bk) ?(tensorize = on_cpu) ~hoist_b ~tile_prec ~a ~b
    (opt : LL.optimized) : Sched.schedule =
  let paths = Ll_test.nest_paths opt.LL.llc in
  let i, j, k =
    match List.find_exn paths ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let stage ~hoisted source tile_loops =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = false;
        cooperative = None;
        hoisted;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec;
      }
  in
  [ sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ sink i_o [ k_o ]
  @ [ stage ~hoisted:hoist_b b [ k_i; j ]; stage ~hoisted:false a [ i_i; k_i ] ]
  @ if tensorize then [ fst (Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 ()) ] else []

(* Which way each compiled [Tile_mma] went is the routine's mma census to say, never the emitted
   text (gh-ocannl-1316): on the C backends the schedule ends in [Tensorize], and every statement
   must render the register tiling rather than the lane-0 scalar fallback whose guard wraps both; on
   GPU backends it stops short of [Tensorize], so no [Tile_mma] may reach the hardware at all. The
   text pins beside these claims are about the text itself — tile declarations, the precision
   bridges and their annotations, the vector typedef. *)
let rendered_as_scheduled ~what census =
  if on_cpu then Ll_test.all_rendered ~what Ir.C_syntax.Mma_register_tiled census
  else Ll_test.none_emitted ~what census

(* The readback, and the compile's census. *)
let run_census ~name ?schedule (out : Tensor.t) =
  let comp = named name (Train.forward out) in
  let transform opt =
    match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
  in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx out.Tensor.value, Ll_test.renderings routine.Context.mma)

let run ~name ?schedule out = fst (run_census ~name ?schedule out)

let () =
  (* === bf16 storage, f32 compute, both panels widened at pack time === Inputs: a in {0, 1/4, 1/2},
     b in {-1, -1/2, 0, 1/2, 1}; every product is a multiple of 1/8 bounded by 1/2, and every
     partial sum over k = 64 is a multiple of 1/8 bounded by 32 — exactly representable in bf16's 8
     significand bits, so parity is bitwise on either rendering and accumulator width. *)
  let mab =
    NTDSL.init ~l:"mab" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:3 ~offset:0. ~stride:0.25)
      ()
  in
  let mbb =
    NTDSL.init ~l:"mbb" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
      ()
  in
  let%op bc0 = mab * mbb in
  Tn.update_prec bc0.Tensor.value Ir.Ops.bfloat16;
  let want_b = nonzero "nrw_bf16_serial" (run ~name:"nrw_bf16_serial" bc0) in
  let%op bc1 = mab * mbb in
  Tn.update_prec bc1.Tensor.value Ir.Ops.bfloat16;
  let got_b, census_b =
    run_census ~name:"nrw_bf16_packed"
      ~schedule:
        (composed_schedule ~hoist_b:false ~tile_prec:(Some Ir.Ops.single) ~a:mab.Tensor.value
           ~b:mbb.Tensor.value)
      bc1
  in
  p_all2 "bf16 packed+tensorized matmul matches the serial twin bitwise" got_b want_b ~f:Float.equal;
  (let src = Generated.read "nrw_bf16_packed" in
   let has s = String.is_substring src ~substring:s in
   let count_sub sub = String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length in
   p "bf16 packing widens into f32 panels; only d crosses the narrow bridge"
     (rendered_as_scheduled ~what:"nrw_bf16_packed" census_b
     &&
     if on_cpu then
       (* Two f32 tile arrays: the widening (the C backends' [bfloat16_to_single]) rides the packing
          copy; the register tiling sees f32 panels, bridging only the accumulator. *)
       count_sub ("float " ^ Generated.node_ident ~facts "tile_") = 2
       && has "bfloat16_to_single"
       && has "narrow storage bridged: d:bfloat16."
       && has "OCANNL_VEC_WIDEN_BFLOAT16" && has "OCANNL_VEC_NARROW_BFLOAT16"
     else count_sub ("float " ^ Generated.node_ident ~facts "tile_") = 2));

  (* === gh-ocannl-639: bf16 with partial sums that are NOT bf16-exact === *)
  (* Products are multiples of 15/128 with a nonzero mean (~0.23 drift per k step), so the running
     sums outgrow bf16's 8 significand bits and per-k-step narrowing would visibly split these
     legs. They stay bitwise equal because the accumulation width is policy, not schedule: the
     serial fallback holds its accumulator at compute precision across the whole k extent and
     narrows once at the store, exactly like the register tiling. The reduction is exact in f32
     (multiples of 1/128, magnitude far below 2^16), so both legs compute the same exact sums —
     and [bk = n] gives the packed leg a single whole-k tile, so both narrow at the same single
     point. (The narrowing POINTS remain a property of the schedule's reduction structure: a
     k-blocked schedule, [bk < k], stores bf16 partials at every block boundary by construction,
     which on these inputs would round where the serial fallback does not — gh-ocannl-639 unifies
     the accumulator's width, not the blocking.) *)
  let mai =
    NTDSL.init ~l:"mai" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:3 ~offset:1. ~stride:0.375)
      ()
  in
  let mbi =
    NTDSL.init ~l:"mbi" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:5 ~offset:(-1.5) ~stride:0.625)
      ()
  in
  let%op ic0 = mai * mbi in
  Tn.update_prec ic0.Tensor.value Ir.Ops.bfloat16;
  let want_i = nonzero "nrw_bf16_inexact_serial" (run ~name:"nrw_bf16_inexact_serial" ic0) in
  let%op ic1 = mai * mbi in
  Tn.update_prec ic1.Tensor.value Ir.Ops.bfloat16;
  let got_i =
    run ~name:"nrw_bf16_inexact_packed"
      ~schedule:
        (composed_schedule ~bk:n ~hoist_b:false ~tile_prec:(Some Ir.Ops.single) ~a:mai.Tensor.value
           ~b:mbi.Tensor.value)
      ic1
  in
  p_all2 "bf16 matmul with inexact partial sums matches the serial twin bitwise (gh-ocannl-639)"
    got_i want_i ~f:Float.equal;

  (* === half storage, hoisted f32 B~ panel (host-side converting pack) + in-kernel f32 A~ === *)
  (* Half leaves are minted at their precision ([ndarray] settles a leaf as [Specified] single);
     the B operand is additionally declared constant so the hoisted pack applies to it. *)
  let mah =
    NTDSL.init ~l:"mah" ~prec:Ir.Ops.half ~i:[ n ] ~o:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:5 ~offset:0. ~stride:0.125)
      ()
  in
  let mbh =
    NTDSL.init ~l:"mbh" ~prec:Ir.Ops.half ~i:[ n ] ~o:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:7 ~offset:(-3.) ~stride:0.25)
      ()
  in
  Tn.set_host_constant mbh.Tensor.value;
  let%op hc0 = mah * mbh in
  Tn.update_prec hc0.Tensor.value Ir.Ops.half;
  let want_h = nonzero "nrw_half_serial" (run ~name:"nrw_half_serial" hc0) in
  let%op hc1 = mah * mbh in
  Tn.update_prec hc1.Tensor.value Ir.Ops.half;
  let got_h, census_h =
    run_census ~name:"nrw_half_hoisted"
      ~schedule:
        (composed_schedule ~hoist_b:true ~tile_prec:(Some Ir.Ops.single) ~a:mah.Tensor.value
           ~b:mbh.Tensor.value)
      hc1
  in
  p_all2 "half hoisted-packed matmul matches the serial twin bitwise" got_h want_h ~f:Float.equal;
  (let src = Generated.read "nrw_half_hoisted" in
   let has s = String.is_substring src ~substring:s in
   let count_sub sub = String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length in
   p "half hoisted pack converts on the host; only d crosses the narrow bridge"
     ((* One in-kernel tile (A~); the B~ panel is a constant-pool buffer packed (and widened) on the
         host, so no in-kernel copy of it exists. *)
      count_sub ("float " ^ Generated.node_ident ~facts "tile_") = 1
     && rendered_as_scheduled ~what:"nrw_half_hoisted" census_h
     && ((not on_cpu)
        || has "narrow storage bridged: d:half."
           && has "OCANNL_VEC_WIDEN_HALF" && has "OCANNL_VEC_NARROW_HALF")));

  (* === pure fp16: native (forced) f16 arithmetic, f16 accumulators, doubled lanes === *)
  let saved_policy = Numerics.get () in
  Numerics.set_policy { saved_policy with fp16_arithmetic = Numerics.Fp16_narrow };
  (let%op fc0 = mah * mbh in
   Tn.update_prec fc0.Tensor.value Ir.Ops.half;
   let%op fc1 = mah * mbh in
   Tn.update_prec fc1.Tensor.value Ir.Ops.half;
   let parity_name = "pure-f16 packed+tensorized matmul matches the serial twin bitwise" in
   let structure_name = "pure-f16 register tiling: half lanes, identity bridges" in
   match
     let want_f = nonzero "nrw_f16_serial" (run ~name:"nrw_f16_serial" fc0) in
     let got_f =
       run_census ~name:"nrw_f16_packed"
         ~schedule:
           (composed_schedule ~hoist_b:false ~tile_prec:None ~a:mah.Tensor.value ~b:mbh.Tensor.value)
         fc1
     in
     (want_f, got_f)
   with
   | want_f, (got_f, census_f) ->
       p_all2 parity_name got_f want_f ~f:Float.equal;
       let src = Generated.read "nrw_f16_packed" in
       let has s = String.is_substring src ~substring:s in
       let count_sub sub =
         String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
       in
       p structure_name
         (rendered_as_scheduled ~what:"nrw_f16_packed" census_f
         && ((not on_cpu)
            (* Half-typed tiles, an [ocannl_vec<2*f32 lanes>h] register file, no narrow bridge —
               storage and compute coincide. *)
            || count_sub "HALF_T tile_" = 2
               && (not (has "narrow storage bridged"))
               && has half_vec_typ))
   | exception e ->
       (* The one condition verified to warrant a skip: the generated kernel failed to COMPILE over
          the half type ([Backend_rejected]'s detail quotes the compiler output, which names
          [HALF_T]/[_Float16] when the type is the problem). In fact even a no-[_Float16] toolchain
          is expected to run this leg — [HALF_T] is then [uint16_t] and every half operation the
          tiling emits routes through the emulation-safe [OCANNL_HALF_FMA] / [HALF_TO_FLOAT] macros,
          the same ones the serial twin uses — so this arm should never fire; it exists so an exotic
          toolchain reads as a loud skip rather than a red suite. Every other failure (schedule
          construction, runtime, parity readback) propagates: a swallowed exception here would
          false-green exactly when the implementation breaks. *)
       let msg = Exn.to_string e in
       let mentions s = String.is_substring msg ~substring:s in
       if mentions "failed to compile" && (mentions "_Float16" || mentions "HALF_T") then (
         Stdio.eprintf "pure-f16 leg unavailable, toolchain rejected the half kernel (%s)\n%!" msg;
         Verdict.skipped ~aggregation:`Environment ~backend:backend_name parity_name;
         Verdict.skipped ~aggregation:`Environment ~backend:backend_name structure_name)
       else raise e);
  Numerics.set_policy saved_policy

(* === gh-ocannl-1975: a virtual cast twin on the tile path ===

   A twin of a wider master reads as [Round_prec] of the master, so the matmul recognizers seed on
   it, and a tile cannot honour the operator: its operands are a node and indices. The rounding is
   the pack's, once per cell, and the micro-kernel reads a plain panel. These masters sit under half
   an f16 ulp above values with few bits, so a tile reading them unrounded computes another product,
   and what the rounded values give is worked out here in double, whatever renders it: multiples of
   1/16 bounded by 4, 64 of them to a cell, which f32 holds as they are. *)
let () =
  let eps = 2. **. -12. in
  let dims = [| n; n |] in
  let a_base = Ll_test.cycle ~dims ~modulus:3 ~offset:4. ~stride:0.25 in
  let b_base = Ll_test.cycle ~dims ~modulus:5 ~offset:4. ~stride:0.25 in
  let master label base =
    NTDSL.init ~l:label ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ] ~f:(fun idcs -> base idcs +. eps) ()
  in
  let twin_at prec (m : Tensor.t) =
    let t = Operation.cast ~grad_spec:Tensor.Prohibit_grad m () in
    Tn.update_prec t.Tensor.value prec;
    Train.set_virtual t.Tensor.value;
    t
  in
  let matmul a_cell b_cell =
    Array.init (n * n) ~f:(fun flat ->
        let i = flat / n and j = flat % n in
        Array.sum
          (module Float)
          (Array.init n ~f:Fn.id)
          ~f:(fun k -> a_cell [| i; k |] *. b_cell [| k; j |]))
  in
  let leg ~label ~round_a ~round_b =
    let ma = master ("ma_" ^ label) a_base and mb = master ("mb_" ^ label) b_base in
    let a = if round_a then twin_at Ir.Ops.half ma else ma in
    let b = if round_b then twin_at Ir.Ops.half mb else mb in
    let%op c = a * b in
    Tn.update_prec c.Tensor.value Ir.Ops.single;
    let name = "nrw_round_" ^ label in
    let got, census =
      run_census ~name
        ~schedule:
          (composed_schedule ~hoist_b:false ~tile_prec:(Some Ir.Ops.single) ~a:ma.Tensor.value
             ~b:mb.Tensor.value)
        c
    in
    let cell rounded base idcs = if rounded then base idcs else base idcs +. eps in
    let want = matmul (cell round_a a_base) (cell round_b b_base) in
    p_all2
      (Printf.sprintf "%s: the tile reads the rounded values, exactly" label)
      got want ~f:Float.equal;
    p
      (Printf.sprintf "%s: an unrounded read would have computed another product" label)
      (not
         (Array.equal Float.equal want
            (matmul (fun idcs -> a_base idcs +. eps) (fun idcs -> b_base idcs +. eps))));
    let src = Generated.read name in
    p
      (Printf.sprintf "%s: rendered as scheduled, the pack holding the rounding" label)
      (rendered_as_scheduled ~what:name census
      && ((not on_cpu) || String.is_substring src ~substring:"FLOAT_TO_HALF"))
  in
  leg ~label:"a" ~round_a:true ~round_b:false;
  leg ~label:"b" ~round_a:false ~round_b:true;
  leg ~label:"both" ~round_a:true ~round_b:true;

  (* A pack's edge guard stays out of the rounding's way: a block size that does not divide the
     extent leaves the load's [Where] live, and the store it sits in renders at compute
     precision. *)
  (let ma = master "ma_edge" a_base and mb = master "mb_edge" b_base in
   let a = twin_at Ir.Ops.half ma in
   let%op c = a * mb in
   Tn.update_prec c.Tensor.value Ir.Ops.single;
   let got =
     run ~name:"nrw_round_edge"
       ~schedule:
         (composed_schedule ~tensorize:false ~bk:24 ~hoist_b:false ~tile_prec:(Some Ir.Ops.single)
            ~a:ma.Tensor.value ~b:mb.Tensor.value)
       c
   in
   p_all2 "edge: a non-dividing pack rounds every cell it loads" got
     (matmul a_base (fun idcs -> b_base idcs +. eps))
     ~f:Float.equal);

  (* An f64 master goes to fp8 in one step on the pack too: through the tile's f32 it would round
     twice, and 1.125 + 2^-40 would read 1. *)
  (let supports_f64 =
     (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.supports_f64
   in
   let leg = Verdict.leg ~when_:supports_f64 ~on:backend_name () in
   in_leg ~leg "f64 master: the pack rounds to fp8 in one step" (fun label ->
       let off_tie = 1.125 +. (2. **. -40.) in
       let scale = Ll_test.weighted ~weights:[| 1; 1 |] ~modulus:2 ~offset:1. ~stride:1. in
       let m =
         NTDSL.init ~l:"m_f64" ~prec:Ir.Ops.double ~i:[ n ] ~o:[ n ]
           ~f:(fun idcs -> off_tie *. scale idcs)
           ()
       in
       let mb = master "mb_f64" b_base in
       let a = twin_at Ir.Ops.fp8 m in
       let%op c = a * mb in
       Tn.update_prec c.Tensor.value Ir.Ops.single;
       let got =
         run ~name:"nrw_round_f64"
           ~schedule:
             (composed_schedule ~hoist_b:false ~tile_prec:(Some Ir.Ops.single) ~a:m.Tensor.value
                ~b:mb.Tensor.value)
           c
       in
       p_all2 label got
         (matmul (fun idcs -> 1.25 *. scale idcs) (fun idcs -> b_base idcs +. eps))
         ~f:Float.equal));

  (* One tile holds one value per cell, and a hoisted panel has no load nest to round in. *)
  let refused name ~because schedule_of (out : Tensor.t) =
    let refusal = ref None in
    let transform opt =
      (refusal :=
         match Sched.apply (schedule_of opt) opt with
         | _ -> Some false
         | exception Invalid_argument why -> Some (String.is_substring why ~substring:because));
      opt
    in
    let comp = named name (Train.forward out) in
    let (_ : Context.t * _) =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    p name (Option.value !refusal ~default:false)
  in
  (let m = master "m_mixed" a_base in
   let t = twin_at Ir.Ops.half m in
   let%op c = t *. m in
   Tn.update_prec c.Tensor.value Ir.Ops.single;
   refused "a rounded and a raw read of one source do not share a tile" ~because:"round differently"
     (fun opt ->
       let i, j =
         match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 2) with
         | [ i; j ] -> (i, j)
         | _ -> assert false
       in
       [
         Sched.Stage
           {
             source = m.Tensor.value;
             tile_loops = [ i; j ];
             shared = false;
             cooperative = None;
             hoisted = false;
             swizzle = None;
             pad_stride = None;
             pipeline_depth = 1;
             tile_prec = None;
           };
       ])
     c);
  (* And a tile reads memory: a rounding it took for an operand would be a rounding it dropped. *)
  (let ma = master "ma_place" a_base and mb = master "mb_place" b_base in
   let a = twin_at Ir.Ops.half ma in
   let%op c = a * mb in
   Tn.update_prec c.Tensor.value Ir.Ops.single;
   refused "Tensorize refuses a rounded operand a pack has not absorbed" ~because:"product of reads"
     (fun opt ->
       match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
       | [ i; j; k ] -> [ fst (Sched.tensorize ~i ~j ~k ~simd_width:1 ()) ]
       | _ -> assert false)
     c);
  let ma = master "ma_hoist" a_base and mb = master "mb_hoist" b_base in
  let b = twin_at Ir.Ops.half mb in
  let%op c = ma * b in
  Tn.update_prec c.Tensor.value Ir.Ops.single;
  refused "a hoisted panel cannot round a read" ~because:"no load nest"
    (composed_schedule ~hoist_b:true ~tile_prec:(Some Ir.Ops.single) ~a:ma.Tensor.value
       ~b:mb.Tensor.value)
    c

(* The tuner's own path to the same tile (gh-ocannl-1975; the default schedule ships none). Every
   matmul seed of a site reading two twins goes through its real pipeline, and then a search ships
   whatever it ships: which seed wins is the box's business, what it computes is not. *)
let () =
  let eps = 2. **. -12. in
  let dims = [| n; n |] in
  let a_base = Ll_test.cycle ~dims ~modulus:3 ~offset:4. ~stride:0.25 in
  let b_base = Ll_test.cycle ~dims ~modulus:5 ~offset:4. ~stride:0.25 in
  let product () =
    let master label base =
      NTDSL.init ~l:label ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ] ~f:(fun idcs -> base idcs +. eps) ()
    in
    let twin (m : Tensor.t) =
      let t = Operation.cast ~grad_spec:Tensor.Prohibit_grad m () in
      Tn.update_prec t.Tensor.value Ir.Ops.half;
      Train.set_virtual t.Tensor.value;
      t
    in
    let a = twin (master "ma_tuned" a_base) and b = twin (master "mb_tuned" b_base) in
    let%op c = a * b in
    Tn.update_prec c.Tensor.value Ir.Ops.single;
    c
  in
  let want =
    Array.init (n * n) ~f:(fun flat ->
        let i = flat / n and j = flat % n in
        Array.sum
          (module Float)
          (Array.init n ~f:Fn.id)
          ~f:(fun k -> a_base [| i; k |] *. b_base [| k; j |]))
  in
  let exact got = Array.equal Float.equal got want in
  let limits = Context.hardware_limits (Context.auto ()) in
  let caps = Context.codegen_capabilities (Context.auto ()) in
  let seeds =
    let found = ref [] in
    let comp = named "nrw_seeds" (Train.forward (product ())) in
    let (_ : Context.t * Context.routine) =
      Context.compile
        ~lowered_transform:(fun opt ->
          found :=
            Autotune.sketch_seed_params
              ~is_gpu:(Ir.Backend_intf.binds_hardware_grid facts)
              ~is_cpu:on_cpu ~limits opt;
          [ opt ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    List.filter !found ~f:(fun p -> p.Autotune.sk_mma && not p.sk_conv)
  in
  let leg = Verdict.leg ~when_:on_cpu ~on:backend_name () in
  in_leg ~leg "tuned: the matmul seeds of a site reading twins round, and some ship a tile"
    (fun label ->
      let runs =
        List.mapi (List.take seeds 6) ~f:(fun i p ->
            let name = Printf.sprintf "nrw_seed_%d" i in
            run_census ~name
              ~schedule:(fun opt ->
                Autotune.sketch_schedule ~accum_prec:caps.Ir.Backend_intf.accum_prec ~p opt)
              (product ()))
      in
      p label
        (List.for_all runs ~f:(fun (got, _) -> exact got)
        && List.exists runs ~f:(fun (_, census) ->
            List.exists census ~f:(Ir.C_syntax.equal_mma_rendering Mma_register_tiled))));
  let cache_dir = "autotune_cache_tile_mma_narrow" in
  if Stdlib.Sys.file_exists cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let reports = ref [] in
  let out = product () in
  let comp = named "nrw_tuned" (Train.forward out) in
  let ctx, routine =
    Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> reports := r :: !reports)
      (Context.auto ()) out comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  p "tuned: the shipped routine computes the rounded values"
    (exact (Context.get_values ctx out.Tensor.value));
  in_leg ~leg "tuned: the search seeds the site through its twins" (fun label ->
      p label (List.exists !reports ~f:(fun r -> r.Autotune.mma_candidates > 0)))
