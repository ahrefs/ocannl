open Base

(* Whether bf16 reduction accumulators reside in f32 on this backend: under [Numerics.Bf16_wide],
   and since gh-ocannl-1051 under [Bf16_auto] too -- only [Bf16_narrow] (the [approximate] profile's
   pin) keeps gfx11's bf16-accumulate WMMA and storage-width serial accumulators. The trade,
   measured on gfx1151 (unified) and gfx1102 (discrete) under a drift-controlled A/B: the narrow
   tensorized arm is not exactly rounded, and a 2048-term staged GEMM cell read 93 against the
   unscheduled reference's 536, where the wide arm's first difference was 34.25 against 34.5; wide
   costs up to ~8% on some tensorized cells (intrinsic to the f32-accumulate path, not the
   gh-ocannl-1064 boundary) and speeds the serial legs up 5-21%. The one predicate behind
   [Hip_backend]'s [accum_prec] and [mma_accumulator], so emission and residency cannot disagree;
   the cache key follows [accum_prec] by derivation
   ([Backend_intf.codegen_capabilities_fingerprint], gh-ocannl-1117). *)
let bf16_accum_wide () =
  match (Numerics.get ()).Numerics.bf16_arithmetic with
  | Numerics.Bf16_auto | Numerics.Bf16_wide -> true
  | Numerics.Bf16_narrow -> false

(* Whether f16 reduction accumulators reside in f32 on this backend: under [Numerics.Fp16_wide], and
   since gh-ocannl-2195 under [Fp16_auto] too -- only an explicit [Fp16_narrow] keeps gfx11's
   f16-accumulate WMMA and storage-width serial accumulators (the profiles' [Fp16_prefer_narrow] is
   floored to auto here, [fp16_narrow_floored]). Why: gh-ocannl-2151's census measured gfx1151's f16
   WMMA narrowing per ordered pair (odd rows exchanging the two 8-term halves of each 16-deep block)
   where the storage-width serial loop narrows per step, so under a narrow auto the value depended
   on whether the tuner crowned a tile. The constant is published to seeding as
   [capability.mma_f16_auto_wide]. *)
let fp16_auto_wide = true

(* Whether this backend floors [Numerics.Fp16_prefer_narrow] -- the [performance] and [approximate]
   profiles' request -- to its auto (wide): yes (gh-ocannl-2233). On gfx11 the f16-accumulate WMMA
   arm buys nothing: on gfx1151 its whole-output checksum misses the exact answer by 7.2% at n = k =
   1024 and 82% at n = k = 2048 (the narrow serial loop by 0.11% and 0.19%), and the wide arm is
   faster (0.57x and 0.60x on gfx1151's [mma_pd2], 0.21-0.34x on gfx1102's tensorized variants;
   gh-ocannl-2195). The floor covers the serial residency too, through the one predicate below, so a
   floored profile stays schedule-uniform. It is HIP-wide, not gfx11-only: these tables are
   device-free, and gfx12, the other rocWMMA family, is unmeasured. An explicit [Fp16_narrow]
   ([fp16_arithmetic=true]) still lifts it. Published to seeding as
   [capability.mma_f16_narrow_floored]. *)
let fp16_narrow_floored = true

let fp16_accum_wide () =
  Numerics.fp16_accum_wide ~auto_wide:fp16_auto_wide ~narrow_floored:fp16_narrow_floored ()

let tile = 16

(* The combination table both hooks consult: per storage triple, the precision the accumulator
   fragment resides at and the wmma leading-dimension multiples for a/b and for d. The accumulator
   and the destination storage coincide on every arm but the wide-f16 and wide-bf16 ones; where they
   differ, [Hip_backend]'s [mma_d_boundary_lines] carries the conversion. Arms and [capability]'s
   STORAGE triples must correspond, which [test/operations/mma_format_arm_tables] checks as a pure
   query under every numerics policy (gh-ocannl-1246). [mma_arm] publishes this table to the
   schedule cache's identity (gh-ocannl-1153). *)
let mma_accumulator ~a_prec ~b_prec ~d_prec =
  match (a_prec, b_prec, d_prec) with
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ -> Some (Ops.single, 8, 4)
  (* The uniform-f16 arm's accumulator follows the [Numerics] policy, and the two arms are mutually
     exclusive by construction. Under [Fp16_narrow] the accumulator fragment is itself f16, so the
     [d] boundary is rocWMMA's own load/store. Under [Fp16_wide] (gh-ocannl-789), and since
     gh-ocannl-2195 under [Fp16_auto] ([fp16_accum_wide]), the accumulator is [float] against the
     same f16 STORAGE, and [mma_d_boundary_lines] converts once at each end -- which is what lets
     [mma_f16_wide_acc_scopes] advertise both scopes and the uniform-f16 seeds survive the wide
     policy on this backend. *)
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when fp16_accum_wide () ->
      Some (Ops.single, 8, 8)
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ -> Some (Ops.half, 8, 8)
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ -> Some (Ops.single, 8, 4)
  (* The uniform-bf16 twin of the wide-f16 arm (gh-ocannl-838): under [Bf16_wide], and since
     gh-ocannl-1051 under the default [Bf16_auto], a [float] accumulator against the bf16 STORAGE
     destination, converted by [mma_d_boundary_lines]. gfx11's bf16-accumulate WMMA is not exactly
     rounded (about a bf16 ulp at the partial-sum scale, see schedule_mma_matmul's table; grossly
     more over long reductions), so this leaves only the narrowing rounding. *)
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ when bf16_accum_wide () ->
      Some (Ops.single, 8, 8)
  (* [Bf16_narrow] only: the bf16-accumulate arm. *)
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ -> Some (Ops.bfloat16, 8, 8)
  | _ -> None

(* Both hooks consult [mma_accumulator] in both scopes, and the gfx11/gfx12 + rocWMMA gate is the
   device's ([hardware_limits.mma], [Hip_backend]'s [mma_supported]), so the arm names no floor. *)
let mma_arm ~a_prec ~b_prec ~d_prec ~scope:_ =
  Option.map (mma_accumulator ~a_prec ~b_prec ~d_prec) ~f:(fun (arm_accumulator, _, _) ->
      { Backend_intf.arm_name = "rocwmma"; arm_accumulator; arm_floor = None })

let capability =
  let t = (tile, tile, tile) in
  {
    Backend_intf.mma_simd_width = 32;
    mma_tile = t;
    (* rocWMMA has both accumulator widths for both operand pairs — the CUDA asymmetry that
       motivated the accumulator key (gh-ocannl-545) does not exist here. *)
    mma_format_tiles =
      [
        ((Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f32), t);
        ((Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f16), t);
        ((Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_f32), t);
        ((Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16), t);
      ];
    (* gh-ocannl-789: rocWMMA's [(f16, f16, f32)] fragments now carry the uniform-f16 arm under
       [Numerics.Fp16_wide] too — [mma_accumulator] pairs a [float] accumulator with the f16 STORAGE
       destination and [mma_d_boundary_lines] converts elementwise at each end — so the wide policy
       no longer costs this backend its f16 tensor-unit legs (gh-ocannl-680's stated remainder). *)
    mma_f16_wide_acc_scopes = [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
    mma_f16_auto_wide = fp16_auto_wide;
    mma_f16_narrow_floored = fp16_narrow_floored;
    (* gh-ocannl-838: the uniform-bf16 arm swaps the same way under [Numerics.Bf16_wide], through
       rocWMMA's [(bf16, bf16, f32)] fragments and the same converted boundary. *)
    mma_bf16_wide_acc_scopes = [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
    (* rocWMMA fragments are opaque like wmma's: no swizzle-aware fragment load here (gh-ocannl-481
       item 3, D3). *)
    mma_staged_layouts = [];
    mma_pipeline_depths = [];
  }

(* {1 Acceptance and operand loads (gh-ocannl-1456)} *)

(* The operand conditions every rocWMMA rendering checks, written once for the three sites that used
   to repeat them ([Hip_backend]'s update-only and self-contained [mma_syntax] arms and its
   [mma_fragment_syntax]): the triple has an arm, every layout is plain (rocWMMA fragments are
   opaque, with no swizzle-aware load: gh-ocannl-481 item 3, D2), the tile divides the extents, the
   leading dimensions meet the arm's multiples, and every addressed operand is device or shared
   memory (generic-address loads cover both; a per-thread stack is not a jointly-owned tile). An
   update-only statement addresses no destination: its accumulator is the resident fragment
   array. *)
let accepts ~a_prec ~b_prec ~d_prec ~m ~n ~k ~d ~a:(lda, (a_space, a_layout))
    ~b:(ldb, (b_space, b_layout)) =
  let plain = function `Plain -> true | `Swizzled_b128 -> false in
  let loadable = function `Device | `Shared -> true | `Thread -> false in
  match mma_accumulator ~a_prec ~b_prec ~d_prec with
  | None -> false
  | Some (_, ab_ld_mult, d_ld_mult) ->
      (match d with
        | `Resident d_layout -> plain d_layout
        | `Addressed (ldd, (d_space, d_layout)) ->
            plain d_layout && loadable d_space && ldd % d_ld_mult = 0)
      && plain a_layout && plain b_layout
      && m % tile = 0
      && n % tile = 0
      && k % tile = 0
      && lda % ab_ld_mult = 0
      && ldb % ab_ld_mult = 0
      && loadable a_space && loadable b_space

let frag_typ kind typ layout =
  Printf.sprintf "rocwmma::fragment<rocwmma::%s, %d, %d, %d, %s%s>" kind tile tile tile typ
    (match layout with Some l -> ", rocwmma::" ^ l | None -> "")

(* The serial [k] extent of one block statement, shared by the self-contained rendering ([acc] = its
   block-local [__mma_acc]) and the update-only one against a resident fragment array
   (gh-ocannl-480): per reduction step, the row of [nt] [b] fragments, then for each of the [mt] [a]
   fragments its updates across that row. Each fragment is one whole [load_matrix_sync], once per
   step, reused across the other operand's free axis: what [operand_fragments] says. Transposed
   storage ([ta] / [tb]): the stored matrix is the role's transpose -- index it at (col, row) and
   declare the fragment [col_major]; the leading dimension stays the operand's own. *)
let k_loop_lines ~ab_typ ~acc ~ta ~tb ~lda ~ldb ~m ~n ~k =
  let mt = m / tile and nt = n / tile and kt = k / tile in
  let a_layout = if ta then "col_major" else "row_major" in
  let b_layout = if tb then "col_major" else "row_major" in
  [
    Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
    Printf.sprintf "  %s __mma_bf[%d];" (frag_typ "matrix_b" ab_typ (Some b_layout)) nt;
    Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
    (if tb then
       Printf.sprintf
         "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d * %d + __ki * %d, %d);"
         tile ldb tile ldb
     else
       Printf.sprintf
         "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d * %d + __ni * %d, %d);"
         tile ldb tile ldb);
    "  }";
    Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
    Printf.sprintf "    %s __mma_af;" (frag_typ "matrix_a" ab_typ (Some a_layout));
    (if ta then
       Printf.sprintf
         "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * %d + __mi * %d, %d);" tile
         lda tile lda
     else
       Printf.sprintf
         "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * %d + __ki * %d, %d);" tile
         lda tile lda);
    Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
    Printf.sprintf
      "      rocwmma::mma_sync(%s[__mi][__ni], __mma_af, __mma_bf[__ni], %s[__mi][__ni]);" acc acc;
    "    }";
    "  }";
    "}";
  ]

(* [k_loop_lines]' loads, as the cost model counts them: the [tile] cube, each fragment once per
   reduction step, loaded whole by one [load_matrix_sync]. The same in both emission scopes, whose
   hooks share [k_loop_lines]. *)
let operand_fragments ~a_prec ~b_prec ~d_prec ~scope:_ =
  Option.map (mma_accumulator ~a_prec ~b_prec ~d_prec) ~f:(fun _ ->
      { Backend_intf.mf_tile = (tile, tile, tile); mf_reloads = false; mf_fragment_loads = true })

(* No [ldmatrix] analogue: [accepts] declines a swizzled operand, and every load [k_loop_lines]
   emits is a whole-fragment [load_matrix_sync]. *)
let operand_loads ~a_prec:_ ~b_prec:_ ~d_prec:_ ~scope:_ ~ta:_ ~tb:_ ~a:_ ~b:_ = (false, false)

(* gh-ocannl-1092: advisory dense matrix-unit peaks for the benchmark report's kernel %peak column
   (reporting-only; never feeds autotune ranking or bound pruning). On RDNA3/RDNA3.5 wave32 (gfx1102
   RX 7700S: 32 CUs x 512 FLOP/cycle/CU x 2.5 GHz = 40.96 TFLOP/s; gfx1151 Radeon 8060S: 40 CUs x
   512 x 2.9 GHz = 59.39 TFLOP/s), [v_wmma] delivers 512 FLOP/cycle/CU for both FP16 and BF16 (4x
   single-issue fp32, matching [Hip_backend]'s [peak_flops = 1.0e13]). *)
let rdna3_wmma_peak fmt =
  {
    Backend_intf.mma_peak_flops = 4.0e13;
    mma_peak_provenance =
      Datasheet_class
        (Printf.sprintf "AMD RDNA3 WMMA class, 512 FLOP/cycle/CU, RX 7700S 41 TFLOP/s dense %s" fmt);
    mma_peak_accum = Any_accum;
  }

let mma_peak_f16 = rdna3_wmma_peak "FP16"
let mma_peak_bf16 = rdna3_wmma_peak "BF16"
