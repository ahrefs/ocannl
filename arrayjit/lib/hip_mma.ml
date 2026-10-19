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
     exclusive by construction. Under [Fp16_auto]/[Fp16_narrow] the accumulator fragment is itself
     f16, so the [d] boundary is rocWMMA's own load/store. Under [Fp16_wide] (gh-ocannl-789) the
     accumulator is [float] against the same f16 STORAGE, and [mma_d_boundary_lines] converts once
     at each end -- which is what lets [mma_f16_wide_acc_scopes] advertise both scopes and the
     uniform-f16 seeds survive the wide policy on this backend. *)
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when Numerics.fp16_accum_wide () ->
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
    (* gh-ocannl-838: the uniform-bf16 arm swaps the same way under [Numerics.Bf16_wide], through
       rocWMMA's [(bf16, bf16, f32)] fragments and the same converted boundary. *)
    mma_bf16_wide_acc_scopes = [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
    (* rocWMMA fragments are opaque like wmma's: no swizzle-aware fragment load here (gh-ocannl-481
       item 3, D3). *)
    mma_staged_layouts = [];
    mma_pipeline_depths = [];
  }
