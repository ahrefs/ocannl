(** CUDA's tensor-core tables as pure functions of storage precisions, the numerics policy and the
    compute capability: the arm resolver the backend's [mma_syntax] and [mma_fragment_syntax] hooks
    dispatch on, the arch floors those arms check, and the capability descriptor
    ({!Backend_intf.mma_capability}) that autotune seeds from.

    They live here, outside the [cudajit]-gated [Cuda_backend], so the agreement between the two
    owners of the 70/80/89 cutoffs -- the descriptor's [entry ~min_cc] and the resolver's floors
    ([wc_min_cc], [mma16_min_cc]) -- is tested on both sides of every cutoff wherever OCANNL builds,
    with no toolkit and no device ([test/operations/cuda_mma_cutoffs], gh-ocannl-1214). The backend
    consumes these definitions directly: [Cuda_backend.Impl] includes them into its syntax
    configuration and builds [hardware_limits.mma] as [capability ~cc] at the attached devices'
    minimum compute capability. Compute capabilities are [major * 10 + minor] throughout. *)

open Base

(* The wmma-supported precision combinations (tensorize-mma T3). Shared by [mma_syntax] and
   [mma_fragment_syntax] so a fragment scope accepts exactly when its nested update-only MMA calls
   would — including the numerics-policy gate on the tf32 arm, which lives here for the same
   reason. *)
type wmma_combo_info = {
  wc_ab_typ : string;
      (** The fragment element type for [matrix_a]/[matrix_b] — a C++ type for the 16-bit
          combinations, the tag type [nvcuda::wmma::precision::tf32] for tf32 (storage stays
          [float]; only the fragments are tagged). *)
  wc_acc_prec : Ops.prec;
      (** The accumulator fragment's precision; its element type is [typ_of_prec] of it. *)
  wc_tm : int;
  wc_tn : int;
  wc_tk : int;
      (** The intrinsic tile shape: 16×16×16 for the 16-bit combinations, 16×16×8 for tf32 (mirrors
          [mma_format_tiles] in the capability descriptor). *)
  wc_ab_ld_mult : int;  (** wmma stride constraint: a/b leading-dim multiple, in elements. *)
  wc_d_ld_mult : int;  (** wmma stride constraint: d leading-dim multiple, in elements. *)
  wc_min_cc : int;
  wc_marker : string;
      (** Marker suffix for the rendering comment (["" | "-bf16" | "-tf32"]); [cuda_to_ptx] greps it
          to select the arch floor. *)
  wc_cvt_tf32 : bool;
      (** Convert loaded a/b fragment elements with [__float_to_tf32]: tf32 fragments load raw f32
          bits, the explicit conversion performs the mantissa truncation (per the CUDA programming
          guide; the intrinsic requires already-converted inputs). *)
  wc_d_cvt : (string * string) option;
      (** The [(widen, narrow)] conversions of a destination whose storage type is not the
          accumulator fragment's — the wide-f16 arm, gh-ocannl-925. [None]: [d] loads and stores
          through [load_matrix_sync]/[store_matrix_sync] directly. [Some]: it crosses the fragment
          boundary element by element, see [wmma_d_boundary_lines]. *)
}

let wmma_combo ~a_prec ~b_prec ~d_prec : wmma_combo_info option =
  let mk ?(tile = (16, 16, 16)) ?(marker = "") ?(cvt_tf32 = false) ?d_cvt ab_typ acc_prec ab_ld d_ld
      cc =
    let wc_tm, wc_tn, wc_tk = tile in
    Some
      {
        wc_ab_typ = ab_typ;
        wc_acc_prec = acc_prec;
        wc_tm;
        wc_tn;
        wc_tk;
        wc_ab_ld_mult = ab_ld;
        wc_d_ld_mult = d_ld;
        wc_min_cc = cc;
        wc_marker = marker;
        wc_cvt_tf32 = cvt_tf32;
        wc_d_cvt = d_cvt;
      }
  in
  match (a_prec, b_prec, d_prec) with
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ -> mk "__half" Ops.single 8 4 70
  (* The f16-accumulate wmma triple must not render under [Numerics.Fp16_wide] (gh-ocannl-680): the
     uniform-f16 combination then goes through the f32-accumulate inline-PTX m16n8k16 arm instead,
     or declines to the scalar fallback, whose accumulator follows [accum_prec] — width-uniform
     either way. *)
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when not (Numerics.fp16_accum_wide ()) ->
      mk "__half" Ops.half 8 8 70
  (* gh-ocannl-925: the wide uniform-f16 combination — the f16 x f16 -> f32 fragments above over an
     f16 STORAGE destination, converted once at the [d] boundary by [wmma_d_boundary_lines]. That
     boundary is element-wise, so [d] has no wmma stride constraint. Only the fragment scope renders
     it: a per-statement statement takes the inline-PTX m16n8k16 arm, whose conditions this arm's
     imply (see [mma_syntax]). sm_80+ like that arm, the floor the capability's
     [mma_f16_wide_acc_scopes] is verified at. *)
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ ->
      mk ~marker:"-f16-wide" ~d_cvt:("__half2float", "__float2half") "__half" Ops.single 8 1 80
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ ->
      mk ~marker:"-bf16" "__nv_bfloat16" Ops.single 8 4 80
  | Ops.Single_prec _, Ops.Single_prec _, Ops.Single_prec _
    when (Numerics.get ()).Numerics.tf32_matmuls ->
      (* gh-ocannl-478: uniform-f32 GEMMs compute in tf32 (m16n16k8, sm_80+) when the numerics
         policy opts in; with the policy off this arm is [None] and the scalar fallback keeps full
         f32 numerics. f32's wmma stride constraint is 4 elements. *)
      mk ~tile:(16, 16, 8) ~marker:"-tf32" ~cvt_tf32:true "nvcuda::wmma::precision::tf32" Ops.single
        4 4 80
  | _ -> None

(* The element-type spellings of the inline-PTX m16n8k16 arm, shared by its two 16-bit forms: the C
   type, the bits-as-ushort intrinsic, the widening and narrowing conversions, the instruction's
   element infix, and the marker [gpu_arch_options] greps for the arch floor (sm_80 for both).
   [None]: the combination has no m16n8k16 form.

   gh-ocannl-545: [nvcuda::wmma] pairs [__nv_bfloat16] operands with a [float] accumulator only —
   [crt/mma.hpp] declares no bf16 accumulator fragment — so a uniformly-bf16 network, where the
   GEMM's destination node is itself bf16, has no wmma combination. The hardware is not the limit:
   [mma.sync] accumulates bf16 operands in per-lane f32 registers, which we can convert at the [d]
   boundary because that layout is architecturally defined.

   gh-ocannl-680: under [Numerics.Fp16_wide] the uniform-f16 combination renders through the same
   arm — the PTX ISA's "Matrix Fragments for mma.m16n8k16" layouts are shared by .f16 and .bf16 —
   which is exactly the residency [accum_prec] gives the serial legs under that policy. Under
   [Fp16_auto]/[Fp16_narrow] the wmma f16-accumulate combo renders instead. *)
let mma16_spellings ~a_prec ~b_prec ~d_prec =
  match (a_prec, b_prec, d_prec) with
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ ->
      Some
        ( "__nv_bfloat16",
          "__bfloat16_as_ushort",
          "__bfloat162float",
          "__float2bfloat16",
          "bf16",
          "mma-bf16" )
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when Numerics.fp16_accum_wide () ->
      Some ("__half", "__half_as_ushort", "__half2float", "__float2half", "f16", "mma-f16")
  | _ -> None

(* The fp8 arm's marker, shared by its statement and register-scope renderings and [mma_arm]. *)
let mma_fp8_marker = "mma-fp8"

let mma_fp8_combo ~a_prec ~b_prec ~d_prec =
  match (a_prec, b_prec, d_prec) with
  | Ops.Fp8_prec _, Ops.Fp8_prec _, Ops.Single_prec _ -> true
  | _ -> false

(* The inline-PTX shapes share the architected m16n8 f32 accumulator layout. Their persistent scope
   returns the destination boundary spellings, so plain and swizzled staged twins use one register
   array. Wide uniform f16 keeps gh-ocannl-925's wmma scope: it has no swizzled twin. *)
let mma16_register_scope ~a_prec ~b_prec ~d_prec =
  match (a_prec, b_prec, d_prec) with
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ ->
      Option.map (mma16_spellings ~a_prec ~b_prec ~d_prec)
        ~f:(fun (elt_typ, _, widen, narrow, _, marker) -> (elt_typ, Some (widen, narrow), marker))
  | Ops.Fp8_prec _, Ops.Fp8_prec _, Ops.Single_prec _ -> Some ("float", None, mma_fp8_marker)
  | _ -> None

(* The inline-PTX arms' own arch floors: m16n8k32 over e5m2 is sm_89+ (Ada), m16n8k16 sm_80+. *)
let mma16_min_cc ~a_prec ~b_prec ~d_prec = if mma_fp8_combo ~a_prec ~b_prec ~d_prec then 89 else 80

(* gh-ocannl-1153: the arm each hook selects at the precision level, in the hooks' own dispatch
   order — [mma_syntax]: fp8's m16n8k32, then the m16n8k16 forms, then wmma, whose self-contained
   statement declines a converted [d] boundary; [mma_fragment_syntax]: the inline-PTX register
   scope, then wmma. The update-only statement that DOES render wmma's converted [-f16-wide] arm
   runs inside that fragment scope, so it is the [Mma_fragment_scope] entry's, not a per-statement
   one. The inline-PTX arms accumulate in f32 per-lane registers whatever the storage ([.f32] in
   both instructions). The tables are the hooks' own; the ORDER is restated here, and pinned against
   the hooks by [schedule_mma_matmul] and [schedule_ldmatrix_matmul], which read every arm marker
   they expect in rendered CUDA from this function. This is how the tf32 gate in [wmma_combo]
   reaches the schedule cache's identity: [accum_prec] says f32 accumulates at f32 with the policy
   on or off. *)
let mma_arm ~a_prec ~b_prec ~d_prec ~scope =
  let inline_ptx arm_name =
    Some
      {
        Backend_intf.arm_name;
        arm_accumulator = Ops.single;
        arm_floor = Some (mma16_min_cc ~a_prec ~b_prec ~d_prec);
      }
  in
  let wmma () =
    Option.bind (wmma_combo ~a_prec ~b_prec ~d_prec) ~f:(fun combo ->
        match (scope, combo.wc_d_cvt) with
        | Backend_intf.Mma_per_statement, Some _ -> None
        | (Backend_intf.Mma_per_statement | Backend_intf.Mma_fragment_scope), _ ->
            Some
              {
                Backend_intf.arm_name = "wmma" ^ combo.wc_marker;
                arm_accumulator = combo.wc_acc_prec;
                arm_floor = Some combo.wc_min_cc;
              })
  in
  match scope with
  | Backend_intf.Mma_per_statement -> (
      if mma_fp8_combo ~a_prec ~b_prec ~d_prec then inline_ptx mma_fp8_marker
      else
        match mma16_spellings ~a_prec ~b_prec ~d_prec with
        | Some (_, _, _, _, _, marker) -> inline_ptx marker
        | None -> wmma ())
  | Backend_intf.Mma_fragment_scope -> (
      match mma16_register_scope ~a_prec ~b_prec ~d_prec with
      | Some (_, _, marker) -> inline_ptx marker
      | None -> wmma ())

(** The [cp.async] staging arm's arch floor (gh-ocannl-487 phase 2): [Cuda_backend]'s
    [Cuda_syntax_config.async_copy] renders it at and above this compute capability, and
    {!capability} proposes software-pipelined depths exactly there. *)
let async_copy_floor = 80

(** The tile-MMA descriptor of a device of compute capability [cc]; [None] below sm_70.

    Tensor cores (tensorize-mma T3): the 32-thread warp cooperates on 16x16x16 wmma tiles from sm_70
    up; [mma_format_tiles] advertises the divergent fp8 16x8x32, tf32 16x16x8 and uniform-bf16
    16x8x16 shapes to typed autotune seeds. Precision combinations are ultimately decided per call
    by [mma_syntax] — but each entry here mirrors an arm of that hook, INCLUDING its accumulator
    format and arch floor (gh-ocannl-545), because a seed the hook will decline is a candidate the
    tuner times as scalar code under a tensorized label. [test/operations/cuda_mma_cutoffs] checks
    that mirroring at every compute capability on either side of a cutoff: what this descriptor
    admits per storage triple, emission scope and policy is what {!mma_arm}'s floors admit
    (gh-ocannl-1214). *)
let capability ~cc =
  let entry ~min_cc key tile = if cc >= min_cc then Some (key, tile) else None in
  if cc >= 70 then
    Some
      {
        Backend_intf.mma_simd_width = 32;
        mma_tile = (16, 16, 16);
        mma_format_tiles =
          List.filter_opt
            [
              (* wmma, sm_70+: f16 operands against either accumulator width. *)
              entry ~min_cc:70
                (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f32)
                (16, 16, 16);
              entry ~min_cc:70
                (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f16)
                (16, 16, 16);
              (* wmma, sm_80+: bf16 operands accumulate in f32 only. *)
              entry ~min_cc:80
                (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_f32)
                (16, 16, 16);
              (* Inline-PTX [mma.sync] m16n8k16, sm_80+: the uniform-bf16 combination wmma cannot
                 express. *)
              entry ~min_cc:80
                (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16)
                (16, 8, 16);
              (* Inline-PTX [mma.sync] m16n8k32, sm_89+. *)
              entry ~min_cc:89
                (Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_f32)
                (16, 8, 32);
              (* wmma tf32, sm_80+; [mma_input_formats_of_prec] additionally gates this on the
                 numerics policy. *)
              entry ~min_cc:80
                (Backend_intf.Mma_tf32, Backend_intf.Mma_tf32, Backend_intf.Mma_f32)
                (16, 16, 8);
            ];
        (* gh-ocannl-680: under [Numerics.Fp16_wide] the uniform-f16 key above renders through the
           f32-accumulate inline-PTX m16n8k16 arm (sm_80+, sharing the bf16 form's body). The
           advertised (16, 16, 16) tile stays valid for it — its divisibility constraints (m%16,
           n%8, k%16) are implied — merely conservative about n. The fragment scope renders through
           wmma instead: f16 x f16 -> f32 accumulator fragments resident across the outer k,
           converting the f16 [d] once at each end through [wmma_d_boundary_lines] (gh-ocannl-925;
           before it, the list held only [Mma_per_statement] and staged seeds were withheld,
           gh-ocannl-836). Below sm_80 the arm cannot render and the list is empty, withholding
           every uniform-f16 candidate under the wide policy instead of timing scalar fallbacks
           under a tensorized label (gh-ocannl-545). *)
        mma_f16_wide_acc_scopes =
          (if cc >= 80 then [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ]
           else []);
        (* gh-ocannl-838: the uniform-bf16 key is the same inline-PTX arm, f32 in hardware whatever
           the policy, so [Numerics.Bf16_wide] changes no rendering here. Since gh-ocannl-1063 a
           staged outer-k split keeps it f32 too: the persistent-fragment scope holds that arm's
           per-lane registers across the outer reduction and converts [d] once at each end
           ([mma16_register_scope]), for the plain and the swizzled staged twins alike — under every
           policy, as [accum_prec] keeps bf16 accumulators f32 under every policy. *)
        mma_bf16_wide_acc_scopes =
          (if cc >= 80 then [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ]
           else []);
        (* Swizzled staged twins share the inline-PTX register scope. bf16 uses [ldmatrix] for both
           operands; row-major fp8 uses it for A and byte gathers through the swizzle map for B
           (gh-ocannl-1073). *)
        mma_staged_layouts =
          List.filter_opt
            [
              entry ~min_cc:80
                (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16)
                Backend_intf.Mma_swizzled_b128;
              entry ~min_cc:89
                (Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_f32)
                Backend_intf.Mma_swizzled_b128;
            ];
        (* gh-ocannl-487 phase 2: the depth-2 twins are worth proposing exactly where the [cp.async]
           arm renders them ({!async_copy_floor}, the gate of [Cuda_syntax_config.async_copy] too) —
           pre-Ampere the twin would be the portable synchronous form, whose occupancy cost was
           measured, not hypothesized (phase 1: ~1.4-1.5x on Metal). Depth 2 only: the wait-all
           emission has single-step lookahead; deeper pipelines need commit_group/wait_group N. *)
        mma_pipeline_depths = (if cc >= async_copy_floor then [ 2 ] else []);
      }
  else None
