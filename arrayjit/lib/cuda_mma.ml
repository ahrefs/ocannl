open Base

(* The arm variant: every tensor-core rendering CUDA's mma hooks can emit, named once. The hooks
   dispatch on [resolve_arm], [mma_arm] projects it, and the floors, markers and tiles below are
   functions of it, so neither the dispatch order nor a floor is written out a second time
   (gh-ocannl-1232). The fields are documented in the interface. *)
type m16n8k16_kind = M16n8k16_bf16 | M16n8k16_f16 [@@deriving equal, sexp_of, enumerate]
type inline_ptx = Fp8_m16n8k32 | M16n8k16 of m16n8k16_kind [@@deriving equal, sexp_of, enumerate]

type wmma_kind = Wmma_f16_f32 | Wmma_f16_f16 | Wmma_f16_wide | Wmma_bf16 | Wmma_tf32
[@@deriving equal, sexp_of, enumerate]

type arm = Inline_ptx of inline_ptx | Wmma of wmma_kind [@@deriving equal, sexp_of, enumerate]

(* The one table from storage precisions, the numerics policy and the emission scope to an arm.

   - fp8 x fp8 -> f32 has no wmma form: inline-PTX m16n8k32 over e5m2 in both scopes. - Uniform bf16
   likewise (gh-ocannl-545): [nvcuda::wmma] declares no bf16 accumulator fragment, but [mma.sync]
   accumulates bf16 operands in per-lane f32 registers whose layout is architecturally defined, so
   the [d] boundary converts — per statement, or once at each end of the register scope
   (gh-ocannl-1063). - Uniform f16 under [Numerics.Fp16_wide] (gh-ocannl-680) is the one triple
   whose arm depends on the scope. A per-statement statement renders through the same m16n8k16 body
   as bf16 (the PTX fragment layouts are shared by .f16 and .bf16). The persistent-fragment scope
   renders wmma's f16 x f16 -> f32 fragments behind a converted f16 [d] boundary (gh-ocannl-925) —
   and so do the update-only statements nested in it, whose arm is the scope's: the hooks resolve a
   [`Fragment] destination at [Mma_fragment_scope]. That converted boundary is element-wise across a
   whole fragment array, which no self-contained statement has, so the per-statement scope never
   resolves to it. Before gh-ocannl-1232 this was a fall-through: the per-statement hook tried the
   inline-PTX arm first and reached wmma only when that declined. - The f16-accumulate wmma triple
   is gated off under that policy: f16 accumulates width-uniform with [accum_prec] either way. -
   Uniform f32 computes in tf32 (m16n16k8) only when the policy opts in (gh-ocannl-478); with the
   policy off there is no arm and the scalar fallback keeps full f32 numerics.

   The cases are disjoint except for the guarded uniform-f16 one, which must precede its unguarded
   twin (swapping them is a fatal unused-case warning), so a reordering cannot silently change an
   arm. *)
let resolve_arm ~a_prec ~b_prec ~d_prec ~scope =
  match (a_prec, b_prec, d_prec) with
  | Ops.Fp8_prec _, Ops.Fp8_prec _, Ops.Single_prec _ -> Some (Inline_ptx Fp8_m16n8k32)
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ ->
      Some (Inline_ptx (M16n8k16 M16n8k16_bf16))
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when Numerics.fp16_accum_wide () -> (
      match scope with
      | Backend_intf.Mma_per_statement -> Some (Inline_ptx (M16n8k16 M16n8k16_f16))
      | Backend_intf.Mma_fragment_scope -> Some (Wmma Wmma_f16_wide))
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ -> Some (Wmma Wmma_f16_f16)
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ -> Some (Wmma Wmma_f16_f32)
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ -> Some (Wmma Wmma_bf16)
  | Ops.Single_prec _, Ops.Single_prec _, Ops.Single_prec _
    when (Numerics.get ()).Numerics.tf32_matmuls ->
      Some (Wmma Wmma_tf32)
  | _ -> None

(* The arch floors: wmma's 16-bit f16 forms are sm_70+ (Volta tensor cores); its bf16, tf32 and
   converted wide-f16 forms, and the m16n8k16 [mma.sync] forms, sm_80+ (Ampere); m16n8k32 over e5m2
   sm_89+ (Ada). The ONLY place CUDA's mma cutoffs are written: [capability], [mma_arm] and
   [gpu_arch_options] read them from here. *)
let arm_floor = function
  | Inline_ptx Fp8_m16n8k32 -> 89
  | Inline_ptx (M16n8k16 (M16n8k16_bf16 | M16n8k16_f16)) -> 80
  | Wmma (Wmma_f16_f32 | Wmma_f16_f16) -> 70
  | Wmma (Wmma_f16_wide | Wmma_bf16 | Wmma_tf32) -> 80

let arm_marker = function
  | Inline_ptx Fp8_m16n8k32 -> "mma-fp8"
  | Inline_ptx (M16n8k16 M16n8k16_bf16) -> "mma-bf16"
  | Inline_ptx (M16n8k16 M16n8k16_f16) -> "mma-f16"
  | Wmma (Wmma_f16_f32 | Wmma_f16_f16) -> "wmma"
  | Wmma Wmma_f16_wide -> "wmma-f16-wide"
  | Wmma Wmma_bf16 -> "wmma-bf16"
  | Wmma Wmma_tf32 -> "wmma-tf32"

let arm_tile = function
  | Inline_ptx Fp8_m16n8k32 -> (16, 8, 32)
  | Inline_ptx (M16n8k16 _) -> (16, 8, 16)
  | Wmma Wmma_tf32 -> (16, 16, 8)
  | Wmma (Wmma_f16_f32 | Wmma_f16_f16 | Wmma_f16_wide | Wmma_bf16) -> (16, 16, 16)

type wmma_info = {
  wc_ab_typ : string;
  wc_acc_prec : Ops.prec;
  wc_ab_ld_mult : int;
  wc_d_ld_mult : int;
  wc_cvt_tf32 : bool;
  wc_d_cvt : (string * string) option;
}

let wmma_info kind =
  let mk ?(cvt_tf32 = false) ?d_cvt ab_typ acc_prec ab_ld d_ld =
    {
      wc_ab_typ = ab_typ;
      wc_acc_prec = acc_prec;
      wc_ab_ld_mult = ab_ld;
      wc_d_ld_mult = d_ld;
      wc_cvt_tf32 = cvt_tf32;
      wc_d_cvt = d_cvt;
    }
  in
  match kind with
  | Wmma_f16_f32 -> mk "__half" Ops.single 8 4
  | Wmma_f16_f16 -> mk "__half" Ops.half 8 8
  (* gh-ocannl-925: the f16 x f16 -> f32 fragments above over an f16 STORAGE destination, converted
     once at the [d] boundary by [wmma_d_boundary_lines]. That boundary is element-wise, so [d] has
     no wmma stride constraint. *)
  | Wmma_f16_wide -> mk ~d_cvt:("__half2float", "__float2half") "__half" Ops.single 8 1
  | Wmma_bf16 -> mk "__nv_bfloat16" Ops.single 8 4
  (* tf32 fragments load raw f32 storage; f32's wmma stride constraint is 4 elements. *)
  | Wmma_tf32 -> mk ~cvt_tf32:true "nvcuda::wmma::precision::tf32" Ops.single 4 4

let arm_accumulator = function
  (* Both inline-PTX instructions accumulate [.f32] in per-lane registers whatever the storage. *)
  | Inline_ptx (Fp8_m16n8k32 | M16n8k16 _) -> Ops.single
  | Wmma kind -> (wmma_info kind).wc_acc_prec

(* The operand conditions of every wmma rendering, written once for the two hooks that used to
   repeat them (gh-ocannl-1456): opaque fragments load with [load_matrix_sync] from row- or
   column-major pointer+stride storage only, so every layout is plain (no supported way feeds them
   from [ldmatrix] registers, gh-ocannl-481 item 3, D2); the arm's tile divides the extents; the
   leading dimensions meet its stride multiples; every addressed operand is device or shared memory
   (generic-address loads cover both; a per-thread stack is not a jointly-owned tile); and the
   device meets the arm's floor. An update-only statement addresses no destination. *)
let wmma_accepts kind ~cc ~m ~n ~k ~d ~a:(lda, (a_space, a_layout)) ~b:(ldb, (b_space, b_layout)) =
  let arm = Wmma kind in
  let { wc_ab_ld_mult; wc_d_ld_mult; _ } = wmma_info kind in
  let tm, tn, tk = arm_tile arm in
  let plain = function `Plain -> true | `Swizzled_b128 -> false in
  let loadable = function `Device | `Shared -> true | `Thread -> false in
  (match d with
    | `Resident d_layout -> plain d_layout
    | `Addressed (ldd, (d_space, d_layout)) ->
        plain d_layout && loadable d_space && ldd % wc_d_ld_mult = 0)
  && plain a_layout && plain b_layout
  && m % tm = 0
  && n % tn = 0
  && k % tk = 0
  && lda % wc_ab_ld_mult = 0
  && ldb % wc_ab_ld_mult = 0
  && loadable a_space && loadable b_space
  && cc >= arm_floor arm

let wmma_frag_typ kind role typ layout =
  let tm, tn, tk = arm_tile (Wmma kind) in
  Printf.sprintf "nvcuda::wmma::fragment<nvcuda::wmma::%s, %d, %d, %d, %s%s>" role tm tn tk typ
    (match layout with Some l -> ", nvcuda::wmma::" ^ l | None -> "")

(* The serial [k] extent of one wmma block statement: per step, the B-row fragment loads, the A
   fragment loads, and the mma updates of the accumulator array [acc]. Shared between the
   self-contained rendering ([acc] = the block-local [__mma_acc]) and the update-only rendering
   against a resident fragment array (gh-ocannl-480). Each fragment is one whole [load_matrix_sync]
   once per step, reused across the other operand's free axis: what [operand_fragments] says of the
   wmma arms ([arm_reloads_operands]). *)
let wmma_k_loop_lines kind ~acc ~ta ~tb ~lda ~ldb ~m ~n ~k =
  let { wc_ab_typ = ab_typ; wc_cvt_tf32 = cvt_tf32; _ } = wmma_info kind in
  let wc_tm, wc_tn, wc_tk = arm_tile (Wmma kind) in
  let mt = m / wc_tm and nt = n / wc_tn and kt = k / wc_tk in
  let frag = wmma_frag_typ kind in
  let a_frag_layout = if ta then "col_major" else "row_major" in
  let b_frag_layout = if tb then "col_major" else "row_major" in
  (* tf32 fragments load raw f32 bits; the explicit elementwise [__float_to_tf32] performs the
     mantissa truncation the intrinsic expects (CUDA programming guide requirement -- without it the
     multiplicand bits are implementation-defined). *)
  let cvt_lines frag_expr =
    if cvt_tf32 then
      [
        Printf.sprintf
          "    for (int __t = 0; __t < %s.num_elements; ++__t) %s.x[__t] = \
           nvcuda::wmma::__float_to_tf32(%s.x[__t]);"
          frag_expr frag_expr frag_expr;
      ]
    else []
  in
  [
    Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
    Printf.sprintf "  %s __mma_bf[%d];" (frag "matrix_b" ab_typ (Some b_frag_layout)) nt;
    Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
    (* Transposed storage ([tb]): the stored matrix is the role's transpose -- index it at (col,
       row) and declare the fragment [col_major]; the leading dimension stays the operand's own.
       Same for [ta] below. *)
    (if tb then
       Printf.sprintf
         "    nvcuda::wmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d * %d + __ki * \
          %d, %d);"
         wc_tn ldb wc_tk ldb
     else
       Printf.sprintf
         "    nvcuda::wmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d * %d + __ni * \
          %d, %d);"
         wc_tk ldb wc_tn ldb);
  ]
  @ cvt_lines "__mma_bf[__ni]"
  @ [
      "  }";
      Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
      Printf.sprintf "    %s __mma_af;" (frag "matrix_a" ab_typ (Some a_frag_layout));
      (if ta then
         Printf.sprintf
           "    nvcuda::wmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * %d + __mi * %d, \
            %d);"
           wc_tk lda wc_tm lda
       else
         Printf.sprintf
           "    nvcuda::wmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * %d + __ki * %d, \
            %d);"
           wc_tm lda wc_tk lda);
    ]
  @ cvt_lines "__mma_af"
  @ [
      Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
      Printf.sprintf
        "      nvcuda::wmma::mma_sync(%s[__mi][__ni], __mma_af, __mma_bf[__ni], %s[__mi][__ni]);"
        acc acc;
      "    }";
      "  }";
      "}";
    ]

type m16n8k16_spelling = {
  ms_elt_typ : string;
  ms_as_ushort : string;
  ms_widen : string;
  ms_narrow : string;
  ms_infix : string;
}

let m16n8k16_spelling = function
  | M16n8k16_bf16 ->
      {
        ms_elt_typ = "__nv_bfloat16";
        ms_as_ushort = "__bfloat16_as_ushort";
        ms_widen = "__bfloat162float";
        ms_narrow = "__float2bfloat16";
        ms_infix = "bf16";
      }
  | M16n8k16_f16 ->
      {
        ms_elt_typ = "__half";
        ms_as_ushort = "__half_as_ushort";
        ms_widen = "__half2float";
        ms_narrow = "__float2half";
        ms_infix = "f16";
      }

(* fp8's destination is f32 storage, moved directly; the m16n8k16 destinations are their 16-bit
   operands' type, converted at the boundary. *)
let inline_ptx_d_storage = function
  | Fp8_m16n8k32 -> ("float", None)
  | M16n8k16 kind ->
      let s = m16n8k16_spelling kind in
      (s.ms_elt_typ, Some (s.ms_widen, s.ms_narrow))

(* fp8's 4-byte registers are contiguous only in A's row-major and B's transposed orientations
   (gh-ocannl-481 item 3, D2); the 16-bit arms' registers hold element PAIRS, contiguous under one
   of [ldmatrix]'s two forms in either orientation; wmma's opaque fragments cannot be fed from
   [ldmatrix] at all. *)
let arm_ldmatrix_operands arm ~ta ~tb ~a ~b =
  match arm with
  | Inline_ptx Fp8_m16n8k32 -> (a && not ta, b && tb)
  | Inline_ptx (M16n8k16 _) -> (a, b)
  | Wmma _ -> (false, false)

(* An operand [ldmatrix] can both reach and de-conflict: the swizzled 16-byte-unit layout in shared
   memory. Everything else keeps per-lane gathers. *)
let ldmatrix_eligible : Backend_intf.mma_operand_place -> bool = function
  | `Shared, `Swizzled_b128 -> true
  | (`Shared | `Device | `Thread), (`Plain | `Swizzled_b128) -> false

(* The emission scope's arm: the census reads it off the statement's destination, the cost model off
   the schedule's simdgroup fragments. *)
let operand_loads ~a_prec ~b_prec ~d_prec ~scope ~ta ~tb ~a ~b =
  match resolve_arm ~a_prec ~b_prec ~d_prec ~scope with
  | None -> (false, false)
  | Some arm -> arm_ldmatrix_operands arm ~ta ~tb ~a:(ldmatrix_eligible a) ~b:(ldmatrix_eligible b)

(* The inline-PTX bodies load an [a] and a [b] fragment inside their [__mi x __ni x __ki] loops, so
   each is reloaded for every tile of the other's free axis; wmma loads its B row of fragments and
   then each A fragment once per [__ki] step, reusing them across [__mi] / [__ni]. Stated beside the
   loops: [Cuda_backend]'s inline-PTX [mma_syntax] bodies and [wmma_k_loop_lines] above, which
   [test/operations/mma_format_arm_tables] counts against it (gh-ocannl-1456). *)
let arm_reloads_operands = function Inline_ptx _ -> true | Wmma _ -> false

let operand_fragments ~a_prec ~b_prec ~d_prec ~scope =
  Option.map (resolve_arm ~a_prec ~b_prec ~d_prec ~scope) ~f:(fun arm ->
      {
        Backend_intf.mf_tile = arm_tile arm;
        mf_reloads = arm_reloads_operands arm;
        (* wmma's opaque fragments load whole through [load_matrix_sync]; the inline-PTX arms gather
           per lane. *)
        mf_fragment_loads = (match arm with Wmma _ -> true | Inline_ptx _ -> false);
      })

let mma_arm ~a_prec ~b_prec ~d_prec ~scope =
  Option.map (resolve_arm ~a_prec ~b_prec ~d_prec ~scope) ~f:(fun arm ->
      {
        Backend_intf.arm_name = arm_marker arm;
        arm_accumulator = arm_accumulator arm;
        arm_floor = Some (arm_floor arm);
      })

let min_floor arms =
  Option.value_exn (List.min_elt (List.map arms ~f:arm_floor) ~compare:Int.compare)

(* Tensor cores at all: the lowest floor of any arm. *)
let tensor_core_floor = min_floor all_of_arm
let async_copy_floor = 80

(* The tile-MMA descriptor of a device of compute capability [cc]; [None] below every arm's floor.

   Tensor cores (tensorize-mma T3): the 32-thread warp cooperates on 16x16x16 wmma tiles from sm_70
   up; [mma_format_tiles] advertises the divergent fp8 16x8x32, tf32 16x16x8 and uniform-bf16
   16x8x16 shapes to typed autotune seeds. Precision combinations are ultimately decided per call by
   the hooks — but each entry here is NAMED by the arm it seeds, whose floor and tile it takes,
   because a seed the hook will decline is a candidate the tuner times as scalar code under a
   tensorized label (gh-ocannl-545). Which arm a format key reaches is still this table's claim, and
   [test/operations/cuda_mma_cutoffs] checks it at every compute capability on either side of a
   cutoff: what this descriptor admits per storage triple, emission scope and policy is what
   {!mma_arm}'s floors admit (gh-ocannl-1214). *)
let capability ~cc =
  let admits arm = cc >= arm_floor arm in
  let entry arm key = Option.some_if (admits arm) (key, arm_tile arm) in
  let staged arm key = Option.some_if (admits arm) (key, Backend_intf.Mma_swizzled_b128) in
  let scopes arms =
    List.filter_map arms ~f:(fun (scope, arm) -> Option.some_if (admits arm) scope)
  in
  let bf16_uniform = Inline_ptx (M16n8k16 M16n8k16_bf16) in
  if cc >= tensor_core_floor then
    Some
      {
        Backend_intf.mma_simd_width = 32;
        mma_tile = (16, 16, 16);
        mma_format_tiles =
          List.filter_opt
            [
              (* wmma: f16 operands against either accumulator width. *)
              entry (Wmma Wmma_f16_f32)
                (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f32);
              entry (Wmma Wmma_f16_f16)
                (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f16);
              (* wmma: bf16 operands accumulate in f32 only. *)
              entry (Wmma Wmma_bf16)
                (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_f32);
              (* Inline-PTX [mma.sync] m16n8k16: the uniform-bf16 combination wmma cannot
                 express. *)
              entry bf16_uniform
                (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16);
              (* Inline-PTX [mma.sync] m16n8k32. *)
              entry (Inline_ptx Fp8_m16n8k32)
                (Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_f32);
              (* wmma tf32; [mma_input_formats_of_prec] additionally gates this on the numerics
                 policy. *)
              entry (Wmma Wmma_tf32)
                (Backend_intf.Mma_tf32, Backend_intf.Mma_tf32, Backend_intf.Mma_f32);
            ];
        (* gh-ocannl-680: under [Numerics.Fp16_wide] the uniform-f16 key above renders per statement
           through the f32-accumulate inline-PTX m16n8k16 arm, sharing the bf16 form's body. The
           advertised (16, 16, 16) tile stays valid for it — its divisibility constraints (m%16,
           n%8, k%16) are implied — merely conservative about n. The fragment scope renders through
           wmma instead: f16 x f16 -> f32 accumulator fragments resident across the outer k,
           converting the f16 [d] once at each end through [wmma_d_boundary_lines] (gh-ocannl-925;
           before it, the list held only [Mma_per_statement] and staged seeds were withheld,
           gh-ocannl-836). Each scope is listed where ITS arm renders, withholding uniform-f16
           candidates under the wide policy below that floor instead of timing scalar fallbacks
           under a tensorized label (gh-ocannl-545). *)
        mma_f16_wide_acc_scopes =
          scopes
            [
              (Backend_intf.Mma_per_statement, Inline_ptx (M16n8k16 M16n8k16_f16));
              (Backend_intf.Mma_fragment_scope, Wmma Wmma_f16_wide);
            ];
        (* gh-ocannl-838: the uniform-bf16 key is the same inline-PTX arm, f32 in hardware whatever
           the policy, so [Numerics.Bf16_wide] changes no rendering here. Since gh-ocannl-1063 a
           staged outer-k split keeps it f32 too: the persistent-fragment scope holds that arm's
           per-lane registers across the outer reduction and converts [d] once at each end, for the
           plain and the swizzled staged twins alike — under every policy, as [accum_prec] keeps
           bf16 accumulators f32 under every policy. *)
        mma_bf16_wide_acc_scopes =
          scopes
            [
              (Backend_intf.Mma_per_statement, bf16_uniform);
              (Backend_intf.Mma_fragment_scope, bf16_uniform);
            ];
        (* Swizzled staged twins share the inline-PTX register scope. bf16 uses [ldmatrix] for both
           operands; row-major fp8 uses it for A and byte gathers through the swizzle map for B
           (gh-ocannl-1073; [arm_ldmatrix_operands]). *)
        mma_staged_layouts =
          List.filter_opt
            [
              staged bf16_uniform
                (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16);
              staged (Inline_ptx Fp8_m16n8k32)
                (Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_f32);
            ];
        (* gh-ocannl-487 phase 2: the depth-2 twins are worth proposing exactly where the [cp.async]
           arm renders them ({!async_copy_floor}, the gate of [Cuda_syntax_config.async_copy] too) —
           pre-Ampere the twin would be the portable synchronous form, whose occupancy cost was
           measured, not hypothesized (phase 1: ~1.4-1.5x on Metal). Depth 2 only: the wait-all
           emission has single-step lookahead; deeper pipelines need commit_group/wait_group N. *)
        mma_pipeline_depths = (if cc >= async_copy_floor then [ 2 ] else []);
      }
  else None

(* The non-mma floors [gpu_arch_options] applies. Half ARITHMETIC intrinsics (unlike the
   conversions, which cuda_fp16.h/cuda_bf16.h emulate on any arch) are declared only for
   [__CUDA_ARCH__ >= 530], the bfloat16 ones for [>= 800]. CUDA 13 dropped offline compilation below
   compute_75. *)
let half_arith_floor = 53
let bf16_arith_floor = 80
let nvrtc_offline_floor = 75
let mxfp8_family_marker = "(mma-mxfp8)"
let mxfp8_family_cc = 120

let gpu_arch_options ~device_cc cu_src =
  (* Two kinds of target, and the difference is the whole point of this function:

     - A FLOOR (every marker but one): the lowest arch whose PTX contains the instruction. PTX
     targeted at a floor is forward-JIT-compiled by the driver on every later GPU, so one compile
     covers the entire range above it. Each arm's rendering carries its [(<arm_marker>)], whose
     floor is that arm's own {!arm_floor} — read from the arm, never restated (gh-ocannl-1232).

     - A FAMILY target ([(mma-mxfp8)]): a [compute_120a]-style architecture-specific arch, which
     only the device family it names can load. Blackwell's block-scaled [kind::mxf8f6f4] forms exist
     ONLY under such a target, so this is the one case where forward-JIT portability has to be given
     up — and therefore the one marker gated on the attached devices' own family. Family PTX is
     never produced for a device that could not load it; a marked kernel reaching a mismatched
     device falls back to floor targeting, which is defense in depth rather than a recovery path (an
     arm emitting the marker is gated on the same family, so it should have declined already). No
     arm emits it yet: block scaling itself is blocked on OCANNL having microscaling storage at all
     (the e8m0 per-32-element scale factors are extra mma OPERANDS with their own layout, and
     [Tile_mma] has no slot for them), and a unit-scale arm would be numerically identical to the
     plain fp8 path while forfeiting forward-JIT. *)
  let has s = String.is_substring cu_src ~substring:s in
  if has mxfp8_family_marker && device_cc / 10 = mxfp8_family_cc / 10 then
    [ Printf.sprintf "--gpu-architecture=compute_%da" device_cc ]
  else
    (* A kernel that reaches [nvcuda::wmma] needs tensor cores even where no marker names the
       arm. *)
    let wmma_floor =
      min_floor (List.filter all_of_arm ~f:(function Wmma _ -> true | Inline_ptx _ -> false))
    in
    let uses_h_arith =
      (* Every half-arith token [unop_syntax]/[binop_syntax]/[ternop_syntax] can emit; [hexp]/[hlog]
         also cover their [2]-suffixed variants and [__hmax]/[__hmin] the [_nan] variants as
         substrings. The bf16 overloads share the half intrinsics' names, so a bf16 kernel is
         recognized by the type name appearing alongside the arithmetic tokens; a kernel mixing half
         arithmetic with bf16 storage-only is conservatively floored at the bf16 floor too. *)
      List.exists ~f:has
        [
          "__hadd";
          "__hsub";
          "__hmul";
          "__hdiv";
          "__hmax";
          "__hmin";
          "__hgt";
          "__hfma";
          "hexp";
          "hlog";
          "hsin";
          "hcos";
          "hsqrt";
          "hrcp";
          "hrsqrt";
          "htanh_approx";
        ]
    in
    (* [compile_batch] concatenates several kernels into one source, so the floors can trigger
       independently (e.g. a half-wmma kernel batched with scalar bf16 arithmetic needs the bf16
       floor even without a bf16 arm marker): take the max, not the first match. *)
    let arch_floor =
      List.filter_map all_of_arm ~f:(fun arm ->
          Option.some_if (has ("(" ^ arm_marker arm ^ ")")) (arm_floor arm))
      @ List.filter_opt
          [
            Option.some_if (has "nvcuda::wmma") wmma_floor;
            (* The [cp.async] staging builtins (gh-ocannl-487 phase 2); emission is gated on the
               devices' own capability, so the floor only ever fires where it can also load. The
               marker is the PTX instruction text, which appears exactly in the prepended helper
               definitions' asm literals — never in an identifier — so a routine or label merely
               NAMED like a helper (which the token-scoped builtin filter declines to activate)
               cannot raise the floor past a pre-Ampere device (Codex P2 on PR #317, round 5). *)
            Option.some_if (has "cp.async.") async_copy_floor;
            Option.some_if uses_h_arith
              (if has "__nv_bfloat16" then bf16_arith_floor else half_arith_floor);
          ]
      |> List.max_elt ~compare:Int.compare
    in
    match arch_floor with
    | Some floor ->
        (* nvrtc 13 rejects the compute_53/compute_70 floors outright: raise a triggered floor to
           the offline floor whenever every attached device can load such PTX (a device below it
           keeps the literal floor — it must be paired with an nvrtc 12.x that still accepts it). We
           deliberately do NOT raise all the way to the device arch (e.g. compute_120 on Blackwell
           GeForce): the sm_89 fp8 [mma.sync] encoding remains valid under a compute_89 target where
           a compute_120 target would demand the family-specific [kind::] forms. *)
        let arch = max floor (min nvrtc_offline_floor device_cc) in
        [ Printf.sprintf "--gpu-architecture=compute_%d" arch ]
    | None -> []
