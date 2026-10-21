open Base

(* The one combination resolver both MMA hooks consult: the precision a storage triple's accumulator
   fragment resides at, every fragment type then following from its precision ([Metal_backend]'s
   [mma_fragment_types]); its [mma_d_boundary_lines] converts at the destination boundary exactly
   where the accumulator's and the destination's differ. Arms and [capability]'s STORAGE triples
   must correspond: an advertised triple with no arm makes autotune time seeds that render the
   scalar fallback, and an arm with no triple is never seeded.
   [test/operations/mma_format_arm_tables] checks the correspondence as a pure query, under every
   numerics policy (gh-ocannl-1246); [schedule_mma_matmul]'s format-triple leg ties it to the
   rendering on the device. [mma_arm] publishes this table to the schedule cache's identity
   (gh-ocannl-1153). *)
let mma_accumulator ~d_prec ~a_prec ~b_prec =
  match (a_prec, b_prec, d_prec) with
  | Ops.Single_prec _, Ops.Single_prec _, Ops.Single_prec _ -> Some Ops.single
  (* gh-ocannl-923: half operands into f32 destination STORAGE. The mixed multiply surface is
     gh-ocannl-837's wide-f16 one ([simdgroup_multiply_accumulate] over half A/B fragments and a
     float accumulator), but the destination already is the accumulator's type, so there is no
     staging fragment and no conversion: [simdgroup_load]/[simdgroup_store] move the float tile
     directly, in both emission scopes. Independent of [fp16_arithmetic], which is about f16
     DESTINATIONS. *)
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ -> Some Ops.single
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when Numerics.fp16_accum_wide () ->
      Some Ops.single
  | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ -> Some Ops.half
  (* The bf16 twins of the two f16 arms above (gh-ocannl-923, completing gh-ocannl-838's [Bf16_wide]
     on Metal): [simdgroup_multiply_accumulate] is generic over its operand element type, and bfloat
     operands into a float accumulator compile and execute on Apple silicon (schedule_mma_matmul's
     bf32 and [Bf16_wide] legs, M4 Max). The f32-storage arm needs no conversion; the wide arm
     converts at the bfloat destination through the coordinate table exactly as the wide-f16 one
     does. *)
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ -> Some Ops.single
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ when Numerics.bf16_accum_wide ()
    ->
      Some Ops.single
  | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ -> Some Ops.bfloat16
  | _ -> None

(* Both hooks consult one table in both scopes, and the Apple7 gate is the device's
   ([hardware_limits.mma]), so the arm names no floor. *)
let mma_arm ~a_prec ~b_prec ~d_prec ~scope:_ =
  Option.map (mma_accumulator ~d_prec ~a_prec ~b_prec) ~f:(fun arm_accumulator ->
      { Backend_intf.arm_name = "simdgroup"; arm_accumulator; arm_floor = None })

(* [simdgroup_matrix] (docs/proposals/tensorize-mma.md): 8x8x8 tiles cooperatively held by the
   32-thread simdgroup. Supported storage precisions are decided per call by [mma_accumulator]
   (f32/f16/bf16 uniform, plus f16 or bf16 operands into f32 storage since gh-ocannl-923); under
   [Fp16_wide], the uniform-f16 storage arm uses an f32 accumulator fragment and converts only at
   the destination boundary (gh-ocannl-837). *)
let capability =
  {
    Backend_intf.mma_simd_width = 32;
    mma_tile = (8, 8, 8);
    (* These triples describe storage formats, one per [mma_accumulator] arm. The wide-f16 exception
       below uses the same f16 storage triple with an f32 accumulator fragment internally. *)
    mma_format_tiles =
      [
        ((Backend_intf.Mma_f32, Backend_intf.Mma_f32, Backend_intf.Mma_f32), (8, 8, 8));
        ((Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f16), (8, 8, 8));
        (* gh-ocannl-923: half fragments into a float accumulator that IS the f32 destination's
           storage — no boundary conversion, so both emission scopes and no policy gate. *)
        ((Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f32), (8, 8, 8));
        ((Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16), (8, 8, 8));
        (* gh-ocannl-923: its bf16 twin, bfloat fragments into the f32 destination. *)
        ((Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_f32), (8, 8, 8));
      ];
    (* MSL's generic [simdgroup_multiply_accumulate] accepts half A/B fragments and an f32
       accumulator. [mma_syntax]/[mma_fragment_syntax] bridge that accumulator to f16 destination
       storage elementwise at the outer boundary (gh-ocannl-837). *)
    mma_f16_wide_acc_scopes = [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
    (* gh-ocannl-923: the uniform-bf16 arm's wide counterpart, the same float accumulator and
       coordinate-table boundary in both hooks. *)
    mma_bf16_wide_acc_scopes = [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
    (* Metal banks too, but [simdgroup_load] takes a plain pointer and leading dimension — no
       [ldmatrix] analogue (gh-ocannl-481 item 3, D3). *)
    mma_staged_layouts = [];
    mma_pipeline_depths = [ 2 ];
  }

let tile = 8

(* One 8x8 operand tile at role-space block [(row_i, col_i)] from [ptr]: transposed storage keeps
   the operand's own leading dimension and indexes the STORED matrix at [(col_i, row_i)];
   [simdgroup_load]'s [transpose_matrix] flag transposes the tile in flight. *)
let load_line frag ptr ~row_i ~col_i ld ~transpose =
  if transpose then
    Printf.sprintf
      "    simdgroup_load(%s, %s + %s * %d * %d + %s * %d, (ulong)%d, ulong2(0), true);" frag ptr
      col_i tile ld row_i tile ld
  else
    Printf.sprintf "    simdgroup_load(%s, %s + %s * %d * %d + %s * %d, (ulong)%d);" frag ptr row_i
      tile ld col_i tile ld

(* The serial [k] extent of one block statement, shared by the self-contained rendering ([acc] = its
   block-local [__mma_acc]) and the update-only one against a resident fragment array
   (gh-ocannl-480): per reduction step, the row of [nt] [b] fragments, then for each of the [mt] [a]
   fragments its updates across that row. Each fragment is one whole [simdgroup_load], once per
   step, reused across the other operand's free axis: what [operand_fragments] says. *)
let k_loop_lines ~a_frag ~b_frag ~acc ~ta ~tb ~lda ~ldb ~m ~n ~k =
  let mt = m / tile and nt = n / tile and kt = k / tile in
  [
    Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
    Printf.sprintf "  %s __mma_bf[%d];" b_frag nt;
    Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
    load_line "__mma_bf[__ni]" "__mma_bp" ~row_i:"__ki" ~col_i:"__ni" ldb ~transpose:tb;
    "  }";
    Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
    Printf.sprintf "    %s __mma_af;" a_frag;
    load_line "__mma_af" "__mma_ap" ~row_i:"__mi" ~col_i:"__ki" lda ~transpose:ta;
    Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
    Printf.sprintf
      "      simdgroup_multiply_accumulate(%s[__mi][__ni], __mma_af, __mma_bf[__ni], \
       %s[__mi][__ni]);"
      acc acc;
    "    }";
    "  }";
    "}";
  ]

(* [k_loop_lines]' loads, as the cost model counts them (gh-ocannl-1456): the arm's own [tile]
   cubed, each fragment once per reduction step, loaded whole by one [simdgroup_load]. The same in
   both emission scopes, whose hooks share [k_loop_lines]. *)
let operand_fragments ~a_prec ~b_prec ~d_prec ~scope:_ =
  Option.map (mma_accumulator ~d_prec ~a_prec ~b_prec) ~f:(fun _ ->
      { Backend_intf.mf_tile = (tile, tile, tile); mf_reloads = false; mf_fragment_loads = true })

(* [simdgroup_load] takes a plain pointer and leading dimension; the hooks decline a swizzled
   operand (gh-ocannl-481 item 3, D2) and every load [k_loop_lines] emits is that whole-fragment
   call, so no operand of an accepted call loads through a swizzle-aware matrix load. *)
let operand_loads ~a_prec:_ ~b_prec:_ ~d_prec:_ ~scope:_ ~ta:_ ~tb:_ ~a:_ ~b:_ = (false, false)
