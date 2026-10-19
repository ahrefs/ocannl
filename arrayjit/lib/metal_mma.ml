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
