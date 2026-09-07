(* Negative zero across the scalar-to-vector splat (gh-ocannl-615).

   Both explicit-SIMD renderings of the C backends have to broadcast a lane-uniform scalar into a
   vector register: the register-tiled [Tile_mma] micro-kernel splats each A element per k step
   ([C_syntax.try_register_tile]), and [try_vectorize] splats an index-free subexpression wherever a
   vector-typed operand is required (an FMA operand, a builtin argument). Both promise their values
   are BITWISE what the scalar twin computes — the peeled edges, the serial remainder loop, the
   unscheduled reference.

   The splat has to be ARITHMETIC-FREE to keep that promise, which is why it renders as an
   initializer of repeated copies of a bound scalar. Both arithmetic spellings alter bits the scalar
   twin keeps: [((vtyp){0} + x)] breaks on negative zero, since IEEE-754 has [(+0.0) + (-0.0) =
   +0.0], so a negative-zero element enters the vector arithmetic as POSITIVE zero; and [(x -
   (vtyp){0})], the first fix for that, is the identity on both zeros but still quiets a signaling
   NaN ([0x7f800001 -> 0x7fc00001] under gcc [-fsignaling-nans]; that it usually survives is only
   the optimizer folding the subtraction away, which nothing requires). The legs below execute the
   negative-zero case — a signaling NaN cannot be planted through the tensor API, since host values
   cross as OCaml doubles and the narrowing conversion quiets it — and the structural pins keep the
   arithmetic spellings out. Its products then differ from the scalar path's in the sign of a zero,
   and that survives into the result whenever the accumulating sum is itself a signed zero. The two
   legs below are those two arrangements, and each is checked with a comparison that can SEE the
   difference: [Float.equal] reports [-0. = +0.], so parity here is on the bits.

   - Register tiling: a 4x16 accumulator preloaded with [-0.0] in every cell (the statement's [+=]
   semantics load it into the C-tile, and a [+0.0] accumulator would absorb the sign of every zero
   product), row 0 of A all [-0.0], B strictly positive. Row 0 of the product is [-0.0] cell by cell
   — [(-0.0) * b = -0.0] and [(-0.0) + (-0.0) = -0.0] — while the normalizing splat yields [+0.0]
   throughout it. The other rows are ordinary nonzero values, so the reference is not vacuous, and
   they also pin that the fix changed nothing else. The preloaded accumulator is why this leg is an
   explicit [=+] into an initialized tensor rather than one of [schedule_mma_matmul]'s
   zero-then-accumulate legs. - Vectorized FMA: [s *. a + b] over a [Vectorized] innermost loop,
   where [s] is [-0.0] per row, broadcast along the vectorized axis (so it is lane-uniform and
   reaches the FMA through the splat), [a] takes both signs and [b] is mostly negative zero.

   Both legs first need the negative zeros to reach the kernel at all, which is a second instance of
   the same normalization: the constant fill of a small array inlines as scalar stores of C
   literals, and [%.16g (-0.)] is ["-0"] — an integer literal, [+0.0] once cast. Hence the
   [c_float_literal] half of gh-ocannl-615 and the first printed check.

   CPU-only: the splat is a `Vec_extensions` construct. GPU backends render the fragment intrinsics
   or the scalar fallback and print the legs as skipped, so the golden stays backend-uniform. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
let skipped = Verdict.skipped ~backend:backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* Parity on the bits, not on the values: [Float.equal (-0.) 0.] is [true], and the sign of a zero
   is the entire difference this test is about. *)
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let is_neg_zero x = Float.equal x 0. && Float.ieee_negative x

(* A reference full of [+0.0] would satisfy the bitwise checks below while covering nothing (the
   normalizing splat produces exactly that). Every reference is pinned to actually carry negative
   zeros, and nonzero values elsewhere. *)
let discriminating name (a : float array) =
  if not (Array.exists a ~f:is_neg_zero) then
    failwith (name ^ ": no negative zero in the reference — the splat checks are vacuous");
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks are vacuous");
  a

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* One context lineage per leg: a tensor's initialization code is consumed by the first computation
   that embeds it, so the second compile of a leg must inherit the context that already holds the
   shared operands' values. *)
let run ?ctx ~name ~transform comp =
  let ctx = match ctx with Some ctx -> ctx | None -> Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx (named name comp) Ir.Indexing.Empty
  in
  Context.run ctx routine

(* The maximal single-child chains of statement-level loops: one symbol list per top-level nest. *)
let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

(* The innermost loop of the first top-level nest. *)
let rec innermost_loop (llc : LL.t) : Ir.Indexing.symbol option =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  match llc with
  | LL.Seq (a, b) -> ( match innermost_loop a with Some r -> Some r | None -> innermost_loop b)
  | LL.For_loop { index; body; _ } -> (
      match strip (LL.flat_lines [ body ]) with
      | [ single ] -> ( match innermost_loop single with Some r -> Some r | None -> Some index)
      | _ -> Some index)
  | LL.If { body; _ } -> innermost_loop body
  | _ -> None

(* === Leg 0: the constant literal, on every backend. ===

   Upstream of both splat legs, and not a CPU concern: the constant fill of a small array lowers to
   scalar stores of C literals on every C-family backend, and [%.16g (-0.)] is ["-0"] — an INTEGER
   literal, [+0.0] once cast. So before gh-ocannl-615 a [-0.0] in host data was normalized before
   any kernel ran, which is what made the first version of the legs below vacuous. Value-only, since
   whether a given backend inlines this particular fill or uploads it is its own business; either
   way the host's bits must come back. *)

let () =
  (* Both zeros and ordinary values, so a buffer that came back all-[+0.0] (the pre-fix behavior) or
     all-[-0.0] (a sign-flipping "fix") both fail. *)
  let zv = Array.init 8 ~f:(fun x -> if x % 2 = 0 then -0.0 else Float.of_int (x + 1) *. 0.5) in
  let z = TDSL.ndarray zv ~label:[ "nzc" ] ~output_dims:[ 8 ] () in
  let%op zc = z *. 1.0 in
  let ctx = run ~name:"nzc_fill" ~transform:(fun opt -> opt) (Train.forward zc) in
  let got = Context.get_values ctx z.Tensor.value in
  p "negative-zero host data survives initialization"
    (Array.length got = 8 && Array.for_all2_exn got zv ~f:bitwise)

(* === Leg 1: the register-tiled [Tile_mma] A splat. === *)

let mi = 4
let mk = 4

(* 16 columns is a whole number of vectors at every lane count in play (4 on NEON, 8 on AVX2, 16 at
   doubled fp16 lanes), so the tiling covers all of them and the peel — which reads A directly and
   was never affected — cannot stand in for the vector body. *)
let mj = 16

let () =
  if on_cpu then (
    (* Row 0 of A is all negative zero, the remaining rows ordinary values; B is strictly positive,
       so row 0's products are [-0.0] rather than [+0.0]. Everything is a multiple of 1/8, hence
       exact in f32: the parity claims are about zero signs, not rounding. *)
    let av =
      Array.init (mi * mk) ~f:(fun x -> if x < mk then -0.0 else Float.of_int ((x % 7) + 1) *. 0.5)
    in
    let bv =
      Array.init (mk * mj)
        ~f:(Ll_test.cycle_flat ~dims:[| mk; mj |] ~modulus:13 ~offset:1. ~stride:0.25)
    in
    (* The accumulator's preloaded value: the C-tile is loaded from [d] at block entry, and a [-0.0]
       accumulator is what makes a [-0.0] product visible ([(+0.0) + (-0.0) = +0.0] would otherwise
       absorb the sign). *)
    let dv = Array.create ~len:(mi * mj) (-0.0) in
    let ma = TDSL.ndarray av ~label:[ "nz_a" ] ~input_dims:[ mk ] ~output_dims:[ mi ] () in
    let mb = TDSL.ndarray bv ~label:[ "nz_b" ] ~input_dims:[ mj ] ~output_dims:[ mk ] () in
    let accumulator tag =
      TDSL.ndarray dv ~label:[ tag ] ~input_dims:[ mj ] ~output_dims:[ mi ] ()
    in
    let d_twin = accumulator "nz_d_twin" in
    let ctx_twin =
      run ~name:"nz_serial" ~transform:(fun opt -> opt) [%cd d_twin =+ ma * mb ~logic:"@"]
    in
    let want = discriminating "nz_serial" (Context.get_values ctx_twin d_twin.Tensor.value) in
    let d_mma = accumulator "nz_d_mma" in
    let transform (opt : LL.optimized) =
      let i, j, k =
        match List.find_exn (nest_paths opt.LL.llc) ~f:(fun q -> List.length q = 3) with
        | [ i; j; k ] -> (i, j, k)
        | _ -> assert false
      in
      let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:mj () in
      Sched.apply [ tz ] opt
    in
    let ctx_mma = run ~ctx:ctx_twin ~name:"nz_mma" ~transform [%cd d_mma =+ ma * mb ~logic:"@"] in
    let got = Context.get_values ctx_mma d_mma.Tensor.value in
    p_all2 "register-tiled Tile_mma preserves negative zero (bitwise vs the serial twin)" got want
      ~f:bitwise;
    let src = Generated.read "nz_mma" in
    let has s = String.is_substring src ~substring:s in
    p "register tiling renders a bit-preserving A splat"
      (has "Tile_mma register tiling" && has "full blocks 4x16 of 4x16"
     (* An initializer of repeated copies of one bound scalar — no arithmetic, so no bit pattern is
        altered. Both arithmetic spellings are pinned out by name: [(vtyp){0} + x] normalizes a
        negative zero, and [x - (vtyp){0}] quiets a signaling NaN. *)
     && has "tmma_as_0__, tmma_as_0__"
      && (not (has "){0} + "))
      && not (has " - (ocannl_vec")))
  else (
    skipped "register-tiled Tile_mma preserves negative zero (bitwise vs the serial twin)";
    skipped "register tiling renders a bit-preserving A splat")

(* === Leg 2: the [Vectorized] rendering's lane-uniform FMA operand. === *)

let () =
  if on_cpu then (
    let rows = 3 and cols = 64 in
    let n = rows * cols in
    (* [a] takes both signs, so the correct product is [-0.0] on the positive cells and [+0.0] on
       the negative ones — the normalizing splat gets BOTH backwards, which a fix that merely
       flipped a sign would not. [b] is negative zero except on a few cells: a zero addend keeps the
       product's zero sign observable, and the nonzero cells keep the reference non-vacuous (and pin
       that ordinary arithmetic is unchanged). *)
    let av =
      Array.init n ~f:(fun x ->
          let mag = Float.of_int ((x % 7) + 1) *. 0.5 in
          if x % 3 = 0 then -.mag else mag)
    in
    let bv =
      Array.init n ~f:(fun x -> if x % 4 = 1 then Float.of_int ((x % 9) + 1) *. 0.25 else -0.0)
    in
    let a = TDSL.ndarray av ~label:[ "nzv_a" ] ~output_dims:[ rows; cols ] () in
    let b = TDSL.ndarray bv ~label:[ "nzv_b" ] ~output_dims:[ rows; cols ] () in
    (* The lane-uniform multiplier: [-0.0] per row, broadcast along the vectorized axis. A scalar
       [-0.0] constant would not do — the simplifier folds a multiply by a zero constant away
       entirely — but a per-row array read mentions only the OUTER index, which is exactly what
       "lane-uniform" means to the vectorizer, so it reaches the fused multiply-add through the
       splat rather than through a binary operator's implicit (and exact) vector-scalar
       broadcast. *)
    let s =
      TDSL.ndarray (Array.create ~len:rows (-0.0)) ~label:[ "nzv_s" ] ~output_dims:[ rows ] ()
    in
    let%op c0 = s +* "i;ij=>ij" a + b in
    let ctx_twin = run ~name:"nzv_twin" ~transform:(fun opt -> opt) (Train.forward c0) in
    let want = discriminating "nzv_twin" (Context.get_values ctx_twin c0.Tensor.value) in
    (* Both zero signs must actually occur, or half the discrimination above is imaginary. *)
    p "vectorized reference carries zeros of both signs"
      (Array.exists want ~f:is_neg_zero
      && Array.exists want ~f:(fun x -> Float.equal x 0. && not (Float.ieee_negative x)));
    let%op c1 = s +* "i;ij=>ij" a + b in
    let transform (opt : LL.optimized) =
      let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
      Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt
    in
    let ctx_vec = run ~ctx:ctx_twin ~name:"nzv_vec" ~transform (Train.forward c1) in
    let got = Context.get_values ctx_vec c1.Tensor.value in
    p_all2 "vectorized FMA preserves a negative-zero uniform operand (bitwise vs the serial twin)"
      got want ~f:bitwise;
    let src = Generated.read "nzv_vec" in
    let has s = String.is_substring src ~substring:s in
    p "vectorized rendering renders a bit-preserving splat"
      (has "vector_size" && has "vunif" && (not (has "){0} + ")) && not (has " - (ocannl_vec")))
  else (
    skipped "vectorized reference carries zeros of both signs";
    skipped "vectorized FMA preserves a negative-zero uniform operand (bitwise vs the serial twin)";
    skipped "vectorized rendering renders a bit-preserving splat")
