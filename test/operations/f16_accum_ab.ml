(* gh-ocannl-2195: the accuracy half of the f16 accumulation A/B on CUDA and HIP.

   Under [fp16_arithmetic=auto] CUDA and HIP resolve f16 reduction accumulators wide (f32 in the
   serial loop and in the tensor-unit arm, one narrowing at the store), while [fp16_arithmetic=true]
   ([Fp16_narrow]) keeps the f16-accumulate tensor units, which narrow per 16-deep block on sm_120
   and per ordered pair on gfx1151 (gh-ocannl-2151), and the storage-width serial loop, which
   narrows per step. This measures both against an f64 reference over contractions whose running
   sums outgrow f16's resolution -- the real-input case the census fixtures exclude -- so the error
   is genuine accumulation error. Three contractions: a GEMM with mixed-sign operands at a short (k
   = 64) and at a long (k = 1024) reduction, and one model contraction, attention's value product
   P.V (a row of softmax-like positive weights against mixed-sign values over a 1024-key sequence).

   Each runs as the hand-tensorized [Tile_mma] schedule (the whole k extent in one statement, as
   [mma_bf16_seed_parity]) and as the serial loop, under each policy. The reference is the f64 sum
   of the products of the actual half operands (every input is an f16 value by construction, so the
   host values ARE the device operands). Per run, the maximum and RMS of the error normalized by the
   cell's sum of absolute products (a condition-free scale), and the count of non-finite cells, go
   to stderr. Stdout holds two-sided claims.

   Under the default, the tile's and the serial loop's normalized errors lie in (0, B], with B =
   2^-11 + k * 2^-23: the final f16 rounding (at most half an f16 ulp of the result, below 2^-11 of
   the absolute-product sum) plus an f32 accumulation term generous enough for tensor-unit f32
   accumulation, which is not an IEEE per-step chain. Nonzero, because the result is stored at f16;
   bounded, because it accumulated wide. Also under the default, tile and serial agree within one
   f16 ulp of the result per cell plus the f32 accumulation term: the same width and the same single
   narrowing point, not bitwise (the f32 summation orders differ). At the long reductions, the
   narrow tile's error is at least the default tile's and at most the per-step bound k * 2^-10: the
   A/B observes the narrowing it is about. And no run produces a non-finite cell.

   The leg runs only where the backend resolves [Fp16_auto] wide and advertises the uniform-f16 tile
   (CUDA, HIP); elsewhere every claim skips. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Numerics = Ir.Numerics
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cuda = String.is_substring backend_name ~substring:"cuda"
let on_hip = String.is_substring backend_name ~substring:"hip"

let capable =
  lazy
    ((on_cuda || on_hip)
    &&
    match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
    | None -> false
    | Some mma ->
        List.Assoc.mem mma.Ir.Backend_intf.mma_format_tiles
          (Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f16)
          ~equal:Ir.Backend_intf.equal_mma_format_triple)

let leg = Verdict.leg ~when_:(Lazy.force capable) ~on:backend_name ()
let n = 32
let bm = 16
let simd_width = 32

(* A deterministic scramble of the indices into [0, m). *)
let scramble ~salt idcs m =
  let h = Array.fold idcs ~init:salt ~f:(fun h i -> ((h * 1_000_003) + i + 7) land 0x3FFFFFFF) in
  h lxor (h lsr 13) * 2_654_435 land 0x3FFFFFFF % m

(* Multiples of 1/64 in [-2, 2): at most 8 significant bits, so every input is an f16 value.
   Products are multiples of 2^-12, and over these depths the running sums outgrow f16's resolution,
   so narrowing points show. *)
let signed ~salt idcs = (Float.of_int (scramble ~salt idcs 256) -. 128.) /. 64.

(* Attention weights: exp(-d/4) for a scrambled distance d in [0, 32), rounded to f16 on the host
   and then divided by 1024 (exact), so the row sums to about one over 1024 keys. a [| i; key |]. *)
let weights idcs =
  Ir.Ops.round_to_half (Float.exp (-.Float.of_int (scramble ~salt:11 idcs 32) /. 4.)) /. 1024.

type contraction = {
  name : string;
  k : int;
  fa : int array -> float;
  fb : int array -> float;
  long : bool;
}

let contractions =
  [
    { name = "gemm_k64"; k = 64; fa = signed ~salt:1; fb = signed ~salt:2; long = false };
    { name = "gemm_k1024"; k = 1024; fa = signed ~salt:3; fb = signed ~salt:4; long = true };
    { name = "attn_pv"; k = 1024; fa = weights; fb = signed ~salt:5; long = true };
  ]

let named name (comp : Ir.Assignments.comp) : Ir.Assignments.comp =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

(* mma_bf16_seed_parity's hand schedule: the zero expanded ahead of one [Tile_mma] statement per
   16-row grid block, the whole k extent inside the statement. *)
let mma_schedule ~out (opt : LL.optimized) : Sched.schedule =
  let i, j, k =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k ~simd_width () in
  [ ez; sp_zi; rz; sp_i; tz ]

(* One compile and run; the tile's destination starts far from every result, so each cell must be
   overwritten. *)
let run ~c ~tag ~tiled =
  let a = NTDSL.init ~l:(tag ^ "_a") ~prec:Ir.Ops.half ~i:[ c.k ] ~o:[ n ] ~f:c.fa () in
  let b = NTDSL.init ~l:(tag ^ "_b") ~prec:Ir.Ops.half ~i:[ n ] ~o:[ c.k ] ~f:c.fb () in
  let%op t = a * b in
  Tn.update_prec t.Tensor.value Ir.Ops.half;
  let transform opt =
    if tiled then Sched.apply (mma_schedule ~out:t.Tensor.value opt) opt else opt
  in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ())
      (named tag (Train.forward t))
      Ir.Indexing.Empty
  in
  let numel = n * n in
  let ctx =
    Context.set_values ctx t.Tensor.value (Array.init numel ~f:(fun c -> Float.of_int (-4096 - c)))
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx t.Tensor.value, Ll_test.renderings routine.Context.mma)

type measured = {
  max_err : float;
  rms_err : float;
  nonfinite : int;
  got : float array;
  scales : float array;  (** Per cell, the sum of absolute products. *)
}

let measure c got =
  let products = Ll_test.matmul_products ~m:n ~n ~k:c.k c.fa c.fb in
  let scales =
    Array.of_list
      (List.map products ~f:(fun terms -> List.fold terms ~init:0. ~f:(fun s x -> s +. Float.abs x)))
  in
  let errs =
    List.mapi products ~f:(fun cell terms ->
        let reference = List.fold terms ~init:0. ~f:( +. ) in
        Float.abs (got.(cell) -. reference)
        /. Float.max scales.(cell) Float.min_positive_normal_value)
  in
  let nonfinite = Array.count got ~f:(fun x -> not (Float.is_finite x)) in
  let max_err = List.fold errs ~init:0. ~f:Float.max in
  let rms_err =
    Float.sqrt (List.fold errs ~init:0. ~f:(fun s e -> s +. (e *. e)) /. Float.of_int (n * n))
  in
  { max_err; rms_err; nonfinite; got; scales }

let pow2 e = Float.ldexp 1. e

(* One f16 ulp of [x]: 2^(e - 10) for |x| in [2^e, 2^(e+1)), the subnormal spacing below. *)
let f16_ulp x =
  let ax = Float.abs x in
  if Float.(ax < pow2 (-14)) then pow2 (-24)
  else
    let _, e = Float.frexp ax in
    (* ax = m * 2^e with m in [0.5, 1): its binade is 2^(e - 1). *)
    pow2 (e - 11)

let policies = [ ("auto", Numerics.Fp16_auto); ("narrow", Numerics.Fp16_narrow) ]

let results =
  lazy
    (List.map contractions ~f:(fun c ->
         let per_policy =
           List.map policies ~f:(fun (pname, mode) ->
               Test_utils.with_policy
                 (fun p -> { p with Numerics.fp16_arithmetic = mode })
                 (fun () ->
                   let one ~tiled =
                     let tag =
                       Printf.sprintf "f16ab_%s_%s_%s" c.name pname
                         (if tiled then "tile" else "serial")
                     in
                     let got, census = run ~c ~tag ~tiled in
                     let m = measure c got in
                     Stdio.eprintf
                       "f16_accum_ab: %s %s %s %s: max normalized error %.3e, rms %.3e, non-finite \
                        %d, census %s (not part of the golden)\n\
                        %!"
                       backend_name c.name pname
                       (if tiled then "tile" else "serial")
                       m.max_err m.rms_err m.nonfinite (Ll_test.census_string census);
                     (m, census)
                   in
                   (pname, (one ~tiled:true, one ~tiled:false))))
         in
         (c, per_policy)))

let find pname per_policy = List.Assoc.find_exn per_policy pname ~equal:String.equal

let () =
  List.iter contractions ~f:(fun c ->
      let get () = List.Assoc.find_exn (Lazy.force results) c ~equal:Poly.equal in
      let bound = pow2 (-11) +. (Float.of_int c.k *. pow2 (-23)) in
      in_legf ~leg "%s: every tile renders tensor-unit intrinsics under both policies" c.name
        (fun label ->
          p_all label policies ~f:(fun (pname, _) ->
              let (_, census), _ = find pname (get ()) in
              Ll_test.all_rendered ~what:label Ir.C_syntax.Mma_intrinsics census));
      in_legf ~leg "%s: no run produces a non-finite cell" c.name (fun label ->
          p_all label policies ~f:(fun (pname, _) ->
              let (t, _), (s, _) = find pname (get ()) in
              t.nonfinite = 0 && s.nonfinite = 0));
      in_legf ~leg
        "%s, default policy: the tile's and the serial loop's normalized errors are nonzero and \
         within the wide bound 2^-11 + k*2^-23"
        c.name (fun label ->
          let (t, _), (s, _) = find "auto" (get ()) in
          p_all label [ t; s ] ~f:(fun m -> Float.(m.max_err > 0. && m.max_err <= bound)));
      (* Both narrow once from an f32 sum; the sums differ by at most the two f32 accumulation
         errors, 2k * 2^-24 of the absolute-product sum (bounded here by the contraction's largest),
         which can exceed an ulp of a cancelling cell's small result -- hence the second term. *)
      in_legf ~leg
        "%s, default policy: tile and serial loop agree within one f16 ulp per cell, plus the f32 \
         accumulation term"
        c.name (fun label ->
          let (t, _), (s, _) = find "auto" (get ()) in
          let max_scale = Array.fold t.scales ~init:0. ~f:Float.max in
          let f32_term = 2. *. Float.of_int c.k *. pow2 (-24) *. max_scale in
          p_close label t.got s.got ~tol:(fun w -> f16_ulp w +. f32_term));
      if c.long then
        in_legf ~leg
          "%s: the narrow tile's error is at least the default tile's and within the per-step \
           bound k*2^-10"
          c.name (fun label ->
            let (narrow, _), _ = find "narrow" (get ()) in
            let (auto, _), _ = find "auto" (get ()) in
            p label
              Float.(
                narrow.max_err >= auto.max_err && narrow.max_err <= Float.of_int c.k *. pow2 (-10))))
