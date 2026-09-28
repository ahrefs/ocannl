(* 16-bit storage with f32 compute on the CPU backends (gh-ocannl-517): a narrow-float tensor node
   stays narrow in memory, but the arithmetic over it runs in f32 -- one widening per load, one
   narrowing per store, instead of a widen/op/narrow round-trip per operator.

   Three things are checked, in the order they can fail:

   1. Accuracy. An elementwise chain whose intermediates are virtual is run at bf16 storage against
   an all-f32 reference, once under the policy and once with [narrow_compute_f32 = false] (the
   per-operator rounding that predates this issue). Only the second rounds the intermediates, so its
   error is the larger one -- the assertion is the comparison, not a tolerance.

   2. Bitwise parity of the vectorized rendering against its serial twin, at bf16 and at half. The
   convert-on-load/store bridge and the scalar path must agree exactly: bf16's vector narrowing
   reimplements [single_to_bfloat16]'s round-to-nearest-even with vector arithmetic, and half's goes
   through [__builtin_convertvector] where the scalar path casts to [_Float16]. Anything approximate
   here would be a real defect, so the comparison is [=], not [approx].

   3. Structure of the generated source: a bf16 [Vectorized] loop must actually vectorize (it was
   gated to f32/f64 before this issue), and its arithmetic must be free of the per-operator
   [single_to_bfloat16] wrapping -- the narrowing appears once, on the way out.

   GPU backends are deliberately unaffected (they have native 16-bit types and arithmetic), so the
   structural checks are CPU-only; the parity checks hold everywhere. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* The structural checks below are CPU-only in substance -- the vector seam they pin is the C
   backends' -- but their claims print on every backend, because the golden is shared. So the read
   is what gets gated: on a GPU backend the kernels these name are never consulted, and asking for
   an artifact that this run had no reason to emit would be a failure, not a skip. *)
let read_on_cpu routine = if on_cpu then Generated.read routine else ""
let src_has src s = String.is_substring src ~substring:s

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

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

let n = 517
let av = Array.init n ~f:(fun i -> 0.125 +. (Float.of_int (i % 23) *. 0.3125))
let bv = Array.init n ~f:(fun i -> 0.5 +. (Float.of_int (i % 17) *. 0.1875))

(* An elementwise chain: every intermediate is virtual, so it lives in a register and is exactly
   what the storage/compute split is about. *)
let chain a b =
  let%op y = ((a *. b) + a) *. ((a + b) *. b) in
  y

let run ~name ~transform ~prec ~label () =
  (* The leaves are minted at [prec] rather than re-tagged: [ndarray] settles a leaf's precision as
     [Specified], which [update_prec] then refuses to change. *)
  let leaf l vals =
    NTDSL.init ~l ~prec ~o:[ n ] ~f:(function [| i |] -> vals.(i) | _ -> assert false) ()
  in
  let a = leaf (label ^ "a") av and b = leaf (label ^ "b") bv in
  let y = chain a b in
  Tn.update_prec y.Tensor.value prec;
  let comp = named name (Train.forward y) in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx y.Tensor.value

let serial _opt = _opt

let vectorize (opt : LL.optimized) =
  let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
  Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt

let max_err x y = Array.foldi x ~init:0. ~f:(fun i acc v -> Float.max acc (Float.abs (v -. y.(i))))

let () =
  Tensor.unsafe_reinitialize ();
  let base = Ir.Numerics.get () in
  (* gh-ocannl-516 task 1: the target capability is read where target capabilities live, not from
     the backend module -- that is the seam the probe exists to fill. *)
  (* Deliberately not printed: whether this machine has native fp16 arithmetic is a property of the
     machine, and every assertion below is written to hold either way -- the structural check at the
     end is what pins that the policy is honored exactly where the capability is reported. *)
  let native_fp16 =
    (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.native_fp16_arithmetic
  in

  (* --- 1. Accuracy: bf16 storage, f32 compute vs. per-operator rounding. --- *)
  let reference = run ~name:"nsc_f32" ~transform:serial ~prec:Ir.Ops.single ~label:"f32_" () in
  Ir.Numerics.set_policy { base with narrow_compute_f32 = true };
  let wide = run ~name:"nsc_bf16_wide" ~transform:serial ~prec:Ir.Ops.bfloat16 ~label:"wide_" () in
  Ir.Numerics.set_policy { base with narrow_compute_f32 = false };
  let per_op =
    run ~name:"nsc_bf16_perop" ~transform:serial ~prec:Ir.Ops.bfloat16 ~label:"perop_" ()
  in
  Ir.Numerics.set_policy { base with narrow_compute_f32 = true };
  let err_wide = max_err reference wide and err_per_op = max_err reference per_op in
  (* [narrow_compute_f32] is a C-backend policy: the GPU backends have native 16-bit arithmetic and
     ignore it, so both legs above are the same computation there and the accuracy comparison has
     nothing to compare. That is the same reason the structural checks at the end are CPU-only —
     said on stderr, as the sibling schedule tests do, so a vacuous pass is not read as coverage.
     The parity checks in section 2 do hold on every backend and stay ungated. *)
  if not on_cpu then
    Stdio.eprintf
      "narrow-compute: %s ignores narrow_compute_f32 (native 16-bit arithmetic) — the accuracy \
       checks are vacuous here\n"
      backend_name;
  p "f32 compute over bf16 storage beats per-operator rounding"
    ((not on_cpu) || Float.(err_wide < err_per_op));
  (* The wide leg's only rounding is the final store, so its relative error is bounded by bf16's
     half-ulp (2^-9); the per-operator leg compounds five of them. *)
  let rel arr =
    max_err reference arr /. Array.fold reference ~init:0. ~f:(fun m v -> Float.max m (Float.abs v))
  in
  p "wide-compute relative error within one bf16 ulp" ((not on_cpu) || Float.(rel wide < 0.004));
  p "per-operator relative error exceeds it" Float.(rel per_op > 0.004);

  (* --- 2. Bitwise parity of the vectorized rendering against the serial twin. --- *)
  List.iter
    [ ("bf16", Ir.Ops.bfloat16); ("half", Ir.Ops.half) ]
    ~f:(fun (name, prec) ->
      let twin = run ~name:("nsc_twin_" ^ name) ~transform:serial ~prec ~label:("t" ^ name) () in
      let vec = run ~name:("nsc_vec_" ^ name) ~transform:vectorize ~prec ~label:("v" ^ name) () in
      p_all2
        (name ^ " vectorized rendering is bitwise identical to the serial twin")
        vec twin ~f:Float.equal);

  (* --- 2a. Every code through the widening bridge (gh-ocannl-1072). --- The chain above loads a
     few dozen distinct values, and the bridge's x86 arms ([OCANNL_VEC_WIDEN_*_X<lanes>], one
     [vcvtph2ps] or [vpmovzxwd] each) are the machine's conversion, not the scalar path's text. So
     every 16-bit code is widened into an f32 node by the vectorized rendering and by its serial
     twin, and the two must agree BIT for bit -- [Float.equal] would take -0 for +0 -- while the
     twin must return each code's exact value, which is read off the format here rather than asked
     of either converter. A NaN code is fed as a NaN (the host init goes through the scalar
     narrowing, which quiets it), and must widen to a NaN. [x *. 1] is exact on every input, so the
     multiplication hides nothing a copy would show. Which arm runs is the host's: the x86 arm at
     the native width here, the portable one elsewhere. *)
  Ir.Numerics.set_policy
    {
      base with
      narrow_compute_f32 = true;
      fp16_arithmetic = Fp16_auto;
      bf16_arithmetic = Bf16_auto;
    };
  let bits x = Int64.bits_of_float x in
  let same_bits a b = Int64.equal (bits a) (bits b) in
  let bf16_value c =
    if c land 0x7F80 = 0x7F80 && c land 0x7F <> 0 then Float.nan
    else Int32.float_of_bits (Int32.of_int_trunc (c lsl 16))
  in
  let half_value c =
    let sign = if c land 0x8000 <> 0 then -1. else 1. in
    let e = (c lsr 10) land 0x1F and m = c land 0x3FF in
    if e = 0x1F then if m = 0 then sign *. Float.infinity else Float.nan
    else if e = 0 then sign *. Float.ldexp (Float.of_int m) (-24)
    else sign *. Float.ldexp (Float.of_int (1024 + m)) (e - 25)
  in
  List.iter
    [ ("bf16", Ir.Ops.bfloat16, bf16_value); ("half", Ir.Ops.half, half_value) ]
    ~f:(fun (name, prec, value) ->
      let codes = Array.init 65536 ~f:value in
      let widen ~transform ~label =
        let x =
          NTDSL.init ~l:(label ^ "x") ~prec ~o:[ 65536 ]
            ~f:(function [| i |] -> codes.(i) | _ -> assert false)
            ()
        in
        let%op y = x *. 1. in
        Tn.update_prec y.Tensor.value Ir.Ops.single;
        let ctx = Context.auto () in
        let ctx, routine =
          Context.compile
            ~lowered_transform:(fun o -> [ transform o ])
            ctx
            (named ("nsc_" ^ label) (Train.forward y))
            Ir.Indexing.Empty
        in
        Context.get_values (Context.run ctx routine) y.Tensor.value
      in
      let twin = widen ~transform:serial ~label:("wtwin_" ^ name) in
      let vec = widen ~transform:vectorize ~label:("wvec_" ^ name) in
      p_all2
        (name ^ " vectorized widening of every code is bitwise identical to the serial twin")
        vec twin ~f:same_bits;
      p_all2 (name ^ " serial widening returns every code's exact value, and a NaN for a NaN code")
        twin codes ~f:(fun got want ->
          if Float.is_nan want then Float.is_nan got else same_bits got want));

  (* --- 2c. Every fp16 rounding boundary through the narrowing bridge (gh-ocannl-1101). --- The
     store-side twin of 2a: the narrowing bridge's x86 arms (one [vcvtps2ph] per width, named by
     [C_syntax.vec_narrow_macro]) replace gcc's per-lane lowering of the portable one. The f32
     inputs are chosen where rounding decides: for every finite half code, its exact value, the
     midpoint to the next code up in magnitude (a tie, which goes to the even code), and the f32
     neighbours on either side of that midpoint; both signs; the top code's next is 65536, so 65520
     and up round to infinity. Plus the infinities, a NaN, the largest f32 and the smallest f32
     subnormal. Each input's code is known by construction, not asked of either converter. bf16's
     narrowing needs no twin here: [bf16_codec_exhaustive] feeds every f32 input through it. *)
  (let boundaries =
     List.concat_map (List.init 0x7C00 ~f:Fn.id) ~f:(fun c ->
         let v = half_value c in
         let next = if c = 0x7BFF then 65536. else half_value (c + 1) in
         let rounded_up = if c = 0x7BFF then Float.infinity else next in
         let mid = (v +. next) /. 2. in
         let mid_bits = Int32.bits_of_float mid in
         [
           (v, v);
           (mid, if c % 2 = 0 then v else rounded_up);
           (Int32.float_of_bits (Int32.pred mid_bits), v);
           (Int32.float_of_bits (Int32.succ mid_bits), rounded_up);
         ])
   in
   let specials =
     [
       (Float.infinity, Float.infinity);
       (Float.nan, Float.nan);
       (Int32.float_of_bits 0x7F7FFFFFl, Float.infinity);
       (Int32.float_of_bits 1l, 0.);
     ]
   in
   let cases =
     List.concat_map (boundaries @ specials) ~f:(fun (x, want) -> [ (x, want); (-.x, -.want) ])
   in
   (* A multiple of every lane count, so the vectorized rendering's remainder loop has no trips and
      every input crosses the bridge. *)
   let padded =
     Array.of_list (cases @ List.init (64 - (List.length cases % 64)) ~f:(fun _ -> (0., 0.)))
   in
   let len = Array.length padded in
   let narrow ~transform ~label =
     let x =
       NTDSL.init ~l:(label ^ "x") ~prec:Ir.Ops.single ~o:[ len ]
         ~f:(function [| i |] -> fst padded.(i) | _ -> assert false)
         ()
     in
     let%op y = x *. 1. in
     Tn.update_prec y.Tensor.value Ir.Ops.half;
     let ctx = Context.auto () in
     let ctx, routine =
       Context.compile
         ~lowered_transform:(fun o -> [ transform o ])
         ctx
         (named ("nsc_" ^ label) (Train.forward y))
         Ir.Indexing.Empty
     in
     Context.get_values (Context.run ctx routine) y.Tensor.value
   in
   let twin = narrow ~transform:serial ~label:"ntwin_half" in
   let vec = narrow ~transform:vectorize ~label:"nvec_half" in
   p_all2
     "half vectorized narrowing of every rounding boundary is bitwise identical to the serial twin"
     vec twin ~f:same_bits;
   p_all2 "half serial narrowing rounds every boundary to nearest even, and a NaN to a NaN" twin
     (Array.map padded ~f:snd) ~f:(fun got want ->
       if Float.is_nan want then Float.is_nan got else same_bits got want));

  (* --- 2b. Native fp16 arithmetic (gh-ocannl-516): same parity obligation, one precision up. ---
     Where the target has genuine 16-bit arithmetic the half legs compute *in* half at twice f32's
     lane count, so the vector rendering is a different kernel from the one checked above -- and
     owes its serial twin the same bitwise equality. Where the target has only promoted or emulated
     fp16 the policy is ignored and this repeats the widening path, which is the point of testing
     the flag rather than the hardware. *)
  Ir.Numerics.set_policy { base with narrow_compute_f32 = true; fp16_arithmetic = Fp16_narrow };
  let twin = run ~name:"nsc_twin_nat" ~transform:serial ~prec:Ir.Ops.half ~label:"tnat" () in
  let vec = run ~name:"nsc_vec_nat" ~transform:vectorize ~prec:Ir.Ops.half ~label:"vnat" () in
  p_all2 "native-fp16 vectorized rendering is bitwise identical to the serial twin" vec twin
    ~f:Float.equal;
  (* Half's range and mantissa are wider than the chain needs, so computing in half must still land
     within a couple of half ulps of the f32 reference -- a wrong lane geometry or a mismatched FMA
     would not. *)
  p "native-fp16 relative error stays within a few half ulps"
    Float.(
      max_err reference vec
      /. Array.fold reference ~init:0. ~f:(fun m v -> Float.max m (Float.abs v))
      < 0.01);
  let native_source = read_on_cpu "nsc_vec_nat" in

  (* Computing *in* fp16 means fp16's 65504 ceiling applies to the intermediates, not just to the
     stored result. [exp 12.] is 162754, which overflows it; scaling that back down recovers a
     finite value only if the exponential was allowed to stay in f32. The check is on a library call
     on purpose: the ring operators compute in [_Float16] whether or not the result of a
     float-returning call is cast back, so an [expf] is what distinguishes the two. *)
  let overflow_leg policy =
    let a = NTDSL.init ~l:"ovf_a" ~prec:Ir.Ops.half ~o:[ 1 ] ~f:(fun _ -> 12.0) () in
    let c = NTDSL.init ~l:"ovf_c" ~prec:Ir.Ops.half ~o:[ 1 ] ~f:(fun _ -> 0.001) () in
    let%op y = exp a *. c in
    Tn.update_prec y.Tensor.value Ir.Ops.half;
    Ir.Numerics.set_policy policy;
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ serial o ])
        ctx
        (named "nsc_ovf" (Train.forward y))
        Ir.Indexing.Empty
    in
    let v = (Context.get_values (Context.run ctx routine) y.Tensor.value).(0) in
    Ir.Numerics.set_policy base;
    v
  in
  let ovf_wide =
    overflow_leg { base with narrow_compute_f32 = true; fp16_arithmetic = Fp16_auto }
  in
  let ovf_native =
    overflow_leg { base with narrow_compute_f32 = true; fp16_arithmetic = Fp16_narrow }
  in
  (* Same gate as the bf16 accuracy legs: both policies are C-backend knobs, so on a GPU backend
     both [overflow_leg] calls compute [exp 12.] in half and overflow — a property of the target's
     native arithmetic, not of the policy this pair exists to pin. *)
  p "f32 compute over half storage keeps the intermediate finite"
    ((not on_cpu) || Float.(is_finite ovf_wide));
  p "fp16 compute applies fp16's ceiling to the intermediate"
    ((not on_cpu)
    || if native_fp16 then not Float.(is_finite ovf_native) else Float.(is_finite ovf_native));

  Ir.Numerics.set_policy { base with narrow_compute_f32 = true; fp16_arithmetic = Fp16_auto };

  (* --- 2c. The shared fp16 FMA must survive an emulated target. --- *)
  (* `narrow_compute_f32 = false` leaves half at half on *any* target, including one without
     `_Float16`, where `HALF_T` is `uint16_t`. `OCANNL_HALF_FMA` therefore cannot cast its operands
     to float directly -- that would compute on the raw half bit pattern (0x3c00 rather than 1.0) --
     nor take the elementwise builtin, which rejects integer operands. The macro text is what
     encodes that, and it travels with the kernel, so checking the emitted definition holds on a
     native machine too. *)
  let fma_leg () =
    let a = NTDSL.init ~l:"fma_a" ~prec:Ir.Ops.half ~o:[ 4 ] ~f:(fun _ -> 1.5) () in
    let b = NTDSL.init ~l:"fma_b" ~prec:Ir.Ops.half ~o:[ 4 ] ~f:(fun _ -> 2.0) () in
    let%op y = (a *. b) + a in
    Tn.update_prec y.Tensor.value Ir.Ops.half;
    Ir.Numerics.set_policy { base with narrow_compute_f32 = false; fp16_arithmetic = Fp16_auto };
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ serial o ])
        ctx
        (named "nsc_half_fma" (Train.forward y))
        Ir.Indexing.Empty
    in
    let v = (Context.get_values (Context.run ctx routine) y.Tensor.value).(0) in
    Ir.Numerics.set_policy base;
    v
  in
  let fma_v = fma_leg () in
  p "per-operator half FMA computes 1.5 * 2 + 1.5" Float.(abs (fma_v - 4.5) < 0.01);
  (let src = read_on_cpu "nsc_half_fma" in
   let has t = String.is_substring src ~substring:t in
   p "the shared half FMA converts rather than bit-casting"
     ((not on_cpu)
     || (has "OCANNL_HALF_FMA" && has "HALF_TO_FLOAT(a)" && not (has "fmaf((float)(a)"))));

  (* --- 3. Structure of the bf16 vectorized source. ---

     CPU-only in substance (the vector seam these pin is the C backends'), but printed on every
     backend: the golden is shared, so a GPU run that emitted fewer lines than cc could never match
     it however it behaved. *)
  let vec_source = read_on_cpu "nsc_vec_bf16" in
  p "bf16 loop vectorizes with a converting load/store"
    ((not on_cpu)
    || src_has vec_source "vector_size"
       && src_has vec_source "OCANNL_VEC_WIDEN_BFLOAT16"
       && src_has vec_source "OCANNL_VEC_NARROW_BFLOAT16");
  (* Section 2a's sweep is about the bridge, so its vectorized kernels must have taken it: an input
     the rendering folded or converted per lane would pass the parity vacuously. *)
  p "the every-code sweep loads through the widening bridge"
    ((not on_cpu)
    || src_has (read_on_cpu "nsc_wvec_bf16") "OCANNL_VEC_WIDEN_BFLOAT16"
       && src_has (read_on_cpu "nsc_wvec_half") "OCANNL_VEC_WIDEN_HALF");
  (* And section 2c's vectorized kernel must have stored through the narrowing bridge. *)
  p "the rounding-boundary sweep stores through the narrowing bridge"
    ((not on_cpu) || src_has (read_on_cpu "nsc_nvec_half") "OCANNL_VEC_NARROW_HALF");
  (* [narrow(op(...))] immediately re-widened is the signature of per-operator rounding; the seam
     makes it unspellable, in the vector body and in the serial remainder alike. *)
  p "no operator narrows only to be widened again"
    ((not on_cpu) || not (src_has vec_source "bfloat16_to_single(single_to_bfloat16("));
  (* f32 arithmetic reached a bf16 kernel: the fused multiply-add is the f32 one. *)
  p "arithmetic is f32"
    ((not on_cpu) || src_has vec_source "fmaf(" || src_has vec_source "__builtin_elementwise_fma");
  (* Under the fp16-arithmetic policy the half kernel's vector element type is HALF_T rather than
     float -- the lane count doubles and no conversion appears at all. On a target without native
     16-bit arithmetic the policy is correctly ignored, and the kernel is the widening one. *)
  p "fp16 policy is honored exactly where the target reports the capability"
    ((not on_cpu)
    ||
    let nhas t = String.is_substring native_source ~substring:t in
    if native_fp16 then nhas "HALF_T ocannl_vec" && not (nhas "OCANNL_VEC_WIDEN_HALF")
    else nhas "OCANNL_VEC_WIDEN_HALF" || nhas "HALF_TO_FLOAT")
