(* 16-bit storage with f32 compute: does the halved memory traffic actually show up? (gh-ocannl-517
   task 3.)

   Three streaming kernels over large 1-D arrays, each run at f32, bf16 and half storage, in the
   default rendering and with the innermost loop explicitly retyped [Vectorized]. Narrow storage
   moves half the bytes of the f32 leg; the compute is f32 either way, so a bandwidth-bound kernel
   should approach 2x and a compute-bound one should not move. That contrast is the point: it is
   what distinguishes a real traffic win from a measurement artifact.

   The reported GB/s counts the bytes the kernel's own precision moves, so the number is directly
   comparable to the machine's stream bandwidth; the "vs f32" column is the speedup that matters for
   the issue's claim.

   Measured on an Apple-Silicon M-series (NEON, single stream, n = 2^22, 100 repeats), explicit
   [Vectorized] rendering, "vs f32" column ("half-nat" is fp16 storage computed in fp16,
   gh-ocannl-516):

   {v kernel f32 GB/s bf16 half half-nat add 127.3 0.91x 1.51x 2.05x mul_add 127.5 0.89x 1.48x 1.98x
   polynomial 87.3 0.63x 0.76x 1.99x v}

   Three findings:

   - The traffic win is real where the kernel is actually bandwidth-bound. "add" at f32 runs at the
   machine's stream ceiling and half storage collects most of the theoretical 2x. The compute-bound
   control stays below 1x with f32 compute, as it must -- there is no traffic to save there, only
   conversions to pay for.

   - It is *fp16*, not bf16, that collects the win, the reverse of gh-ocannl-517's expectation.
   bf16's conversion is cheap in instruction count but not free: widening is a zero-extend plus a
   shift and narrowing is four vector ops (the round-to-nearest-even of [single_to_bfloat16]), which
   at 130 GB/s costs more than halving the bytes saves. The route to making bf16 competitive is a
   hardware convert ([BFCVT] on ARMv8.6-A, AVX512-BF16 on x86) rather than portable bit arithmetic
   -- with the caveat that it would have to be shown to agree with [single_to_bfloat16] bitwise, NaN
   payloads included, or it breaks the parity the vectorized rendering owes its serial twin.

   - Native fp16 arithmetic (gh-ocannl-516) is what makes the compute-bound case work at all: the
   polynomial goes from 0.76x to 1.99x, because the lane count doubles instead of the loads being
   widened back to f32. It also removes the conversions from the streaming kernels, taking them from
   1.5x to ~2x. Reaching it needed the arch-flag fix in the same change: Apple clang accepts
   [-march=native] on arm64 and downgrades the target with it, so the probe saw a machine without
   16-bit arithmetic.

   Usage: dune exec bin/narrow_storage_bench.exe -- [n] [repeats] [threads] (defaults 4194304, 50,
   1).

   As in {!Cpu_vectorization_bench}, with no backend configured this pins the single-stream C
   backend rather than letting [Context.auto]'s metal/cuda-first order land a "CPU" benchmark on a
   GPU. An explicitly configured backend is respected -- but note the seam is CPU-only by design
   ({!Ir.Numerics.narrow_compute_f32}), so a GPU run measures native 16-bit arithmetic instead.

   Cross-check against the analytic cost model (gh-ocannl-491): its footprint extraction reads
   [prec_in_bytes] off each node, so the narrow legs' modelled bytes are exactly half, and the
   measured ratio below is the calibration of that prediction.

   Every leg's whole output is checked against a double-precision reference (gh-ocannl-1567), within
   a tolerance derived per leg from the kernel's rounding count and the leg's storage and compute
   precisions -- see [tolerance] below. A leg that fails its check is reported as failed and the
   process exits 1, so [benchmarks/bin_cells.py] fails the cell rather than publishing the timing of
   a wrong answer. The row prints the leg's tolerance, its worst observed error as a fraction of
   that tolerance, and how many elements overflowed the storage format (allowed exactly where the
   reference itself rounds to infinity: the polynomial control exceeds half's 65504 for most of its
   inputs, so its half legs mostly store [inf]). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Tn = Ir.Tnode

(* Flushed per line ([Bench_out]): see gh-ocannl-829 -- a buffered table is invisible until the
   process exits, which on a slow leg reads as a hang. *)
let p fmt = Bench_out.p fmt

(* The output oracle (gh-ocannl-1567). The reference evaluates the kernel in double over the values
   the kernel actually read -- the inputs read back from their storage, so their rounding into the
   narrow format is not error the check has to allow for. What remains is the kernel's own
   arithmetic, rounded at the compute precision once per operation, plus the final store's rounding
   into the storage precision when that is narrower than the compute.

   [unit_roundoff] is u = 2^-p for a format with p significand bits (round to nearest), and
   [overflow_at] the magnitude from which round-to-nearest gives infinity: 2^(emax+1) (1 - u/2). *)
let unit_roundoff prec =
  match prec with
  | Ir.Ops.Single_prec _ -> Float.ldexp 1. (-24)
  | Ir.Ops.Bfloat16_prec _ -> Float.ldexp 1. (-8)
  | Ir.Ops.Half_prec _ -> Float.ldexp 1. (-11)
  | Ir.Ops.Double_prec _ -> Float.ldexp 1. (-53)
  | _ -> invalid_arg ("narrow_storage_bench: no float format for " ^ Ir.Ops.prec_string prec)

let overflow_at prec =
  let u = unit_roundoff prec in
  match prec with
  | Ir.Ops.Half_prec _ -> Float.ldexp (1. -. (u /. 2.)) 16
  | Ir.Ops.Single_prec _ | Ir.Ops.Bfloat16_prec _ -> Float.ldexp (1. -. (u /. 2.)) 128
  | _ -> Float.infinity

(* The relative error bound of a leg whose every output element is the result of [ops] roundings at
   the compute precision, chained along the kernel's longest dependency path. Every operand and
   intermediate here is positive (the inputs lie in [0.5, 1.5)), so an addition does not amplify
   relative error -- (x(1+t1) + y(1+t2))(1+d) has |error| at most max(t1, t2) plus one rounding --
   and a multiplication adds its operands' counts plus one. Higham's gamma_k = k u / (1 - k u)
   (Accuracy and Stability of Numerical Algorithms, Lemma 3.1) then bounds the product of the k
   factors (1 + d_i), |d_i| <= u, rigorously rather than to first order -- which matters for the
   polynomial's 78 roundings in fp16, where k u is 0.04. The final store into a narrower storage
   format is one more rounding of the compute result. The reference's own double rounding (at most
   gamma_78 at 2^-53, ~1e-14) is below every tolerance by eight orders and left out. *)
let tolerance ~ops ~storage ~compute =
  let gamma k u = Float.of_int k *. u /. (1. -. (Float.of_int k *. u)) in
  let in_compute = gamma ops (unit_roundoff compute) in
  if Ir.Ops.equal_prec storage compute then in_compute
  else ((1. +. in_compute) *. (1. +. unit_roundoff storage)) -. 1.

(* The whole-output comparison: the worst |got - ref| / (tol |ref|) over the elements stored finite
   (above 1 when one of them fails), and how many stored [inf] where the reference overflows (within
   its tolerance). A NaN, a sign error, an [inf] the reference does not reach, and an element
   outside the tolerance each fail the leg. *)
type verdict = {
  worst : float;
  overflowed : int;
  bad : int;
  first_bad : (int * float * float) option;
}

let compare_output ~tol ~overflow ~got ~reference =
  let worst = ref 0. and overflowed = ref 0 and bad = ref 0 and first_bad = ref None in
  Array.iteri got ~f:(fun i g ->
      let r = reference i in
      let fail () =
        Int.incr bad;
        if Option.is_none !first_bad then first_bad := Some (i, g, r)
      in
      if Float.is_nan g then fail ()
      else if Float.is_inf g then
        if Float.(g > 0. && r *. (1. +. tol) >= overflow) then Int.incr overflowed else fail ()
      else
        let ratio = Float.abs (g -. r) /. (tol *. Float.abs r) in
        worst := Float.max !worst ratio;
        if Float.(ratio > 1.) then fail ());
  { worst = !worst; overflowed = !overflowed; bad = !bad; first_bad = !first_bad }

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

let () =
  (* Positional args beside the [--ocannl_*] config flags, split and range-checked by [Bench_args]
     (gh-ocannl-634): a negative extent is a positional here too, so it is reported by name rather
     than reaching [Array.init] below. *)
  let args = Bench_args.create "narrow_storage_bench" in
  let n = Bench_args.int args 0 ~name:"n" ~default:(1 lsl 22) in
  let repeats = Bench_args.int args 1 ~name:"repeats" ~default:50 in
  (* Threads: 1 (the default) measures one core's stream; 0 uses the whole pool, which is where a
     CPU actually reaches its memory ceiling and where a traffic win can show at all. *)
  let threads = Bench_args.int args 2 ~name:"threads" ~least:0 ~default:1 in
  (* O(1) magnitudes so bf16's ~3 significant decimal digits still leave the result recognizable. *)
  let av = Array.init n ~f:(fun i -> 0.5 +. (Float.of_int (i % 1009) /. 1009.)) in
  let bv = Array.init n ~f:(fun i -> 0.5 +. (Float.of_int (i % 997) /. 997.)) in
  let make_ctx () =
    match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
    | "" -> if threads > 0 then Context.cpu ~threads () else Context.cpu ()
    | _ -> Context.auto ()
  in
  let named name (comp : Ir.Assignments.comp) =
    { comp with Ir.Assignments.asgns = Ir.Assignments.Block_comment (name, comp.asgns) }
  in
  let serial opt = opt in
  let vectorize (opt : LL.optimized) =
    match innermost_loop opt.LL.llc with
    | None -> opt
    | Some j -> Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt
  in
  (* One (kernel, precision, rendering) measurement. [build] gets fresh leaves so each leg owns its
     own storage; [reads] is how many n-element arrays the kernel streams in (plus one written);
     [ops] and [reference] are the kernel's rounding count and double-precision twin, for the output
     check. *)
  let bench ~build ~reads ~ops ~reference ~prec ~transform ~label =
    (* Minted at [prec] rather than re-tagged: [ndarray] settles a leaf's precision as [Specified],
       which [update_prec] then refuses to change. *)
    let leaf l vals =
      NTDSL.init ~l ~prec ~o:[ n ] ~f:(function [| i |] -> vals.(i) | _ -> assert false) ()
    in
    let a = leaf (label ^ "_a") av and b = leaf (label ^ "_b") bv in
    let t : Tensor.t = build a b in
    Tn.update_prec t.Tensor.value prec;
    let comp = named label (Train.forward t) in
    let ctx = make_ctx () in
    let ctx, routine =
      Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
    in
    (* The compute precision is the one the backend resolves the storage to: on the CPU backends
       [Numerics.cpu_compute_prec] (the predicate [Cc_backend] emits through, under the policy this
       leg set), elsewhere the storage precision itself -- a GPU computes 16-bit formats natively,
       and assuming so only loosens the bound if it in fact widens. *)
    let compute =
      if Sched.backend_is_cpu (Context.backend_name ctx) then
        Ir.Numerics.cpu_compute_prec
          ~native_fp16_arithmetic:
            (Context.hardware_limits ctx).Ir.Backend_intf.native_fp16_arithmetic prec
      else prec
    in
    let ctx = Context.run ctx routine in
    let tol = tolerance ~ops ~storage:prec ~compute in
    let verdict =
      let got = Context.get_values ctx t.Tensor.value in
      let a_read = Context.get_values ctx a.Tensor.value
      and b_read = Context.get_values ctx b.Tensor.value in
      compare_output ~tol ~overflow:(overflow_at prec) ~got ~reference:(fun i ->
          reference a_read.(i) b_read.(i))
    in
    (* [get_values] is deliberately outside the timed region: it converts the whole device buffer
       into an OCaml [float array] element by element, which costs O(n) *independently of the
       storage precision* and would swamp -- and exactly mask -- the traffic difference this
       benchmark exists to measure. The cc backend's scheduler is synchronous, so the fold below
       needs no separate await. *)
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx (Stdlib.Array.make repeats ())
    in
    (* An explicitly configured accelerator enqueues asynchronously, so the timed region has to be
       fenced: without this it would report enqueue time, and the readback below -- which fences on
       its own -- happens after the clock stops. Cheap and a no-op on the synchronous cc
       scheduler. *)
    Context.sync ctx;
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let (_ : float array) = Context.get_values ctx t.Tensor.value in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    let bytes = Float.of_int ((reads + 1) * n * Ir.Ops.prec_in_bytes prec) in
    (secs, bytes /. secs /. 1e9, tol, verdict)
  in
  (* Each kernel with its double twin and its rounding count along the longest dependency path (the
     [ops] of [tolerance]): a sum of positives costs max(operand counts) + 1, a product their sum +
     1. A kernel edited without its twin and count fails its check loudly or, if the count is left
     too high, checks less than it says -- change the three together. *)
  let kernels =
    [
      (* Pure streaming: one add per 3 array touches -- the bandwidth-bound case the issue is
         about. *)
      ("add", (fun a b -> [%op a + b]), 2, 1, fun a b -> a +. b);
      (* Still streaming, twice the arithmetic per byte. *)
      ("mul_add", (fun a b -> [%op (a *. b) + a]), 2, 2, fun a b -> (a *. b) +. a);
      (* Compute-bound control: the intermediates are virtual, so the byte traffic is unchanged from
         "add" while the FLOPs per byte are ~15x. Narrow storage should NOT speed this up much -- if
         it does, the "add" number is measuring something other than traffic. *)
      ( "polynomial",
        (fun a b ->
          let%op t = ((a *. b) + a) *. ((a *. b) + b) *. ((a + b) *. a) in
          let%op u = (t *. t) + t in
          let%op v = (u *. u) + u in
          [%op (v *. v) + v]),
        2,
        (* ab: 1; ab + a, ab + b, (a + b) a: 2 each; their product t: 2 + 2 + 1, then + 2 + 1 = 8; t
           t + t: 17, 18; u u + u: 37, 38; v v + v: 77, 78. *)
        78,
        fun a b ->
          let t = ((a *. b) +. a) *. ((a *. b) +. b) *. ((a +. b) *. a) in
          let u = (t *. t) +. t in
          let v = (u *. u) +. u in
          (v *. v) +. v );
    ]
  in
  (* The last leg is half storage computed *in* half where the target has native 16-bit arithmetic
     (gh-ocannl-516): same bytes as "half", twice the lanes and no conversions. On a target without
     it the policy is ignored and the leg repeats "half". *)
  let base = Ir.Numerics.get () in
  let precs =
    [
      ("f32", Ir.Ops.single, base);
      ("bf16", Ir.Ops.bfloat16, base);
      ("half", Ir.Ops.half, base);
      ("half-nat", Ir.Ops.half, { base with Ir.Numerics.fp16_arithmetic = Ir.Numerics.Fp16_narrow });
    ]
  in
  let renderings = [ ("default", serial); ("vectorized", vectorize) ] in
  p "narrow storage bench: n = %d, %d repeats, threads = %d, narrow_compute_f32 = %b\n" n repeats
    threads base.Ir.Numerics.narrow_compute_f32;
  p "native fp16 arithmetic: %b\n"
    (Context.hardware_limits (make_ctx ())).Ir.Backend_intf.native_fp16_arithmetic;
  p "%-11s %-11s %-8s %10s %10s %8s %9s %7s %8s  %s\n" "kernel" "rendering" "prec" "us" "GB/s"
    "vs f32" "tol" "err/tol" "inf" "check";
  let failed = ref 0 in
  List.iter kernels ~f:(fun (name, build, reads, ops, reference) ->
      List.iter renderings ~f:(fun (rname, transform) ->
          let baseline = ref None in
          List.iter precs ~f:(fun (pname, prec, policy) ->
              Ir.Numerics.set_policy policy;
              let label = Printf.sprintf "nsb_%s_%s_%s" name rname pname in
              let secs, gbs, tol, v = bench ~build ~reads ~ops ~reference ~prec ~transform ~label in
              Ir.Numerics.set_policy base;
              let ratio =
                match !baseline with
                | None ->
                    baseline := Some secs;
                    1.0
                | Some base -> base /. secs
              in
              let check =
                match v.first_bad with
                | None -> Bench_out.Passed
                | Some (i, g, r) ->
                    Int.incr failed;
                    Bench_out.Failed
                      (Printf.sprintf
                         "%d of %d elements outside relative tolerance %.3g of the double \
                          reference; first at %d: got %.9g, reference %.9g"
                         v.bad n tol i g r)
              in
              p "%-11s %-11s %-8s %10.1f %10.2f %8.2fx %9.2e %7.3f %8d  %s\n" name rname pname
                (secs *. 1e6) gbs ratio tol v.worst v.overflowed
                (match check with Bench_out.Passed -> "ok" | _ -> "FAILED");
              Bench_out.result ~variant:label ~seconds:secs ~check)));
  if !failed > 0 then begin
    p "%d variant(s) failed their output check -- see the lines above.\n" !failed;
    Stdlib.exit 1
  end
