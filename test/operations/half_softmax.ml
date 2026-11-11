(* Regression test for half-precision softmax, in particular a causally masked one -- the shape
   every attention model has.

   Two constants used to make this unlowerable at half precision, both raised from
   [Low_level.simplify_llc.check_constant] before any backend saw the code (hence the failure was
   backend-independent -- it was reproduced on cc, metal, cuda and hip alike):

   1. The max-subtraction's [-inf]. A [Max] accumulation initializes with its neutral element
   [Float.neg_infinity], and [Ops.exceeds_fp16_cutoff] rejected it: [abs c >= cutoff] is trivially
   true for an infinity. But [-inf] is exactly representable in binary16 and the code generator
   emits it deliberately ([C_syntax] renders [(-INFINITY)]), so the cutoff -- a headroom guard
   against overflow during arithmetic -- had no business rejecting it. Infinities are now exempt.
   Part 1 below pins the bare max-reduction; part 2 pins it inside a softmax.

   2. The mask fill. It was [-1e9], four orders of magnitude past binary16's largest finite value,
   so the guard was doing exactly its job. The fill is now [Float.neg_infinity]
   ([Nn_blocks.default_mask_fill], overridable per call via [?mask_fill]), which is
   precision-independent -- see the comment there for why not a smaller finite magic number.

   Fixing (1) alone only moved the failure to (2), so the masked-softmax case below is the one that
   actually pins the workload; the max-reduction case isolates (1).

   Values are chosen exactly representable in half (10 mantissa bits) where they are printed. *)

open Base
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
module Precision_policy = Ocannl.Precision_policy
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

(* [Min] reductions initialize with [Float.infinity] and so hit the very same guard, but the DSL has
   no [einmin1] to reach that path with -- so here is a local forward-only one. (gh-ocannl-547 asks
   for the [+inf] side explicitly; the exemption is [Float.is_finite], which is sign-agnostic.) *)
let einmin1 spec =
  let%cd op_asn ~t ~t1 ~projections = v =:@- v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop
    ~transpose_op:(Shape.Permute (Shape.einsum_spec spec, []))
    ~op_asn ~grad_asn ~op_label:"@-=>"

let show ctx label t =
  Stdio.printf "%s:" label;
  Stdio.printf " ";
  Test_utils.print_floats ~prec:4 (Array.to_list (Context.get_values ctx t));
  Stdio.printf "\n"

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in

  (* === Part 1: bare max- and min-reductions at half === The [-inf] / [+inf] neutral elements of
     the [Max] / [Min] accumulations land in [m]'s and [mn]'s own half nodes. *)
  let x = Tensor.term_init [| 1.5; -2.; 3.25; 0.5 |] ~label:[ "x" ] ~output_dims:[ 4 ] () in
  let%op m = x @^^ "i => 0" in
  let mn = einmin1 "i => 0" ~grad_spec:Tensor.Prohibit_grad x () in
  List.iter [ x; m; mn ] ~f:(fun t ->
      Tn.update_prec t.Tensor.value Ir.Ops.half;
      Train.set_materialized t.Tensor.value);
  let ctx = Train.forward_once ctx m in
  let ctx = Train.forward_once ctx mn in
  show ctx "max of [1.5; -2; 3.25; 0.5] at half" m.Tensor.value;
  show ctx "min of [1.5; -2; 3.25; 0.5] at half" mn.Tensor.value;

  (* === Part 2: a causally masked softmax at half === Mirrors [multi_head_attention]'s use: a 0/1
     causal mask selects between the scores and [Nn_blocks.default_mask_fill], and the result goes
     through the max-subtracting [softmax]. Batch axis [s] is the query position, input axis [t] the
     key position; the softmax reduces [t]. *)
  let n = 4 in
  let scores =
    NTDSL.init ~l:"scores" ~prec:Ir.Ops.half ~b:[ n ] ~i:[ n ] ~o:[]
      ~f:(function [| s; t |] -> Float.of_int ((2 * s) + t) *. 0.25 | _ -> assert false)
      ()
  in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.half ~b:[ n ] ~i:[ n ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let%op masked = where mask scores !.Nn_blocks.default_mask_fill in
  let probs = Nn_blocks.softmax ~spec:" ... | t -> ..." () masked in
  (* Half throughout -- params, activations and the constants folded into them -- so the sentinels
     have to survive in half nodes rather than in a single-precision temporary. *)
  Precision_policy.apply (Precision_policy.uniform Ir.Ops.half) probs;
  Train.set_materialized probs.Tensor.value;
  let ctx = Train.forward_once ctx probs in
  let p = Context.get_values ctx probs.Tensor.value in
  (* Row [s] has [s + 1] live positions. The masked-out entries must be exactly zero (not merely
     small): that is what distinguishes a working sentinel from one that saturated.

     The table is printed coarsely on purpose. Unlike the reduction identities above, softmax
     outputs are [exp] results, and no choice of inputs makes those exactly representable in half —
     so a backend whose [exp] rounds the last mantissa bit differently disagrees by an ulp (HIP
     yields 2.1215e-1 where cc, metal and cuda yield 2.1228e-1: one ulp at 2^-13). Rather than
     letting that decide a golden diff, the table stays above the divergence and the numeric check
     is the tolerance comparison below, against a reference evaluated in double. *)
  Stdio.printf "\ncausal softmax at half (row s reduces over t):\n";
  for s = 0 to n - 1 do
    Stdio.printf "  s=%d:" s;
    Test_utils.print_floats ~prec:2 (List.init n ~f:(fun t -> p.((s * n) + t)));
    Stdio.printf "\n"
  done;
  let row_sum s = List.sum (module Float) (List.init n ~f:(fun t -> p.((s * n) + t))) ~f:Fn.id in
  Verdict.p_close "rows sum to 1 (within half's resolution)" (Array.init n ~f:row_sum)
    (Array.create ~len:n 1.) ~tol:(fun _ -> 0.01);
  Verdict.p_all "masked-out entries are exactly zero" (List.init n ~f:Fn.id) ~f:(fun s ->
      List.for_all (List.init n ~f:Fn.id) ~f:(fun t -> s >= t || Float.equal p.((s * n) + t) 0.));
  Verdict.p_all "all entries finite" (Array.to_list p) ~f:Float.is_finite;
  (* Tolerance against the same softmax evaluated in double. Half's resolution below 1 is ~1e-3, so
     5e-4 admits the ulp of backend [exp] disagreement and nothing looser -- a saturated sentinel or
     a lost max-subtraction moves these by far more. *)
  let live_got, live_want =
    List.concat_map (List.init n ~f:Fn.id) ~f:(fun s ->
        let live = List.init (s + 1) ~f:(fun t -> Float.of_int ((2 * s) + t) *. 0.25) in
        let mx = List.reduce_exn live ~f:Float.max in
        let denom = List.sum (module Float) live ~f:(fun v -> Float.exp (v -. mx)) in
        List.mapi live ~f:(fun t v -> (p.((s * n) + t), Float.exp (v -. mx) /. denom)))
    |> List.unzip
  in
  Verdict.p_close "matches a double-precision softmax within 5e-4" (Array.of_list live_got)
    (Array.of_list live_want) ~tol:(fun _ -> 5e-4);
  Stdio.printf "probs prec: %s\n"
    (Ir.Ops.prec_string (Lazy.force probs.Tensor.value.Tn.storage_prec))
