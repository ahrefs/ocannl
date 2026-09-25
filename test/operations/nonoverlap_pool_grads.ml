(* The non-overlapping pooling gradient-gate specialization, gh-ocannl-527.

   gh-512 made the max-family gradient gates exact for overlapping windows by moving them into the
   (result x contracted) product space; the conv benchmarks pool non-overlapping and paid 1.8-2.6x
   for an exactness their geometry cannot exercise. [Operation.einmax1]/[Operation.tropical] now
   take [?nonoverlapping] restoring the input-space ([_rhs1]) gate on the domain where the two
   formulations agree exactly — each RHS1 position feeds at most one result position — and
   [Nn_blocks.max_pool2d]/[max_pool2d_copy] dispatch on [stride >= window_size] by shadowing the
   [tropical] the [@^+] operator expands to.

   Pinned here, backend-independent:

   - Dispatch: the lowered backward of a stride>=window pool contains the [_rhs1] gate proxies and
   no [_pspace] ones; a stride<window pool keeps the product-space gate. - Parity: on the
   non-overlapping domain the two gates produce BITWISE-equal pooled values and input gradients —
   ties included — across no-padding, gapped (stride > window), and clamped padded odd-extent
   configurations, and for [max_pool2d_copy]. This is the issue's "documented choice": on this
   domain there is no semantic difference to choose, only cost. - [einmax1]'s flag: a full reduction
   (trivially non-overlapping) with ties matches the product-space gradient bitwise. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Stdio
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
open Verdict.Claims

(* Non-empty by construction (gh-ocannl-746): [Array.for_all2_exn] answers [true] on two empty
   arrays, and a readback and the reference it is compared against go empty TOGETHER -- the
   reference went through the same path. {!Verdict.p_all2} is the claim-shaped form; the sites
   reached through this helper keep a boolean, so the guard lives here. *)
let bitwise a b = (not (Array.is_empty a)) && Array.for_all2_exn a b ~f:Float.equal
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(* Whether the update computation mentions a tensor whose label contains [sub] — a pre-lowering
   structural pin of which gate formulation was built (the cheap gate's proxies typically virtualize
   away entirely in the lowered code, which is the point of the specialization). *)
let mentions_label ~sub (update : Asgns.comp) : bool =
  let nodes, guessed = Asgns.collect_nodes_guess_output update.Asgns.asgns in
  let mem set =
    Set.exists set ~f:(fun tn ->
        List.exists tn.Tn.label ~f:(fun l -> String.is_substring l ~substring:sub))
  in
  mem nodes || mem guessed

(* [max_pool2d] with the gradient gate forced, mirroring its dispatch shadow — the product-space
   twin for the parity legs (and, forced true, a canary that the shadow mechanism itself works). *)
let%op pool_gate ?(nonoverlapping = false) ?(stride = 2) ?(window_size = 2) ?(use_padding = false)
    () x =
  let tropical ?label ?capture_dims spec t1 t2 =
    tropical ?label ?capture_dims ~nonoverlapping spec t1 t2
  in
  Shape.set_dim wh window_size;
  Shape.set_dim ww window_size;
  Shape.set_dim pwh window_size;
  Shape.set_dim pww window_size;
  if use_padding then
    x
    @^+ "... | stride*oh= + pwh, stride*ow= + pww, ..c..; |pwh, pww => ... | oh, ow, ..c.."
          [ "pwh"; "pww" ] (stretch 0.0)
  else
    x
    @^+ "... | stride*oh< + wh, stride*ow< + ww, ..c..; |wh, ww => ... | oh, ow, ..c.."
          [ "wh"; "ww" ] (stretch 0.0)

(* Deterministic data with in-window ties: coarse quantization repeats values. The operand is [h; w]
   or [h; w; c], so the weights are cut to its rank. *)
let input_f idcs =
  Ll_test.weighted
    ~weights:(Array.sub [| 7; 3; 5 |] ~pos:0 ~len:(Array.length idcs))
    ~modulus:4 ~offset:0. ~stride:1. idcs

(* Build pool(x) with [pool], run one update of the summed pool, return (y, gx). *)
let run_pool ~label ~dims pool =
  Tensor.unsafe_reinitialize ();
  let x =
    Operation.init ~l:("np_x_" ^ label) ~prec:Ir.Ops.single ~b:[] ~o:dims ~f:input_f
      ~grad_spec:Tensor.Require_grad ()
  in
  let y = pool x in
  let%op loss = y ++ "... | ... => 0" in
  let ctx = Context.auto () in
  Train.set_materialized y.Tensor.value;
  Train.set_materialized (grad_of x);
  let ctx = Train.update_once ~output_cd_file:false ctx loss in
  (Context.get_values ctx y.Tensor.value, Context.get_values ctx (grad_of x))

(* === Leg 1: dispatch — the lowered backward's gate proxies name the formulation. === *)

let () =
  let census ~stride ~window_size =
    Tensor.unsafe_reinitialize ();
    let x =
      Operation.init ~l:"np_census_x" ~prec:Ir.Ops.single ~b:[] ~o:[ 6; 6; 2 ] ~f:input_f
        ~grad_spec:Tensor.Require_grad ()
    in
    let pool = Nn_blocks.max_pool2d ~stride ~window_size () in
    let y = pool x in
    let%op loss = y ++ "... | ... => 0" in
    let update = Train.grad_update loss in
    (mentions_label ~sub:"rhs1" update, mentions_label ~sub:"pspace" update)
  in
  let rhs1, pspace = census ~stride:2 ~window_size:2 in
  p "non-overlapping pool dispatches to the input-space gate" (rhs1 && not pspace);
  let rhs1, pspace = census ~stride:1 ~window_size:2 in
  p "overlapping pool keeps the product-space gate" (pspace && not rhs1)

(* === Leg 2: parity — bitwise-equal values and gradients on the non-overlapping domain. === *)

let () =
  let parity name ~dims ~stride ~window_size ~use_padding =
    let y_ref, g_ref =
      run_pool ~label:(name ^ "_pspace") ~dims
        (pool_gate ~nonoverlapping:false ~stride ~window_size ~use_padding ())
    in
    let y_new, g_new =
      run_pool ~label:(name ^ "_dispatch") ~dims
        (Nn_blocks.max_pool2d ~stride ~window_size ~use_padding ())
    in
    p
      (Printf.sprintf "%s: input-space gate matches the product-space gate bitwise" name)
      (bitwise y_ref y_new && bitwise g_ref g_new)
  in
  parity "pool 2/2 no-padding with ties" ~dims:[ 6; 6; 2 ] ~stride:2 ~window_size:2
    ~use_padding:false;
  parity "pool 3/2 gapped no-padding" ~dims:[ 8; 8; 2 ] ~stride:3 ~window_size:2 ~use_padding:false;
  parity "pool 2/2 padded" ~dims:[ 8; 8; 2 ] ~stride:2 ~window_size:2 ~use_padding:true;
  (* Padded window 3 at stride 3: the centered anchoring clamps the edge windows (gh-504), so the
     gate parity covers clamped out-of-range window positions. *)
  parity "pool 3/3 padded (clamped edge windows)" ~dims:[ 9; 9; 2 ] ~stride:3 ~window_size:3
    ~use_padding:true;
  (* The copy variant carries the same dispatch. *)
  let y_ref, g_ref =
    run_pool ~label:"copy_pspace" ~dims:[ 8; 8; 2 ]
      (pool_gate ~nonoverlapping:false ~stride:2 ~window_size:2 ~use_padding:true ())
  in
  let y_new, g_new =
    run_pool ~label:"copy_dispatch" ~dims:[ 8; 8; 2 ]
      (Nn_blocks.max_pool2d_copy ~stride:2 ~window_size:2 ~use_padding:true ())
  in
  p "max_pool2d_copy padded: gates match bitwise" (bitwise y_ref y_new && bitwise g_ref g_new)

(* === Leg 3: einmax1's flag — a full reduction with ties, both gates bitwise-equal. === *)

let%op emax_gate ?(nonoverlapping = false) () x =
  let einmax1 ?label ?capture_dims spec t1 = einmax1 ?label ?capture_dims ~nonoverlapping spec t1 in
  x @^^ "i => 0"

let () =
  let run ~label pool =
    Tensor.unsafe_reinitialize ();
    let x =
      Operation.init ~l:("np_emax_" ^ label) ~prec:Ir.Ops.single ~b:[] ~o:[ 5 ]
        ~f:(fun idcs -> if idcs.(0) = 2 then 3. else 7.)
        ~grad_spec:Tensor.Require_grad ()
    in
    let loss = pool x in
    let ctx = Context.auto () in
    Train.set_materialized loss.Tensor.value;
    Train.set_materialized (grad_of x);
    let ctx = Train.update_once ~output_cd_file:false ctx loss in
    (Context.get_values ctx loss.Tensor.value, Context.get_values ctx (grad_of x))
  in
  let l_ref, g_ref = run ~label:"pspace" (emax_gate ~nonoverlapping:false ()) in
  let l_new, g_new = run ~label:"nonoverlap" (emax_gate ~nonoverlapping:true ()) in
  p "einmax1 full reduction with ties: gates match bitwise"
    (bitwise l_ref l_new && bitwise g_ref g_new);
  p "tie gradient gates every achieving position" (bitwise g_new [| 1.; 1.; 0.; 1.; 1. |]);
  printf "\nDone.\n%!"
