(* gh-ocannl-1476: one tensor named by several operands of one operation.

   [(w, w) ++^ "s, c; s, d => s, c^d"] used to fail in BOTH passes with "Ambiguous indices in
   concatenation: multiple blocks viable for same position". The cause sat upstream of the
   concatenation lowering: projection ids live on the [Dim]s of a shape's rows and
   [Shape.derive_projections] freshened each operand's shape IN PLACE, so two operands that are one
   tensor -- one shape object -- ended up with the same ids. The solver then equated the axes the
   spec keeps apart, here the two concatenated segments: the projections carried [Concat [s; s]]
   over a component of ONE segment, which neither the forward block selection nor the gradient
   scatter can attribute. The same collapse made a self-product silently compute its diagonal: [m +*
   "a, b; b, c => a, c" m] equated the contracted axis with both free ones and returned
   [m.(i).(i)^2] on the diagonal and zeros elsewhere, with no error at all. [derive_projections] now
   gives every repeated operand a shallow copy of its shape ([Shape.dealias_operands]), so each
   occurrence has its own ids.

   Checked here, executed with every node read back materialized: the 2-way self-concatenation and a
   3-way one with a repeated operand ([(w, v, w)]), forward (the copies side by side) and backward
   (each block's upstream gradient summed into the one node); the projections the concatenation
   derives (one segment per operand, with pairwise-distinct symbols); and the self-product, as an
   explicit einsum and as [*], forward and backward. Every weight is distinct, so a gradient that
   took the wrong block, took one block twice or dropped one differs from the oracle; the oracles
   are computed here from the definitions. All values are small integers, exact in single
   precision. *)

open Base
open Ocannl
open Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module LL = Ir.Low_level
open Verdict.Claims

(* The first [Accum_op] of [asgns] whose right-hand side satisfies [f]. *)
let rec find_accum ~f = function
  | Asgns.Accum_op { rhs; _ } as a -> Option.some_if (f rhs) a
  | Asgns.Seq (a, b) -> Option.first_some (find_accum ~f a) (find_accum ~f b)
  | Asgns.Block_comment (_, body) -> find_accum ~f body
  | Asgns.Noop | Asgns.Set_vec_unop _ | Asgns.Fetch _ -> None

let ndarray ~grad dims values =
  Tensor.ndarray
    ~grad_spec:(if grad then Tensor.Require_grad else Tensor.Prohibit_grad)
    values ~batch_dims:[] ~input_dims:[] ~output_dims:dims ()

(* Distinct primes: no two weights are equal, and no two pairs share a sum. *)
let primes = [| 2.; 3.; 5.; 7.; 11.; 13.; 17.; 19.; 23.; 29.; 31.; 37.; 41.; 43.; 47.; 53. |]
let weights n = Array.sub primes ~pos:0 ~len:n
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(* Runs one training step of [sum (out *. weight)] and returns the context to read from. *)
let run_step ~out ~weight ~materialize =
  let%op loss = (out *. weight) ++ "...|... => |->0" in
  List.iter materialize ~f:Train.set_materialized;
  Train.update_once ~output_cd_file:false (Context.auto ()) loss

let rows = 2

(* [segments] are [(operand index, width)] in concatenation order; [operands] are the distinct
   tensors' widths. The forward is each row's segments side by side, and an operand's gradient is
   the sum, over the segments naming it, of that segment's slice of the weights. *)
let concat_case ~label ~segments ~operands ~make =
  Tensor.unsafe_reinitialize ();
  let values =
    Array.mapi operands ~f:(fun k width ->
        Array.init (rows * width) ~f:(fun i -> Float.of_int ((100 * (k + 1)) + i)))
  in
  let tensors =
    Array.mapi operands ~f:(fun k width -> ndarray ~grad:true [ rows; width ] values.(k))
  in
  let cat = make tensors in
  let total = List.sum (module Int) segments ~f:snd in
  let w = weights (rows * total) in
  let weight = ndarray ~grad:false [ rows; total ] w in
  let block_projections =
    match
      find_accum cat.Tensor.forward.Asgns.asgns ~f:(function Asgns.Block _ -> true | _ -> false)
    with
    | Some (Asgns.Accum_op { projections; _ }) -> projections
    | _ -> failwith "expected the concatenation's Block forward"
  in
  let scatter =
    Option.value_exn ~here:[%here]
      (find_accum (Option.value_exn ~here:[%here] cat.Tensor.diff).Tensor.backprop.asgns
         ~f:(function
        | Asgns.Rev_sides _ -> true
        | _ -> false))
  in
  let ctx =
    run_step ~out:cat ~weight
      ~materialize:(cat.Tensor.value :: Array.to_list (Array.map tensors ~f:grad_of))
  in
  let expected_cat =
    Array.concat
      (List.init rows ~f:(fun s ->
           Array.concat
             (List.map segments ~f:(fun (k, width) ->
                  Array.sub values.(k) ~pos:(s * width) ~len:width))))
  in
  pf_all2 "%s: the forward is the operands side by side, per row" label
    (Context.get_values ctx cat.value)
    expected_cat ~f:Float.equal;
  Array.iteri tensors ~f:(fun k t ->
      let width = operands.(k) in
      let expected =
        Array.init (rows * width) ~f:(fun i ->
            let s = i / width and c = i % width in
            snd
              (List.fold segments ~init:(0, 0.) ~f:(fun (offset, acc) (k', width') ->
                   (offset + width', if k' = k then acc +. w.((s * total) + offset + c) else acc))))
      in
      pf_all2 "%s: operand %d's gradient sums its segments' slices of the upstream gradient" label k
        (Context.get_values ctx (grad_of t))
        expected ~f:Float.equal);
  (* The lowering behind the gradient: the scatter writes each operand's gradient once per segment
     naming it, each time accumulating (reading the cell it writes) -- a node named by two segments
     is never attributed a plain store, which would keep only the later contribution. *)
  let scatter_llc = Asgns.to_low_level scatter in
  Array.iteri tensors ~f:(fun k t ->
      let g = grad_of t in
      let stores ~reads =
        Ll_test.count_stmt scatter_llc ~f:(function
          | LL.Set { tn; llsc; _ } ->
              Ir.Tnode.equal tn g && Bool.equal reads (LL.scalar_touches_tn g llsc)
          | _ -> false)
      in
      pf "%s: the scatter accumulates into operand %d's gradient once per segment naming it" label k
        (stores ~reads:true = List.count segments ~f:(fun (k', _) -> k' = k)
        && stores ~reads:false = 0));
  (* The projections behind it: one segment per operand in the concatenation's component, with
     pairwise-distinct symbols -- the collapse put both [w] segments on one symbol. *)
  let projections : Idx.projections = Lazy.force block_projections in
  let concat_syms =
    Array.find_map projections.project_lhs ~f:(function Idx.Concat syms -> Some syms | _ -> None)
    |> Option.value_exn ~here:[%here]
  in
  pf "%s: the concatenated axis names one symbol per segment" label
    (List.length concat_syms = List.length segments);
  p_pairwise_distinct
    (label ^ ": the concatenated segments' symbols")
    concat_syms ~equal:Idx.equal_symbol ~to_string:Idx.symbol_ident;
  let sorted syms = List.sort syms ~compare:Idx.compare_symbol in
  p_exists (label ^ ": one component iterates exactly the concatenated segments")
    (Array.to_list projections.components) ~f:(fun comp ->
      List.equal Idx.equal_symbol (sorted (List.map comp ~f:snd)) (sorted concat_syms))

(* [m] is [[1, 2], [3, 4]]: its square differs from its elementwise square on every cell, and its
   off-diagonal cells are nonzero, so the collapse onto the diagonal fails both claims. *)
let product_case ~label ~(axes : [ `Output_only | `Matrix ]) ~make =
  Tensor.unsafe_reinitialize ();
  let m_vals = [| 1.; 2.; 3.; 4. |] in
  let m, out = make m_vals in
  let w = weights 4 in
  (* The weight has [out]'s own axes, so [out *. weight] is pointwise rather than an outer
     broadcast. *)
  let weight =
    match axes with
    | `Output_only -> ndarray ~grad:false [ 2; 2 ] w
    | `Matrix ->
        Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad w ~batch_dims:[] ~input_dims:[ 2 ]
          ~output_dims:[ 2 ] ()
  in
  let ctx = run_step ~out ~weight ~materialize:[ out.Tensor.value; grad_of m ] in
  let mv i j = m_vals.((2 * i) + j) and wv i j = w.((2 * i) + j) in
  let sum2 f = f 0 +. f 1 in
  pf_all2 "%s: the forward is the matrix square" label
    (Context.get_values ctx out.value)
    (Array.init 4 ~f:(fun idx -> sum2 (fun b -> mv (idx / 2) b *. mv b (idx % 2))))
    ~f:Float.equal;
  (* d/dm[i, j] of sum_{a, c} w[a, c] * sum_b m[a, b] m[b, c]: m as the left factor gives sum_c w[i,
     c] m[j, c], as the right one sum_a m[a, i] w[a, j]. *)
  pf_all2 "%s: the gradient sums both factors' contributions" label
    (Context.get_values ctx (grad_of m))
    (Array.init 4 ~f:(fun idx ->
         let i = idx / 2 and j = idx % 2 in
         sum2 (fun c -> wv i c *. mv j c) +. sum2 (fun a -> mv a i *. wv a j)))
    ~f:Float.equal

let () =
  concat_case ~label:"(w, w) ++^ \"s, c; s, d => s, c^d\""
    ~segments:[ (0, 3); (0, 3) ]
    ~operands:[| 3 |]
    ~make:(fun ts ->
      let w = ts.(0) in
      let%op cat = (w, w) ++^ "s, c; s, d => s, c^d" in
      cat);
  concat_case ~label:"(w, v, w) ++^ \"s, a; s, b; s, c => s, a^b^c\""
    ~segments:[ (0, 3); (1, 2); (0, 3) ]
    ~operands:[| 3; 2 |]
    ~make:(fun ts ->
      let w = ts.(0) and v = ts.(1) in
      let%op cat = (w, v, w) ++^ "s, a; s, b; s, c => s, a^b^c" in
      cat);
  product_case ~label:"m +* \"a, b; b, c => a, c\" m" ~axes:`Output_only ~make:(fun vals ->
      let m = ndarray ~grad:true [ 2; 2 ] vals in
      let%op out = m +* "a, b; b, c => a, c" m in
      (m, out));
  product_case ~label:"m * m" ~axes:`Matrix ~make:(fun vals ->
      let m =
        Tensor.ndarray ~grad_spec:Tensor.Require_grad vals ~batch_dims:[] ~input_dims:[ 2 ]
          ~output_dims:[ 2 ] ()
      in
      let%op out = m * m in
      (m, out))
