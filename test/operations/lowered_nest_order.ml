(* gh-ocannl-1151: the loop order [Assignments.lower] gives a FORWARD contraction — output loops
   outside reduction loops — pinned, with the scope it actually has.

   The rebuilt peel input [Schedule.apply_privatize] used before gh-ocannl-1151 (the privatized
   loop's subtree with the loops the accumulated cell mentions dropped) was sound only under this
   invariant, which nothing tested; review round 7 of staging#880 rested its rebuttal on it. The
   reconstruction is gone — [Privatize] now judges the program it was handed with the same
   accumulator-width decision code generation makes ([Low_level.decide_accum_width]) — but other
   readers of a lowered nest still lean on the shape ("the reduction loops are the innermost run"),
   so the order is pinned where it holds and where it does not.

   Roles come from the accesses, not from names: in each top-level nest ([Ll_test.nest_paths]) the
   accumulating write at the bottom of the chain decides, per loop, OUTPUT (the written cell
   mentions its symbol) or REDUCTION (it does not). The claim is that no reduction loop precedes an
   output loop. Five forward shapes — a matmul, a transposed output, a contraction axis in the
   middle of its operand, a two-axis reduction and a batched matmul — on the raw lowering
   ([opt.source]) and on the optimized routine ([opt.llc]).

   Where it does NOT hold, also pinned: backprop lowers a weight gradient over the FORWARD product
   space, so [dw[o,i] += dy[b,o] * x[b,i]] iterates its batch reduction OUTERMOST — the shape
   [Autotune.detect_matmul_canonical]'s interchange exists for (gh-ocannl-1183). A reader of lowered
   code may rely on the order for forward contractions only.

   The negative control interchanges the matmul's reduction loop above an output loop with a
   [Schedule.Swap] and asks the same predicate, which must refuse it: a predicate that cannot fail
   pins nothing. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
open Verdict.Claims

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

type role = Output | Reduction

(* Each accumulating nest of [llc] — a top-level statement whose loop chain ([Ll_test.nest_paths])
   ends in exactly one accumulating write — with its loops' roles, outermost first, and the node it
   writes. *)
let accumulating_nests_with_node (llc : LL.t) : (Ir.Tnode.t * (Idx.symbol * role) list) list =
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      let writes = ref [] in
      Ll_test.walk stmt ~on_stmt:(fun st ->
          match st with
          | LL.Set { tn; idcs; _ } when LL.has_accumulating_cell st ->
              writes := (tn, idcs) :: !writes
          | _ -> ());
      match (Ll_test.nest_paths stmt, !writes) with
      | [ (_ :: _ as syms) ], [ (tn, idcs) ] ->
          let mentioned s = Array.exists idcs ~f:(Idx.axis_index_mentions_symbol s) in
          Some (tn, List.map syms ~f:(fun s -> (s, if mentioned s then Output else Reduction)))
      | _ -> None)

let accumulating_nests llc = List.map (accumulating_nests_with_node llc) ~f:snd

(* The invariant: once a reduction loop opens, no output loop follows. *)
let outputs_first (nest : (Idx.symbol * role) list) =
  let rec go seen_reduction = function
    | [] -> true
    | (_, Output) :: _ when seen_reduction -> false
    | (_, Output) :: rest -> go seen_reduction rest
    | (_, Reduction) :: rest -> go true rest
  in
  go false nest

let has_both nest =
  List.exists nest ~f:(fun (_, r) -> Poly.equal r Output)
  && List.exists nest ~f:(fun (_, r) -> Poly.equal r Reduction)

let describe nest =
  String.concat ~sep:" "
    (List.map nest ~f:(fun (s, r) ->
         Idx.symbol_ident s ^ match r with Output -> ":out" | Reduction -> ":red"))

(* Lower and optimize [comp] through the ordinary compile path, capturing the optimized record. *)
let lowered ~name (comp : Asgns.comp) : LL.optimized =
  let captured = ref None in
  let (_ : Context.t * Context.routine) =
    Context.compile
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      (Context.auto ()) (named name comp) Idx.Empty
  in
  Option.value_exn !captured

let nests_of (opt : LL.optimized) = (accumulating_nests opt.source, accumulating_nests opt.llc)

let report name (source, llc) =
  Stdio.eprintf "lowered_nest_order: %s: source [%s] llc [%s] (not part of the golden)\n%!" name
    (String.concat ~sep:"; " (List.map source ~f:describe))
    (String.concat ~sep:"; " (List.map llc ~f:describe))

let init ~l ?(b = []) ?(i = []) o =
  NTDSL.init ~l ~prec:Ir.Ops.single ~b ~i ~o ~f:(fun idcs -> Float.of_int (1 + idcs.(0))) ()

let () =
  let forward ~name out = lowered ~name (Train.forward out) in
  let cases =
    [
      ( "matmul",
        fun () ->
          let a = init ~l:"lno_mm_a" ~i:[ 5 ] [ 4 ] and b = init ~l:"lno_mm_b" ~i:[ 6 ] [ 5 ] in
          let%op c = a * b in
          forward ~name:"lno_matmul" c );
      ( "transposed output",
        fun () ->
          let a = init ~l:"lno_tr_a" [ 4; 5 ] and b = init ~l:"lno_tr_b" [ 5; 6 ] in
          let%op c = a +* "ik;kj=>ji" b in
          forward ~name:"lno_transposed" c );
      ( "middle contraction",
        fun () ->
          let x = init ~l:"lno_mid_x" [ 4; 5; 6 ] and v = init ~l:"lno_mid_v" [ 5 ] in
          let%op c = x +* "ikj;k=>ij" v in
          forward ~name:"lno_middle" c );
      ( "two-axis reduction",
        fun () ->
          let x = init ~l:"lno_two_x" [ 4; 5; 6 ] in
          let%op c = x ++ "irs=>i" in
          forward ~name:"lno_two_axis" c );
      ( "batched matmul",
        fun () ->
          let a = init ~l:"lno_bmm_a" [ 3; 4; 5 ] and b = init ~l:"lno_bmm_b" [ 3; 5; 6 ] in
          let%op c = a +* "bik;bkj=>bij" b in
          forward ~name:"lno_batched" c );
    ]
  in
  let results =
    List.map cases ~f:(fun (name, build) ->
        let nests = nests_of (build ()) in
        report name nests;
        (name, nests))
  in
  (* Non-empty and discriminating: every case contributes a contraction nest with loops of both
     roles on both programs, or the [p_all] below would pass over nests that cannot violate it. *)
  p_all
    "every forward case lowers a contraction nest with output and reduction loops, raw and \
     optimized"
    results ~f:(fun (_, (source, llc)) ->
      List.exists source ~f:has_both && List.exists llc ~f:has_both);
  p_all "forward lowering opens every output loop before any reduction loop, raw and optimized"
    results ~f:(fun (_, (source, llc)) ->
      List.for_all source ~f:outputs_first && List.for_all llc ~f:outputs_first);
  (* The negative control: the matmul with its reduction loop swapped above the inner output loop.
     The predicate must refuse it. *)
  let interchanged =
    let a = init ~l:"lno_neg_a" ~i:[ 5 ] [ 4 ] and b = init ~l:"lno_neg_b" ~i:[ 6 ] [ 5 ] in
    let%op c = a * b in
    let opt = forward ~name:"lno_negative" c in
    let swapped =
      match List.find (accumulating_nests opt.llc) ~f:has_both with
      | Some [ _; (j, Output); (k, Reduction) ] ->
          Some (Sched.apply [ Sched.Swap { outer = j; inner = k } ] opt)
      | _ -> None
    in
    Option.map swapped ~f:(fun o -> accumulating_nests o.LL.llc)
  in
  Option.iter interchanged ~f:(fun nests -> report "interchanged matmul" ([], nests));
  p "the predicate refuses the matmul with its reduction loop interchanged above an output loop"
    (match interchanged with
    | Some nests -> List.exists nests ~f:has_both && not (List.for_all nests ~f:outputs_first)
    | None -> false);
  (* Where the order does not hold: a weight gradient iterates the forward product space. *)
  let wgrad =
    let x = init ~l:"lno_bw_x" ~b:[ 3 ] [ 5 ] in
    let w =
      Operation.init ~l:"lno_bw_w" ~prec:Ir.Ops.single ~i:[ 5 ] ~o:[ 4 ]
        ~f:(fun idcs -> Float.of_int (1 + idcs.(0)))
        ~grad_spec:Tensor.Require_grad ()
    in
    let%op y = w * x in
    let%op loss = y ++ "... => 0" in
    let opt = lowered ~name:"lno_wgrad" (Train.grad_update loss) in
    let grad = Option.value_exn (Option.map w.Tensor.diff ~f:(fun d -> d.Tensor.grad)) in
    let nests_of llc =
      List.filter_map (accumulating_nests_with_node llc) ~f:(fun (tn, nest) ->
          Option.some_if (Ir.Tnode.equal tn grad && has_both nest) nest)
    in
    (nests_of opt.source, nests_of opt.llc)
  in
  report "weight gradient" wgrad;
  p
    "backprop lowers a weight gradient's batch reduction loop OUTSIDE its output loops (the order \
     is a forward-lowering property only)"
    (let source, _ = wgrad in
     (not (List.is_empty source)) && List.for_all source ~f:(fun nest -> not (outputs_first nest)))
