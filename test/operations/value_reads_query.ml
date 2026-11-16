(* gh-ocannl-2205: [Assignments.collect_value_reads] is the set of nodes whose values a program
   reads as operands -- a slice's parent included, whether the slice copies or aliases -- while
   [guess_materialized] keeps its private subtraction term; [collect_mentioned] names all nodes.
   Every lazy dimension and projection below fails when forced: the query is syntactic. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

let leaf name =
  NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[ 2 ] ~i:[] ~o:[ 4 ]
    ~f:(function [| b; j |] -> Float.of_int ((10 * b) + j + 1) | _ -> assert false)
    ()

let node name = (leaf name).Tensor.value
let unforced what = lazy (failwith (what ^ " forced"))
let set = Set.of_list (module Tn)

let accum lhs rhs =
  Asgns.Accum_op
    {
      initialize_neutral = false;
      accum = Ir.Ops.Add;
      lhs;
      rhs;
      projections = unforced "projections";
      projections_debug = "";
    }

let rec seq = function [] -> Asgns.Noop | [ c ] -> c | c :: cs -> Asgns.Seq (c, seq cs)

let () =
  let index, _bindings = Ir.Indexing.get_static_symbol ~static_range:2 Ir.Indexing.Empty in
  let const = node "vrq_const" and dst = node "vrq_dst" and merged = node "vrq_merged" in
  let rev_src = node "vrq_rev_src" and rev_dst = node "vrq_rev_dst" in
  let vec_src = node "vrq_vec_src" and vec_dst = node "vrq_vec_dst" in
  let parent = node "vrq_parent" and view = node "vrq_view" in
  let code =
    seq
      [
        Asgns.Fetch { array = const; fetch_op = Constant 1.; dims = unforced "dims" };
        accum dst (Binop { op = Add; rhs1 = Node const; rhs2 = Merge_buffer merged });
        (* Absent targets retain positions but contribute no read or materialization output. *)
        Block_comment
          ( "reversed",
            accum rev_src
              (Rev_sides { op = Identity; lhses = [| None; Some (Node rev_dst); None |] }) );
        Set_vec_unop
          {
            op = Uint4x32_to_prec_uniform;
            lhs = vec_dst;
            rhs = Node vec_src;
            projections = unforced "projections";
            projections_debug = "";
          };
        Fetch
          {
            array = view;
            fetch_op = Slice { batch_idx = index; sliced = parent };
            dims = unforced "dims";
          };
      ]
  in
  match Asgns.collect_value_reads code with
  | exception Failure msg -> fail ("collect_value_reads forced a lazy field: " ^ msg)
  | reads ->
      p "the operands, the reversed source and the slice parent are read; nothing else is"
        (Set.equal reads (set [ const; rev_src; vec_src; parent ]));
      p "a constant fetch's target, written then read, stays a read" (Set.mem reads const);
      p_none "no destination is a read" [ dst; rev_dst; vec_dst; view ] ~f:(Set.mem reads);
      p "a merge buffer is not a read of its node" (not (Set.mem reads merged));
      p "the mentions are all named nodes, merge buffer aside"
        (Set.equal (Asgns.collect_mentioned code)
           (set [ const; dst; rev_src; rev_dst; vec_src; vec_dst; parent; view ]))

(* The materialization guess is unchanged: a slice parent is an output to materialize, unless the
   sequence also reads it, which the guess's [Seq] subtracts. *)
let () =
  let index, _bindings = Ir.Indexing.get_static_symbol ~static_range:2 Ir.Indexing.Empty in
  let parent = node "vrq_m_parent" and view = node "vrq_m_view" and dst = node "vrq_m_dst" in
  let slice =
    Asgns.Fetch
      {
        array = view;
        fetch_op = Slice { batch_idx = index; sliced = parent };
        dims = unforced "dims";
      }
  in
  let outputs = Asgns.guess_materialized slice in
  p "an unread slice parent is materialized with its view"
    (Set.equal outputs (set [ parent; view ]));
  p "a slice mentions its parent and view"
    (Set.equal (Asgns.collect_mentioned slice) (set [ parent; view ]));
  let read_later = Asgns.Seq (slice, accum dst (Unop { op = Identity; rhs = Node parent })) in
  let outputs = Asgns.guess_materialized read_later in
  p "a slice parent read later in the sequence leaves the materialization set"
    (Set.equal outputs (set [ view; dst ]));
  p "the sequence mentions its slice, parent and destination"
    (Set.equal (Asgns.collect_mentioned read_later) (set [ parent; view; dst ]));
  p "while the value reads keep it"
    (Set.equal (Asgns.collect_value_reads read_later) (set [ parent ]))

(* Both lowerings of a slice -- an alias view and, under a precision mismatch, a copy -- read the
   parent's value. *)
let () =
  let index, _bindings = Ir.Indexing.get_static_symbol ~static_range:2 Ir.Indexing.Empty in
  let data = leaf "vrq_slice_data" in
  let alias = NTDSL.slice index data () and copy = NTDSL.slice index data () in
  Tn.update_prec copy.Tensor.value Ir.Ops.half;
  let eligible t = Asgns.slice_alias_eligible ~array:t.Tensor.value ~sliced:data.Tensor.value in
  p "one slice aliases its parent and the other copies it" (eligible alias && not (eligible copy));
  p_all "each slice's forward code reads the parent" [ alias; copy ] ~f:(fun t ->
      Set.mem (Asgns.collect_value_reads t.Tensor.forward.asgns) data.Tensor.value)
