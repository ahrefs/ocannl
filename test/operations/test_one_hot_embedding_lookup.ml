(* gh-ocannl-343 / task-73617488: end-to-end test for the one-hot embedding optimization.

   The embedding lookup [emb[b,o] = C[o, ids[b]]] is written logically as a one-hot reduction
   [emb = C * (range vocab == ids)]. [rewrite_one_hot_reductions] collapses the vocabulary loop into
   a guarded [Get_dynamic]. task-73617488 makes this fire under the default [virtualize_max_visits =
   1] via a narrow virtualizer exemption for one-hot selector producers.

   Pinned invariants:
   - Forward equivalence: the logical one-hot embedding equals a direct gather of table rows.
   - Optimization observability: the optimized low-level IR contains [Get_dynamic] and no reduction
     loop over the vocabulary axis.
   - Out-of-bounds: an index outside [0, vocab) yields a zero embedding row (one-hot semantics).
   - Read tracking: the index tensor is a routine input of the optimized gather.
   - Fallback: an ordinary (non one-hot) contraction is left unchanged (no [Get_dynamic]).
   - Negative (task-73617488): non-one-hot complex pointwise tensors still respect [max_visits]. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level

(* task-73617488: the gather rewrite now fires under the default max_visits = 1; no override. *)
let () = assert (LL.virtualize_settings.max_visits = 1)

(* Keep the generated C-family source in [build_files/<name>_fwd.c] so we can inspect it. *)
let () = Utils.settings.output_debug_files_in_build_directory <- true

let vocab = 4
let embed = 3

(* Read the generated C source for a forward kernel, if the C-family backend wrote one. *)
let read_generated_c base_name =
  let path = Stdlib.Filename.concat "build_files" (base_name ^ ".c") in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

(* Embedding table C with C[o,i] = o*vocab + i, so row i (over the input axis) is distinctive. *)
let cvals = Array.init (embed * vocab) ~f:Float.of_int

let approx a b = Float.(abs (a - b) < 1e-4)

(* Lower the (shape-forced) forward comp and report (#Get_dynamic, #For_loop, index-tensor-is-input). *)
let inspect (t : Tensor.t) (index_tn : Ir.Tnode.t) : int * int * bool =
  let comp = t.Tensor.forward in
  let optim_ctx = { LL.computations = Hashtbl.create (module Ir.Tnode) } in
  let opt =
    Ir.Assignments.lower optim_ctx ~unoptim_ll_source:None ~ll_source:None ~cd_source:None
      ~name:"probe" [] comp.Ir.Assignments.asgns
  in
  let dyn = ref 0 and loops = ref 0 and index_read = ref false in
  (* Does scalar [s] read [index_tn] via a plain [Get]? Exercises that traversals descend into a
     [Get_dynamic]'s [dyn_value] sub-expression (gh-343 read-tracking). *)
  let rec reads_index (s : LL.scalar_t) =
    match s with
    | Get (g, _) -> Ir.Tnode.equal g index_tn
    | Get_dynamic { dyn_value = v, _; _ } -> reads_index v
    | Local_scope { body; _ } -> proc_reads body
    | Ternop (_, (a, _), (b, _), (c, _)) -> reads_index a || reads_index b || reads_index c
    | Binop (_, (a, _), (b, _)) -> reads_index a || reads_index b
    | Unop (_, (a, _)) -> reads_index a
    | _ -> false
  and proc_reads (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) -> proc_reads a || proc_reads b
    | For_loop { body; _ } -> proc_reads body
    | Set { llsc; _ } -> reads_index llsc
    | Set_from_vec { arg = s, _; _ } -> reads_index s
    | Set_local (_, s) -> reads_index s
    | _ -> false
  in
  let rec proc (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) -> proc a; proc b
    | For_loop { body; _ } -> Int.incr loops; proc body
    | Set { llsc; _ } -> scal llsc
    | Set_from_vec { arg = s, _; _ } -> scal s
    | Set_local (_, s) -> scal s
    | _ -> ()
  and scal (s : LL.scalar_t) =
    match s with
    | Get_dynamic { dyn_value = v, _; _ } ->
        Int.incr dyn;
        if reads_index v then index_read := true;
        scal v
    | Local_scope { body; _ } -> proc body
    | Ternop (_, (a, _), (b, _), (c, _)) -> scal a; scal b; scal c
    | Binop (_, (a, _), (b, _)) -> scal a; scal b
    | Unop (_, (a, _)) -> scal a
    | _ -> ()
  in
  proc opt.LL.llc;
  (!dyn, !loops, !index_read)

let build_embedding id_values =
  let ids =
    TDSL.ndarray id_values ~label:[ "ids" ]
      ~batch_dims:[ Array.length id_values ]
      ~output_dims:[] ()
  in
  let c = TDSL.ndarray cvals ~label:[ "C" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] () in
  let classes = TDSL.range vocab in
  let%op one_hot = classes = ids in
  let%op embedded = c * one_hot in
  (ids, embedded)

let p name b = Stdio.printf "%s: %b\n" name b

let () =
  (* --- Forward equivalence + observability (in-range indices) ---
     Run first so the forward comp is fully assembled and memory modes are fixed, then re-lower the
     same graph to inspect the optimized IR. *)
  let id_values = [| 1.; 3.; 0. |] in
  let ids, embedded = build_embedding id_values in
  let ctx = Context.cpu () in
  let ctx = Train.forward_once ctx embedded in
  let got = Context.get_values ctx embedded.Tensor.value in
  let dyn, loops, index_is_input = inspect embedded ids.Tensor.value in
  (* expected: emb[b,o] = C[o, ids[b]] = o*vocab + ids[b] *)
  let expected =
    Array.concat_map id_values ~f:(fun idf ->
        Array.init embed ~f:(fun o -> Float.of_int ((o * vocab) + Int.of_float idf)))
  in
  p "forward equals direct gather of table rows" (Array.for_all2_exn got expected ~f:approx);
  p "optimized IR contains a Get_dynamic gather" (dyn >= 1);
  (* The logical embedding nests batch x output x vocab loops; after the rewrite the vocabulary
     reduction loop is gone, leaving only the 2 batch/output loops (the small literals are inlined as
     constant fills, contributing no loops). *)
  p "vocabulary reduction loop is eliminated" (loops <= 2);
  p "gather's dynamic index reads the token-id tensor" index_is_input;
  (* Proposal AC: the generated C for the optimized kernel contains a guarded dynamic table read and
     no reduction loop over the vocabulary axis. The dynamic index renders as an [((int)(...))] cast
     inside a ternary guard; a vocabulary loop would iterate up to [vocab - 1] ([<= 3] here), which
     is distinct from the batch/output loop bounds ([<= 2]). *)
  (match read_generated_c "embedded_fwd" with
  | None -> p "generated C: guarded dynamic table read present (skipped: non-C backend)" true
  | Some c ->
      p "generated C contains a guarded dynamic table read"
        (String.is_substring c ~substring:"((int)(" && String.is_substring c ~substring:"?");
      p "generated C has no vocabulary reduction loop"
        (not (String.is_substring c ~substring:"<= 3")));

  (* --- Out-of-bounds index yields a zero embedding row --- *)
  let oob = [| 1.; Float.of_int vocab (* == vocab, out of [0,vocab) *) |] in
  let _ids2, embedded2 = build_embedding oob in
  let ctx2 = Context.cpu () in
  let ctx2 = Train.forward_once ctx2 embedded2 in
  let got2 = Context.get_values ctx2 embedded2.Tensor.value in
  let row1 = Array.init embed ~f:(fun o -> Float.of_int ((o * vocab) + 1)) in
  let in_range_ok = Array.for_alli got2 ~f:(fun i v -> i >= embed || approx v row1.(i)) in
  let oob_zero = Array.for_alli got2 ~f:(fun i v -> i < embed || approx v 0.) in
  p "in-range row of OOB batch is correct" in_range_ok;
  p "out-of-range index gives a zero embedding row" oob_zero;

  (* --- Fractional index yields a zero row (one-hot semantics: every [k == 1.5] is false). The
     guard's integrality check ([idx == trunc(idx)]) must reject it rather than gather row 1. --- *)
  let frac = [| 2.; 1.5 |] in
  let _ids3, embedded3 = build_embedding frac in
  let ctx_f = Context.cpu () in
  let ctx_f = Train.forward_once ctx_f embedded3 in
  let got3 = Context.get_values ctx_f embedded3.Tensor.value in
  let row2 = Array.init embed ~f:(fun o -> Float.of_int ((o * vocab) + 2)) in
  let frac_in_range_ok = Array.for_alli got3 ~f:(fun i v -> i >= embed || approx v row2.(i)) in
  let frac_zero = Array.for_alli got3 ~f:(fun i v -> i < embed || approx v 0.) in
  p "integer row of fractional batch is correct" frac_in_range_ok;
  p "fractional index gives a zero embedding row" frac_zero;

  (* --- Fallback: an ordinary matmul (not a one-hot reduction) is not rewritten --- *)
  let a = TDSL.ndarray (Array.create ~len:(2 * vocab) 1.) ~label:[ "A" ] ~input_dims:[ vocab ]
      ~output_dims:[ 2 ] () in
  let x = TDSL.ndarray (Array.create ~len:vocab 2.) ~label:[ "x" ] ~output_dims:[ vocab ] () in
  let%op plain = a * x in
  let ctx3 = Context.cpu () in
  let ctx3 = Train.forward_once ctx3 plain in
  ignore (Context.get_values ctx3 plain.Tensor.value : float array);
  let dyn_plain, _, _ = inspect plain x.Tensor.value in
  p "ordinary matmul is not rewritten to Get_dynamic" (dyn_plain = 0);

  (* --- Helper migration: Nn_blocks.one_hot_of_int_list is now logical (no dense host data) but
     still produces the correct one-hot values, and class_ids_of_int_list keeps compact IDs. --- *)
  let id_list = [ 1; 3 ] in
  let oh = Ocannl.Nn_blocks.one_hot_of_int_list ~num_classes:vocab id_list in
  let ctx4 = Context.cpu () in
  let ctx4 = Train.forward_once ctx4 oh in
  let oh_vals = Context.get_values ctx4 oh.Tensor.value in
  let oh_expected =
    Array.concat_map (Array.of_list id_list) ~f:(fun idx ->
        Array.init vocab ~f:(fun k -> if k = idx then 1. else 0.))
  in
  p "logical one_hot_of_int_list matches a dense one-hot"
    (Array.for_all2_exn oh_vals oh_expected ~f:approx);
  let cids = Ocannl.Nn_blocks.class_ids_of_int_list id_list in
  let ctx5 = Context.cpu () in
  let ctx5 = Train.forward_once ctx5 cids in
  let cid_vals = Context.get_values ctx5 cids.Tensor.value in
  p "class_ids_of_int_list stores compact ids"
    (Array.length cid_vals = List.length id_list
    && Array.for_all2_exn cid_vals (Array.of_list id_list) ~f:(fun v i -> approx v (Float.of_int i)));

  (* --- Negative (task-73617488): non-one-hot complex pointwise tensors still obey max_visits ---
     [not_hot[b,k] = range[k] + ids_neg[b]] is complex (reads an accessing tensor) but NOT a
     one-hot [Cmpeq], so the virtualizer exemption does NOT apply: the tensor stays materialized
     ([Never_virtual]) and the vocabulary reduction loop survives in the consumer.  *)
  let ids_neg =
    TDSL.ndarray [| 1.; 0. |] ~label:[ "ids_neg" ] ~batch_dims:[ 2 ] ~output_dims:[] ()
  in
  let c_neg =
    TDSL.ndarray cvals ~label:[ "C_neg" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  let classes_neg = TDSL.range vocab in
  let%op not_hot = classes_neg + ids_neg in
  let%op emb_neg = c_neg * not_hot in
  let ctx_neg = Context.cpu () in
  let ctx_neg = Train.forward_once ctx_neg emb_neg in
  ignore (Context.get_values ctx_neg emb_neg.Tensor.value : float array);
  let dyn_neg, loops_neg, _ = inspect emb_neg ids_neg.Tensor.value in
  p "non-one-hot complex tensor does not produce Get_dynamic" (dyn_neg = 0);
  (* Without the one-hot exemption the vocab reduction loop is NOT collapsed: batch + output loops
     are 2, so total > 2 means the vocabulary loop survived. *)
  p "non-one-hot complex tensor keeps the vocab reduction loop" (loops_neg > 2)
