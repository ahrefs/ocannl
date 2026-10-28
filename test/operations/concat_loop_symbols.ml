(* gh-ocannl-765: a concatenation component's segments are lowered as SIBLING [For_loop]s with
   different bounds. Lowering used to mint one loop symbol per product component and reuse it for
   every segment, so the emitted nest bound the same symbol twice at two different widths.
   [Low_level]'s per-branch [loop_ranges] threading reads such a nest correctly, but every FLAT
   scanner keyed by symbol misreads it: [def_loop_ranges] (the inlining path) keeps only the last
   segment's width, [affine_accesses] collects two ranges for one symbol, and the canonical renderer
   -- the walk both digests share -- reports the second binder as shadowed, which makes the analysis
   cache decline the routine and the schedule cache call its rendering incomplete.

   The fix is a fresh symbol per SEGMENT loop, which costs nothing at lowering time and removes the
   hazard class. This test pins the emitted shape (distinct binders over the concat's segment
   widths), the canonical renderer's verdict on it, and -- since a symbol change is exactly the kind
   of rewrite that can lower to something structurally pretty and numerically wrong -- the executed
   values of both the forward concatenation and the [Rev_sides] gradient scatter. *)

open Base
open Ocannl
open Operation.DSL_modules
open Stdio
module Asgns = Ir.Assignments
module LL = Ir.Low_level
module Idx = Ir.Indexing
open Verdict.Claims

(** Every [For_loop] binder of [llc], as [(symbol, width)] in preorder. *)
let binders (llc : LL.t) : (Idx.symbol * int) list =
  let acc = ref [] in
  Ll_test.walk llc ~on_stmt:(function
    | LL.For_loop { index; from_; to_; _ } -> acc := (index, to_ - from_ + 1) :: !acc
    | _ -> ());
  List.rev !acc

(** The canonical renderer's binder events: [(symbol, shadowed)] per [For_loop] it walks. This is
    the walk shared by the analysis cache's key and the schedule cache's identity, so a [true] here
    is what makes both decline the routine. *)
let render_binder_events (llc : LL.t) : (Idx.symbol * bool) list =
  let events = ref [] in
  let buf = Buffer.create 4096 in
  LL.Canonical_render.emit ~buf
    {
      LL.Canonical_render.emit_tn = (fun tn -> Buffer.add_string buf (Ir.Tnode.label_name tn));
      emit_free_sym = (fun s -> Buffer.add_string buf (Idx.symbol_ident s));
      on_bind_loop = (fun s ~id:_ ~shadowed -> events := (s, shadowed) :: !events);
      mark_incomplete = (fun () -> ());
      mma = LL.Canonical_render.Structural_mma;
      initial_tokens = [];
    }
    llc;
  List.rev !events

let occurrences syms s = List.count syms ~f:(fun s' -> Idx.equal_symbol s' s)

(** The symbols each [Set] target is indexed by, as [(tnode id, symbol)] pairs. This is what
    [Low_level.track_symbol] builds [reverse_node_map] from, so it says whether one loop symbol can
    still own several tensor nodes -- the shape the virtualizer's shared-loop candidate list exists
    for. *)
let write_index_symbols (llc : LL.t) : (int * Idx.symbol) list =
  let acc = ref [] in
  let of_idx : Idx.axis_index -> Idx.symbol list = function
    | Idx.Iterator s -> [ s ]
    | Idx.Affine { symbols; _ } -> List.map symbols ~f:snd
    | Idx.Concat syms -> syms
    | Idx.Fixed_idx _ | Idx.Sub_axis -> []
  in
  Ll_test.walk llc ~on_stmt:(function
    | LL.Set { tn; idcs; _ } ->
        Array.iter idcs ~f:(fun idx ->
            List.iter (of_idx idx) ~f:(fun s -> acc := (tn.Ir.Tnode.id, s) :: !acc))
    | _ -> ());
  !acc

let () =
  Tensor.unsafe_reinitialize ();

  (* [x1] (dim 3) and [x2] (dim 2) concatenated into a dim-5 axis: one product component with two
     segments of DIFFERENT widths, which is what makes a shared binder observable at all. *)
  let x1 =
    Tensor.ndarray ~grad_spec:Tensor.Require_grad [| 1.0; 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[]
      ~output_dims:[ 3 ] ()
  in
  let x2 =
    Tensor.ndarray ~grad_spec:Tensor.Require_grad [| 10.0; 20.0 |] ~batch_dims:[] ~input_dims:[]
      ~output_dims:[ 2 ] ()
  in
  let%op cat = (x1, x2) ++^ "a; b => a^b" in
  let fwd = Train.forward cat in
  let fwd_llc = Asgns.to_low_level fwd.Asgns.asgns in
  let fwd_binders = binders fwd_llc in
  let fwd_syms = List.map fwd_binders ~f:fst in
  let fwd_widths = List.map fwd_binders ~f:snd in

  (* Non-vacuity: the lowering really did emit the two segment loops this test is about. *)
  p "concat forward emits a loop of the first segment's width (3)"
    (List.mem fwd_widths 3 ~equal:Int.equal);
  p "concat forward emits a loop of the second segment's width (2)"
    (List.mem fwd_widths 2 ~equal:Int.equal);
  Verdict.p_all ~min:2 "every concat forward loop binder is bound exactly once" fwd_syms
    ~f:(fun s -> occurrences fwd_syms s = 1);
  let fwd_events = render_binder_events fwd_llc in
  Verdict.p_none ~min:2 "no concat forward binder is shadowed in the canonical render" fwd_events
    ~f:snd;

  (* The gradient of a concatenation is a [Rev_sides] scatter: the same product-loop walker, the
     other role assignment, and its own copy of the segment loops. *)
  let%op loss = cat ++ "...|... => |->0" in
  let grad = Train.grad_update loss in
  let grad_llc = Asgns.to_low_level grad.Asgns.asgns in
  let grad_binders = binders grad_llc in
  let grad_syms = List.map grad_binders ~f:fst in
  p "concat backward emits a loop of the first segment's width (3)"
    (List.mem (List.map grad_binders ~f:snd) 3 ~equal:Int.equal);
  p "concat backward emits a loop of the second segment's width (2)"
    (List.mem (List.map grad_binders ~f:snd) 2 ~equal:Int.equal);
  Verdict.p_all ~min:2 "every concat backward loop binder is bound exactly once" grad_syms
    ~f:(fun s -> occurrences grad_syms s = 1);
  let grad_events = render_binder_events grad_llc in
  Verdict.p_none ~min:2 "no concat backward binder is shadowed in the canonical render" grad_events
    ~f:snd;

  (* The segment loops no longer share a binder, but a symbol can still index writes to SEVERAL
     tensor nodes -- an ENCLOSING product level's iterator appears in every segment's store. That is
     the shape `Low_level.reverse_node_map` is list-valued for and the virtualizer's shared-loop
     candidate list handles, so it is worth knowing it is still reachable from the DSL. A batched
     concatenation's gradient is the witness: one batch axis outside the concat component. *)
  let z1 =
    Tensor.ndarray ~grad_spec:Tensor.Require_grad
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
      ~batch_dims:[ 2 ] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let z2 =
    Tensor.ndarray ~grad_spec:Tensor.Require_grad [| 7.0; 8.0; 9.0; 10.0 |] ~batch_dims:[ 2 ]
      ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op zcat = (z1, z2) ++^ "...|a; ...|b => ...|a^b" in
  let%op zloss = zcat ++ "...|... => |->0" in
  let zgrad = Train.grad_update zloss in
  let zwrites = write_index_symbols (Asgns.to_low_level zgrad.Asgns.asgns) in
  let nodes_indexed_by s =
    List.filter_map zwrites ~f:(fun (id, s') -> if Idx.equal_symbol s s' then Some id else None)
    |> List.dedup_and_sort ~compare:Int.compare
  in
  let write_syms = List.map zwrites ~f:snd |> List.dedup_and_sort ~compare:Idx.compare_symbol in
  Verdict.p_exists "a batched concat gradient still indexes writes to several nodes by one symbol"
    write_syms ~f:(fun s -> List.length (nodes_indexed_by s) > 1);

  (* Executed parity: the values, not just the shape of the nest. The oracle has to DISCRIMINATE the
     loop indices, which a plain sum-of-the-concatenation loss does not: its derivative is 1.0
     everywhere, so a scatter that wrote the right cells for the wrong reason -- a swapped segment,
     a dropped enclosing batch index, a duplicated store -- still lands all-ones gradients. So the
     loss is weighted by a tensor of pairwise-distinct constants over BOTH the concatenated axis and
     the batch axis: every gradient cell then has its own value, and the expected arrays below fail
     under any permutation, duplication or omission of an index. Fresh tensors, because the
     inspections above already consumed the forward and backprop code of the ones they lowered. *)
  let y1 =
    Tensor.ndarray ~grad_spec:Tensor.Require_grad
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
      ~batch_dims:[ 2 ] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let y2 =
    Tensor.ndarray ~grad_spec:Tensor.Require_grad [| 7.0; 8.0; 9.0; 10.0 |] ~batch_dims:[ 2 ]
      ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  (* Distinct primes: no two subsets share a sum either, so even a gradient built by accumulating
     the wrong cells is distinguishable from the right one. *)
  let w =
    Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad
      [| 2.0; 3.0; 5.0; 7.0; 11.0; 13.0; 17.0; 19.0; 23.0; 29.0 |]
      ~batch_dims:[ 2 ] ~input_dims:[] ~output_dims:[ 5 ] ()
  in
  let%op cat2 = (y1, y2) ++^ "...|a; ...|b => ...|a^b" in
  let%op loss2 = (cat2 *. w) ++ "...|... => |->0" in
  let ctx = Context.auto () in
  Train.set_materialized cat2.value;
  Train.set_materialized (Option.value_exn ~here:[%here] y1.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] y2.diff).grad;
  let ctx = Train.update_once ~output_cd_file:false ctx loss2 in
  let cat_v = Context.get_values ctx cat2.value in
  let g1 = Context.get_values ctx (Option.value_exn ~here:[%here] y1.diff).grad in
  let g2 = Context.get_values ctx (Option.value_exn ~here:[%here] y2.diff).grad in
  (* Row-major over (batch, concatenated axis): each batch row is the first segment then the second,
     so a swapped segment or a lost batch index changes the array. *)
  p "batched concat forward interleaves the segments within each batch row"
    (Array.equal Float.equal cat_v [| 1.0; 2.0; 3.0; 7.0; 8.0; 4.0; 5.0; 6.0; 9.0; 10.0 |]);
  (* d(sum(cat *. w))/d(y1) is w restricted to the first segment's cells, batch row by batch row --
     and likewise for y2, from the second segment's. *)
  p "first segment's gradient is its own weights, per batch row"
    (Array.equal Float.equal g1 [| 2.0; 3.0; 5.0; 13.0; 17.0; 19.0 |]);
  p "second segment's gradient is its own weights, per batch row"
    (Array.equal Float.equal g2 [| 7.0; 11.0; 23.0; 29.0 |]);

  (* Descriptive, and deterministic: the widths come from the declared dimensions. *)
  printf "concat loop widths: fwd %s / bwd %s\n%!"
    (String.concat ~sep:"," (List.map fwd_widths ~f:Int.to_string))
    (String.concat ~sep:"," (List.map grad_binders ~f:(fun (_, w) -> Int.to_string w)))
