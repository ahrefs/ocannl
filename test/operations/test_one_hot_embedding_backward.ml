(* gh-ocannl-466: deterministic scatter for the embedding-table gradient.

   The embedding backward [d_C[o,v] += Σ_pos (v == ids[pos]) * d_emb[pos,o]] lowers as a dense
   one-hot reduction — a loop over the vocabulary inside the position loops, O(V·positions) work.
   [rewrite_one_hot_reductions] now recognizes this transposed one-hot pattern and replaces the
   vocabulary loop with a guarded scatter-accumulate ([Set_dynamic]) at the dynamic row [ids[pos]] —
   O(positions) work, llm.c's deterministic no-atomics encoder backward (llmc-lessons.md B5).
   Positions accumulate in their original serial order (the scheduler treats the dynamic write as
   never-parallelizable), so results match the dense form.

   Pinned invariants: - Backward equivalence: the scatter-accumulated gradient equals the dense
   one-hot gradient computed analytically. - Optimization observability: the optimized IR of the
   gradient update contains [Set_dynamic] and no loop over the vocabulary axis. - Guard flavors
   mirror the forward gather (gh-343): float ids carry the integrality [Trunc] conjunct in the
   scatter guard; integer (uint32) ids carry none. - Out-of-range and fractional ids contribute
   nothing (one-hot semantics: no row of the table gradient is touched). - Fallback: an ordinary
   matmul backward is left unchanged (no [Set_dynamic]). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Asgns = Ir.Assignments
module IDX = Train.IDX

let () = Utils.settings.output_debug_files_in_build_directory <- true

(* Distinct extents so the vocabulary loop bound is recognizable: positions -> [<= 2], embed -> [<=
   3], vocab -> [<= 4]. [embed * vocab] must be divisible by 4 for the [uniform ()] table init
   (uint4x32 yields 4 values per draw). *)
let vocab = 5
let embed = 4
let positions = 3

open Verdict.Claims

let approx a b = Float.(abs (a - b) < 1e-4)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* Every kernel here is compiled on an explicit [Context.cpu ()] lineage, so the artifacts are the C
   backend's regardless of the run's configured backend: the read below pins [.c] rather than
   deriving the extension from the configured backend. *)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let grad_of t = (Option.value_exn t.Tensor.diff).Tensor.grad

let param_by_label l name =
  List.find_exn (Set.to_list l.Tensor.params) ~f:(fun t ->
      String.equal (Ir.Tnode.debug_name t.Tensor.value) name)

(* Lower the (already run, so shape/memory-mode-fixed) update comp and summarize the optimized IR:
   (#Set_dynamic, #For_loop over the vocabulary axis, #Trunc inside guards of scatters). *)
let inspect (update : Asgns.comp) : int * int * int =
  let optim_ctx = LL.empty_optimize_ctx () in
  let opt =
    Asgns.lower optim_ctx ~unoptim_ll_source:None ~ll_source:None ~cd_source:None
      ~name:"onehot_bwd_probe" [] update.Asgns.asgns
  in
  let dyn = ref 0 and vocab_loops = ref 0 and guard_truncs = ref 0 in
  let rec truncs_of (s : LL.scalar_t) =
    match s with
    | Unop (Ir.Ops.Trunc, (a, _)) -> 1 + truncs_of a
    | Unop (_, (a, _)) -> truncs_of a
    | Binop (_, (a, _), (b, _)) -> truncs_of a + truncs_of b
    | Ternop (_, (a, _), (b, _), (c, _)) -> truncs_of a + truncs_of b + truncs_of c
    | Get_dynamic { dyn_value = v, _; _ } -> truncs_of v
    | _ -> 0
  in
  let rec proc_has_scatter (llc : LL.t) =
    match llc with
    | LL.Set_dynamic _ -> true
    | LL.Seq (a, b) -> proc_has_scatter a || proc_has_scatter b
    | LL.For_loop { body; _ } | LL.If { body; _ } -> proc_has_scatter body
    | _ -> false
  in
  let rec proc (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        proc a;
        proc b
    | LL.For_loop { body; to_; from_ = 0; _ } ->
        if to_ = vocab - 1 then Int.incr vocab_loops;
        proc body
    | LL.For_loop { body; _ } -> proc body
    | LL.If { cond = c, _; body } ->
        if proc_has_scatter body then guard_truncs := !guard_truncs + truncs_of c;
        proc body
    | LL.Set_dynamic { dyn_value = v, _; llsc; _ } ->
        Int.incr dyn;
        scal v;
        scal llsc
    | LL.Set { llsc; _ } -> scal llsc
    | LL.Set_from_vec { arg = s, _; _ } -> scal s
    | LL.Set_local (_, s) -> scal s
    | _ -> ()
  and scal (s : LL.scalar_t) =
    match s with
    | Get_dynamic { dyn_value = v, _; _ } -> scal v
    | Local_scope { body; _ } -> proc body
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scal a;
        scal b;
        scal c
    | Binop (_, (a, _), (b, _)) ->
        scal a;
        scal b
    | Unop (_, (a, _)) -> scal a
    | _ -> ()
  in
  proc opt.LL.llc;
  (!dyn, !vocab_loops, !guard_truncs)

(* Distinct per-(position, channel) upstream gradients so the scatter parity is not vacuous:
   d_emb[pos, o] = coeff[pos, o]. *)
let coeff_values = Array.init (positions * embed) ~f:(fun i -> 0.1 *. Float.of_int (i + 1))

(* Analytic dense one-hot gradient: d_C[o, v] = Σ_{pos : ids[pos] = v} coeff[pos, o]. Ids that are
   fractional or out of [0, vocab) contribute nothing. *)
let expected_grads id_values =
  let g = Array.create ~len:(embed * vocab) 0. in
  Array.iteri id_values ~f:(fun pos idf ->
      let v = Int.of_float idf in
      if Float.(idf = Float.of_int v) && v >= 0 && v < vocab then
        for o = 0 to embed - 1 do
          g.((o * vocab) + v) <- g.((o * vocab) + v) +. coeff_values.((pos * embed) + o)
        done);
  g

(* Build the embedding + loss, run one gradient update, and return (table grads, update comp). *)
let run_backward ?name ids =
  let classes = TDSL.range vocab in
  let%op one_hot = classes = ids in
  let%op embedded = { c = uniform (); i = [ vocab ]; o = [ embed ] } * one_hot in
  let coeff =
    TDSL.ndarray coeff_values ~label:[ "coeff" ] ~batch_dims:[ positions ] ~output_dims:[ embed ] ()
  in
  let%op loss = (embedded *. coeff) ++ "...|... => |->0" in
  let update = Train.grad_update loss in
  let update = match name with Some n -> named n update | None -> update in
  let ctx = Context.cpu () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, routine = Train.to_routine ctx IDX.empty update in
  let ctx = Context.run ctx routine in
  let c = param_by_label loss "c" in
  (Context.get_values ctx (grad_of c), update)

let () =
  case "float ids" (fun () ->
      (* --- Float ids, all in range --- *)
      let id_values = [| 1.; 3.; 0. |] in
      let ids =
        TDSL.ndarray id_values ~label:[ "ids" ] ~batch_dims:[ positions ] ~output_dims:[] ()
      in
      let grads, update = run_backward ~name:"onehot_scatter_bwd" ids in
      p_all2 "float ids: table gradient equals the dense one-hot gradient" grads
        (expected_grads id_values) ~f:approx;
      let dyn, vocab_loops, guard_truncs = inspect update in
      p "optimized IR contains a Set_dynamic scatter" (dyn >= 1);
      p "vocabulary loop is eliminated from the whole update" (vocab_loops = 0);
      p "float ids: scatter guard contains the integrality Trunc" (guard_truncs > 0);
      (* Generated C: the backward writes at a dynamically computed offset (cast to the index
         precision), and the only loop over the vocabulary axis ([<= 4]) is the gradient's zero-init
         expansion — the dense reduction nest would add a second one. *)
      let src = Generated.read ~ext:".c" "onehot_scatter_bwd" in
      let has_index_prec_cast =
        String.is_substring src ~substring:"((int)("
        || String.is_substring src ~substring:"((long long)("
      in
      p "generated C contains a dynamically indexed write" has_index_prec_cast;
      let vocab_loop_count =
        List.length (String.substr_index_all src ~may_overlap:false ~pattern:"<= 4")
      in
      p "generated C has no vocabulary reduction loop (zero-init only)" (vocab_loop_count <= 1));

  case "out-of-range and fractional ids" (fun () ->
      (* --- Out-of-range and fractional ids contribute nothing --- *)
      let id_oob = [| 2.; Float.of_int vocab; 1.5 |] in
      let ids_oob =
        TDSL.ndarray id_oob ~label:[ "ids_oob" ] ~batch_dims:[ positions ] ~output_dims:[] ()
      in
      let grads_oob, _ = run_backward ids_oob in
      p_all2 "OOB and fractional ids: untouched rows stay zero, in-range rows correct" grads_oob
        (expected_grads id_oob) ~f:approx);

  case "uint32 ids" (fun () ->
      (* --- uint32 ids: integer guard flavor (no Trunc), OOB still skipped --- *)
      let id_ints = [ 1; 3; vocab (* out of [0, vocab) *) ] in
      let ids_int = Nn_blocks.class_ids_of_int_list ~label:"ids_int" id_ints in
      let ids_int_oh = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids_int in
      (* Explicit float table values: with a [uniform ()] init, top-down precision inference would
         join the table with the uint32 one-hot and make the param integer-precision. *)
      let c_int =
        TDSL.param
          ~values:(Array.init (embed * vocab) ~f:Float.of_int)
          "c_int" ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
      in
      let%op embedded_int = c_int * ids_int_oh in
      let coeff_int =
        TDSL.ndarray coeff_values ~label:[ "coeff_int" ] ~batch_dims:[ positions ]
          ~output_dims:[ embed ] ()
      in
      let%op loss_int = (embedded_int *. coeff_int) ++ "...|... => |->0" in
      let update_int = Train.grad_update loss_int in
      let ctx = Context.cpu () in
      let ctx = Train.init_params ctx IDX.empty loss_int in
      let ctx, routine = Train.to_routine ctx IDX.empty update_int in
      let ctx = Context.run ctx routine in
      let c_int = param_by_label loss_int "c_int" in
      let grads_int = Context.get_values ctx (grad_of c_int) in
      let expected_int = expected_grads (Array.of_list_map id_ints ~f:Float.of_int) in
      p_all2 "uint32 ids: table gradient equals the dense one-hot gradient (OOB id skipped)"
        grads_int expected_int ~f:approx;
      let dyn_int, vocab_loops_int, guard_truncs_int = inspect update_int in
      p "uint32 ids: optimized IR contains a Set_dynamic scatter" (dyn_int >= 1);
      p "uint32 ids: vocabulary loop is eliminated" (vocab_loops_int = 0);
      p "uint32 ids: scatter guard has no integrality Trunc" (guard_truncs_int = 0));

  case "ordinary matmul" (fun () ->
      (* --- Fallback: an ordinary matmul backward is not rewritten --- *)
      let x =
        TDSL.ndarray
          (Array.init (positions * vocab) ~f:(fun i -> 0.01 *. Float.of_int i))
          ~label:[ "x" ] ~batch_dims:[ positions ] ~output_dims:[ vocab ] ()
      in
      let%op y = { w = uniform (); i = [ vocab ]; o = [ embed ] } * x in
      let%op loss_mm = y ++ "...|... => |->0" in
      let update_mm = Train.grad_update loss_mm in
      let ctx = Context.cpu () in
      let ctx = Train.init_params ctx IDX.empty loss_mm in
      let _, routine = Train.to_routine ctx IDX.empty update_mm in
      ignore (Context.run routine.Context.context routine : Context.t);
      let dyn_mm, _, _ = inspect update_mm in
      p "ordinary matmul backward is not rewritten to a scatter" (dyn_mm = 0))
