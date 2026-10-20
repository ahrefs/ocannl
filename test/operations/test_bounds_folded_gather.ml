(* Interval analysis, Phase B v1 (docs/proposals/interval-analysis-scalar-t.md): executed parity of
   a kernel whose gather guard folded away against [Tnode] bounds (binding constraint 10).

   The gh-343 embedding gather [emb[b,o] = C[o, ids[b]]] carries an in-range guard on the dynamic
   index. When [ids] is a host-initialized, never-device-written uint32 tensor, its upload scans and
   proposes value bounds; a reader compiled AFTER the upload proves the whole guard (the unsigned
   precision proves the lower bound and integrality; the proposed [0, max_id] bounds prove the upper
   bound), settles the bounds, and emits a bare gather.

   Pinned invariants: - Baseline: a reader compiled BEFORE any upload of [ids] keeps the upper-bound
   guard (its generated C contains the guard ternary). - Fold: a reader compiled AFTER the upload
   emits no guard at all (no [Where], no [Trunc] in the re-lowered IR; no ternary in its generated
   C), and the ids bounds are settled. - Executed parity: BOTH kernels run, and both match the
   direct table gather -- a fold that produced wrong values fails here, not just structurally. -
   Post-settlement widening: uploading an id above the settled range raises (the compiled guard-free
   kernel would read out of bounds). - Execution anchoring: a device-written ids tensor (bounds
   pinned to top at lowering) does NOT fold its guard, even though its runtime values are in
   range. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true
let vocab = 6
let embed = 3
let cvals = Array.init (embed * vocab) ~f:Float.of_int
let id_ints = [ 1; 3; 0; 5 ]
let batch = List.length id_ints
let approx_tol _ = 1e-4

open Verdict.Claims

(* emb[b,o] = C[o, ids[b]] = o*vocab + ids[b] *)
let expected =
  Array.concat_map (Array.of_list id_ints) ~f:(fun idx ->
      Array.init embed ~f:(fun o -> Float.of_int ((o * vocab) + idx)))

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* Every kernel here is compiled on an explicit [Context.cpu ()] lineage, so the artifacts are the C
   backend's regardless of the run's configured backend: the reads below pin [.c] rather than
   deriving the extension from the configured backend. *)

(* Count (#Get_dynamic, #Where, #Trunc) in the freshly re-lowered forward comp of [t]. Because
   lowering consults the CURRENT bounds state, this reflects post-upload structure -- use the
   generated C files (written at original compile time) for pre-upload structure. *)
let inspect (t : Tensor.t) : int * int * int =
  let comp = t.Tensor.forward in
  let optim_ctx = LL.empty_optimize_ctx () in
  let opt =
    Ir.Assignments.lower optim_ctx ~unoptim_ll_source:None ~ll_source:None ~cd_source:None
      ~name:"bfg_probe" [] comp.Ir.Assignments.asgns
  in
  let dyn = ref 0 and wheres = ref 0 and truncs = ref 0 in
  let rec proc (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        proc a;
        proc b
    | For_loop { body; _ } -> proc body
    | Set { llsc; _ } -> scal llsc
    | Set_from_vec { arg = s, _; _ } -> scal s
    | Set_local (_, s) -> scal s
    | _ -> ()
  and scal (s : LL.scalar_t) =
    match s with
    | Get_dynamic { dyn_value = v, _; _ } ->
        Int.incr dyn;
        scal v
    | Local_scope { body; _ } -> proc body
    | Ternop (op, (a, _), (b, _), (c, _)) ->
        (match op with Ir.Ops.Where -> Int.incr wheres | _ -> ());
        scal a;
        scal b;
        scal c
    | Binop (_, (a, _), (b, _)) ->
        scal a;
        scal b
    | Unop (op, (a, _)) ->
        (match op with Ir.Ops.Trunc -> Int.incr truncs | _ -> ());
        scal a
    | _ -> ()
  in
  proc opt.LL.llc;
  (!dyn, !wheres, !truncs)

let () =
  (* Shared uint32 ids tensor. Its host-init data is scanned and proposed as bounds the first time a
     routine using it is linked (the [Host_inits] upload). *)
  let ids = Nn_blocks.class_ids_of_int_list id_ints in

  (* --- Baseline: reader compiled BEFORE any upload of [ids] --- *)
  let c_a = TDSL.ndarray cvals ~label:[ "C_a" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] () in
  let%op guarded_emb = c_a * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let ctx_a = Context.cpu () in
  (* [forward_once] compiles (no bounds proposed yet: guard present), links (uploads [ids],
     proposing bounds [0, 5]), and runs. *)
  let ctx_a = Train.forward_once ctx_a guarded_emb in
  let got_a = Context.get_values ctx_a guarded_emb.Tensor.value in
  p_close "guarded baseline executes the direct table gather" got_a expected ~tol:approx_tol;
  let has_guard_ternary c =
    (* [Where] renders as [(cond != 0 ? ... : ...)] (or [!= 0.0] at float guard precisions). *)
    String.is_substring c ~substring:"!= 0 ?" || String.is_substring c ~substring:"!= 0.0 ?"
  in
  (let c = Generated.read ~ext:".c" "guarded_emb_fwd" in
   p "guarded baseline C contains the guard ternary" (has_guard_ternary c));

  (* --- Fold: reader compiled AFTER the upload --- *)
  let c_b = TDSL.ndarray cvals ~label:[ "C_b" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] () in
  let%op folded_emb = c_b * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let ctx_b = Context.cpu () in
  let ctx_b = Train.forward_once ctx_b folded_emb in
  let got_b = Context.get_values ctx_b folded_emb.Tensor.value in
  (* Binding constraint 10: the bounds-folded kernel EXECUTES and its values match the unfolded
     baseline. *)
  p_close "bounds-folded kernel values equal the guarded baseline" got_b got_a ~tol:approx_tol;
  p_close "bounds-folded kernel values equal the host reference" got_b expected ~tol:approx_tol;
  let dyn_b, wheres_b, truncs_b = inspect folded_emb in
  p "folded IR keeps the Get_dynamic gather" (dyn_b >= 1);
  p "folded IR has no Where guard and no Trunc" (wheres_b = 0 && truncs_b = 0);
  (let c = Generated.read ~ext:".c" "folded_emb_fwd" in
   p "folded C has no guard ternary" (not (has_guard_ternary c)));
  p "ids bounds are settled by the fold"
    (match ids.Tensor.value.Tn.bounds with Tn.Bounds_settled _ -> true | _ -> false);

  (* --- Post-settlement widening must be rejected --- *)
  let widened = Array.init batch ~f:(fun i -> if i = 0 then Float.of_int vocab else 0.) in
  (try
     ignore (Context.set_values ctx_b ids.Tensor.value widened : Context.t);
     p "post-settlement widening host write is rejected" false
   with Utils.User_error msg ->
     p "post-settlement widening host write is rejected"
       (String.is_substring msg ~substring:"settled"));
  (* An in-range re-upload validates against the settled bounds and succeeds. *)
  let shifted = Array.init batch ~f:(fun i -> Float.of_int ((i + 1) % vocab)) in
  ignore (Context.set_values ctx_b ids.Tensor.value shifted : Context.t);
  p "in-range re-upload validates against the settled bounds" true;

  (* --- Execution anchoring: device-written ids never fold --- *)
  (* Compiling (not even running) a routine that writes [ids_dev] pins its bounds to top at
     lowering ([pin_device_written_bounds]); a reader compiled afterwards must keep the guard,
     even though the host-initialized runtime values are all in [0, vocab), because compiled
     writes are not execution-anchored facts (they may run later, with any values). *)
  let ids_dev = Nn_blocks.class_ids_of_int_list id_ints in
  let ctx_d = Context.cpu () in
  let writer =
    [%cd
      ~~("pin ids_dev writer";
         ids_dev =: ids_dev)]
  in
  let ctx_d, writer_routine = Context.compile ctx_d writer Ir.Indexing.Empty in
  p "device-written ids are pinned to top"
    (match Tn.bounds_candidate ids_dev.Tensor.value with
    | Some iv -> Ir.Interval.is_top iv
    | None -> false);
  (* Run the (identity) writer so the execution ledger lets the reader run after it. *)
  let ctx_d = Context.run ctx_d writer_routine in
  let c_d = TDSL.ndarray cvals ~label:[ "C_d" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] () in
  let%op dev_emb = c_d * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids_dev in
  let ctx_d = Train.forward_once ctx_d dev_emb in
  let got_d = Context.get_values ctx_d dev_emb.Tensor.value in
  p_close "device-written ids kernel executes the direct table gather" got_d expected
    ~tol:approx_tol;
  let dyn_d, wheres_d, _ = inspect dev_emb in
  p "device-written ids keep the Get_dynamic gather" (dyn_d >= 1);
  p "device-written ids keep the guard (bounds pinned to top)" (wheres_d >= 1)
