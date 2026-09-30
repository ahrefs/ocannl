(* gh-ocannl-1126: schedule-aware fission. A merge the race analysis admits is still refused when
   some statement of the merged kernel would get less of its OWN hardware mapping -- groups or
   active threads of its own loops under the default GPU schedule -- than it gets in a kernel of its
   own. Three merges the legality rules alone take, each losing a nest's mapping:

   1. the composed attention backward's [v.grad] (reducing over the query rows) merged with
   [w_v.grad], which reads it and reduces over the positions [v.grad]'s chain owns: the aligned
   merge trims [v.grad] to the common prefix -- at batch 1 x seq 1024 a 256-thread kernel (the
   issue's 86 ms per layer on Metal); 2. the fused backward's dV nest, a lane nest, merged with dK
   (conflict-free, so merged unconditionally before), which is no lane nest: the kernel keeps the
   plain plans and dV loses its lanes -- since gh-ocannl-1124 dK IS a lane nest by default, so the
   case pins the preamble reduction refused, and the default's claims say that neither segmentation
   takes dV or dK off its lanes any more; 3. the lm_head's logits accumulation beside the row max
   that reads it: the alignment trims the logits' [(b, s, v)] chain to [(b, s)].

   Per case, structurally on the GPU pipeline (the schedule every hardware loop of the statement
   ITSELF carries, so hardware loops elsewhere in its kernel cannot satisfy the claim): the affected
   statement gets its standalone mapping, and the legality-only segmentation gives it less -- the
   negative control that makes the claim discriminating. Then executed: the step (gradients and the
   SGD update) under the schedule-aware segmentation agrees with the legality-only one, at batch 1
   and 2. *)

open Base
open Stdio
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module S = Ir.Schedule
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
module Online_softmax = Ir.Online_softmax

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* The GPU pipeline is what the cases are about: on a C backend it is still built (and executed, its
   hardware loops rendered serially) under a GPU backend's name. *)
let gpu_name = if S.backend_is_gpu backend_name then backend_name else "metal"

(* At the economics measured on Metal and CUDA (gh-ocannl-1124): the configured [auto] gives the
   fused backward's dK and dQ their lanes, as on those devices. *)
let limits =
  { Ir.Backend_intf.no_hardware_limits with Ir.Backend_intf.lane_scalar_recompute_cheap = true }

(* A hermetic copy: fission promotes placements of the record it is given. *)
let copy (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* The GPU pipeline's segments as (pre-schedule, scheduled) pairs, on a copy unless [in_place];
   [keep] selects the schedule-aware merge rule ([Schedule.fission_keep_mapping], what the default
   pipeline passes), otherwise the legality rules alone decide. [preamble_reduction] pins the lane
   geometry's treatment of a preamble reduction in both the rule and the preset (the configured one
   otherwise, which is what [Schedule.fission_keep_mapping] reads). *)
let segments ?(in_place = false) ?preamble_reduction ~keep (o : LL.optimized) =
  let keep_mapping =
    match preamble_reduction with
    | None -> if keep then S.fission_keep_mapping ~is_gpu:true ~limits else None
    | Some _ -> Option.some_if keep (fun o -> S.default_gpu ~limits ?preamble_reduction o)
  in
  S.fission_scheduled ~promote_locals:true ?keep_mapping
    ~preset:(fun o -> S.default_gpu ~limits ?preamble_reduction o)
    ~zero_sched:(S.zero_expansion ~limits) ~static_indices:[]
    (if in_place then o else copy o)
  |> List.map ~f:(fun (_, pre, _, post) -> (pre, post))

let writes_named ~f stmt =
  List.exists (LL.affine_accesses stmt) ~f:(fun (a : Ir.Tnode.t Ir.Affine.access) ->
      a.a_write && f (Ir.Tnode.debug_name a.a_tn))

let reads_something stmt =
  List.exists (LL.affine_accesses stmt) ~f:(fun (a : Ir.Tnode.t Ir.Affine.access) -> not a.a_write)

let non_glue llc =
  List.filter (LL.flat_lines [ llc ]) ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true)

let axis_of (site : L.loop_site) =
  match site.ls_stmt with LL.For_loop { axis; _ } -> axis | _ -> LL.Serial

let is_axis ty site = LL.equal_axis_type (axis_of site) ty

(* The threads a statement's OWN hardware loops span (Unrolled and Vectorized loops still run inside
   one thread): what its kernel's launch dimensions cannot vouch for. *)
let own_threads stmt =
  List.fold (L.loop_sites stmt) ~init:1 ~f:(fun n site ->
      if List.exists [ LL.Grid; LL.Workgroup; LL.Workgroup_reduce ] ~f:(fun ty -> is_axis ty site)
      then n * site.ls_extent
      else n)

(* Whether a statement runs on lanes (gh-ocannl-1003): a [Workgroup] loop inside a [Serial] loop
   that is itself inside a [Grid] loop -- not merely hardware loops under a serial reduction. *)
let on_lanes stmt =
  List.exists (L.loop_sites stmt) ~f:(fun site ->
      is_axis LL.Workgroup site
      && List.exists site.ls_ancestors ~f:(fun serial ->
          is_axis LL.Serial serial && List.exists serial.ls_ancestors ~f:(is_axis LL.Grid)))

(* The statements computing a node named by [target] (writing it and reading something, which leaves
   out initializations) of a pipeline's segments, in statement order, taken from the scheduled
   ([post]) or the pre-schedule ([pre]) side. *)
let targets ~target ~side segs =
  List.concat_map segs ~f:(fun seg ->
      List.filter (non_glue (side seg).LL.llc) ~f:(fun stmt ->
          writes_named ~f:target stmt && reads_something stmt))

(* The same statements scheduled in a kernel of their own: the pre-schedule slices of the
   schedule-aware pipeline (with its placements), one statement at a time. *)
let standalone ?preamble_reduction ~target segs =
  List.concat_map segs ~f:(fun ((pre : LL.optimized), _) ->
      List.filter_map (non_glue pre.LL.llc) ~f:(fun stmt ->
          if writes_named ~f:target stmt && reads_something stmt then
            let solo = { pre with LL.llc = stmt } in
            Some (S.apply (S.default_gpu ~limits ?preamble_reduction solo) solo).LL.llc
          else None))

(* The structural claims of one case over a lowering [o]. [lanes]: the standalone mapping is the
   lane geometry, and the schedule-aware one keeps it. [control]: the legality-only segmentation
   loses the mapping (where it does, the first claim is discriminating). *)
let check_mapping ~what ~target ?(lanes = false) ?(control = true) ?preamble_reduction
    (o : LL.optimized) =
  let keep = segments ?preamble_reduction ~keep:true o
  and legacy = segments ?preamble_reduction ~keep:false o in
  let alone = standalone ?preamble_reduction ~target keep in
  let kept = targets ~target ~side:snd keep and merged = targets ~target ~side:snd legacy in
  let n = List.length alone in
  p (what ^ ": the lowering computes the target") (n > 0);
  p
    (what ^ ": both pipelines schedule every target statement")
    (List.length kept = n && List.length merged = n);
  let pairs l = List.zip_exn (List.take l n) (List.take alone n) in
  p_exists (what ^ ": some target statement carries a parallel chain alone") alone ~f:(fun s ->
      own_threads s > 1);
  if lanes then p_exists (what ^ ": alone, some target statement runs on lanes") alone ~f:on_lanes;
  p_all (what ^ ": schedule-aware fission gives every target statement its standalone mapping")
    (pairs kept) ~f:(fun (k, a) ->
      own_threads k >= own_threads a && Bool.equal (on_lanes k) (on_lanes a));
  if control then
    p_exists
      (what ^ ": the legality-only segmentation loses some target statement's mapping (control)")
      (pairs merged) ~f:(fun (m, a) -> own_threads m < own_threads a);
  eprintf
    "%s: %d / %d kernels; own threads alone [%s], schedule-aware [%s], legality-only [%s] (not \
     part of the golden)\n\
     %!"
    what (List.length keep) (List.length legacy)
    (String.concat ~sep:" " (List.map alone ~f:(fun s -> Int.to_string (own_threads s))))
    (String.concat ~sep:" " (List.map kept ~f:(fun s -> Int.to_string (own_threads s))))
    (String.concat ~sep:" " (List.map merged ~f:(fun s -> Int.to_string (own_threads s))))

(* One training step (gradients, then the SGD update) of [build ()]'s loss, compiled with
   [transform] on the run's backend and (unless [run] is false) run once: the lowering (a copy taken
   before the transform) and every trainable parameter's gradient and updated value. *)
let step ?(run = true) ~name ~build ~transform () =
  Tensor.unsafe_reinitialize ();
  let loss = build () in
  let params =
    Set.to_list (Train.trainable_params loss)
    |> List.sort ~compare:(fun a b ->
        Int.compare a.Tensor.value.Ir.Tnode.id b.Tensor.value.Ir.Tnode.id)
  in
  List.iter params ~f:(fun p -> Train.set_materialized (Option.value_exn p.Tensor.diff).Tensor.grad);
  let update = Train.grad_update loss in
  let%op learning_rate = 0.5 in
  let sgd = Train.sgd_update ~learning_rate loss in
  let init = Train.init_params (Context.auto ()) Ir.Indexing.Empty loss in
  let captured = ref None in
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun o ->
        captured := Some (copy o);
        transform o)
      init
      (Ir.Assignments.sequence [ update; sgd ])
      Ir.Indexing.Empty
  in
  let ctx = if run then Context.run ctx routine else ctx in
  let read f = Array.concat (List.map params ~f:(fun p -> Context.get_values ctx (f p))) in
  let grads = read (fun p -> (Option.value_exn p.Tensor.diff).Tensor.grad)
  and values = read (fun p -> p.Tensor.value) in
  (* Device buffers are rooted in the backend's pool tables, not reclaimed by the GC: release the
     compiled leaf, then its initialization parent, before the next case allocates. *)
  Context.release ctx;
  Context.release init;
  (Option.value_exn !captured, grads, values)

let lowering ~name ~build =
  (* Compiled unscheduled and never run: only the lowering is wanted. *)
  let o, _, _ = step ~run:false ~name ~build ~transform:(fun o -> [ o ]) () in
  o

(* As a compile's transform: in place, so the context allocates what the promotions made
   materialized. *)
let pipeline ~keep o = List.map (segments ~in_place:true ~keep o) ~f:snd

(* Executed parity of the schedule-aware step with the legality-only one: the same gradients and the
   same updated parameters. *)
let check_parity ~what ~name ~build =
  let _, g_keep, v_keep = step ~name:(name ^ "_keep") ~build ~transform:(pipeline ~keep:true) () in
  let _, g_legacy, v_legacy =
    step ~name:(name ^ "_legacy") ~build ~transform:(pipeline ~keep:false) ()
  in
  let close g w = Float.(abs (g -. w) <= 1e-5 *. max 1. (abs w)) in
  p
    (what ^ ": the gradients are not identically zero")
    (Array.exists g_keep ~f:(fun v -> Float.(v <> 0.)));
  p_all2
    (what ^ ": the gradients agree with the legality-only segmentation's")
    g_keep g_legacy ~f:close;
  p_all2
    (what ^ ": the updated parameters agree with the legality-only segmentation's")
    v_keep v_legacy ~f:close

(* The attention block of gpu_serial_lanes leg 6, at [heads] heads of width [width]. The caller
   picks the forward and backward forms through [Online_softmax]. *)
let attention ~batch ~seq ~heads ~width () =
  let d_model = heads * width in
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  (* Scaled into [0, 1): a saturated softmax would make the gradient parity vacuous. *)
  let scale = Float.of_int (batch * seq * d_model) in
  let%op x = x /. !.scale in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let block =
    Nn_blocks.multi_head_attention ~label:[ "attn" ] ~num_heads:heads ~d_k:width ~d_v:width ()
  in
  let%op y = x + block ~train_step:None ~mask x in
  let%op loss = (y *. y) ++ "... | ... => 0" in
  loss

let with_attention_forms ~online ~fused f =
  Online_softmax.set_enabled (Some online);
  Online_softmax.set_backward_enabled (Some fused);
  Exn.protect ~f ~finally:(fun () ->
      Online_softmax.set_enabled None;
      Online_softmax.set_backward_enabled None)

(* A tied lm_head and the cross-entropy loss over it, as in gpt2_mini. *)
let lm_head ~batch ~seq ~d ~vocab () =
  let h =
    TDSL.range_of_shape ~label:[ "h" ] ~batch_dims:[ batch; seq ] ~input_dims:[] ~output_dims:[ d ]
      ()
  in
  let scale = Float.of_int (batch * seq * d) in
  let ids =
    NTDSL.init ~l:"ids" ~prec:Ir.Ops.single ~b:[ batch; seq ] ~i:[] ~o:[]
      ~f:(function [| b; s |] -> Float.of_int (((b * 7) + (s * 3)) % vocab) | _ -> assert false)
      ()
  in
  let targets = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let%op hfinal = h /. !.scale in
  let%op logits = { wte; i = [ vocab ]; o = [ d ] } +* "|v -> d; ... | d => ... | v" hfinal in
  Nn_blocks.cross_entropy_loss ~spec:"...|v" () ~logits ~targets

let v_grad n = String.is_suffix n ~suffix:"v.grad"
let logits n = String.equal n "logits"

let () =
  eprintf "gpu_fission_mapping backend: %s, GPU pipeline of %s (not part of the golden)\n%!"
    backend_name gpu_name;
  printf "--- case 1: the composed backward's v.grad against w_v.grad (gh-ocannl-1126) ---\n";
  with_attention_forms ~online:false ~fused:false (fun () ->
      check_mapping ~what:"composed v.grad, batch 1 x seq 1024" ~target:v_grad
        (lowering ~name:"gfm_composed_s1024"
           ~build:(attention ~batch:1 ~seq:1024 ~heads:8 ~width:32));
      List.iter [ 1; 2 ] ~f:(fun batch ->
          let what = Printf.sprintf "composed step, batch %d x seq 128" batch in
          let name = Printf.sprintf "gfm_composed_b%d" batch in
          let build = attention ~batch ~seq:128 ~heads:8 ~width:32 in
          (* At batch 2 the legality-only merge keeps [v.grad]'s mapping too: the issue's collapse
             is specific to batch 1, where the chain's leading batch loop has extent 1. *)
          check_mapping ~what ~target:v_grad ~control:(batch = 1) (lowering ~name ~build);
          check_parity ~what ~name ~build));
  printf "--- case 2: the fused backward's dV lanes against dK (gh-ocannl-1124) ---\n";
  (* The case needs dK to be no lane nest: since gh-ocannl-1124 it is one by default (its [dp]
     preamble reduction is admitted), so the merge the legality rules take no longer costs dV its
     lanes -- which the default-mode claims below pin. The scenario itself, a lane nest merged with
     a plain one, is kept with the preamble reduction refused. *)
  with_attention_forms ~online:true ~fused:true (fun () ->
      let preamble_reduction = S.Preamble_refused in
      check_mapping ~what:"fused dV, batch 1 x seq 1024" ~target:v_grad ~lanes:true
        ~preamble_reduction
        (lowering ~name:"gfm_fused_s1024" ~build:(attention ~batch:1 ~seq:1024 ~heads:8 ~width:32));
      List.iter [ 1; 2 ] ~f:(fun batch ->
          let what = Printf.sprintf "fused step, batch %d x seq 128" batch in
          let name = Printf.sprintf "gfm_fused_b%d" batch in
          let build = attention ~batch ~seq:128 ~heads:8 ~width:32 in
          check_mapping ~what ~target:v_grad ~lanes:true ~preamble_reduction (lowering ~name ~build);
          check_parity ~what ~name ~build);
      (* The configured default: dK on lanes too, so even the legality-only segmentation keeps every
         dV and dK statement on its lanes. *)
      let o =
        lowering ~name:"gfm_fused_default" ~build:(attention ~batch:1 ~seq:1024 ~heads:8 ~width:32)
      in
      let k_grad n = String.is_suffix n ~suffix:"k.grad" in
      List.iter
        [ ("dV", v_grad); ("dK", k_grad) ]
        ~f:(fun (name, target) ->
          List.iter
            [ ("schedule-aware", true); ("legality-only", false) ]
            ~f:(fun (rule, keep) ->
              let stmts = targets ~target ~side:snd (segments ~keep o) in
              p
                (Printf.sprintf "default, fused %s: the %s pipeline schedules it" name rule)
                (not (List.is_empty stmts));
              p_all
                (Printf.sprintf "default, fused %s: the %s segmentation keeps it on lanes" name rule)
                stmts ~f:on_lanes)));
  printf "--- case 3: the lm_head's logits against the row max ---\n";
  List.iter [ 1; 2 ] ~f:(fun batch ->
      let what = Printf.sprintf "lm_head step, batch %d x seq 128" batch in
      let name = Printf.sprintf "gfm_lm_b%d" batch in
      let build = lm_head ~batch ~seq:128 ~d:64 ~vocab:512 in
      check_mapping ~what ~target:logits (lowering ~name ~build);
      check_parity ~what ~name ~build)
