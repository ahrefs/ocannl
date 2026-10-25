(* gh-ocannl-1395: the census of [Schedule.Privatize]'s v1 refusal, pinned.

   [Privatize] refuses a target accessed at two index vectors under its accumulation loop
   ([Schedule.privatize_index_vectors_refusal]). [Stage]'s look-alike refusal declined every
   whole-routine seed of a ResNet block until gh-ocannl-1147 relaxed it; this one was censused over
   the tuned benchmark cells and the tuning test corpus twice (the issue's comments) and never
   fired: [Privatize] scans only its own loop's subtree, where a sketch's perfectly nested
   accumulation and the privatized preset's self-reading [Set] leave one vector. The relaxation is
   parked until a workload reaches the refusal, and this test is the tripwire that says when.

   The control leg, on hand-built IR through the privatized preset's own extension
   ([Autotune.privatize_extension]): [c[0] += c[1]] under a serial loop is proposed and dropped by
   exactly this refusal, while its twin [c[0] += tick k] is privatized. The census sees a dropped
   proposal only because this seam returns it; the tuner drops it silently.

   The corpus leg: the training steps of an MLP, a downsampling ResNet block (the strided 1x1
   shortcut beside a 3x3 conv: gh-ocannl-1147's shape) and a decoder block (Q, K, V over one input;
   layer norms; softmax), each lowered as the tuner sees it. Per workload, the privatized preset's
   proposals on every segment of the default segmentation, and every sketch seed's schedule on the
   whole routine and on each segment of the sketches' finer segmentation, are applied; no attempt
   may be refused for two index vectors. Counts and every other refusal go to stderr. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Outcome = Ir.Schedule_outcome
module Ll = Ll_test
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let scratch (seg : LL.optimized) =
  {
    seg with
    LL.traced_store = Hashtbl.copy seg.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx seg.LL.optimize_ctx;
  }

let privatize_targets (sched : Sched.schedule) =
  List.filter_map sched ~f:(function Sched.Privatize { target; _ } -> Some target | _ -> None)

let detail_of (cause : Outcome.cause) =
  match cause with
  | Outcome.Illegal_schedule { detail; _ } -> Some detail
  | Unsupported { detail; _ }
  | Resource_exceeded { detail; _ }
  | Backend_rejected { detail; _ }
  | Unclassified { detail; _ }
  | Seed_evicted { detail; _ }
  | Not_dispatched { detail; _ } ->
      Some detail

(* Whether [cause] is the v1 refusal of one of [targets] -- by the library's own text for it. *)
let is_vector_refusal ~targets cause =
  match (cause, detail_of cause) with
  | Outcome.Illegal_schedule _, Some detail ->
      List.exists targets ~f:(fun t ->
          String.equal detail (Sched.privatize_index_vectors_refusal t))
  | _ -> false

(* One [Privatize] attempt: a preset proposal or a sketch seed's schedule carrying one. *)
type attempt = { a_where : string; a_targets : Ir.Tnode.t list; a_refusal : Outcome.cause option }

let show_attempt a =
  Printf.sprintf "%s [%s]: %s" a.a_where
    (String.concat ~sep:"," (List.map a.a_targets ~f:Ir.Tnode.debug_name))
    (match a.a_refusal with
    | None -> "applied"
    | Some cause -> Option.value (detail_of cause) ~default:"refused")

let multi_vector a =
  match a.a_refusal with
  | Some cause -> is_vector_refusal ~targets:a.a_targets cause
  | None -> false

(* The privatized preset's attempts on one segment: what it kept, then what it dropped. *)
let preset_attempts ~accum_prec ~where preset (seg : LL.optimized) =
  let base = preset seg in
  let sched, drops = Autotune.privatize_extension ~accum_prec ~static_indices:[] base seg in
  let kept =
    List.map
      (List.drop (privatize_targets sched) (List.length (privatize_targets base)))
      ~f:(fun t -> { a_where = where; a_targets = [ t ]; a_refusal = None })
  in
  ( sched,
    kept
    @ List.map drops ~f:(fun (d : Autotune.privatize_drop) ->
        { a_where = where; a_targets = [ d.pd_target ]; a_refusal = Some d.pd_cause }) )

(* --- Leg 1: the control. --- *)
let () =
  let accum_prec = Ll.backend_accum_prec in
  let mk = Ll.node_factory ~first_id:913950 ~dims:[| 2 |] () in
  let run ~name ~addend =
    let c = mk name in
    Ll.materialize c;
    let k = Ll.sym () in
    let llc =
      Ll.loop_n k 8 (Ll.set c [| Ll.fixed 0 |] (Ll.add (Ll.get c [| Ll.fixed 0 |]) (addend c k)))
    in
    let o = Ll.optimize ~name llc in
    let _sched, attempts = preset_attempts ~accum_prec ~where:name (fun _ -> []) o in
    (c, attempts)
  in
  let c2, two_vectors = run ~name:"pc_two_vectors" ~addend:(fun c _ -> Ll.get c [| Ll.fixed 1 |]) in
  let _, one_vector = run ~name:"pc_one_vector" ~addend:(fun _ k -> Ll.tick k) in
  p "control: c[0] += c[1] is proposed once and dropped for its two index vectors"
    (match two_vectors with
    | [ ({ a_refusal = Some cause; _ } as a) ] ->
        multi_vector a && is_vector_refusal ~targets:[ c2 ] cause
    | _ -> false);
  p "control: its one-vector twin c[0] += tick k is privatized"
    (match one_vector with [ { a_refusal = None; _ } ] -> true | _ -> false)

(* --- Leg 2: the corpus. --- *)

let init_input ~label ~b ~o =
  NTDSL.init ~l:label ~prec:Ir.Ops.single ~b ~i:[] ~o
    ~f:(fun idx ->
      Float.of_int ((Array.fold idx ~init:7 ~f:(fun acc i -> (acc * 31) + i) % 13) - 6) /. 8.)
    ()

let lowered_step ~name (loss : Tensor.t) =
  let step = Train.grad_update loss in
  let comp = { step with asgns = Ir.Assignments.Block_comment (name, step.asgns) } in
  let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty loss in
  let captured = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      ctx comp Ir.Indexing.Empty
  in
  (ctx, Option.value_exn ~here:[%here] !captured)

let census ~what ctx (opt : LL.optimized) =
  let is_gpu = Sched.backend_is_gpu backend_name and is_cpu = Sched.backend_is_cpu backend_name in
  let limits = Context.hardware_limits ctx in
  let capabilities = Context.codegen_capabilities ctx in
  let accum_prec = capabilities.Ir.Backend_intf.accum_prec in
  let preset seg =
    if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits seg
    else if is_cpu then Sched.default_cpu ~min_work:0 seg
    else []
  in
  let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
  (* The privatized preset flavor: the default segmentation, each [`Normal] segment's preset
     extended. *)
  let preset_seen = ref [] in
  let privatized seg =
    let sched, attempts =
      preset_attempts ~accum_prec ~where:(what ^ " preset") preset (scratch seg)
    in
    preset_seen := !preset_seen @ attempts;
    sched
  in
  ignore
    (Sched.fission_scheduled ~promote_locals:is_gpu
       ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
       ~preset:privatized ~zero_sched ~static_indices:[] (scratch opt));
  (* The sketch flavors: every seed on the whole routine and on each segment of the finer
     segmentation the per-segment sketches are enumerated under. *)
  let sketch_seen = ref [] and unbuilt = ref 0 in
  let seed_attempts ~where (seg : LL.optimized) =
    let seeds =
      Autotune.sketch_seed_params ~capabilities ~is_gpu ~is_cpu ~limits seg
      @ Autotune.coalesced_seed_params ~capabilities ~is_gpu ~is_cpu ~limits seg
    in
    List.iter seeds ~f:(fun p ->
        match Autotune.sketch_schedule ~accum_prec ~p seg with
        | exception Outcome.Cause_at _ -> Int.incr unbuilt
        | sched -> (
            match privatize_targets sched with
            | [] -> ()
            | targets ->
                let a_where = where ^ " " ^ Autotune.sketch_params_label p in
                let a_refusal =
                  match Sched.apply_classified ~static_indices:[] sched (scratch seg) with
                  | (_ : LL.optimized) -> None
                  | exception Outcome.Cause_at (_, cause) -> Some cause
                in
                sketch_seen := !sketch_seen @ [ { a_where; a_targets = targets; a_refusal } ]))
  in
  seed_attempts ~where:(what ^ " whole") (scratch opt);
  List.iter
    (Sched.fission_scheduled ~promote_locals:is_gpu ~arity_cuts:true ~fold_zeros:true ~preset
       ~zero_sched ~static_indices:[] (scratch opt))
    ~f:(fun (kind, seg, _, _) ->
      match kind with
      | `Normal -> seed_attempts ~where:(what ^ " segment") (scratch seg)
      | `Zeros | `Solo -> ());
  let report label attempts =
    let refused = List.filter attempts ~f:(fun a -> Option.is_some a.a_refusal) in
    Stdio.eprintf "%s: %d %s attempts, %d refused (not part of the golden)\n" what
      (List.length attempts) label (List.length refused);
    List.iter refused ~f:(fun a -> Stdio.eprintf "  refused: %s\n" (show_attempt a))
  in
  report "privatized-preset" !preset_seen;
  report "sketch-seed" !sketch_seen;
  if !unbuilt > 0 then Stdio.eprintf "%s: %d sketch seeds did not build\n" what !unbuilt;
  (!preset_seen, !sketch_seen)

let mlp_loss () =
  let x = init_input ~label:"pc_mlp_x" ~b:[ 8 ] ~o:[ 16 ] in
  let mlp = Nn_blocks.mlp ~label:[ "pc_mlp" ] ~hid_dims:[ 32 ] () in
  let target = init_input ~label:"pc_mlp_target" ~b:[ 8 ] ~o:[ 4 ] in
  let y = mlp x in
  let%op d = y - target in
  let%op loss = (d *. d) ++ "...|... => |->0" in
  loss

let resnet_loss () =
  let x = init_input ~label:"pc_res_x" ~b:[ 2 ] ~o:[ 8; 8; 4 ] in
  let block = Nn_blocks.resnet_block ~label:[ "pc_res" ] ~stride:2 ~out_channels:8 () in
  (* A computed operand, so the padded convs can still widen its margins (a data-backed input's
     layout is committed). *)
  let%op x = relu x in
  let y = Nn_blocks.max_pool2d () (block ~train_step:None x) in
  let%op loss = (y *. y) ++ "...|... => |->0" in
  loss

let decoder_loss () =
  let seq = 4 in
  let x = init_input ~label:"pc_dec_x" ~b:[ 2; seq ] ~o:[ 16 ] in
  let mask =
    NTDSL.init ~l:"pc_dec_mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> 0.)
      ()
  in
  let block =
    Nn_blocks.decoder_only_block ~label:[ "pc_dec" ] ~num_heads:2 ~d_k:8 ~d_v:8 ~d_ff:32 ()
  in
  let y = block ~train_step:None x ~mask in
  let%op loss = (y *. y) ++ "...|... => |->0" in
  loss

let () =
  let presets, sketches =
    List.unzip
      (List.map
         [ ("mlp", mlp_loss); ("resnet_block", resnet_loss); ("decoder_block", decoder_loss) ]
         ~f:(fun (what, loss) ->
           let ctx, opt = lowered_step ~name:("pc_" ^ what) (loss ()) in
           census ~what ctx opt))
  in
  let presets = List.concat presets and sketches = List.concat sketches in
  (* Pooled over the corpus: how many proposals each workload yields is the backend's preset's. *)
  p_none "corpus: no privatized-preset proposal is refused for two index vectors" presets
    ~f:multi_vector;
  (* Which sketch seeds privatize is the backend's: on cc only the scalar packing pipelines do, and
     a backend whose seeds carry none for these shapes cannot evaluate the claim. *)
  let claim = "corpus: no sketch seed's Privatize is refused for two index vectors" in
  if List.is_empty sketches then skipped ~backend:backend_name claim
  else p_none claim sketches ~f:multi_vector
