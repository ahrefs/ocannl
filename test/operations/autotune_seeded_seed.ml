(* gh-ocannl-1388: the autotuner seeds the seeded untuned plan FIRST, so a search cut to one
   candidate ships the step the untuned compile ships.

   Since gh-ocannl-1324 the recipes' untuned compile on CUDA and HIP is [Autotune.seeded_default]:
   the default pipeline with its large contractions seeded with a reuse-tiled sketch, its long
   narrow reductions split (gh-ocannl-1325, CUDA) and the rest of its reductions sunk
   (gh-ocannl-1348). Before this issue the search's pool started from the plain default, so a search
   that timed only its first candidate shipped a plan the untuned compile had already beaten (1720
   ms against 563 ms on gpt2_mini_train b256, rog).

   The fixture is one dense layer with a bias, [y = w * x + b], backpropagated through [loss = sum
   (y *. dy)] over [nb * ns] = 4096 rows: the weight gradient's reduction (4096) and the forward
   product's (96) both reach the default seeding threshold (32), and the bias gradient's serial
   chain (4096 rows per output cell) and the loss sum's reach the default split threshold (4096), so
   the plan seeds AND splits. Every setting is the configuration's default: the search reads them,
   as the untuned compile does.

   The search's measurements are scripted through [Autotune.on_candidate_measured] (gh-ocannl-1027)
   with the windows opted out of host contention ([with_uncontended_test_windows], gh-ocannl-1156):
   the first admitted window keeps its reading and every later one is refused, which is the search
   cut to its first candidate. The claims: that first window is the seeded plan's, its code is the
   untuned compile's segment for segment (the digests the search keys candidates by), the search did
   try later candidates and crowned the seeded plan, and the crowned routine computes bitwise what
   the untuned compile computes, on inputs whose every partial sum rounds -- and which tell the
   plans apart: the split reassociates the bias gradient, so the plain default pipeline computes it
   differently in some cell, which is claimed as the control. Off CUDA/HIP the untuned compile is
   the plain default and the search seeds no plan, which is claimed instead.

   The CUDA/HIP legs need a device that seeds; elsewhere they are skipped. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module SC = Ir.Schedule_cache

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let seeds_here =
  String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

(* The batch axis outermost and long: the split detector splits a reduction over its outermost loop,
   which at [nb = 4] would be too short to propose (the gpt2_mini_train b8 case). *)
let nb = 1024
and ns = 4
and no = 64
and ni = 96

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"ss_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"ss_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x =
  NTDSL.ndarray
    (values ~n:(nb * ns * ni) ~stride:389)
    ~label:[ "ss_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()

let dy =
  NTDSL.ndarray
    (values ~n:(nb * ns * no) ~stride:577)
    ~label:[ "ss_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

let%op y = (w * x) + b
let%op loss = (y *. dy) ++ "... => 0"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let wgrad = grad_of w
let bgrad = grad_of b

let () =
  Train.set_materialized wgrad;
  Train.set_materialized bgrad

let comp = Train.grad_update loss
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let static_indices = Ir.Indexing.bound_symbols Ir.Indexing.Empty

(* The digest the search keys a compiled candidate by: each compiled segment's canonical digest, in
   order. *)
let digest_of posts =
  String.concat ~sep:"+"
    (List.map posts ~f:(fun post -> SC.digest (SC.canonicalize ~static_indices post)))

let run ctx routine =
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx wgrad, Context.get_values ctx bgrad)

(* The untuned compile, through the transform [Autotune.seeded_default] compiles with, capturing the
   code it hands the backend and what it decided. *)
let untuned () =
  let ctx = Context.auto () in
  let digest = ref None and choice = ref None in
  let ctx, routine =
    match
      Autotune.seeded_transform ~on_choice:(fun c -> choice := Some c) ctx Ir.Indexing.Empty
    with
    | None -> Context.compile ctx comp Ir.Indexing.Empty
    | Some transform ->
        Context.compile
          ~lowered_transform:(fun opt ->
            let posts = transform opt in
            digest := Some (digest_of posts);
            posts)
          ctx comp Ir.Indexing.Empty
  in
  (run ctx routine, !digest, !choice)

(* The search cut to its first candidate: every admitted window after the first refused. *)
let cut_search () =
  let windows = ref [] and report = ref None in
  let old_measured = Autotune.Seam.get Autotune.on_candidate_measured in
  let ctx, routine =
    Exn.protect
      ~finally:(fun () -> Autotune.Seam.set Autotune.on_candidate_measured old_measured)
      ~f:(fun () ->
        ( Autotune.Seam.set Autotune.on_candidate_measured @@ fun ~label ~digest ms ->
          let first = List.is_empty !windows in
          windows := (label, digest, ms) :: !windows;
          if first then ms else Float.nan );
        Autotune.with_uncontended_test_windows (fun () ->
            Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
              ~cache_dir:"" ~log:false
              ~report:(fun r -> report := Some r)
              (Context.auto ()) comp Ir.Indexing.Empty))
  in
  (run ctx routine, List.rev !windows, Option.value_exn ~here:[%here] !report)

let () =
  let (w_untuned, b_untuned), untuned_digest, choice = untuned () in
  let claim label evaluate = Verdict.gate ~when_:seeds_here ~on:backend_name label evaluate in
  claim "on CUDA/HIP the untuned plan seeds a contraction and splits a reduction" (fun () ->
      p "on CUDA/HIP the untuned plan seeds a contraction and splits a reduction"
        (match choice with
        | Some c -> (not (List.is_empty c.Autotune.sd_seeded)) && not (List.is_empty c.sd_split)
        | None -> false));
  Verdict.gated ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the untuned compile is the plain default" (Option.is_none untuned_digest);
  Verdict.gate ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the search derives no seeded plan" (fun () ->
      (* The search's own derivation, on the lowering the plain default compiles: no plan. *)
      let plan = ref None in
      let ctx = Context.auto () in
      let limits = Context.hardware_limits ctx in
      let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
      ignore
        (Context.compile
           ~lowered_transform:(fun opt ->
             plan :=
               Some
                 (Autotune.seeded_plan ~backend:backend_name ~limits ~accum_prec ~static_indices opt);
             [ opt ])
           ctx comp Ir.Indexing.Empty
          : Context.t * Context.routine);
      p "off CUDA/HIP the search derives no seeded plan"
        (match !plan with Some None -> true | Some (Some _) | None -> false));
  let tuned = if seeds_here then Some (cut_search ()) else None in
  let with_tuned label f =
    claim label (fun () ->
        match tuned with Some t -> f t | None -> p (label ^ " (no search ran)") false)
  in
  let first_window (_, windows, _) = List.hd windows in
  with_tuned "the search's first timed window is the seeded untuned plan" (fun t ->
      Option.iter (first_window t) ~f:(fun (label, _, _) ->
          Stdio.eprintf "first window: %s (not part of the golden)\n%!" label);
      p "the search's first timed window is the seeded untuned plan"
        (match first_window t with
        | Some (label, _, _) -> String.is_prefix label ~prefix:"F_seeded["
        | None -> false));
  with_tuned "its code is the untuned compile's, segment for segment" (fun t ->
      p "its code is the untuned compile's, segment for segment"
        (match (first_window t, untuned_digest) with
        | Some (_, digest, _), Some d -> String.equal digest d
        | _ -> false));
  with_tuned "the search went on to later candidates, every one of them refused"
    (fun (_, windows, r) ->
      p "the search went on to later candidates, every one of them refused"
        (List.length windows >= 2 && r.Autotune.timings_contended = List.length windows - 1));
  with_tuned "the search crowned the seeded untuned plan" (fun ((_, _, r) as t) ->
      p "the search crowned the seeded untuned plan"
        (match first_window t with
        | Some (label, _, _) -> String.equal r.Autotune.best_label label
        | None -> false));
  with_tuned "the report's seeded_ms is that window's reading, and the crowned time" (fun t ->
      let _, _, r = t in
      p "the report's seeded_ms is that window's reading, and the crowned time"
        (match first_window t with
        | Some (_, _, ms) ->
            Option.equal Float.equal r.Autotune.seeded_ms (Some ms) && Float.equal r.best_ms ms
        | None -> false));
  with_tuned "the crowned weight gradient is bitwise the untuned compile's"
    (fun ((w_tuned, _), _, _) ->
      p_all2 "the crowned weight gradient is bitwise the untuned compile's" w_tuned w_untuned
        ~f:bitwise);
  with_tuned "the crowned bias gradient is bitwise the untuned compile's"
    (fun ((_, b_tuned), _, _) ->
      p_all2 "the crowned bias gradient is bitwise the untuned compile's" b_tuned b_untuned
        ~f:bitwise);
  claim "the control: the plain default pipeline's bias gradient differs in some cell" (fun () ->
      let ctx, routine = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
      let _, b_default = run ctx routine in
      p_exists "the control: the plain default pipeline's bias gradient differs in some cell"
        (List.zip_exn (Array.to_list b_default) (Array.to_list b_untuned))
        ~f:(fun (a, b) -> not (bitwise a b)))
