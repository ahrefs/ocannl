(* gh-ocannl-1164: a fissioned schedule-cache winner persists its segmentation and replays it.

   The entry records every segment of the winner's fission -- kind, length in units, pre-schedule
   digest, schedule (zero expansions included) -- and a replay cuts the routine there instead of
   re-deriving the cuts under the replaying process's policy. So the policy's inputs are not in the
   cache key, and a winner replays identically after they change.

   The workload is the lm_head shape (a materialized rank-3 GEMM and the row max that reads it),
   whose finer [arity_cuts] segmentation (gh-ocannl-574) is not what the default policy derives: the
   stored winner is that fine segmentation, so every replay below applies a segmentation the current
   policy would not produce.

   1. The policy change is real: the default segmentation differs from the stored one, and on GPU
   backends the changed configuration re-derives a different zero expansion than the stored one. 2.
   The cache key does not move with the segmentation policy. 3. The winner replays from the cache
   before and after the configuration change, computes the right values both times, and emits the
   same code byte for byte, up to the numbering of fresh names. 4. Validity: a segmentation that
   does not partition the routine is refused by [Schedule.fission_segmented]; one that partitions it
   at other boundaries applies, but its segments are not the ones the schedules were saved against,
   so the replay declines and the tuner re-searches. 5. Segments are matched by position: two
   segments with one structural digest (the digest leaves placements out, and placements decide a
   segment's kind and schedule) keep their own saved schedules through a replay. 6. A folded segment
   (gh-ocannl-1175: a reduction's zero expanded into its kernel) replays after the zero policy stops
   proposing that fold, with or without search: the recorded fold is rebuilt from its proven target,
   never re-asked of the current policy. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Generated = Test_utils.Generated
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Generated.init ~backend_name
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name
let cache_dir = "autotune_cache_fission_replay"

(* The dune sandbox persists across runs: a previous run's entry would turn the first lookup into a
   replay of whatever that run stored. *)
let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let approx x y = Float.(abs (x - y) < 1e-2)

(* Emitted code up to the numbering of fresh names: loop symbols ([i40], [i40_lo]) and value
   temporaries ([v5_z]) carry process-wide counters, so two compiles of one schedule in one process
   differ in exactly those digits. Each stem -- the letter and its digits -- is renumbered by first
   occurrence; everything else must match byte for byte. *)
let alpha_normalize src =
  let is_ident c = Char.is_alphanum c || Char.equal c '_' in
  let stems = Hashtbl.create (module String) in
  let buf = Buffer.create (String.length src) in
  let len = String.length src in
  let rec go i =
    if i < len then
      if is_ident src.[i] && (i = 0 || not (is_ident src.[i - 1])) then (
        let j = ref (i + 1) in
        while !j < len && Char.is_digit src.[!j] do
          Int.incr j
        done;
        let stem = String.sub src ~pos:i ~len:(!j - i) in
        let fresh =
          (Char.equal src.[i] 'i' || Char.equal src.[i] 'v')
          && !j > i + 1
          && (!j = len || not (Char.is_alpha src.[!j]))
        in
        if fresh then
          Buffer.add_string buf
            (Printf.sprintf "%c#%d" src.[i]
               (Hashtbl.find_or_add stems stem ~default:(fun () -> Hashtbl.length stems)))
        else Buffer.add_string buf stem;
        go !j)
      else (
        Buffer.add_char buf src.[i];
        go (i + 1))
  in
  go 0;
  Buffer.contents buf

(* A hermetic copy: fission promotes placements of the record it is given. *)
let copy (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* A routine under test: its base lowering (what [Autotune.tune] replays against) and the context
   facts its cache key is made of. *)
type routine = {
  base_ctx : Context.t;
  limits : Ir.Backend_intf.hardware_limits;
  caps : Ir.Backend_intf.codegen_capabilities;
  base_opt : LL.optimized;
  base_canon : SC.canonical;
}

let lower comp =
  let base_ctx = Context.auto () in
  let captured = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some (copy opt);
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let base_opt = Option.value_exn ~here:[%here] !captured in
  {
    base_ctx;
    limits = Context.hardware_limits base_ctx;
    caps = Context.codegen_capabilities base_ctx;
    base_opt;
    base_canon = SC.canonicalize base_opt;
  }

(* The autotuner's fissioned pipeline: its placements ([promote_locals] on GPU) and the search's
   aggressive presets. *)
let preset rut o =
  if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits:rut.limits o
  else if is_cpu then Sched.default_cpu ~min_parallel:1 o
  else []

let zero_sched rut tns = if is_gpu then Sched.zero_expansion ~limits:rut.limits tns else []

let fission rut ?arity_cuts ?fold_zeros ?keep_mapping ?segmentation () =
  (* A recorded segmentation is applied with empty schedules: these checks are about the cuts. *)
  let replay = Option.map segmentation ~f:(fun plan -> (plan, fun _ _ -> [])) in
  Sched.fission_segmented ~promote_locals:is_gpu ?arity_cuts ?fold_zeros ?keep_mapping ?replay
    ~preset:(preset rut) ~zero_sched:(zero_sched rut) ~static_indices:[] (copy rut.base_opt)

let cache_key rut =
  SC.cache_key
    ~timing_identity:(Context.timing_identity rut.base_ctx)
    ~limits:rut.limits ~capabilities:rut.caps rut.base_canon
    ~backend:(Context.backend_name rut.base_ctx)

(* A fissioned winner stored the way [Autotune.tune] stores one. *)
let store ~cache_dir rut segments =
  SC.store ~dir:cache_dir ~key:(cache_key rut)
    {
      SC.version = SC.entry_version;
      backend = Context.backend_name rut.base_ctx;
      numerics = SC.numerics_tag ();
      codegen = Some (SC.codegen_tag ~limits:rut.limits ~capabilities:rut.caps ());
      objective = Some (SC.objective_tag ());
      source_digest = SC.digest rut.base_canon;
      saved = [];
      segments = Some segments;
      best_ms = 0.;
      baseline_ms = 0.;
      default_ms = None;
      mma_best_ms = None;
      default_fingerprint = None;
      best_steps = None;
    }

let tune ?search ~name ~cache_dir comp ~read =
  let report = ref None in
  let ctx, routine =
    Autotune.tune ?search ~name ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun rep -> report := Some rep)
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let values = read ctx in
  Context.release ctx;
  (values, !report)

(* Timed-cache reuse needs a concrete device identity; without one every cache claim is vacuous. *)
let cache_available = Option.is_some (Context.timing_identity (Context.auto ()))

let () =
  if not cache_available then
    Stdio.eprintf "timed-cache reuse unavailable: concrete device identity missing\n"

let cache_claim label value =
  gated ~aggregation:`Environment ~when_:cache_available ~on:"no-device-identity" label value

let () =
  clean_cache cache_dir;
  let b = 4 and n = 32 and m = 64 and k = 16 in
  let xv =
    Array.init
      (b * n * k)
      ~f:(Ll_test.cycle_flat ~dims:[| b; n; k |] ~modulus:7 ~offset:0. ~stride:0.25)
  in
  let wv =
    Array.init (k * m) ~f:(Ll_test.cycle_flat ~dims:[| k; m |] ~modulus:5 ~offset:0. ~stride:0.125)
  in
  let z_expected =
    Array.init
      (b * n * m)
      ~f:(fun idx ->
        let bi = idx / (n * m) in
        let i = idx % (n * m) / m and j = idx % m in
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc := !acc +. (xv.((bi * n * k) + (i * k) + kk) *. wv.((kk * m) + j))
        done;
        !acc)
  in
  let r_expected =
    Array.init (b * n) ~f:(fun row ->
        Array.fold
          (Array.sub z_expected ~pos:(row * m) ~len:m)
          ~init:Float.neg_infinity ~f:Float.max)
  in
  let x = TDSL.ndarray xv ~label:[ "x" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w = TDSL.ndarray wv ~label:[ "w" ] ~output_dims:[ k; m ] () in
  let%op z = x +* "b|ik;kj=>b|ij" w in
  Train.set_materialized z.Tensor.value;
  let%op r = z @^^ "b|ij => b|i" in
  let comp = Train.forward r in
  let name = "seg_replay" in
  let rut = lower comp in
  let limits = rut.limits in
  let zero_sched = zero_sched rut in
  let fission ?arity_cuts ?keep_mapping ?segmentation () =
    fission rut ?arity_cuts ?keep_mapping ?segmentation ()
  in
  let fine_segmentation, fine_tuples = fission ~arity_cuts:true () in
  let saved = List.map (SC.save_segments fine_segmentation fine_tuples) ~f:fst in
  let store = store ~cache_dir rut in
  let tune () =
    tune ~name ~cache_dir comp ~read:(fun ctx ->
        (Context.get_values ctx z.Tensor.value, Context.get_values ctx r.Tensor.value))
  in
  let expected = Array.append z_expected r_expected in
  let correct label (zv, rv) = p_all2 label (Array.append zv rv) expected ~f:approx in
  (* The emitted source of the fissioned routine: the batch of its segment kernels. *)
  let fissioned_source = name ^ "__seg" in

  (* --- 1. The stored segmentation is one the policy would not derive. --- *)
  p "the stored segmentation fissions the routine" (List.length saved >= 2);
  p_exists "the stored segmentation carries a zeros segment" saved ~f:(fun s ->
      match s.SC.seg_kind with `Zeros -> true | `Normal | `Solo -> false);
  let key_before = cache_key rut in
  store saved;

  (* --- 2-3. Replay under the configuration the entry was stored under. --- *)
  Generated.arm fissioned_source;
  let values_a, report_a = tune () in
  let source_a = if cache_available then Some (Generated.read fissioned_source) else None in
  cache_claim "the stored winner replays from the cache"
    (Option.value_map report_a ~default:false ~f:replayed);
  cache_claim "the replay is fissioned"
    (Option.value_map report_a ~default:false ~f:(fun r -> r.Autotune.fissioned));
  correct "the replay computes the right values" values_a;

  (* The segmentation policy changes: block size, minimum parallelism and workgroup fill feed the
     default GPU schedule, hence the schedule-aware merge decision and the zero expansion; the CPU
     minimum parallelism feeds the default CPU schedule. *)
  List.iter
    [
      ("OCANNL_GPU_SCHEDULE_BLOCK_SIZE", "32");
      ("OCANNL_GPU_SCHEDULE_MIN_PARALLEL", "1000000");
      ("OCANNL_GPU_SCHEDULE_WORKGROUP_FILL", "1");
      ("OCANNL_CPU_SCHEDULE_MIN_PARALLEL", "1000000");
    ]
    ~f:(fun (var, value) -> Unix.putenv var value);
  let policy_segmentation, _ =
    fission ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits) ()
  in
  p "the current policy segments the routine differently from the stored segmentation"
    (not (Sched.equal_segmentation policy_segmentation (SC.segmentation_of saved)));
  (* What a re-deriving replay would give a zeros segment now: its zero expansion under the changed
     configuration, saved against the segment as the stored schedule is. *)
  let rederived_zeros_differ (s, (_, pre, _, _)) =
    match s.SC.seg_kind with
    | `Normal | `Solo -> false
    | `Zeros ->
        let tns =
          List.filter_map (LL.flat_lines [ pre.LL.llc ]) ~f:(function
            | LL.Zero_out tn -> Some tn
            | _ -> None)
        in
        let pre_canon = SC.canonicalize ~with_placements:false pre in
        let rederived, _ = SC.to_saved (SC.base_registry pre_canon) (zero_sched tns) in
        not (SC.equal_saved_schedule rederived s.SC.seg_saved)
  in
  gated ~when_:is_gpu ~on:backend_name
    "the changed configuration re-derives a different zero expansion than the stored one"
    (List.exists (List.zip_exn saved fine_tuples) ~f:rederived_zeros_differ);
  p "the cache key does not move with the segmentation policy"
    (Option.equal String.equal key_before (cache_key rut));
  Generated.arm fissioned_source;
  let values_b, report_b = tune () in
  cache_claim "after the policy change the winner still replays from the cache"
    (Option.value_map report_b ~default:false ~f:replayed);
  correct "after the policy change the replay computes the right values" values_b;
  cache_claim "after the policy change the replay emits identical code"
    (match source_a with
    | Some a -> String.equal (alpha_normalize a) (alpha_normalize (Generated.read fissioned_source))
    | None -> false);

  (* --- 4. Replay validity. --- *)
  let refused = function
    | Ir.Schedule_outcome.Cause_at (_, Ir.Schedule_outcome.Illegal_schedule { check; _ }) ->
        String.equal check "Schedule.fission_segmented"
    | _ -> false
  in
  let raises segmentation =
    match fission ~segmentation () with _ -> false | exception exn -> refused exn
  in
  p "a segmentation covering fewer statements than the routine is refused"
    (raises (List.drop_last_exn fine_segmentation));
  p "a segmentation with an empty segment is refused" (raises ((`Normal, 0) :: fine_segmentation));
  (* Other boundaries over the same statements: the first two segments as one. *)
  let merged =
    match saved with
    | s0 :: s1 :: rest -> { s0 with SC.seg_units = s0.SC.seg_units + s1.SC.seg_units } :: rest
    | _ -> saved
  in
  let applied =
    match fission ~segmentation:(SC.segmentation_of merged) () with
    | segmentation, tuples -> Some (List.map (SC.save_segments segmentation tuples) ~f:fst)
    | exception exn when refused exn -> None
  in
  p "a segmentation at other boundaries still partitions the routine" (Option.is_some applied);
  p "its segments are not the ones the schedules were saved against"
    (match applied with
    | Some segs ->
        not
          (List.equal String.equal
             (List.map segs ~f:(fun s -> s.SC.seg_digest))
             (List.map merged ~f:(fun s -> s.SC.seg_digest)))
    | None -> false);
  store merged;
  (* Legs 2-3 showed this store path hitting under this key, and the entry differs from that one
     only in its boundaries, so a search here is the replay declining, not a lookup miss. *)
  let values_c, report_c = tune () in
  cache_claim "a stored segmentation that no longer applies declines to a re-search"
    (Option.value_map report_c ~default:false ~f:completed);
  correct "the re-searched routine computes the right values" values_c

(* --- 5. Segments are matched by position, not by digest. --- *)
let () =
  let cache_dir = "autotune_cache_fission_replay_positional" in
  clean_cache cache_dir;
  let b = 2 and n = 16 and m = 32 in
  let xv =
    Array.init
      (b * n * m)
      ~f:(Ll_test.cycle_flat ~dims:[| b; n; m |] ~modulus:7 ~offset:0. ~stride:0.25)
  in
  let wv =
    Array.init (m * m) ~f:(Ll_test.cycle_flat ~dims:[| m; m |] ~modulus:5 ~offset:0. ~stride:0.125)
  in
  let matmul a =
    Array.init
      (b * n * m)
      ~f:(fun idx ->
        let row = idx / m and j = idx % m in
        let acc = ref 0. in
        for kk = 0 to m - 1 do
          acc := !acc +. (a.((row * m) + kk) *. wv.((kk * m) + j))
        done;
        !acc)
  in
  let z2_expected = matmul (matmul (Array.map xv ~f:(fun v -> 2. *. v))) in
  let x0 = TDSL.ndarray xv ~label:[ "x0" ] ~batch_dims:[ b ] ~output_dims:[ n; m ] () in
  let w1 = TDSL.ndarray wv ~label:[ "w1" ] ~output_dims:[ m; m ] () in
  let w2 = TDSL.ndarray wv ~label:[ "w2" ] ~output_dims:[ m; m ] () in
  let%op x = x0 + x0 in
  Train.set_materialized x.Tensor.value;
  let%op z1 = x +* "b|ik;kj=>b|ij" w1 in
  Train.set_materialized z1.Tensor.value;
  let%op z2 = z1 +* "b|ik;kj=>b|ij" w2 in
  Train.set_materialized z2.Tensor.value;
  let comp = Train.forward z2 in
  let rut = lower comp in
  let segmentation, tuples = fission rut () in
  let derived = List.map (SC.save_segments segmentation tuples) ~f:fst in
  (* Two segments with one digest: the zeroings of [z1] and [z2], one shape and precision. *)
  let twins =
    List.find_mapi derived ~f:(fun i s ->
        List.find_mapi derived ~f:(fun j s' ->
            Option.some_if (j > i && String.equal s.SC.seg_digest s'.SC.seg_digest) (i, j)))
  in
  p "the routine has two segments with one structural digest" (Option.is_some twins);
  let i, j = Option.value twins ~default:(0, 0) in
  (* The later twin gets a schedule of its own: its zeroing expanded into a plain serial nest, which
     no preset gives the earlier one. *)
  let own =
    let _, pre, _, _ = List.nth_exn tuples j in
    let expand =
      List.filter_map (LL.flat_lines [ pre.LL.llc ]) ~f:(function
        | LL.Zero_out tn -> Some (fst (Sched.expand_zero ~tn))
        | _ -> None)
    in
    fst (SC.to_saved (SC.base_registry (SC.canonicalize ~with_placements:false pre)) expand)
  in
  let stored =
    List.mapi derived ~f:(fun k s -> if k = j then { s with SC.seg_saved = own } else s)
  in
  p "the twins carry different schedules"
    (not
       (SC.equal_saved_schedule (List.nth_exn stored i).SC.seg_saved
          (List.nth_exn stored j).SC.seg_saved));
  store ~cache_dir rut stored;
  let got, report =
    tune ~name:"seg_twins" ~cache_dir comp ~read:(fun ctx -> Context.get_values ctx z2.Tensor.value)
  in
  cache_claim "the twins' entry replays from the cache"
    (Option.value_map report ~default:false ~f:replayed);
  cache_claim "each twin replays its own schedule"
    (Option.value_map report ~default:false ~f:(fun r ->
         SC.equal_saved_schedule r.Autotune.best_schedule
           (List.concat_map stored ~f:(fun s -> s.SC.seg_saved))));
  p_all2 "the twins' replay computes the right values" got z2_expected ~f:approx

(* --- 6. A folded segment replays under a zero policy that no longer proposes the fold. --- *)
let () =
  let cache_dir = "autotune_cache_fission_replay_folded" in
  clean_cache cache_dir;
  (* Sections 1-4 left the GPU threshold raised; the fold is derived under a permissive one. *)
  Unix.putenv "OCANNL_GPU_SCHEDULE_MIN_PARALLEL" "1";
  let b = 2 and s = 16 and h = 2 and j = 16 and k = 32 in
  let xv =
    Array.init
      (b * s * k)
      ~f:(Ll_test.cycle_flat ~dims:[| b; s; k |] ~modulus:7 ~offset:0. ~stride:0.25)
  in
  let wv =
    Array.init
      (h * j * k)
      ~f:(Ll_test.cycle_flat ~dims:[| h; j; k |] ~modulus:5 ~offset:0. ~stride:0.125)
  in
  let expected =
    Array.init
      (b * s * h * j)
      ~f:(fun idx ->
        let o = Ll_test.unflat ~dims:[| b; s; h; j |] idx in
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc :=
            !acc
            +. 2.
               *. xv.(Ll_test.flat ~dims:[| b; s; k |] [| o.(0); o.(1); kk |])
               *. wv.(Ll_test.flat ~dims:[| h; j; k |] [| o.(2); o.(3); kk |])
        done;
        !acc)
  in
  let x0 = TDSL.ndarray xv ~label:[ "fx0" ] ~output_dims:[ b; s; k ] () in
  let w = TDSL.ndarray wv ~label:[ "fw" ] ~output_dims:[ h; j; k ] () in
  let%op x = x0 + x0 in
  Train.set_materialized x.Tensor.value;
  let%op q = x +* "bsk;hjk=>bshj" w in
  Train.set_materialized q.Tensor.value;
  let comp = Train.forward q in
  let rut = lower comp in
  let folded_label = "the folded segmentation opens a segment with the projection's zero" in
  let segmentation, tuples = fission rut ~fold_zeros:true () in
  let folded =
    List.exists tuples ~f:(fun (kind, pre, _, _) ->
        Poly.equal kind `Normal
        && List.exists (LL.flat_lines [ pre.LL.llc ]) ~f:(fun stmt ->
            Option.exists (LL.zero_initializer_target stmt) ~f:(Ir.Tnode.equal q.Tensor.value)))
  in
  if not is_gpu then Verdict.skipped ~backend:backend_name folded_label
  else p folded_label (folded && List.length tuples >= 2);
  let stored = List.map (SC.save_segments segmentation tuples) ~f:fst in
  store ~cache_dir rut stored;
  let name = "seg_folded" in
  let source = name ^ "__seg" in
  let tune ?search () =
    tune ?search ~name ~cache_dir comp ~read:(fun ctx -> Context.get_values ctx q.Tensor.value)
  in
  Generated.arm source;
  let got_a, report_a = tune () in
  let source_a = if cache_available then Some (Generated.read source) else None in
  cache_claim "the folded winner replays from the cache"
    (Option.value_map report_a ~default:false ~f:replayed);
  p_all2 "the folded replay computes the right values" got_a expected ~f:approx;
  Unix.putenv "OCANNL_GPU_SCHEDULE_MIN_PARALLEL" "1000000";
  let refolds, _ = fission rut ~fold_zeros:true () in
  gated ~when_:is_gpu ~on:backend_name "the raised threshold no longer proposes the fold"
    (not (Sched.equal_segmentation refolds segmentation));
  List.iter [ None; Some false ] ~f:(fun search ->
      let leg = match search with None -> "with search" | Some _ -> "with search disabled" in
      Generated.arm source;
      let got, report = tune ?search () in
      cache_claim
        ("under the raised threshold the folded winner replays from the cache, " ^ leg)
        (Option.value_map report ~default:false ~f:replayed);
      p_all2
        ("under the raised threshold the folded replay computes the right values, " ^ leg)
        got expected ~f:approx;
      cache_claim
        ("under the raised threshold the folded replay emits identical code, " ^ leg)
        (match source_a with
        | Some a -> String.equal (alpha_normalize a) (alpha_normalize (Generated.read source))
        | None -> false))
