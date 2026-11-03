(* gh-ocannl-1664: a search that seeds the seeded untuned plan times its per-segment singles inside
   that plan.

   On CUDA and HIP the search's first candidate is the seeded untuned plan ([F_seeded],
   gh-ocannl-1388). Its per-segment singles used to be timed over the search's own segmentation with
   every other segment on its preset: a backdrop several times the plan's time, so a single was
   comparable only with its own key's singles (gh-ocannl-1528). With config
   [autotune_seeded_singles] (the default) each single is instead the plan with one segment code's
   choice swapped for a sketch ([F_swap], {!Autotune.seeded_swap}), over the plan's own segmentation
   ({!Autotune.seeded_single_seeds}), so it times on the plan's basis.

   Two parts. The structural part runs on every backend: it derives a plan for "cuda" on synthetic
   limits from this fixture's lowering, enumerates the seeded singles' menu on it, and claims the
   menu's keys are the plan's own segments (a key's digest is the one the plan decided that segment
   on) and that a swap changes the plan's decision on the key's segments and nowhere else -- the
   split prelude, the segmentation and every other decision stay the plan's.

   The search part runs where the untuned compile seeds ([Autotune.seeded_default_backend]): a
   scripted search over the same fixture, the plan's windows at 10 ms, the first seeded single's at
   9 ms and every other single's slower than the plan. Its claims are the timing basis, read off the
   digests the search keys candidates by (one structural digest per compiled segment): every seeded
   single compiles to the plan's segment count, a single swapped alone differs from the plan in at
   most its key's segments, and the singles over the search's own segmentation are not seeded. Then
   the sweep: the 9 ms single joins the running seeded composite, every single of a later key
   carries it as its staff, and the search crowns it. Elsewhere those claims are skipped.

   gh-ocannl-1697 adds the split singles and the hopeless cut. Structurally, on every backend: a
   site's other choice ({!Autotune.seeded_resplit}: another block count, unsplit, or split where the
   plan leaves it whole) re-derives a plan other than the plan, deciding every segment code the plan
   decides as the plan does, and the plan's own split re-derives the plan. On CUDA/HIP, two more
   scripted searches -- the split singles timed as [F_resplit] before the segment singles and none
   around presets; a faster one joining, after which every seeded single names the plan it
   re-derived -- and two on a scripted clock: with the cut at its default (off), every seeded single
   takes its whole window; with {!Autotune.seeded_hopeless_cut} on, a seeded single over
   {!Autotune.seeded_hopeless_ratio} times the plan is cut after three samples and one under it
   takes its whole window. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* The seeding gate pinned at threshold 32, as in [autotune_seeded_seed]: the ranked [auto] gate
   seeds nothing on this fixture on HIP, and every claim needs a plan that seeds. *)
let () = Unix.putenv "OCANNL_GPU_DEFAULT_SKETCH_MIN_REDUCTION" "32"
let seeds_here = Autotune.seeded_default_backend (Context.backend_facts_of_name backend_name)

(* [autotune_seeded_seed]'s fixture: one dense layer with a bias, backpropagated, whose weight
   gradient (4096 rows) and forward product (96) both reach the seeding threshold. *)
let nb = 1024
and ns = 4
and no = 64
and ni = 96

let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"sw_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"sw_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x =
  NTDSL.ndarray
    (values ~n:(nb * ns * ni) ~stride:389)
    ~label:[ "sw_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()

let dy =
  NTDSL.ndarray
    (values ~n:(nb * ns * no) ~stride:577)
    ~label:[ "sw_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

let%op y = (w * x) + b
let%op loss = (y *. dy) ++ "... => 0"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let () =
  Train.set_materialized (grad_of w);
  Train.set_materialized (grad_of b)

let comp = Train.grad_update loss
let static_indices = Ir.Indexing.bound_symbols Ir.Indexing.Empty

(* === The structural part: the menu and the swap, on a plan derived for "cuda". === *)

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* The routine's lowering as a compile's transform receives it. *)
let lowering () =
  let captured = ref None in
  let _ctx, _routine =
    Context.compile ~name:"sw_lowering"
      ~lowered_transform:(fun opt ->
        captured := Some (scratch opt);
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !captured

let limits = Ir.Backend_intf.no_hardware_limits

(* Each segment's first unit in a segmentation, by position. *)
let starts segmentation = List.folding_map segmentation ~init:0 ~f:(fun at (_, n) -> (at + n, at))

let decision_digest = function
  | Autotune.Seeded_sketch { digest; _ } | Autotune.Seeded_sunk { digest; _ } -> Some digest
  | Autotune.Seeded_preset -> None

let () =
  let opt = lowering () in
  let plan =
    Autotune.seeded_plan ~min_reduction:1 ~facts:Ir.Backend_intf.cuda_facts ~limits
      ~accum_prec:Fn.id ~static_indices opt
  in
  match plan with
  | Some ({ Autotune.sp_segments = Some (segmentation, decisions); _ } as plan) ->
      p "a plan derived for cuda has a segmentation of its own" true;
      let seeds =
        Autotune.seeded_single_seeds ~capabilities:Ir.Backend_intf.no_codegen_capabilities
          ~is_gpu:true ~is_cpu:false ~limits ~static_indices plan (scratch opt)
      in
      let at = starts segmentation in
      let decision_at unit =
        List.find_map (List.zip_exn at decisions) ~f:(fun (u, d) ->
            if u = unit then Some d else None)
      in
      (* Every (key, unit) pair of the menu, with the plan's decision on that segment. *)
      let keyed =
        List.concat_map seeds ~f:(fun (key, _, _) ->
            List.map key.Autotune.key_units ~f:(fun u -> (key, u, decision_at u)))
      in
      p_all "every unit of a menu key starts a segment of the plan's segmentation" keyed
        ~f:(fun (_, _, d) -> Option.is_some d);
      p_exists "the menu keys a segment the plan decided on its code" keyed ~f:(fun (_, _, d) ->
          Option.exists d ~f:(fun d -> Option.is_some (decision_digest d)));
      p_all "a menu key's digest is the one the plan decided its segment on" keyed
        ~f:(fun (key, _, d) ->
          match Option.bind d ~f:decision_digest with
          | None -> true
          | Some digest -> String.equal digest key.Autotune.key_digest);
      let swaps =
        List.concat_map seeds ~f:(fun (key, _, params) ->
            List.map params ~f:(fun params -> (key, params)))
      in
      p_all "a swap keeps the plan's split prelude, segmentation and admission" swaps
        ~f:(fun choice ->
          let swapped = Autotune.seeded_swap plan [ choice ] in
          phys_equal swapped.sp_split plan.sp_split
          && Bool.equal swapped.sp_admit_narrow plan.sp_admit_narrow
          &&
          match swapped.sp_segments with
          | Some (s, _) -> phys_equal s segmentation
          | None -> false);
      p_all "a swap decides exactly its key's segments, on the key's digest, and no other" swaps
        ~f:(fun ((key, params) as choice) ->
          match (Autotune.seeded_swap plan [ choice ]).sp_segments with
          | None -> false
          | Some (_, swapped) ->
              List.for_all
                (List.zip_exn at (List.zip_exn decisions swapped))
                ~f:(fun (u, (before, after)) ->
                  if List.mem key.key_units u ~equal:Int.equal then
                    match after with
                    | Autotune.Seeded_sketch { digest; params = q } ->
                        String.equal digest key.key_digest && phys_equal q params
                    | Autotune.Seeded_preset | Autotune.Seeded_sunk _ -> false
                  else phys_equal before after));
      p "an empty swap is the plan itself"
        (String.equal
           (Autotune.seeded_plan_digest (Autotune.seeded_swap plan []))
           (Autotune.seeded_plan_digest plan));
      p_all "a seeded single's label names the plan and does not read as the plan's" swaps
        ~f:(fun choice ->
          let label = Autotune.seeded_swap_label plan [ choice ] in
          String.is_prefix label ~prefix:"F_swap["
          && String.is_substring label
               ~substring:("; plan " ^ String.prefix (Autotune.seeded_plan_digest plan) 8 ^ "]")
          && not (String.is_prefix label ~prefix:"F_seeded"));
      (* gh-ocannl-1697: the seeded split singles -- the plan re-derived by its own rule over a
         split with one site's choice changed. *)
      let plan_digest = Autotune.seeded_plan_digest plan in
      let resplit ?memo split = Autotune.seeded_resplit ?memo plan split (scratch opt) in
      let digest_of = Option.map ~f:Autotune.seeded_plan_digest in
      p "re-deriving the plan with its own split reproduces it"
        (Option.equal String.equal (digest_of (resplit plan.sp_split)) (Some plan_digest));
      let same = Autotune.same_split_site in
      let sites = Autotune.split_reduce_sites ~static_indices (scratch opt) in
      p_exists "the plan splits a site of the fixture" plan.sp_split ~f:(fun (t, _) ->
          List.exists sites ~f:(same t));
      (* Every site at every other choice of the menu: a block count, or unsplit. *)
      let alternatives =
        List.concat_map sites ~f:(fun site ->
            let current =
              List.find_map plan.sp_split ~f:(fun (t, b) -> if same t site then Some b else None)
            in
            List.filter_map
              (None :: List.map [ 32; 128; 512 ] ~f:Option.some)
              ~f:(fun blocks ->
                if
                  Option.equal Int.equal blocks current
                  || Option.exists blocks ~f:(fun b -> 2 * b > site.Autotune.sr_red)
                then None
                else
                  let split =
                    List.filter_map plan.sp_split ~f:(fun ((t, _) as entry) ->
                        if same t site then Option.map blocks ~f:(fun b -> (t, b)) else Some entry)
                    @
                    match (current, blocks) with
                    | None, Some b -> [ (site, b) ]
                    | Some _, _ | None, None -> []
                  in
                  Some (site, blocks, split, resplit split)))
      in
      p_exists "some site's other choice unsplits a site the plan splits" alternatives
        ~f:(fun (site, blocks, _, _) ->
          Option.is_none blocks && List.exists plan.sp_split ~f:(fun (t, _) -> same t site));
      p_all "every other choice of a site re-derives a plan other than the plan" alternatives
        ~f:(fun (_, _, _, derived) ->
          Option.exists (digest_of derived) ~f:(fun d -> not (String.equal d plan_digest)));
      (* The decisions a plan records on segment codes, by digest. *)
      let recorded (pl : Autotune.seeded_plan) =
        match pl.sp_segments with
        | None -> []
        | Some (_, ds) ->
            List.filter_map ds ~f:(fun d -> Option.map (decision_digest d) ~f:(fun g -> (g, d)))
      in
      let shared derived =
        List.filter (recorded derived) ~f:(fun (g, _) ->
            List.exists (recorded plan) ~f:(fun (g', _) -> String.equal g g'))
      in
      p_all "a re-derived plan decides every segment code the plan decides as the plan does"
        alternatives ~f:(fun (_, _, _, derived) ->
          match derived with
          | None -> false
          | Some derived ->
              List.for_all (shared derived) ~f:(fun (g, d) ->
                  List.for_all (recorded plan) ~f:(fun (g', d') ->
                      (not (String.equal g g')) || Poly.equal d d')));
      p_exists "... and some re-derived plan shares a decided segment code with the plan"
        alternatives ~f:(fun (_, _, _, derived) ->
          Option.exists derived ~f:(fun d -> not (List.is_empty (shared d))));
      let memo = Autotune.seeded_memo () in
      let memoized =
        Autotune.seeded_plan ~memo ~min_reduction:1 ~facts:Ir.Backend_intf.cuda_facts ~limits
          ~accum_prec:Fn.id ~static_indices opt
      in
      p "a memo shared with the plan's derivation changes no derivation of the plan"
        (Option.equal String.equal (digest_of memoized) (Some plan_digest));
      p_all "... nor of a split single" alternatives ~f:(fun (_, _, split, derived) ->
          Option.equal String.equal (digest_of (resplit ~memo split)) (digest_of derived));
      p_all "a seeded split single's label names the plan, and reads as no plan or segment single"
        alternatives ~f:(fun (site, blocks, _, _) ->
          let label = Autotune.seeded_resplit_label ~from:plan_digest site blocks in
          String.is_prefix label ~prefix:"F_resplit["
          && String.is_substring label ~substring:("; plan " ^ String.prefix plan_digest 8 ^ "]")
          && (not (String.is_prefix label ~prefix:"F_seeded"))
          && not (String.is_prefix label ~prefix:"F_swap["))
  | Some { sp_segments = None; _ } | None ->
      p "a plan derived for cuda has a segmentation of its own" false

(* === The search part: a scripted search where the untuned compile seeds. === *)

let is_seeded label = String.is_prefix label ~prefix:"F_seeded["
let is_swap label = String.is_prefix label ~prefix:"F_swap["
let segments digest = String.split digest ~on:'+'

(* A seeded single's focus -- what precedes its staff or plan -- and its staff, as labelled. *)
let focus_and_staff label =
  let body = String.chop_prefix_exn label ~prefix:"F_swap[" in
  let body = fst (Option.value_exn ~here:[%here] (String.lsplit2 body ~on:';')) in
  match String.substr_index body ~pattern:" | staff " with
  | None -> (body, None)
  | Some i -> (String.prefix body i, Some (String.drop_prefix body (i + String.length " | staff ")))

(* The units of a focus' key tag: [<sketch> @<digest tag>:u3+u7] has two. *)
let focus_units focus =
  match String.rsplit2 focus ~on:':' with
  | Some (_, units) -> List.length (String.split units ~on:'+')
  | None -> 0

let differing a b =
  List.count (List.zip_exn (segments a) (segments b)) ~f:(fun (x, y) -> not (String.equal x y))

let () =
  let first_swap = ref None in
  let measure (w : Scripted_search.window) =
    if is_seeded w.label then 10.
    else if is_swap w.label then (
      if Option.is_none !first_swap then first_swap := Some w.label;
      if Option.exists !first_swap ~f:(String.equal w.label) then 9.
      else 11. +. (Float.of_int w.ordinal /. 1000.))
    else 50. +. (Float.of_int w.ordinal /. 1000.)
  in
  (* Each timing call that returns, under the attempt it timed: a window the host refuses still
     returns one but never reaches the script, so the report's window count is read against
     these. *)
  let latest = ref None and timed = ref [] in
  let search () =
    let report = ref None in
    let _routine, record =
      Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label -> latest := Some label)
      @@ fun () ->
      Autotune.with_timing_call_observer (fun (_ : Autotune.timing_call) ->
          Option.iter !latest ~f:(fun label -> timed := label :: !timed))
      @@ fun () ->
      (* The beam finals off, so every timing call after the base compile is a candidate's own. *)
      Scripted_search.scripted
        ~settings:[ ("autotune_finals_margin", "0") ]
        ~measure
        (fun () ->
          Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
            ~cache_dir:"" ~log:false
            ~report:(fun r -> report := Some r)
            (Context.auto ()) comp Ir.Indexing.Empty)
    in
    (Option.value_exn ~here:[%here] !report, record.Scripted_search.windows)
  in
  let searched = if seeds_here then Some (search ()) else None in
  let searched_claim label f =
    Verdict.gate ~when_:seeds_here ~on:backend_name label (fun () ->
        match searched with
        | None -> p (label ^ " (no search ran)") false
        | Some (report, windows) ->
            let windows =
              List.map windows ~f:(fun (w, ms) -> (w.Scripted_search.label, w.digest, ms))
            in
            f report windows)
  in
  let swaps windows = List.filter windows ~f:(fun (label, _, _) -> is_swap label) in
  let plan_digest windows =
    List.find_map windows ~f:(fun (label, digest, _) ->
        if is_seeded label then Some digest else None)
  in
  Option.iter searched ~f:(fun ((r : Autotune.report), windows) ->
      Stdio.eprintf
        "scripted search (not part of the golden): %d windows, %d seeded singles proposed, %d \
         timed, %d joined, %d refused in all\n\
         %!"
        (List.length windows) r.seeded_single_candidates r.seeded_single_timed r.seeded_single_joins
        r.timings_contended);
  searched_claim "the search seeds seeded singles, and no single over its own segmentation"
    (fun r _ ->
      p "the search seeds seeded singles, and no single over its own segmentation"
        (r.Autotune.seeded_single_candidates > 0 && r.fiss_sketch_candidates = 0));
  searched_claim "no window is a single over the search's own segmentation" (fun _ windows ->
      p_none "no window is a single over the search's own segmentation" windows
        ~f:(fun (label, _, _) -> String.is_prefix label ~prefix:"F_sketch["));
  searched_claim "the report counts every seeded single's window" (fun r windows ->
      p "the report counts every seeded single's window"
        (r.Autotune.seeded_single_timed = List.count !timed ~f:is_swap
        && List.count !timed ~f:is_swap >= List.length (swaps windows)
        && not (List.is_empty (swaps windows))));
  searched_claim "every seeded single compiles to the plan's segment count" (fun _ windows ->
      match plan_digest windows with
      | None -> p "every seeded single compiles to the plan's segment count (no plan window)" false
      | Some plan ->
          p_all "every seeded single compiles to the plan's segment count" (swaps windows)
            ~f:(fun (_, digest, _) -> List.length (segments digest) = List.length (segments plan)));
  searched_claim "a single swapped alone differs from the plan in at most its key's segments"
    (fun _ windows ->
      match plan_digest windows with
      | None -> p "a single swapped alone differs from the plan (no plan window)" false
      | Some plan ->
          p_all "a single swapped alone differs from the plan in at most its key's segments"
            (List.filter (swaps windows) ~f:(fun (label, _, _) ->
                 Option.is_none (snd (focus_and_staff label))))
            ~f:(fun (label, digest, _) ->
              let d = differing plan digest in
              d >= 1 && d <= focus_units (fst (focus_and_staff label))));
  searched_claim "the faster single joined the running seeded composite and was crowned"
    (fun r windows ->
      p "the faster single joined the running seeded composite and was crowned"
        (r.Autotune.seeded_single_joins = 1
        && Option.exists !first_swap ~f:(String.equal r.best_label)
        && Float.equal r.best_ms 9.
        && Option.exists
             (List.hd (swaps windows))
             ~f:(fun (label, _, _) -> Option.exists !first_swap ~f:(String.equal label))));
  (* The joined key's focus tag: the part after its sketch's [" @"]. *)
  let key_of focus = Option.map (String.rsplit2 focus ~on:'@') ~f:snd in
  searched_claim "every later key's single carries the joined single as its staff" (fun _ windows ->
      let joined = Option.map !first_swap ~f:(fun l -> fst (focus_and_staff l)) in
      let joined_key = Option.bind joined ~f:key_of in
      let after_join =
        List.drop_while (swaps windows) ~f:(fun (label, _, _) ->
            not (Option.exists !first_swap ~f:(String.equal label)))
      in
      let later =
        List.filter after_join ~f:(fun (label, _, _) ->
            not (Option.equal String.equal (key_of (fst (focus_and_staff label))) joined_key))
      in
      (* The fixture keys at least two segments, so some key is swept after the join. *)
      p_all "every later key's single carries the joined single as its staff" later
        ~f:(fun (label, _, _) -> Option.equal String.equal (snd (focus_and_staff label)) joined))

(* === gh-ocannl-1697: the seeded split singles, and the seeded singles' hopeless cut. === *)

let is_resplit label = String.is_prefix label ~prefix:"F_resplit["
let is_seeded_single label = is_resplit label || is_swap label

(* The plan prefix a label names: [...; plan <prefix>]] for a seeded single, [..., plan <prefix>]]
   for the plan itself. *)
let plan_named label =
  Option.map
    (List.last (String.substr_index_all label ~may_overlap:false ~pattern:" plan "))
    ~f:(fun i ->
      String.chop_suffix_if_exists ~suffix:"]"
        (String.drop_prefix label (i + String.length " plan ")))

(* A scripted search of the fixture as the first one, under [measure], with its report, its record
   and the labels of the timing calls that returned, in order. *)
let scripted_search measure =
  let latest = ref None and timed = ref [] in
  let report = ref None in
  let _routine, record =
    Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label -> latest := Some label)
    @@ fun () ->
    Autotune.with_timing_call_observer (fun (_ : Autotune.timing_call) ->
        Option.iter !latest ~f:(fun label -> timed := label :: !timed))
    @@ fun () ->
    Scripted_search.scripted
      ~settings:[ ("autotune_finals_margin", "0") ]
      ~measure
      (fun () ->
        Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
          ~cache_dir:"" ~log:false
          ~report:(fun r -> report := Some r)
          (Context.auto ()) comp Ir.Indexing.Empty)
  in
  (Option.value_exn ~here:[%here] !report, record, List.rev !timed)

(* The searches' leg: open where the untuned compile seeds. One claim per label, written once
   (gh-ocannl-1703): evaluated there over the search, skipped elsewhere. *)
let seeded_leg = Verdict.leg ~when_:seeds_here ~on:backend_name ()

let claim searched label f =
  in_leg ~leg:seeded_leg label (fun label ->
      match searched with None -> p label false | Some s -> f label s)

let seeded_plan_named (record : Scripted_search.record) =
  List.find_map record.attempts ~f:(fun l -> if is_seeded l then plan_named l else None)

(* Every split single slower than the plan: none joins, and the segment singles swap into the plan
   itself. *)
let () =
  let measure (w : Scripted_search.window) =
    if is_seeded w.label then 10.
    else if is_seeded_single w.label then 11. +. (Float.of_int w.ordinal /. 1000.)
    else 50. +. (Float.of_int w.ordinal /. 1000.)
  in
  let searched = if seeds_here then Some (scripted_search measure) else None in
  Option.iter searched ~f:(fun ((r : Autotune.report), _, _) ->
      Stdio.eprintf
        "split search (not part of the golden): %d seeded split singles proposed, %d timed, %d \
         joined; %d split-reduce singles\n\
         %!"
        r.seeded_split_candidates r.seeded_split_timed r.seeded_split_joins
        r.split_reduce_candidates);
  let claim = claim searched in
  claim "the search seeds seeded split singles, and no split single around presets"
    (fun label (r, _, _) ->
      p label (r.Autotune.seeded_split_candidates > 0 && r.split_reduce_candidates = 0));
  claim "no window is a split single around presets" (fun label (_, record, _) ->
      p_none label record.Scripted_search.windows ~f:(fun (w, _) ->
          String.is_prefix w.label ~prefix:"F_split["));
  claim "the report counts every seeded split single's window" (fun label (r, record, timed) ->
      let windows =
        List.count record.Scripted_search.windows ~f:(fun (w, _) -> is_resplit w.label)
      in
      let calls = List.count timed ~f:is_resplit in
      p label (r.Autotune.seeded_split_timed = calls && calls >= windows && windows > 0));
  claim "every seeded split single is attempted before every seeded segment single"
    (fun label (_, record, _) ->
      let seeded = List.filter record.Scripted_search.attempts ~f:is_seeded_single in
      let first_swap = List.findi seeded ~f:(fun _ l -> is_swap l) in
      p_alli label (List.filter seeded ~f:is_resplit) ~f:(fun i _ ->
          Option.exists first_swap ~f:(fun (j, _) -> i < j)));
  claim "with no split joined, every seeded single names the plan" (fun label (_, record, _) ->
      let plan = seeded_plan_named record in
      p_all label (List.filter record.Scripted_search.attempts ~f:is_seeded_single) ~f:(fun l ->
          Option.is_some plan && Option.equal String.equal (plan_named l) plan));
  claim "with no split joined, the search crowns the plan" (fun label (r, _, _) ->
      p label
        (r.Autotune.seeded_split_joins = 0 && is_seeded r.best_label && Float.equal r.best_ms 10.))

(* The first split single faster than the plan: it joins, and every later seeded single changes the
   plan it re-derived. *)
let () =
  let first = ref None in
  let measure (w : Scripted_search.window) =
    if is_seeded w.label then 10.
    else if is_resplit w.label && Option.is_none !first then (
      first := Some w.label;
      9.)
    else if is_seeded_single w.label then 11. +. (Float.of_int w.ordinal /. 1000.)
    else 50. +. (Float.of_int w.ordinal /. 1000.)
  in
  let searched = if seeds_here then Some (scripted_search measure) else None in
  let claim = claim searched in
  (* What a split single's label names as its site: the target and the split loop, the first two
     words after the bracket. *)
  let site_of label =
    String.chop_prefix label ~prefix:"F_resplit["
    |> Option.map ~f:(fun body -> List.take (String.split body ~on:' ') 2)
  in
  (* The seeded singles after the joined site's sweep: the later sites' split singles and every
     segment single. The joined site's other choices were timed against the plan it changed. *)
  let after_join (record : Scripted_search.record) =
    let joined = Option.bind !first ~f:site_of in
    List.drop_while record.attempts ~f:(fun l -> not (Option.exists !first ~f:(String.equal l)))
    |> List.filter ~f:(fun l ->
        is_swap l
        || (is_resplit l && not (Option.equal (List.equal String.equal) (site_of l) joined)))
  in
  claim "the faster split single joined the running seeded plan and was crowned"
    (fun label (r, _, _) ->
      p label
        (r.Autotune.seeded_split_joins = 1
        && Option.exists !first ~f:(String.equal r.best_label)
        && Float.equal r.best_ms 9.));
  claim "every seeded single after the join names a plan other than the seeded plan"
    (fun label (_, record, _) ->
      let plan = seeded_plan_named record in
      p_all label (after_join record) ~f:(fun l ->
          Option.is_some (plan_named l) && not (Option.equal String.equal (plan_named l) plan)));
  claim "a segment single is attempted after the join" (fun label (_, record, _) ->
      p_exists label (after_join record) ~f:is_swap)

(* The hopeless cut ([Autotune.seeded_hopeless_cut] on), on a scripted clock: every window physical,
   so the cut is live. The plan reads 10 ms; the seeded singles alternate between 12 ms, over 1.05
   times the plan (the running composite, and with the finals off the beam's head: the reference),
   and 10.4 ms, under it; every other candidate reads 50 ms. Nothing beats the plan, so the
   reference stays 10 ms. *)
let () =
  let cut_ms = 12. and near_ms = 10.4 in
  (* Each attempt, newest first: its label, and once a batch is timed, its reading and batches. *)
  let attempts = ref [] and cycled = ref 0 in
  let clock ~depth ~ordinal:_ _wall =
    match !attempts with
    | (label, reads) :: rest ->
        let ms =
          match reads with
          | Some (ms, _) -> ms
          | None ->
              if is_seeded label then 10.
              else if is_seeded_single label then (
                Int.incr cycled;
                if !cycled % 2 = 1 then cut_ms else near_ms)
              else 50.
        in
        attempts :=
          (label, Some (ms, Option.value_map reads ~default:1 ~f:(fun (_, n) -> n + 1))) :: rest;
        ms *. Float.of_int depth
    | [] -> 10. *. Float.of_int depth
  in
  let search ~cut =
    attempts := [];
    cycled := 0;
    let report = ref None in
    let _ctx, _routine =
      Autotune.Seam.with_value Autotune.seeded_hopeless_cut cut @@ fun () ->
      Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
          attempts := (label, None) :: !attempts)
      @@ fun () ->
      Autotune.Seam.with_value Autotune.on_batch_wall clock @@ fun () ->
      Scripted_search.with_settings [ ("autotune_finals_margin", "0") ] @@ fun () ->
      Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
        ~timing:Autotune.Isolated ~cache_dir:"" ~log:false
        ~report:(fun r -> report := Some r)
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    let windows =
      List.rev !attempts
      |> List.filter_map ~f:(fun (label, reads) ->
          match reads with
          | Some (ms, batches) when is_seeded_single label -> Some (label, ms, batches)
          | Some _ | None -> None)
    in
    (Option.value_exn ~here:[%here] !report, windows)
  in
  (* The cut is off by default (gh-ocannl-1697: its ratio lost tuned step p50 on gpt2_mini/HIP): the
     same clock with the seam at its default cuts nothing. *)
  let uncut = if seeds_here then Some (search ~cut:false) else None in
  claim uncut "with the seeded cut at its default, every seeded single takes its whole window"
    (fun label (r, windows) ->
      p_all label windows ~f:(fun (_, _, batches) ->
          batches >= 16 && r.Autotune.seeded_hopeless = 0));
  let searched = if seeds_here then Some (search ~cut:true) else None in
  Option.iter searched ~f:(fun ((r : Autotune.report), windows) ->
      List.iter windows ~f:(fun (label, ms, batches) ->
          Stdio.eprintf "hopeless search (not part of the golden): %s reads %g ms, %d batches\n%!"
            label ms batches);
      Stdio.eprintf
        "hopeless search (not part of the golden): %d seeded windows cut, %d cut in all, best %g ms\n\
         %!"
        r.seeded_hopeless r.timings_hopeless r.best_ms);
  let claim = claim searched in
  let reading ms windows = List.filter windows ~f:(fun (_, m, _) -> Float.equal m ms) in
  claim "a seeded single over 1.05 times the plan is cut after three samples"
    (fun label (_, windows) ->
      p_all label (reading cut_ms windows) ~f:(fun (_, _, batches) ->
          batches = Autotune.hopeless_min_samples));
  claim "a split single is among the cut" (fun label (_, windows) ->
      p_exists label (reading cut_ms windows) ~f:(fun (l, _, _) -> is_resplit l));
  claim "a segment single is among the cut" (fun label (_, windows) ->
      p_exists label (reading cut_ms windows) ~f:(fun (l, _, _) -> is_swap l));
  claim "a seeded single under 1.05 times the plan takes its whole window"
    (fun label (_, windows) ->
      p_all label (reading near_ms windows) ~f:(fun (_, _, batches) -> batches >= 16));
  claim "a segment single takes its whole window" (fun label (_, windows) ->
      p_exists label (reading near_ms windows) ~f:(fun (l, _, _) -> is_swap l));
  claim "the report counts the cut seeded windows, and the search crowns the plan"
    (fun label (r, windows) ->
      let cut = List.count windows ~f:(fun (_, _, batches) -> batches < 16) in
      p label
        (r.Autotune.seeded_hopeless = cut && r.timings_hopeless >= cut && is_seeded r.best_label
       && Float.equal r.best_ms 10.))

(* gh-ocannl-1697: a seeded single calibrates at the seeded plan's settled depth
   ({!Autotune.prior_depth_agrees}), on a scripted clock under [Queued] timing. Every launch is
   linear in the batch depth: the plan 15 ms, over the batch target and under twice it, so its own
   calibration takes the depth-2 probe and settles at depth 1; the seeded singles 15.5 ms, so none
   joins; every other candidate 50 ms. Each timing call is attributed to the attempt in flight. *)
let () =
  let plan_ms = 15. and single_ms = 15.5 in
  let attempts = ref [] and calls = ref [] in
  let clock ~depth ~ordinal:_ _wall =
    let ms =
      match !attempts with
      | label :: _ when is_seeded label -> plan_ms
      | label :: _ when is_seeded_single label -> single_ms
      | _ :: _ | [] -> 50.
    in
    ms *. Float.of_int depth
  in
  let search () =
    let report = ref None in
    let _ctx, _routine =
      Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
          attempts := label :: !attempts)
      @@ fun () ->
      Autotune.Seam.with_value Autotune.on_batch_wall clock @@ fun () ->
      Autotune.with_timing_call_observer (fun (c : Autotune.timing_call) ->
          Option.iter (List.hd !attempts) ~f:(fun label -> calls := (label, c) :: !calls))
      @@ fun () ->
      Scripted_search.with_settings [ ("autotune_finals_margin", "0") ] @@ fun () ->
      Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
        ~timing:Autotune.Queued ~cache_dir:"" ~log:false
        ~report:(fun r -> report := Some r)
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    (Option.value_exn ~here:[%here] !report, List.rev !calls)
  in
  let searched = if seeds_here then Some (search ()) else None in
  Option.iter searched ~f:(fun (_, calls) ->
      List.iter calls ~f:(fun (label, (c : Autotune.timing_call)) ->
          Stdio.eprintf
            "prior-depth search (not part of the golden): %s settles %s at depth %d, %d probes\n%!"
            label
            (Autotune.batch_settle_name c.decision.settle)
            c.decision.depth
            (List.length c.calibration_probes)));
  let claim = claim searched in
  let of_kind kind calls = List.filter calls ~f:(fun (label, _) -> kind label) in
  claim "the plan calibrates itself: its depth-2 probe settles depth 1" (fun label (_, calls) ->
      p_exists label (of_kind is_seeded calls) ~f:(fun (_, (c : Autotune.timing_call)) ->
          Poly.equal c.decision.settle Autotune.Singles_reach_target
          && c.decision.depth = 1
          && List.length c.calibration_probes = 1));
  claim "every seeded single settles at the plan's depth with no probe" (fun label (_, calls) ->
      p_all label (of_kind is_seeded_single calls) ~f:(fun (_, (c : Autotune.timing_call)) ->
          Poly.equal c.decision.settle Autotune.Prior_depth
          && c.decision.depth = 1
          && List.is_empty c.calibration_probes));
  claim "a segment single is timed" (fun label (_, calls) ->
      p_exists label calls ~f:(fun (l, _) -> is_swap l));
  claim "a split single is timed" (fun label (_, calls) ->
      p_exists label calls ~f:(fun (l, _) -> is_resplit l));
  claim "no other candidate settles at a prior depth" (fun label (_, calls) ->
      p_none label
        (List.filter calls ~f:(fun (l, _) -> not (is_seeded_single l)))
        ~f:(fun (_, (c : Autotune.timing_call)) ->
          Poly.equal c.decision.settle Autotune.Prior_depth));
  claim "the search crowns the plan at its reading" (fun label (r, _) ->
      p label (is_seeded r.Autotune.best_label && Float.equal r.best_ms plan_ms))
